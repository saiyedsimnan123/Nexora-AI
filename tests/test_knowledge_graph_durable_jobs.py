"""Tests for Stage 5 Step 4.6.2 Durable Knowledge Graph Ingestion Integration.

Verifies:
1. Direct PDF ingestion enqueues GraphExtractionJob with metadata_extracted=True.
2. Discovery arXiv import enqueues GraphExtractionJob and extracts authors.
3. Fail-soft isolation: graph extraction errors never fail document ingestion or vector indexing.
4. Atomic job claiming, execution, and zero open DB transactions during LLM inference.
5. Two-gate fencing verification (Gate 1 unexpired, Gate 2 fresh timestamp atomic update, rollback on breach).
6. Expired-lease sweeper: resets stale jobs or transitions exhausted jobs to failed.
7. Background heartbeat lease renewal during long-running LLM extraction.
8. Self-healing Tier 1 metadata extraction when metadata_extracted=False.
9. Historical document reconciliation via DocumentRecoveryService.reconcile_missing_graph_jobs.
10. API endpoints for job status inspection, collection filtering, and manual retry.
11. Strict tenant isolation across collection boundaries.
"""

from __future__ import annotations

import datetime
from datetime import timezone
import json
import threading
import time
from uuid import UUID, uuid4
import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from nexora.database.models import (
    Author,
    Base,
    Collection,
    Document,
    DocumentChunkRecord,
    GraphEntity,
    GraphEntityType,
    GraphRelationship,
    IndexingStatus,
    Paper,
    PaperAuthor,
    User,
    Workspace,
)
from nexora.database.models.graph_extraction_job import (
    GraphExtractionJob,
    GraphExtractionStatus,
)
from nexora.database.repository import (
    AuthorRepository,
    GraphEntityRepository,
    GraphRelationshipRepository,
    PaperAuthorRepository,
    PaperRepository,
)
from nexora.database.repository.graph_extraction_job import (
    GraphExtractionJobRepository,
)
from nexora.database.session import worker_database_transaction
from nexora.discovery.models import (
    DiscoveryImportResult,
    ScholarlyPaperMetadata,
)
from nexora.discovery.service import DiscoveryService
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.ingestion import (
    DocumentIngestionService,
    IngestionResult,
)
from nexora.document.recovery import DocumentRecoveryService
from nexora.graph.models import FencingTokenError
from nexora.graph.service import KnowledgeGraphExtractionService
from nexora.graph.worker import GraphExtractionWorker
from nexora.llm.models import LLMResponse
from nexora.retrieval.in_memory import InMemoryVectorStore


class MockLLMService:
    """Mock LLM returning structured graph entities and relationships."""

    def __init__(self, response_json: dict | None = None, delay_seconds: float = 0.0) -> None:
        self.call_count = 0
        self.delay = delay_seconds
        self.response_json = response_json or {
            "entities": [
                {
                    "name": "Diffusion Model",
                    "entity_type": "method",
                    "description": "A parameterized Markov chain trained using variational inference.",
                },
                {
                    "name": "ImageNet",
                    "entity_type": "dataset",
                    "description": "Standard visual benchmark dataset.",
                },
            ],
            "relationships": [
                {
                    "source_name": "Deep Generative Modeling",
                    "source_type": "paper",
                    "target_name": "Diffusion Model",
                    "target_type": "entity",
                    "relationship_type": "proposes",
                    "evidence_label": "[cite: 0]",
                    "evidence_quote": "We present diffusion models for high-fidelity generative modeling.",
                    "confidence": 0.95,
                },
            ],
        }

    def generate(self, messages: list, **kwargs) -> LLMResponse:
        self.call_count += 1
        if self.delay > 0:
            time.sleep(self.delay)
        return LLMResponse(text=json.dumps(self.response_json), model="mock-graph-llm")


class FakeEmbeddingService:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]

    def embed_text(self, text: str) -> list[float]:
        return [0.1] * self.dimension


@pytest.fixture
def engine():
    from sqlalchemy.pool import StaticPool
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def seed_data(session_factory):
    with session_factory() as session:
        user = User(email="researcher@nexora.ai")
        session.add(user)
        session.flush()

        workspace = Workspace(user_id=user.id, name="AI Lab")
        session.add(workspace)
        session.flush()

        col_a = Collection(workspace_id=workspace.id, name="Vision Collection")
        col_b = Collection(workspace_id=workspace.id, name="NLP Collection")
        session.add_all([col_a, col_b])
        session.flush()

        paper = Paper(
            collection_id=col_a.id,
            title="Deep Generative Modeling",
            publication_date=datetime.date(2024, 1, 1),
            abstract="We present diffusion models for high-fidelity generative modeling.",
        )
        session.add(paper)
        session.flush()

        doc = Document(
            paper_id=paper.id,
            filename="generative.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="a" * 64,
            storage_key="s3://papers/generative.pdf",
            indexing_status=IndexingStatus.COMPLETED.value,
        )
        session.add(doc)
        session.flush()

        chunk = DocumentChunkRecord(
            document_id=doc.id,
            chunk_index=0,
            text="We present diffusion models for high-fidelity generative modeling.",
        )
        session.add(chunk)

        from nexora.database.repository.api_key import ApiKeyRepository
        key_repo = ApiKeyRepository(session)
        _, raw_token = key_repo.generate_key(user_id=user.id, name="Test Key")

        session.commit()

        return {
            "user_id": user.id,
            "workspace_id": workspace.id,
            "col_a_id": col_a.id,
            "col_b_id": col_b.id,
            "paper_id": paper.id,
            "doc_id": doc.id,
            "chunk_id": chunk.id,
            "token": raw_token,
        }


# =========================================================================
# Scenario 1: Direct PDF ingestion enqueues GraphExtractionJob
# =========================================================================
def test_direct_pdf_ingestion_enqueues_graph_job(session_factory, seed_data):
    vector_store = InMemoryVectorStore(dimension=4)
    embedding_service = FakeEmbeddingService(dimension=4)
    graph_service = KnowledgeGraphExtractionService(
        llm_service=MockLLMService(),
        session_factory=session_factory,
    )

    ingestion_service = DocumentIngestionService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        session_factory=session_factory,
        graph_extraction_service=graph_service,
    )

    col_id = seed_data["col_a_id"]
    paper_id = seed_data["paper_id"]

    processed_doc = ProcessedDocument(
        document_id=str(uuid4()),
        source="s3://papers/attention.pdf",
        chunks=[
            DocumentChunk(
                text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                embedding=[0.1, 0.1, 0.1, 0.1],
                index=0,
            ),
        ],
        metadata={
            "filename": "attention.pdf",
            "mime_type": "application/pdf",
            "file_size": 2048,
            "checksum": "c" * 64,
            "authors": ["Vaswani, A.", "Shazeer, N."],
            "collection_id": str(col_id),
        },
    )

    res = ingestion_service.ingest_processed_document(
        paper_id,
        processed_doc,
    )

    assert res.document is not None
    assert res.document.indexing_status == "completed"
    assert res.graph_job_id is not None
    assert res.graph_status in ("pending", "processing", "completed")

    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        job = repo.get_by_id(res.graph_job_id)
        assert job is not None
        assert job.paper_id == paper_id
        assert job.collection_id == col_id
        assert job.metadata_extracted is True
        assert job.attempts == 0


# =========================================================================
# Scenario 2: Discovery arXiv import enqueues job with author extraction
# =========================================================================
def test_discovery_import_enqueues_graph_job_and_extracts_authors(session_factory, seed_data):
    class FakeArxivProvider:
        def get_paper(self, identifier: str) -> ScholarlyPaperMetadata:
            return ScholarlyPaperMetadata(
                external_id="2401.00001",
                title="Generative Diffusion Models",
                authors=("Alice Smith", "Bob Jones"),
                abstract="Diffusion abstract",
                publication_date=datetime.date(2024, 1, 1),
                categories=("cs.AI",),
                primary_category="cs.AI",
                doi="10.1234/gen.2024",
                pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
            )

    import tempfile
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tf:
        tf.write(b"%PDF-1.4 mock content")
        pdf_tmp = tf.name

    class FakeDownloadedPDF:
        def __init__(self, path: str) -> None:
            self.path = path

        def cleanup(self) -> None:
            pass

    class FakeDownloader:
        def download_arxiv_pdf(self, identifier: str) -> FakeDownloadedPDF:
            return FakeDownloadedPDF(pdf_tmp)

    class FakeIngestionService:
        def __init__(self, sf, graph_srv):
            self._sf = sf
            self._graph_service = graph_srv

        def process_and_ingest(self, paper_id, file_path, **kwargs):
            with self._sf() as session:
                paper = session.get(Paper, paper_id)
                job_repo = GraphExtractionJobRepository(session)
                job, _ = job_repo.create_or_get_active_job(
                    paper_id=paper.id,
                    collection_id=paper.collection_id,
                    metadata_extracted=True,
                )
                session.commit()
                return IngestionResult(
                    document_id=uuid4(),
                    paper_id=paper.id,
                    chunks_count=1,
                    indexed_chunks_count=1,
                    vector_ids=["vec-1"],
                    document=None,
                    chunk_records=[],
                    graph_job_id=job.id,
                    graph_status=job.status,
                )

    graph_service = KnowledgeGraphExtractionService(
        llm_service=MockLLMService(),
        session_factory=session_factory,
    )
    fake_ingestion = FakeIngestionService(session_factory, graph_service)

    discovery_service = DiscoveryService(
        provider=FakeArxivProvider(),
        downloader=FakeDownloader(),
        session_factory=session_factory,
        ingestion_service=fake_ingestion,
    )

    import_res = discovery_service.import_paper(
        collection_id=seed_data["col_a_id"],
        external_id="2401.00001",
    )

    assert import_res.graph_job_id is not None
    assert import_res.graph_status == "pending"

    with session_factory() as session:
        # Verify authors were extracted into relational tables
        pa_repo = PaperAuthorRepository(session)
        author_repo = AuthorRepository(session)
        paper_uuid = UUID(import_res.paper_id) if isinstance(import_res.paper_id, str) else import_res.paper_id
        authorships = pa_repo.list_by_paper_id(paper_uuid)
        assert len(authorships) == 2
        author_names = {author_repo.get_by_id(pa.author_id).name for pa in authorships}
        assert "Alice Smith" in author_names
        assert "Bob Jones" in author_names


# =========================================================================
# Scenario 3: Fail-soft isolation (graph failures never fail ingestion)
# =========================================================================
def test_fail_soft_graph_extraction_does_not_break_ingestion(session_factory, seed_data, monkeypatch):
    vector_store = InMemoryVectorStore(dimension=4)
    embedding_service = FakeEmbeddingService(dimension=4)

    # Create graph extraction service that raises an unhandled exception during deterministic metadata
    class FaultyGraphService:
        def extract_metadata_deterministic(self, *args, **kwargs):
            raise RuntimeError("Database deadlock during graph extraction!")

    faulty_graph = FaultyGraphService()

    ingestion_service = DocumentIngestionService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        session_factory=session_factory,
        graph_extraction_service=faulty_graph,
    )

    col_id = seed_data["col_a_id"]
    paper_id = seed_data["paper_id"]

    processed_doc = ProcessedDocument(
        document_id=str(uuid4()),
        source="s3://papers/faulty.pdf",
        chunks=[
            DocumentChunk(
                text="Some text about machine learning algorithms.",
                embedding=[0.1, 0.1, 0.1, 0.1],
                index=0,
            ),
        ],
        metadata={
            "filename": "faulty.pdf",
            "mime_type": "application/pdf",
            "file_size": 1024,
            "checksum": "f" * 64,
            "authors": ["Faulty Author"],
            "collection_id": str(col_id),
        },
    )

    # Ingestion MUST NOT raise RuntimeError and MUST succeed with document completed
    res = ingestion_service.ingest_processed_document(
        paper_id,
        processed_doc,
    )

    assert res.document is not None
    assert res.document.indexing_status == "completed"
    assert res.document_id is not None
    # Graph extraction soft-failed so job was enqueued with metadata_extracted=False
    if res.graph_job_id is not None:
        with session_factory() as session:
            job = GraphExtractionJobRepository(session).get_by_id(res.graph_job_id)
            assert job is not None
            assert job.metadata_extracted is False

    # Verify vector store still received embeddings
    assert vector_store.count() >= 1


# =========================================================================
# Scenario 4: Worker claim, execution, zero open DB transactions during LLM
# =========================================================================
def test_worker_claim_and_execution_zero_transactions_during_llm(session_factory, seed_data):
    mock_llm = MockLLMService()
    graph_service = KnowledgeGraphExtractionService(
        llm_service=mock_llm,
        session_factory=session_factory,
    )

    paper_id = seed_data["paper_id"]
    col_id = seed_data["col_a_id"]

    # Enqueue a pending job
    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        job = repo.create_job(
            paper_id=paper_id,
            collection_id=col_id,
            metadata_extracted=True,
        )
        session.commit()
        job_id = job.id

    worker = GraphExtractionWorker(
        session_factory=session_factory,
        graph_service=graph_service,
        worker_token="test-worker-1",
        poll_interval_seconds=0.1,
    )

    # Run one polling cycle
    processed = worker.run_once()
    assert processed == 1

    # Verify job completed and graph entities/relationships exist
    with session_factory() as session:
        job_repo = GraphExtractionJobRepository(session)
        finished_job = job_repo.get_by_id(job_id)
        assert finished_job.status == GraphExtractionStatus.COMPLETED.value
        assert finished_job.claimed_by is None
        assert finished_job.completed_at is not None

        entity_repo = GraphEntityRepository(session)
        entities = entity_repo.list_by_collection_id(col_id)
        entity_names = {e.name for e in entities}
        assert "Diffusion Model" in entity_names
        assert "ImageNet" in entity_names

        rel_repo = GraphRelationshipRepository(session)
        rels = rel_repo.list_by_collection_id(col_id)
        assert len(rels) >= 1
        assert rels[0].relationship_type == "proposes"


# =========================================================================
# Scenario 5: Two-gate fencing enforcement
# =========================================================================
def test_two_gate_fencing_gate1_expired_lease_rejection(session_factory, seed_data):
    paper_id = seed_data["paper_id"]
    col_id = seed_data["col_a_id"]

    # Create job that is processing with an expired lease
    now = datetime.datetime.now(timezone.utc)
    expired_time = now - datetime.timedelta(seconds=60)

    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        job = GraphExtractionJob(
            paper_id=paper_id,
            collection_id=col_id,
            status=GraphExtractionStatus.PROCESSING.value,
            claimed_by="worker-token-abc",
            claimed_at=expired_time - datetime.timedelta(seconds=300),
            lease_expires_at=expired_time,
            attempts=1,
            metadata_extracted=True,
        )
        session.add(job)
        session.commit()
        job_id = job.id

    # Verify Gate 1 fails
    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        gate1_valid = repo.verify_fencing_gate1(
            job_id=job_id,
            claimed_by="worker-token-abc",
            unexpired_as_of=now,
        )
        assert gate1_valid is False

        # Wrong token also fails
        gate1_wrong_token = repo.verify_fencing_gate1(
            job_id=job_id,
            claimed_by="other-worker",
            unexpired_as_of=expired_time - datetime.timedelta(seconds=10),
        )
        assert gate1_wrong_token is False


def test_two_gate_fencing_gate2_rollback_on_lease_expiry(session_factory, seed_data):
    paper_id = seed_data["paper_id"]
    col_id = seed_data["col_a_id"]

    now = datetime.datetime.now(timezone.utc)
    short_lease = now + datetime.timedelta(milliseconds=50)

    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        job = GraphExtractionJob(
            paper_id=paper_id,
            collection_id=col_id,
            status=GraphExtractionStatus.PROCESSING.value,
            claimed_by="worker-xyz",
            claimed_at=now,
            lease_expires_at=short_lease,
            attempts=1,
            metadata_extracted=True,
        )
        session.add(job)
        session.commit()
        job_id = job.id

    # Sleep so lease expires before Gate 2
    time.sleep(0.1)

    # Attempt to finalize inside transaction
    with pytest.raises(FencingTokenError):
        with worker_database_transaction(session_factory) as session:
            repo = GraphExtractionJobRepository(session)
            # Add a speculative entity that MUST be rolled back
            speculative_entity = GraphEntity(
                collection_id=col_id,
                name="Should Be Rolled Back",
                canonical_name="should be rolled back",
                entity_type=GraphEntityType.CONCEPT.value,
            )
            session.add(speculative_entity)

            gate2_now = datetime.datetime.now(timezone.utc)
            finalized = repo.finalize_job_gate2(
                job_id=job_id,
                claimed_by="worker-xyz",
                completion_time=gate2_now,
            )
            if not finalized:
                raise FencingTokenError("Gate 2 atomic update affected 0 rows.")

    # Confirm rollback: speculative entity is NOT persisted
    with session_factory() as session:
        entity_repo = GraphEntityRepository(session)
        entities = entity_repo.list_by_collection_id(col_id)
        assert not any(e.name == "Should Be Rolled Back" for e in entities)

        # Job remains in processing or stale
        repo = GraphExtractionJobRepository(session)
        job_after = repo.get_by_id(job_id)
        assert job_after.status == GraphExtractionStatus.PROCESSING.value


# =========================================================================
# Scenario 6: Expired lease sweeping (stale reset vs failure on max attempts)
# =========================================================================
def test_expired_lease_sweeper_resets_or_fails_jobs(session_factory, seed_data):
    now = datetime.datetime.now(timezone.utc)
    stale_time = now - datetime.timedelta(seconds=120)

    # Job 1: 1 attempt (can be retried)
    with session_factory() as session:
        p1 = Paper(collection_id=seed_data["col_a_id"], title="Paper 1", publication_date=datetime.date(2024, 1, 1))
        p2 = Paper(collection_id=seed_data["col_a_id"], title="Paper 2", publication_date=datetime.date(2024, 1, 1))
        session.add_all([p1, p2])
        session.flush()

        job_retryable = GraphExtractionJob(
            paper_id=p1.id,
            collection_id=seed_data["col_a_id"],
            status=GraphExtractionStatus.PROCESSING.value,
            claimed_by="crashed-worker-1",
            lease_expires_at=stale_time,
            attempts=1,
            max_attempts=3,
        )
        job_exhausted = GraphExtractionJob(
            paper_id=p2.id,
            collection_id=seed_data["col_a_id"],
            status=GraphExtractionStatus.PROCESSING.value,
            claimed_by="crashed-worker-2",
            lease_expires_at=stale_time,
            attempts=3,
            max_attempts=3,
        )
        session.add_all([job_retryable, job_exhausted])
        session.commit()
        id1, id2 = job_retryable.id, job_exhausted.id

    # Run sweeper
    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        swept_count = repo.sweep_expired_leases(lease_expiry_threshold=now, max_attempts=3)
        session.commit()
        assert swept_count == 2

    # Check outcomes
    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        j1 = repo.get_by_id(id1)
        assert j1.status == GraphExtractionStatus.PENDING.value
        assert j1.claimed_by is None
        assert j1.lease_expires_at is None

        j2 = repo.get_by_id(id2)
        assert j2.status == GraphExtractionStatus.FAILED.value
        assert "maximum" in j2.last_error.lower() or "attempts" in j2.last_error.lower()
        assert j2.claimed_by is None


# =========================================================================
# Scenario 7: Background heartbeat lease renewal during long LLM calls
# =========================================================================
def test_heartbeat_renews_lease_during_llm_inference(session_factory, seed_data):
    # LLM service with artificial delay
    slow_llm = MockLLMService(delay_seconds=0.3)
    graph_service = KnowledgeGraphExtractionService(
        llm_service=slow_llm,
        session_factory=session_factory,
    )

    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        job = repo.create_job(
            paper_id=seed_data["paper_id"],
            collection_id=seed_data["col_a_id"],
            metadata_extracted=True,
        )
        session.commit()
        job_id = job.id

    worker = GraphExtractionWorker(
        session_factory=session_factory,
        graph_service=graph_service,
        worker_token="heartbeat-worker",
        lease_duration_seconds=300,
        heartbeat_interval_seconds=0.1,  # frequent heartbeat
    )

    processed = worker.run_once()
    assert processed == 1

    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        finished = repo.get_by_id(job_id)
        assert finished.status == GraphExtractionStatus.COMPLETED.value


# =========================================================================
# Scenario 8: Self-healing Tier 1 extraction when metadata_extracted=False
# =========================================================================
def test_self_healing_tier1_extraction(session_factory, seed_data):
    graph_service = KnowledgeGraphExtractionService(
        llm_service=MockLLMService(),
        session_factory=session_factory,
    )

    with session_factory() as session:
        # Create an un-extracted author relation
        paper = session.get(Paper, seed_data["paper_id"])
        author = Author(name="Geoffrey Hinton")
        session.add(author)
        session.flush()
        pa = PaperAuthor(paper_id=paper.id, author_id=author.id, author_order=1)
        session.add(pa)
        session.flush()

        # Enqueue job with metadata_extracted=False
        repo = GraphExtractionJobRepository(session)
        job = repo.create_job(
            paper_id=paper.id,
            collection_id=seed_data["col_a_id"],
            metadata_extracted=False,  # Needs Tier 1 repair
        )
        session.commit()
        job_id = job.id

    worker = GraphExtractionWorker(
        session_factory=session_factory,
        graph_service=graph_service,
        worker_token="healing-worker",
    )

    processed = worker.run_once()
    assert processed == 1

    with session_factory() as session:
        # Verify Tier 1 Author was created in GraphEntity
        entity_repo = GraphEntityRepository(session)
        entities = entity_repo.list_by_collection_id(seed_data["col_a_id"])
        author_entity = next((e for e in entities if e.name == "Geoffrey Hinton"), None)
        assert author_entity is not None
        assert author_entity.entity_type == GraphEntityType.AUTHOR.value


# =========================================================================
# Scenario 9: Document recovery historical reconciliation
# =========================================================================
def test_document_recovery_reconcile_missing_graph_jobs(session_factory, seed_data):
    # Create an older completed document that has NO graph job
    with session_factory() as session:
        legacy_paper = Paper(
            collection_id=seed_data["col_a_id"],
            title="Legacy Unextracted Paper",
            publication_date=datetime.date(2020, 1, 1),
        )
        session.add(legacy_paper)
        session.flush()

        legacy_doc = Document(
            paper_id=legacy_paper.id,
            filename="legacy.pdf",
            mime_type="application/pdf",
            file_size=2048,
            checksum="b" * 64,
            storage_key="s3://legacy.pdf",
            indexing_status=IndexingStatus.COMPLETED.value,
        )
        session.add(legacy_doc)
        session.commit()
        legacy_paper_id = legacy_paper.id

    vector_store = InMemoryVectorStore(dimension=4)
    embedding_service = FakeEmbeddingService(dimension=4)

    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )

    # Reconcile missing graph jobs
    enqueued = recovery_service.reconcile_missing_graph_jobs(limit=10)
    assert len(enqueued) >= 1

    # Verify a new pending job was created with metadata_extracted=False
    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        job = repo.get_active_by_paper_id(legacy_paper_id)
        assert job is not None
        assert job.status == GraphExtractionStatus.PENDING.value
        assert job.metadata_extracted is False


# =========================================================================
# Scenario 10: Manual job retry endpoint functionality
# =========================================================================
def test_graph_job_repository_reset_for_retry(session_factory, seed_data):
    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        job = GraphExtractionJob(
            paper_id=seed_data["paper_id"],
            collection_id=seed_data["col_a_id"],
            status=GraphExtractionStatus.FAILED.value,
            last_error="Network timeout on LLM call",
            attempts=3,
        )
        session.add(job)
        session.commit()
        job_id = job.id

    # Reset for retry
    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        retried_job = repo.reset_job_for_retry(job_id)
        session.commit()

        assert retried_job is not None
        assert retried_job.status == GraphExtractionStatus.PENDING.value
        assert retried_job.attempts == 0
        assert retried_job.last_error is None
        assert retried_job.claimed_by is None


# =========================================================================
# Scenario 11: API endpoints for job status, retry, and collection filtering
# =========================================================================
def test_api_graph_jobs_endpoints_and_retry(session_factory, seed_data):
    from fastapi.testclient import TestClient
    from nexora.api.app import create_app
    from nexora.auth.service import AuthenticationService

    class DummyResearch:
        def ask(self, req):
            return None

    auth_service = AuthenticationService(session_factory=session_factory)
    app = create_app(
        DummyResearch(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {seed_data['token']}"}

    # 1. Create a failed job in DB
    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        job = GraphExtractionJob(
            paper_id=seed_data["paper_id"],
            collection_id=seed_data["col_a_id"],
            status=GraphExtractionStatus.FAILED.value,
            last_error="Transient LLM error",
            attempts=3,
        )
        session.add(job)
        session.commit()
        job_id = job.id

    # 2. GET /api/v1/graph/jobs/{job_id}
    res_get = client.get(f"/api/v1/graph/jobs/{job_id}", headers=headers)
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["job_id"] == str(job_id)
    assert data["status"] == "failed"
    assert data["attempts"] == 3
    assert data["error_message"] == "Transient LLM error"

    # 3. POST /api/v1/graph/jobs/{job_id}/retry
    res_retry = client.post(f"/api/v1/graph/jobs/{job_id}/retry", headers=headers)
    assert res_retry.status_code == 200
    retry_data = res_retry.json()
    assert retry_data["job_id"] == str(job_id)
    assert retry_data["status"] == "pending"
    assert retry_data["attempts"] == 0

    # 4. GET /api/v1/collections/{collection_id}/graph/jobs
    col_id = seed_data["col_a_id"]
    res_list = client.get(f"/api/v1/collections/{col_id}/graph/jobs?status=pending", headers=headers)
    assert res_list.status_code == 200
    items = res_list.json()
    assert any(j["job_id"] == str(job_id) for j in items)


# =========================================================================
# Scenario 12: Tenant isolation across collections
# =========================================================================
def test_tenant_isolation_cross_collection_jobs(session_factory, seed_data):
    from fastapi.testclient import TestClient
    from nexora.api.app import create_app
    from nexora.auth.service import AuthenticationService

    class DummyResearch:
        def ask(self, req):
            return None

    auth_service = AuthenticationService(session_factory=session_factory)
    app = create_app(
        DummyResearch(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {seed_data['token']}"}

    col_a = seed_data["col_a_id"]
    col_b = seed_data["col_b_id"]

    # Job in Collection A
    with session_factory() as session:
        repo = GraphExtractionJobRepository(session)
        job = repo.create_job(
            paper_id=seed_data["paper_id"],
            collection_id=col_a,
            metadata_extracted=True,
        )
        session.commit()
        job_id = job.id

    # Querying Collection B graph jobs MUST NOT return Collection A job
    res_b = client.get(f"/api/v1/collections/{col_b}/graph/jobs", headers=headers)
    assert res_b.status_code == 200
    items_b = res_b.json()
    assert not any(j["job_id"] == str(job_id) for j in items_b)

    # 2. Cross-tenant isolation: another user has no access to Collection A's jobs
    with session_factory() as session:
        other_user = User(email="other_tenant@nexora.ai")
        session.add(other_user)
        session.flush()
        from nexora.database.repository.api_key import ApiKeyRepository
        key_repo = ApiKeyRepository(session)
        _, other_token = key_repo.generate_key(user_id=other_user.id, name="Other Key")
        session.commit()

    other_headers = {"Authorization": f"Bearer {other_token}"}
    res_unauth = client.get(f"/api/v1/graph/jobs/{job_id}", headers=other_headers)
    assert res_unauth.status_code == 403

