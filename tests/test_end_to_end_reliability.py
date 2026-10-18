"""End-to-end backend reliability, workflow tracing, and recovery tests (Milestone 24).

Validates:
1. Complete scholarly research workflow:
   - Discovery search -> paper metadata -> explicit import into caller's collection.
   - Document extraction -> chunking -> vector indexing -> status completed.
   - Grounded RAG query -> hybrid retrieval -> answer generation with evidence attribution.
   - Structured paper analysis.
   - Idempotent deduplication (already_imported).
2. Multi-tenant authorization & isolation:
   - Cross-user collection import rejection (403 Forbidden).
   - Cross-user collection, document, and paper analysis access rejection (403 Forbidden).
   - Unauthenticated caller rejection (401 Unauthorized).
3. Partial failure atomicity, cleanup, and recovery:
   - Non-durable mode: vector store failure triggers atomic DB rollback, partial vector purge,
     and temp file cleanup.
   - Durable mode: vector store failure marks document as 'failed', followed by successful
     out-of-band reindexing recovery.
4. Production Health & Readiness probe behavior:
   - 200 'ready' when database, auth, and research services are operational.
   - 503 'unavailable' when required database connectivity is interrupted.
   - 200 'ok' on liveness probe /health even during database blips (true decoupling).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.api.app import create_app
from nexora.api.errors import (
    AuthenticationError,
    DocumentNotFoundError,
    ForbiddenError,
    InvalidRequestError,
)
from nexora.auth.bootstrap import bootstrap_user_and_key
from nexora.auth.models import AuthenticatedUser
from nexora.auth.service import AuthenticationService
from nexora.database.models.base import Base
from nexora.database.session import create_session_factory
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document, IndexingStatus
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.paper import PaperRepository
from nexora.discovery.downloader import DownloadedPDF, SafePDFDownloader
from nexora.discovery.models import (
    DiscoveryImportResult,
    ScholarlyPaperMetadata,
    ScholarlySearchQuery,
    ScholarlySearchResult,
)
from nexora.discovery.service import DiscoveryService
from nexora.document.ingestion import (
    DocumentIngestionService,
    IngestionResult,
    VectorIndexingError,
)
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.rag.generator import AnswerGenerator, GeneratedAnswer
from nexora.rag.models import AttributedSource, RAGResponse
from nexora.rag.pipeline import RAGPipeline, create_rag_pipeline
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk
from nexora.retrieval.service import RetrievalService

VALID_PDF_BYTES = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< >>\n%%EOF"


# ---------------------------------------------------------------------------
# Test Doubles
# ---------------------------------------------------------------------------

class DeterministicMockEmbeddingService:
    """Deterministic offline embedding service producing unit vectors."""

    dimension = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.25, 0.25, 0.25, 0.25] for _ in texts]

    def embed_text(self, text: str) -> list[float]:
        return [0.25, 0.25, 0.25, 0.25]


class DeterministicMockAnswerGenerator:
    """Deterministic offline answer generator for grounded RAG testing."""

    is_mock: bool = True
    model_name: str = "mock-deterministic-llm"

    def generate(
        self,
        query: str,
        context: Any,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GeneratedAnswer:
        items = getattr(context, "items", [])
        citations = " ".join(f"[E{i + 1}]" for i, _ in enumerate(items[:2]))
        excerpts = " ".join(getattr(item, "text", "")[:100] for item in items[:2])
        return GeneratedAnswer(
            text=f"LoRA frozen weights adaptation. {excerpts} {citations}".strip(),
            model=model or self.model_name,
            is_mock=True,
            usage={"input_tokens": 150, "output_tokens": 60, "total_tokens": 210},
        )


class MockDocumentPipeline:
    """Mock document extraction pipeline returning structured chunks with metadata."""

    def process_pdf(self, file_path: str) -> ProcessedDocument:
        chunk1 = DocumentChunk(
            text="Low-Rank Adaptation (LoRA) freezes pre-trained model weights and injects trainable rank decomposition matrices.",
            embedding=[0.25, 0.25, 0.25, 0.25],
            index=0,
        )
        chunk2 = DocumentChunk(
            text="LoRA reduces the number of trainable parameters by 10,000 times and GPU memory requirement by 3 times.",
            embedding=[0.25, 0.25, 0.25, 0.25],
            index=1,
        )
        return ProcessedDocument(
            document_id=str(uuid4()),
            source=file_path,
            chunks=[chunk1, chunk2],
            metadata={"title": "LoRA: Low-Rank Adaptation of Large Language Models", "chunk_pages": [1, 2]},
        )


class MockSafeDownloader:
    """Mock safe downloader writing deterministic valid PDF bytes to temporary files."""

    def __init__(self, pdf_bytes: bytes = VALID_PDF_BYTES) -> None:
        self.pdf_bytes = pdf_bytes
        self.download_calls: list[str] = []
        self.created_temp_files: list[Path] = []

    def download_arxiv_pdf(
        self, arxiv_id: str, *, destination_dir: Path | None = None
    ) -> DownloadedPDF:
        import tempfile

        self.download_calls.append(arxiv_id)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".pdf",
            dir=str(destination_dir) if destination_dir else None,
            delete=False,
        )
        tmp.write(self.pdf_bytes)
        tmp.close()
        path = Path(tmp.name)
        self.created_temp_files.append(path)
        return DownloadedPDF(
            path=path,
            file_size=len(self.pdf_bytes),
            arxiv_id=arxiv_id,
            source_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        )


class MockArxivProvider:
    """Mock scholarly provider serving normalized metadata for research papers."""

    def __init__(self) -> None:
        self.paper = ScholarlyPaperMetadata(
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            authors=("Edward Hu", "Yelong Shen", "Phillip Wallis"),
            abstract="We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights...",
            doi="10.48550/arXiv.2106.09685",
            external_id="2106.09685",
            pdf_url="https://arxiv.org/pdf/2106.09685.pdf",
            primary_category="cs.CL",
            categories=("cs.CL", "cs.AI"),
            publication_date=datetime.date(2021, 6, 17),
            abstract_url="https://arxiv.org/abs/2106.09685",
        )

    def search(self, query: ScholarlySearchQuery) -> ScholarlySearchResult:
        return ScholarlySearchResult(
            provider="arxiv",
            query=query.query,
            total_results=1,
            offset=0,
            limit=10,
            papers=(self.paper,),
        )

    def get_paper(self, external_id: str) -> ScholarlyPaperMetadata | None:
        if "2106.09685" in external_id:
            return self.paper
        return None


class MockAnalysisService:
    """Mock paper analysis service producing structured academic sections."""

    def analyze_paper(self, paper_id: UUID, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_id": str(paper_id),
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "problem_and_objectives": {"summary": "Overcoming parameter-heavy fine-tuning of LLMs."},
                "methodology_and_algorithms": {"summary": "Injecting low-rank decomposition matrices."},
                "models_and_datasets": {"summary": "Evaluated on RoBERTa, DeBERTa, and GPT-3."},
                "experiments_and_metrics": {"summary": "Evaluated accuracy and perplexity across benchmarks."},
                "results_and_findings": {"summary": "Matches or exceeds fine-tuning quality with 10000x fewer params."},
                "limitations": {"summary": "Batching overhead when serving multiple distinct adapters."},
                "future_work": {"summary": "Extending to cross-attention layers and quantized weights."},
                "evidence_count": 2,
                "model": "mock-analysis-llm",
                "usage": None,
            }
        )


class MockResearchApplicationService:
    """Mock ResearchApplicationService adapting ask(request) to RAGPipeline."""

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self._rag_pipeline = rag_pipeline

    def ask(self, request: Any) -> Any:
        from nexora.research.models import ResearchAnswer

        query = request.query if hasattr(request, "query") else str(request)
        resp = self._rag_pipeline.answer(query)
        return ResearchAnswer(
            text=resp.answer,
            query=query,
            model=resp.model,
            usage=resp.usage,
            evidence_count=resp.evidence_count,
        )

    def verify(self, request: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "text": "Claim verified against literature.",
                "query": getattr(request, "query", ""),
                "model": "mock-verify-llm",
                "usage": None,
                "evidence_count": 1,
                "verification": {"supported": True, "confidence": 0.95},
            }
        )


class MockGapService:
    """Mock research gap discovery service."""

    def discover_gaps_for_papers(self, paper_ids: list[str]) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_ids": paper_ids,
                "collection_id": None,
                "documented_gaps": [],
                "inferred_gaps": [],
                "total_gaps_count": 0,
                "summary": "No significant gaps found.",
                "limitations": [],
                "model": "mock-gap-llm",
                "usage": None,
            }
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_db_factory():
    """In-memory SQLite database initialized with all Alembic/SQLAlchemy models."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return create_session_factory(engine)


@pytest.fixture
def test_environment(sqlite_db_factory):
    """Provisions two distinct users with workspaces, collections, and API keys."""
    # User 1
    user1, ws1, key_obj1, raw_key1 = bootstrap_user_and_key(
        sqlite_db_factory,
        email="alice@research.edu",
        workspace_name="Alice Lab",
        key_name="Alice Production Key",
    )
    with sqlite_db_factory() as session:
        col1 = Collection(workspace_id=ws1.id, name="Alice NLP Collection")
        session.add(col1)
        session.commit()
        session.refresh(col1)
        col1_id = col1.id

    # User 2
    user2, ws2, key_obj2, raw_key2 = bootstrap_user_and_key(
        sqlite_db_factory,
        email="bob@institution.org",
        workspace_name="Bob Lab",
        key_name="Bob Production Key",
    )
    with sqlite_db_factory() as session:
        col2 = Collection(workspace_id=ws2.id, name="Bob Robotics Collection")
        session.add(col2)
        session.commit()
        session.refresh(col2)
        col2_id = col2.id

    vector_store = InMemoryVectorStore(dimension=4)
    embedding_service = DeterministicMockEmbeddingService()
    retrieval_service = RetrievalService(embedding_service, vector_store)
    generator = DeterministicMockAnswerGenerator()
    rag_pipeline = create_rag_pipeline(
        retrieval_service=retrieval_service,
        generator=generator,
        default_mode="hybrid",
    )
    research_service = MockResearchApplicationService(rag_pipeline)

    ingestion_service = DocumentIngestionService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        session_factory=sqlite_db_factory,
    )

    downloader = MockSafeDownloader()
    provider = MockArxivProvider()
    doc_pipeline = MockDocumentPipeline()

    discovery_service = DiscoveryService(
        provider=provider,
        downloader=downloader,
        session_factory=sqlite_db_factory,
        ingestion_service=ingestion_service,
        document_pipeline=doc_pipeline,
    )

    analysis_service = MockAnalysisService()
    gap_service = MockGapService()
    auth_service = AuthenticationService(session_factory=sqlite_db_factory)

    app = create_app(
        research_service=research_service,
        rag_pipeline=rag_pipeline,
        ingestion_service=ingestion_service,
        discovery_service=discovery_service,
        analysis_service=analysis_service,
        gap_service=gap_service,
        session_factory=sqlite_db_factory,
        auth_service=auth_service,
        enforce_auth=True,
    )

    return SimpleNamespace(
        app=app,
        client=TestClient(app),
        sqlite_db_factory=sqlite_db_factory,
        vector_store=vector_store,
        embedding_service=embedding_service,
        ingestion_service=ingestion_service,
        discovery_service=discovery_service,
        downloader=downloader,
        user1=user1,
        col1=col1,
        key1=raw_key1,
        user2=user2,
        col2=col2,
        key2=raw_key2,
    )


# ---------------------------------------------------------------------------
# End-to-End Reliability Tests
# ---------------------------------------------------------------------------

def test_complete_scholarly_workflow_end_to_end(test_environment):
    """Trace complete research workflow: search -> fetch -> import -> chunk -> index -> query -> analyze -> dedup."""
    client = test_environment.client
    key1 = test_environment.key1
    col1_id = str(test_environment.col1.id)
    headers = {"Authorization": f"Bearer {key1}"}

    # Step 1: Scholarly Discovery Search
    res_search = client.get("/api/v1/discovery/search?query=LoRA", headers=headers)
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert search_data["total_results"] == 1
    assert search_data["papers"][0]["external_id"] == "2106.09685"
    assert "LoRA" in search_data["papers"][0]["title"]

    # Step 2: Retrieve Paper Details
    res_paper = client.get("/api/v1/discovery/paper/2106.09685", headers=headers)
    assert res_paper.status_code == 200
    paper_data = res_paper.json()
    assert paper_data["external_id"] == "2106.09685"
    assert paper_data["doi"] == "10.48550/arXiv.2106.09685"

    # Step 3: Explicit Import into Caller's Collection
    res_import = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": col1_id, "external_id": "2106.09685"},
        headers=headers,
    )
    assert res_import.status_code == 200
    import_data = res_import.json()
    assert import_data["status"] == "imported"
    assert import_data["collection_id"] == col1_id
    assert import_data["external_id"] == "2106.09685"
    assert import_data["chunks_count"] == 2
    assert import_data["indexed_chunks_count"] == 2
    doc_id = import_data["document_id"]
    paper_id = import_data["paper_id"]
    assert doc_id is not None
    assert paper_id is not None

    # Verify vector store holds the 2 indexed vectors
    assert test_environment.vector_store.count() == 2

    # Step 4: Verify Document Details and Indexing Status
    res_doc = client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert res_doc.status_code == 200
    doc_details = res_doc.json()
    assert doc_details["document_id"] == doc_id
    assert doc_details["paper_id"] == paper_id
    assert doc_details["indexing_status"] == "completed"
    assert doc_details["file_size"] > 0
    assert doc_details["checksum"] is not None

    # Step 5: Grounded RAG Query across Indexed Research Literature
    res_rag = client.post(
        "/api/v1/research/query",
        json={"query": "What is LoRA and how does it work?", "retrieval_mode": "vector"},
        headers=headers,
    )
    assert res_rag.status_code == 200
    rag_data = res_rag.json()
    assert "LoRA" in rag_data["answer"]
    assert rag_data["evidence_count"] >= 1
    assert len(rag_data["sources"]) >= 1
    assert rag_data["model"] == "mock-deterministic-llm"

    # Step 6: Paper Analysis
    res_analysis = client.post(f"/api/v1/papers/{paper_id}/analyze", headers=headers)
    assert res_analysis.status_code == 200
    analysis_data = res_analysis.json()
    assert analysis_data["paper_id"] == paper_id
    assert "methodology_and_algorithms" in analysis_data
    assert "results_and_findings" in analysis_data

    # Step 7: Idempotent Deduplication
    res_dedup = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": col1_id, "external_id": "2106.09685"},
        headers=headers,
    )
    assert res_dedup.status_code == 200
    dedup_data = res_dedup.json()
    assert dedup_data["status"] == "already_imported"
    assert dedup_data["document_id"] == doc_id
    assert dedup_data["chunks_count"] == 2
    # Verify no duplicate vectors added
    assert test_environment.vector_store.count() == 2


def test_multi_tenant_authorization_isolation(test_environment):
    """Verify that User 2 cannot access, import into, or analyze User 1's resources."""
    client = test_environment.client
    key1 = test_environment.key1
    key2 = test_environment.key2
    col1_id = str(test_environment.col1.id)

    # 1. User 1 successfully imports paper into Collection 1
    res_import = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": col1_id, "external_id": "2106.09685"},
        headers={"Authorization": f"Bearer {key1}"},
    )
    assert res_import.status_code == 200
    doc_id = res_import.json()["document_id"]
    paper_id = res_import.json()["paper_id"]

    # 2. User 2 attempts to import into User 1's collection
    res_cross_import = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": col1_id, "external_id": "2106.09685"},
        headers={"Authorization": f"Bearer {key2}"},
    )
    assert res_cross_import.status_code == 403
    assert "forbidden" in res_cross_import.json()["error"]["code"]

    # 3. User 2 attempts to access User 1's collection metadata
    res_cross_col = client.get(
        f"/api/v1/collections/{col1_id}",
        headers={"Authorization": f"Bearer {key2}"},
    )
    assert res_cross_col.status_code == 403

    # 4. User 2 attempts to access User 1's document metadata
    res_cross_doc = client.get(
        f"/api/v1/documents/{doc_id}",
        headers={"Authorization": f"Bearer {key2}"},
    )
    assert res_cross_doc.status_code == 403

    # 5. User 2 attempts to analyze User 1's paper
    res_cross_analyze = client.post(
        f"/api/v1/papers/{paper_id}/analyze",
        headers={"Authorization": f"Bearer {key2}"},
    )
    assert res_cross_analyze.status_code == 403

    # 6. User 2 attempts to discover research gaps on User 1's paper
    res_cross_gaps = client.post(
        f"/api/v1/papers/{paper_id}/gaps",
        headers={"Authorization": f"Bearer {key2}"},
    )
    assert res_cross_gaps.status_code == 403

    # 7. User 2 attempts claim verification on User 1's paper
    res_cross_verify = client.post(
        "/api/v1/research/verify",
        json={"query": "LoRA uses low rank matrices", "expected_paper_id": str(paper_id)},
        headers={"Authorization": f"Bearer {key2}"},
    )
    assert res_cross_verify.status_code == 403

    # 8. Unauthenticated caller access rejection
    res_unauth = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": col1_id, "external_id": "2106.09685"},
    )
    assert res_unauth.status_code == 401
    assert "www-authenticate" in res_unauth.headers


def test_partial_failure_atomic_rollback_non_durable(test_environment, monkeypatch):
    """Verify that vector indexing failure in standard mode rolls back DB and purges partial vectors."""
    client = test_environment.client
    key1 = test_environment.key1
    col1_id = str(test_environment.col1.id)

    # Force vector store to fail during upsert
    def failing_upsert(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Simulated vector storage connection drop")

    monkeypatch.setattr(test_environment.vector_store, "upsert", failing_upsert)

    res_fail = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": col1_id, "external_id": "2106.09685", "durable": False},
        headers={"Authorization": f"Bearer {key1}"},
    )
    assert res_fail.status_code == 500
    assert res_fail.json()["error"]["code"] == "indexing_failed"

    # Verify no Document records exist in database for this paper
    with test_environment.sqlite_db_factory() as session:
        docs = session.query(Document).all()
        assert len(docs) == 0

        chunks = session.query(DocumentChunkRecord).all()
        assert len(chunks) == 0

    # Verify vector store is empty
    assert test_environment.vector_store.count() == 0

    # Verify all temporary files created by downloader were cleanly removed
    for path in test_environment.downloader.created_temp_files:
        assert not path.exists()


def test_durable_failure_and_reindex_recovery(test_environment, monkeypatch):
    """Verify durable mode preserves failed document and out-of-band reindex recovers it."""
    client = test_environment.client
    key1 = test_environment.key1
    col1_id = str(test_environment.col1.id)

    # Simulate vector store failure during initial ingestion
    fail_state = {"should_fail": True}
    real_upsert = test_environment.vector_store.upsert

    def maybe_failing_upsert(records: Any) -> None:
        if fail_state["should_fail"]:
            raise RuntimeError("Transient vector store outage")
        real_upsert(records)

    monkeypatch.setattr(test_environment.vector_store, "upsert", maybe_failing_upsert)

    res_fail = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": col1_id, "external_id": "2106.09685", "durable": True},
        headers={"Authorization": f"Bearer {key1}"},
    )
    assert res_fail.status_code == 500
    assert res_fail.json()["error"]["code"] == "indexing_failed"

    # In durable mode, document record was preserved with status 'failed'
    with test_environment.sqlite_db_factory() as session:
        doc = session.query(Document).first()
        assert doc is not None
        assert doc.indexing_status == IndexingStatus.FAILED.value
        assert "Transient vector store outage" in (doc.indexing_error or "")
        doc_id = doc.id

    # Restore vector store and perform recovery / reindexing
    fail_state["should_fail"] = False
    recovery_result = test_environment.ingestion_service.reindex_document(doc_id)

    assert recovery_result.indexed_chunks_count == 2
    assert test_environment.vector_store.count() == 2

    # Verify database status is now completed
    with test_environment.sqlite_db_factory() as session:
        updated_doc = session.query(Document).filter_by(id=doc_id).first()
        assert updated_doc.indexing_status == IndexingStatus.COMPLETED.value
        assert updated_doc.indexing_error is None


def test_readiness_probe_operational_and_failure_reporting(test_environment, monkeypatch):
    """Verify /ready reports ready status when dependencies are healthy and 503 when broken."""
    client = test_environment.client

    # 1. Healthy state
    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    body = res_ready.json()
    assert body["status"] == "ready"
    assert body["checks"]["database"]["status"] == "ready"
    assert body["checks"]["auth"]["status"] == "ready"
    assert body["checks"]["research"]["status"] == "ready"

    # Alias /health/ready
    res_ready_alias = client.get("/health/ready")
    assert res_ready_alias.status_code == 200
    assert res_ready_alias.json()["status"] == "ready"

    # 2. Database outage state
    def broken_session_factory():
        class BrokenSession:
            def __enter__(self):
                raise RuntimeError("Database connection pool exhausted")
            def __exit__(self, *args):
                pass
        return BrokenSession()

    test_environment.app.state.session_factory = broken_session_factory

    res_broken = client.get("/ready")
    assert res_broken.status_code == 503
    broken_body = res_broken.json()
    assert broken_body["status"] == "unavailable"
    assert broken_body["checks"]["database"]["status"] == "unavailable"
    assert broken_body["checks"]["database"]["error"] == "RuntimeError"

    # 3. Liveness probe /health remains 200 OK during database blip
    res_live = client.get("/health")
    assert res_live.status_code == 200
    assert res_live.json() == {"status": "ok"}
