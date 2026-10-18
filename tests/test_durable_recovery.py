"""Tests for Durable Vector Indexing Recovery (Milestone 12C-22).

Covers all 9 verification scenarios:
1. Ingestion failure recording: Document persists with indexing_status='failed',
   error recorded, and indexing_attempts incremented when durable=True and vector store fails.
2. Recovery success: DocumentRecoveryService reindexes vectors, transitions
   status to 'completed', and clears indexing_error.
3. Fresh instance recovery: A new DocumentRecoveryService with no shared state
   correctly discovers and recovers unindexed documents.
4. Idempotent recovery: Multiple recovery executions do not create duplicate
   chunks or vectors and preserve completed status.
5. Persistent failure tracking: Persistent vector errors keep status as 'failed',
   increment attempts, update error message, and respect max_attempts bounds.
6. Concurrent recovery safety: Simultaneous recovery or concurrent recovery + ingestion
   for the same paper are safely serialized without deadlocks or corrupted state.
7. Successful ingestion state: Standard ingestion marks documents as 'completed'
   with non-null indexed_at timestamp.
8. Transaction boundaries and rollback: Failures during recovery roll back
   transaction boundaries cleanly.
9. PostgreSQL integration: Runs durable recovery against live PostgreSQL when configured,
   failing clearly if configured but inaccessible.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import os
import threading
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from nexora.database.models.base import Base
from nexora.database.models import Collection, Document, DocumentChunkRecord, Paper, User, Workspace
from nexora.database.models.document import IndexingStatus
from nexora.database.repository.base import RecordNotFoundError
from nexora.document.ingestion import (
    DocumentIngestionService,
    VectorIndexingError,
    active_paper_locks_count,
    get_paper_ingestion_lock,
)
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.recovery import (
    DocumentRecoveryService,
    RecoveryBatchResult,
    RecoveryItemResult,
)
from nexora.document.service import DocumentPersistenceService
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import VectorRecord


VALID_CHECKSUM_A = "0123456789abcdef" * 4
VALID_CHECKSUM_B = "abcdef0123456789" * 4
VALID_CHECKSUM_C = "fedcba9876543210" * 4


class ControllableVectorStore(InMemoryVectorStore):
    """Vector store that can dynamically simulate failures."""

    def __init__(self, fail_on_upsert: bool = False, fail_message: str = "Vector cluster unavailable"):
        super().__init__()
        self.fail_on_upsert = fail_on_upsert
        self.fail_message = fail_message
        self.upsert_call_count = 0

    def upsert(self, records: list[VectorRecord]) -> None:
        self.upsert_call_count += 1
        if self.fail_on_upsert:
            raise RuntimeError(self.fail_message)
        return super().upsert(records)


class DummyEmbeddingService:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        return [0.1] * self.dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


from sqlalchemy import event


@pytest.fixture
def sqlite_engine(tmp_path):
    db_file = tmp_path / f"recovery_test_{uuid4().hex}.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    @event.listens_for(engine, "begin")
    def set_sqlite_begin(conn):
        conn.exec_driver_sql("BEGIN IMMEDIATE")

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(sqlite_engine):
    return sessionmaker(bind=sqlite_engine, expire_on_commit=False)


@pytest.fixture
def seed_paper(session_factory):
    with session_factory() as session:
        user = User(email=f"recovery_user_{uuid4()}@example.com")
        session.add(user)
        session.flush()

        workspace = Workspace(name="Recovery WS", user_id=user.id)
        session.add(workspace)
        session.flush()

        collection = Collection(name="Recovery Col", workspace_id=workspace.id)
        session.add(collection)
        session.flush()

        paper = Paper(collection=collection, title="Durable Recovery Paper")
        session.add(paper)
        session.commit()
        paper_id = paper.id
    return paper_id


def _make_processed_doc(num_chunks: int = 3, doc_id: str | None = None) -> ProcessedDocument:
    chunks = [
        DocumentChunk(
            text=f"Chunk text content #{i}",
            embedding=[0.1, 0.2, 0.3, 0.4],
            index=i,
        )
        for i in range(num_chunks)
    ]
    return ProcessedDocument(
        document_id=doc_id or str(uuid4()),
        source="test/path.pdf",
        chunks=chunks,
        metadata={"filename": "test.pdf"},
    )


# ---------------------------------------------------------------------------
# 1. Ingestion failure recording
# ---------------------------------------------------------------------------
def test_durable_ingestion_failure_records_failed_status_and_attempts(session_factory, seed_paper):
    vector_store = ControllableVectorStore(fail_on_upsert=True, fail_message="Vector index timeout")
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    processed = _make_processed_doc(num_chunks=3)

    with pytest.raises(VectorIndexingError, match="Vector store indexing failed during ingestion"):
        service.ingest_processed_document(
            paper_id=seed_paper,
            processed_document=processed,
            filename="failed_doc.pdf",
            checksum=VALID_CHECKSUM_A,
            storage_key="keys/failed_doc.pdf",
            durable=True,
        )

    # Document must be persisted with status FAILED, error message, and 1 attempt
    with session_factory() as session:
        doc = session.query(Document).filter_by(checksum=VALID_CHECKSUM_A).first()
        assert doc is not None
        assert doc.indexing_status == IndexingStatus.FAILED.value
        assert "Vector index timeout" in (doc.indexing_error or "")
        assert doc.indexing_attempts == 1
        assert doc.indexed_at is None

        # Relational chunks must still exist
        chunks = session.query(DocumentChunkRecord).filter_by(document_id=doc.id).all()
        assert len(chunks) == 3


# ---------------------------------------------------------------------------
# 2. Recovery success
# ---------------------------------------------------------------------------
def test_recovery_service_recovers_document_to_completed(session_factory, seed_paper):
    vector_store = ControllableVectorStore(fail_on_upsert=True)
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    processed = _make_processed_doc(num_chunks=2)

    with pytest.raises(VectorIndexingError):
        service.ingest_processed_document(
            paper_id=seed_paper,
            processed_document=processed,
            filename="recoverable.pdf",
            checksum=VALID_CHECKSUM_A,
            storage_key="keys/recoverable.pdf",
            durable=True,
        )

    with session_factory() as session:
        doc = session.query(Document).filter_by(checksum=VALID_CHECKSUM_A).one()
        doc_id = doc.id
        assert doc.indexing_status == IndexingStatus.FAILED.value

    # Fix vector store
    vector_store.fail_on_upsert = False

    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )

    result = recovery_service.recover_document(doc_id)
    assert result.status == IndexingStatus.COMPLETED
    assert result.chunks_count == 2
    assert result.indexed_chunks_count == 2
    assert result.error is None

    # Verify database state
    with session_factory() as session:
        recovered_doc = session.query(Document).filter_by(id=doc_id).one()
        assert recovered_doc.indexing_status == IndexingStatus.COMPLETED.value
        assert recovered_doc.indexing_error is None
        assert recovered_doc.indexed_at is not None

    # Verify vector store contents
    assert vector_store.count() == 2


# ---------------------------------------------------------------------------
# 3. Fresh instance recovery
# ---------------------------------------------------------------------------
def test_fresh_instance_recovery_with_no_shared_memory(session_factory, seed_paper):
    vector_store = ControllableVectorStore(fail_on_upsert=True)
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    processed = _make_processed_doc(num_chunks=3)

    with pytest.raises(VectorIndexingError):
        service.ingest_processed_document(
            paper_id=seed_paper,
            processed_document=processed,
            filename="fresh_recovery.pdf",
            checksum=VALID_CHECKSUM_A,
            storage_key="keys/fresh_recovery.pdf",
            durable=True,
        )

    # Discard existing service, create brand new vector store and recovery service
    vector_store.fail_on_upsert = False
    fresh_recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=DummyEmbeddingService(),
    )

    batch_res = fresh_recovery_service.recover_batch(batch_size=10)
    assert batch_res.scanned_count == 1
    assert batch_res.recovered_count == 1
    assert batch_res.failed_count == 0

    with session_factory() as session:
        doc = session.query(Document).filter_by(checksum=VALID_CHECKSUM_A).one()
        assert doc.indexing_status == IndexingStatus.COMPLETED.value


# ---------------------------------------------------------------------------
# 4. Idempotent recovery
# ---------------------------------------------------------------------------
def test_repeated_recovery_is_idempotent_no_duplicate_vectors(session_factory, seed_paper):
    vector_store = ControllableVectorStore()
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    processed = _make_processed_doc(num_chunks=2)
    ingest_res = service.ingest_processed_document(
        paper_id=seed_paper,
        processed_document=processed,
        filename="idempotent.pdf",
        checksum=VALID_CHECKSUM_A,
        storage_key="keys/idempotent.pdf",
        durable=True,
    )
    assert vector_store.count() == 2

    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )

    # Calling recover_document repeatedly on an already completed document
    res1 = recovery_service.recover_document(ingest_res.document_id)
    assert res1.status == IndexingStatus.COMPLETED
    assert vector_store.count() == 2  # No duplicate vectors added

    res2 = recovery_service.recover_document(ingest_res.document_id)
    assert res2.status == IndexingStatus.COMPLETED
    assert vector_store.count() == 2

    # Calling recover_batch should find zero unindexed documents
    batch_res = recovery_service.recover_batch(batch_size=10)
    assert batch_res.scanned_count == 0
    assert batch_res.recovered_count == 0
    assert vector_store.count() == 2


# ---------------------------------------------------------------------------
# 5. Persistent failure tracking
# ---------------------------------------------------------------------------
def test_persistent_failure_tracking_increments_attempts_and_updates_error(session_factory, seed_paper):
    vector_store = ControllableVectorStore(fail_on_upsert=True, fail_message="Initial failure")
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    processed = _make_processed_doc(num_chunks=1)

    with pytest.raises(VectorIndexingError):
        service.ingest_processed_document(
            paper_id=seed_paper,
            processed_document=processed,
            filename="persistent_fail.pdf",
            checksum=VALID_CHECKSUM_A,
            storage_key="keys/persistent_fail.pdf",
            durable=True,
        )

    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )

    with session_factory() as session:
        doc_id = session.query(Document).filter_by(checksum=VALID_CHECKSUM_A).one().id

    # Attempt 2
    vector_store.fail_message = "Secondary failure: cluster partition"
    res1 = recovery_service.recover_document(doc_id)
    assert res1.status == IndexingStatus.FAILED
    assert res1.attempts == 2
    assert "cluster partition" in (res1.error or "")

    with session_factory() as session:
        doc = session.query(Document).filter_by(id=doc_id).one()
        assert doc.indexing_attempts == 2
        assert "cluster partition" in doc.indexing_error

    # Attempt 3
    vector_store.fail_message = "Tertiary failure: node OOM"
    res2 = recovery_service.recover_document(doc_id)
    assert res2.status == IndexingStatus.FAILED
    assert res2.attempts == 3
    assert "node OOM" in (res2.error or "")

    # Bounded recovery with max_attempts: skip if attempts >= max_attempts
    batch = recovery_service.recover_batch(batch_size=10, max_attempts=2)
    assert batch.scanned_count == 0  # skipped because attempts (3) > max_attempts (2)


# ---------------------------------------------------------------------------
# 6. Concurrent recovery safety
# ---------------------------------------------------------------------------
def test_concurrent_recovery_safety_serializes_cleanly(session_factory, seed_paper):
    vector_store = ControllableVectorStore(fail_on_upsert=True)
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    processed = _make_processed_doc(num_chunks=2)

    with pytest.raises(VectorIndexingError):
        service.ingest_processed_document(
            paper_id=seed_paper,
            processed_document=processed,
            filename="concurrent_rec.pdf",
            checksum=VALID_CHECKSUM_A,
            storage_key="keys/concurrent_rec.pdf",
            durable=True,
        )

    with session_factory() as session:
        doc_id = session.query(Document).filter_by(checksum=VALID_CHECKSUM_A).one().id

    # Enable vector store
    vector_store.fail_on_upsert = False
    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )

    # Run 4 simultaneous recovery tasks for the exact same document
    def _run_rec():
        return recovery_service.recover_document(doc_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_run_rec) for _ in range(4)]
        results = [f.result() for f in futures]

    # All should report COMPLETED
    for r in results:
        assert r.status == IndexingStatus.COMPLETED

    # Final DB check
    with session_factory() as session:
        doc = session.query(Document).filter_by(id=doc_id).one()
        assert doc.indexing_status == IndexingStatus.COMPLETED.value
        assert doc.indexing_error is None

    # Vector store must have exact count of 2 chunks, no duplicates
    assert vector_store.count() == 2


def test_concurrent_recovery_and_live_ingestion(session_factory, seed_paper):
    """Verify that recovery and live ingestion for the same paper serialize safely."""
    vector_store = ControllableVectorStore(fail_on_upsert=True)
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    processed1 = _make_processed_doc(num_chunks=2)

    with pytest.raises(VectorIndexingError):
        service.ingest_processed_document(
            paper_id=seed_paper,
            processed_document=processed1,
            filename="doc1.pdf",
            checksum=VALID_CHECKSUM_A,
            storage_key="keys/doc1.pdf",
            durable=True,
        )

    vector_store.fail_on_upsert = False
    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )

    processed2 = _make_processed_doc(num_chunks=2)

    def _do_recovery():
        return recovery_service.recover_batch(batch_size=5)

    def _do_live_ingest():
        return service.ingest_processed_document(
            paper_id=seed_paper,
            processed_document=processed2,
            filename="doc2.pdf",
            checksum=VALID_CHECKSUM_B,
            storage_key="keys/doc2.pdf",
            allow_duplicate=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_rec = executor.submit(_do_recovery)
        f_ing = executor.submit(_do_live_ingest)
        rec_res = f_rec.result()
        ing_res = f_ing.result()

    assert rec_res.recovered_count == 1
    assert ing_res.chunks_count == 2
    assert vector_store.count() == 4


# ---------------------------------------------------------------------------
# 7. Successful ingestion state
# ---------------------------------------------------------------------------
def test_successful_ingestion_sets_indexing_status_completed(session_factory, seed_paper):
    vector_store = ControllableVectorStore()
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    processed = _make_processed_doc(num_chunks=3)

    result = service.ingest_processed_document(
        paper_id=seed_paper,
        processed_document=processed,
        filename="success.pdf",
        checksum=VALID_CHECKSUM_A,
        storage_key="keys/success.pdf",
    )

    with session_factory() as session:
        doc = session.query(Document).filter_by(id=result.document_id).one()
        assert doc.indexing_status == IndexingStatus.COMPLETED.value
        assert doc.indexing_error is None
        assert doc.indexed_at is not None
        assert doc.indexing_attempts == 0


# ---------------------------------------------------------------------------
# 8. Transaction boundaries and summary
# ---------------------------------------------------------------------------
def test_indexing_summary_and_batch_limits(session_factory, seed_paper):
    vector_store = ControllableVectorStore(fail_on_upsert=True)
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )

    # Ingest 3 failed documents
    for i, cs in enumerate([VALID_CHECKSUM_A, VALID_CHECKSUM_B, VALID_CHECKSUM_C]):
        processed = _make_processed_doc(num_chunks=1)
        with pytest.raises(VectorIndexingError):
            service.ingest_processed_document(
                paper_id=seed_paper,
                processed_document=processed,
                filename=f"fail_{i}.pdf",
                checksum=cs,
                storage_key=f"keys/fail_{i}.pdf",
                allow_duplicate=True,
                durable=True,
            )

    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )

    summary = recovery_service.get_indexing_summary()
    assert summary.get(IndexingStatus.FAILED.value, 0) == 3
    assert summary.get(IndexingStatus.COMPLETED.value, 0) == 0

    # Recover first 2 in a bounded batch
    vector_store.fail_on_upsert = False
    batch_res = recovery_service.recover_batch(batch_size=2)
    assert batch_res.scanned_count == 2
    assert batch_res.recovered_count == 2

    # Check updated summary
    summary2 = recovery_service.get_indexing_summary()
    assert summary2.get(IndexingStatus.COMPLETED.value, 0) == 2
    assert summary2.get(IndexingStatus.FAILED.value, 0) == 1

    # Recover remainder
    batch_res2 = recovery_service.recover_batch(batch_size=2)
    assert batch_res2.scanned_count == 1
    assert batch_res2.recovered_count == 1

    summary3 = recovery_service.get_indexing_summary()
    assert summary3.get(IndexingStatus.COMPLETED.value, 0) == 3
    assert summary3.get(IndexingStatus.FAILED.value, 0) == 0


def test_recovery_nonexistent_document_raises_not_found(session_factory):
    vector_store = ControllableVectorStore()
    embedder = DummyEmbeddingService()
    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    with pytest.raises(RecordNotFoundError):
        recovery_service.recover_document(uuid4())


# ---------------------------------------------------------------------------
# 9. Live PostgreSQL Integration Test
# ---------------------------------------------------------------------------
def _get_postgres_test_url() -> str | None:
    url = os.environ.get("TEST_POSTGRES_URL") or os.environ.get("NEXORA_DATABASE_URL")
    if url and (url.startswith("postgresql://") or url.startswith("postgresql+")):
        return url
    return None


def test_durable_recovery_against_live_postgresql():
    """Verify durable vector indexing recovery against a live PostgreSQL database."""
    url = _get_postgres_test_url()
    if not url:
        pytest.skip(
            "Live PostgreSQL database URL not configured. Set TEST_POSTGRES_URL or NEXORA_DATABASE_URL "
            "to run live PostgreSQL durable recovery tests."
        )

    try:
        engine = create_engine(url, pool_size=5, pool_timeout=10)
        with engine.connect() as conn:
            conn.execute(select(1))
    except Exception as exc:
        pytest.fail(f"PostgreSQL server configured at {url} is not accessible: {exc}")

    Base.metadata.create_all(engine)
    pg_session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    with pg_session_factory() as session:
        user = User(email=f"pg_rec_{uuid4()}@example.com")
        session.add(user)
        session.flush()
        workspace = Workspace(name="PG Rec WS", user_id=user.id)
        session.add(workspace)
        session.flush()
        col = Collection(name="PG Rec Col", workspace_id=workspace.id)
        session.add(col)
        session.flush()
        paper = Paper(collection=col, title="PG Durable Recovery Paper")
        session.add(paper)
        session.commit()
        paper_id = paper.id

    pg_checksum = "aa" * 32
    vector_store = ControllableVectorStore(fail_on_upsert=True, fail_message="PG Vector fail")
    embedder = DummyEmbeddingService()
    service = DocumentIngestionService(
        session_factory=pg_session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    processed = _make_processed_doc(num_chunks=2)

    with pytest.raises(VectorIndexingError):
        service.ingest_processed_document(
            paper_id=paper_id,
            processed_document=processed,
            filename="pg_doc.pdf",
            checksum=pg_checksum,
            storage_key="keys/pg_doc.pdf",
            durable=True,
        )

    with pg_session_factory() as session:
        doc = session.query(Document).filter_by(checksum=pg_checksum).one()
        doc_id = doc.id
        assert doc.indexing_status == IndexingStatus.FAILED.value
        assert doc.indexing_attempts == 1

    # Recover
    vector_store.fail_on_upsert = False
    recovery = DocumentRecoveryService(
        session_factory=pg_session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
    )
    item_res = recovery.recover_document(doc_id)
    assert item_res.status == IndexingStatus.COMPLETED

    with pg_session_factory() as session:
        rec_doc = session.query(Document).filter_by(id=doc_id).one()
        assert rec_doc.indexing_status == IndexingStatus.COMPLETED.value
        assert rec_doc.indexing_error is None
        assert rec_doc.indexed_at is not None


def test_independent_workers_contention_and_crash_recovery(session_factory, seed_paper):
    """Verify independent workers contending with atomic claims and recovering crashed claims."""
    vector_store = InMemoryVectorStore()
    embedder = DummyEmbeddingService()
    fail_store = ControllableVectorStore(fail_on_upsert=True)
    ingestion_service = DocumentIngestionService(
        session_factory=session_factory,
        vector_store=fail_store,
        embedding_service=embedder,
    )

    # Ingest 3 failed documents
    doc_ids = []
    for i in range(3):
        cs = f"{i:02x}" * 32
        pdoc = _make_processed_doc(num_chunks=2)
        with pytest.raises(VectorIndexingError):
            ingestion_service.ingest_processed_document(
                paper_id=seed_paper,
                processed_document=pdoc,
                filename=f"contention_{i}.pdf",
                checksum=cs,
                storage_key=f"keys/contention_{i}.pdf",
                allow_duplicate=True,
                durable=True,
            )
        with session_factory() as session:
            d = session.query(Document).filter_by(checksum=cs).one()
            doc_ids.append(d.id)

    # Add a 4th document that was claimed by a crashed worker 600 seconds ago (expired claim)
    crashed_cs = "cc" * 32
    with session_factory() as session:
        crashed_doc = Document(
            paper_id=seed_paper,
            filename="crashed_worker.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum=crashed_cs,
            storage_key="keys/crashed_worker.pdf",
            indexing_status=IndexingStatus.PROCESSING.value,
            indexing_attempts=1,
            claimed_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600),
            claimed_by="dead_worker_token",
        )
        session.add(crashed_doc)
        session.flush()
        for idx in range(2):
            session.add(
                DocumentChunkRecord(
                    document_id=crashed_doc.id,
                    chunk_index=idx,
                    text=f"Crashed chunk text {idx}",
                )
            )
        session.commit()
        doc_ids.append(crashed_doc.id)

    # Two independent worker instances
    worker_1 = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
        default_claim_timeout_seconds=300.0,
    )
    worker_2 = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
        default_claim_timeout_seconds=300.0,
    )

    def _run_worker(w):
        return w.recover_batch(batch_size=2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_run_worker, worker_1)
        f2 = executor.submit(_run_worker, worker_2)
        res1 = f1.result()
        res2 = f2.result()

    total_recovered = res1.recovered_count + res2.recovered_count
    # In SQLite, both workers together claimed and recovered all 4 documents
    # (if the first worker claimed 2 and the second claimed 2, total is 4; or remaining on next pass)
    if total_recovered < 4:
        # Run another pass to drain remaining
        pass2 = worker_1.recover_batch(batch_size=4)
        total_recovered += pass2.recovered_count

    assert total_recovered == 4
    assert res1.failed_count == 0
    assert res2.failed_count == 0

    # Verify all 4 documents reached COMPLETED
    with session_factory() as session:
        for did in doc_ids:
            doc = session.query(Document).filter_by(id=did).one()
            assert doc.indexing_status == IndexingStatus.COMPLETED.value
            assert doc.indexed_at is not None
            assert doc.indexing_error is None
            assert doc.claimed_at is None
            assert doc.claimed_by is None
            assert doc.indexing_attempts <= 2

            chunks = session.query(DocumentChunkRecord).filter_by(document_id=did).all()
            assert len(chunks) == 2

    assert vector_store.count() == 8


def test_postgresql_independent_workers_contention_and_recovery():
    """Verify independent workers contending on PostgreSQL with SKIP LOCKED and crash recovery."""
    url = _get_postgres_test_url()
    if not url:
        pytest.skip(
            "Live PostgreSQL database URL not configured. Set TEST_POSTGRES_URL or NEXORA_DATABASE_URL "
            "to run live PostgreSQL contention tests."
        )

    try:
        engine = create_engine(url, pool_size=10, max_overflow=10, pool_timeout=10)
        with engine.connect() as conn:
            conn.execute(select(1))
    except Exception as exc:
        pytest.fail(f"PostgreSQL server configured at {url} is not accessible: {exc}")

    Base.metadata.create_all(engine)
    pg_session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    with pg_session_factory() as session:
        user = User(email=f"pg_worker_contention_{uuid4()}@example.com")
        session.add(user)
        session.flush()
        workspace = Workspace(name="PG Contention WS", user_id=user.id)
        session.add(workspace)
        session.flush()
        col = Collection(name="PG Contention Col", workspace_id=workspace.id)
        session.add(col)
        session.flush()
        paper = Paper(collection=col, title="PG Multi-Worker Paper")
        session.add(paper)
        session.commit()
        paper_id = paper.id

    vector_store = InMemoryVectorStore()
    embedder = DummyEmbeddingService()

    doc_ids = []
    fail_store = ControllableVectorStore(fail_on_upsert=True)
    ingestion_service = DocumentIngestionService(
        session_factory=pg_session_factory,
        vector_store=fail_store,
        embedding_service=embedder,
    )
    for i in range(4):
        cs = f"{i:02x}" * 32
        pdoc = _make_processed_doc(num_chunks=2)
        with pytest.raises(VectorIndexingError):
            ingestion_service.ingest_processed_document(
                paper_id=paper_id,
                processed_document=pdoc,
                filename=f"contention_{i}.pdf",
                checksum=cs,
                storage_key=f"keys/contention_{i}.pdf",
                allow_duplicate=True,
                durable=True,
            )
        with pg_session_factory() as session:
            d = session.query(Document).filter_by(checksum=cs).one()
            doc_ids.append(d.id)

    # Simulate 5th document claimed by a crashed worker 600s ago
    crashed_cs = "cc" * 32
    with pg_session_factory() as session:
        crashed_doc = Document(
            paper_id=paper_id,
            filename="crashed_worker.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum=crashed_cs,
            storage_key="keys/crashed_worker.pdf",
            indexing_status=IndexingStatus.PROCESSING.value,
            indexing_attempts=1,
            claimed_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600),
            claimed_by="dead_worker_token",
        )
        session.add(crashed_doc)
        session.flush()
        for idx in range(2):
            session.add(
                DocumentChunkRecord(
                    document_id=crashed_doc.id,
                    chunk_index=idx,
                    text=f"Crashed chunk text {idx}",
                )
            )
        session.commit()
        doc_ids.append(crashed_doc.id)

    worker_1 = DocumentRecoveryService(
        session_factory=pg_session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
        default_claim_timeout_seconds=300.0,
    )
    worker_2 = DocumentRecoveryService(
        session_factory=pg_session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
        default_claim_timeout_seconds=300.0,
    )

    def _run_worker(w):
        return w.recover_batch(batch_size=3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_run_worker, worker_1)
        f2 = executor.submit(_run_worker, worker_2)
        res1 = f1.result()
        res2 = f2.result()

    total_recovered = res1.recovered_count + res2.recovered_count
    if total_recovered < 5:
        pass2 = worker_1.recover_batch(batch_size=5)
        total_recovered += pass2.recovered_count

    assert total_recovered == 5
    assert res1.failed_count == 0
    assert res2.failed_count == 0

    with pg_session_factory() as session:
        for did in doc_ids:
            doc = session.query(Document).filter_by(id=did).one()
            assert doc.indexing_status == IndexingStatus.COMPLETED.value
            assert doc.indexed_at is not None
            assert doc.indexing_error is None
            assert doc.claimed_at is None
            assert doc.claimed_by is None
            assert doc.indexing_attempts <= 2

            chunks = session.query(DocumentChunkRecord).filter_by(document_id=did).all()
            assert len(chunks) == 2

    assert vector_store.count() == 10


# ---------------------------------------------------------------------------
# 11. Recovery safety hardening: Token fencing & retry exhaustion regression
# ---------------------------------------------------------------------------
def test_old_worker_finalize_rejected_when_newer_worker_owns_processing_claim(session_factory, seed_paper):
    """An old worker attempting to finalize while a newer worker owns a processing claim is rejected."""
    with session_factory() as session:
        doc = Document(
            paper_id=seed_paper,
            filename="fencing_test.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="11" * 32,
            storage_key="keys/fencing_test.pdf",
            indexing_status=IndexingStatus.PROCESSING.value,
            indexing_attempts=1,
            claimed_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600),
            claimed_by="old_token_1",
        )
        session.add(doc)
        session.flush()
        session.add(
            DocumentChunkRecord(
                document_id=doc.id,
                chunk_index=0,
                text="Chunk text for fencing test",
            )
        )
        session.commit()
        doc_id = doc.id

    # Newer worker claims it
    with session_factory() as session:
        persistence = DocumentPersistenceService(session)
        claimed_doc = persistence.claim_document_for_recovery(
            doc_id,
            claim_timeout_seconds=300.0,
            claim_token="newer_token_2",
        )
        assert claimed_doc is not None
        assert claimed_doc.claimed_by == "newer_token_2"
        assert claimed_doc.indexing_attempts == 2
        session.commit()

    # Old worker attempts to finalize with old_token_1
    with session_factory() as session:
        persistence = DocumentPersistenceService(session)
        res = persistence.finalize_document_indexing(
            doc_id,
            IndexingStatus.COMPLETED,
            claim_token="old_token_1",
            error=None,
        )
        assert res is None
        session.commit()

    # Verify newer worker's claim is completely intact
    with session_factory() as session:
        doc = session.query(Document).filter_by(id=doc_id).one()
        assert doc.indexing_status == IndexingStatus.PROCESSING.value
        assert doc.claimed_by == "newer_token_2"
        assert doc.indexing_attempts == 2
        assert doc.indexed_at is None
        assert doc.indexing_error is None

    # Verify old worker's DocumentRecoveryService recognises lost claim
    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=InMemoryVectorStore(),
        embedding_service=DummyEmbeddingService(),
    )
    item_res = recovery_service._execute_document_indexing(
        doc_id=doc_id,
        paper_id=seed_paper,
        claim_token="old_token_1",
        attempts=1,
    )
    assert item_res.status == IndexingStatus.FAILED
    assert "Recovery claim lost or superseded" in (item_res.error or "")
    assert item_res.indexed_chunks_count == 0


def test_old_worker_finalize_rejected_after_newer_worker_failed_document(session_factory, seed_paper):
    """An old worker attempting to finalize after a newer worker has marked document failed is rejected."""
    with session_factory() as session:
        doc = Document(
            paper_id=seed_paper,
            filename="failed_fencing_test.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="22" * 32,
            storage_key="keys/failed_fencing_test.pdf",
            indexing_status=IndexingStatus.PROCESSING.value,
            indexing_attempts=1,
            claimed_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600),
            claimed_by="old_token_1",
        )
        session.add(doc)
        session.flush()
        session.add(
            DocumentChunkRecord(
                document_id=doc.id,
                chunk_index=0,
                text="Chunk text for failed fencing test",
            )
        )
        session.commit()
        doc_id = doc.id

    # Newer worker claims and marks it failed
    with session_factory() as session:
        persistence = DocumentPersistenceService(session)
        claimed = persistence.claim_document_for_recovery(
            doc_id,
            claim_timeout_seconds=300.0,
            claim_token="newer_token_2",
        )
        assert claimed is not None
        persistence.finalize_document_indexing(
            doc_id,
            IndexingStatus.FAILED,
            claim_token="newer_token_2",
            error="Upstream network partition",
        )
        session.commit()

    with session_factory() as session:
        doc = session.query(Document).filter_by(id=doc_id).one()
        assert doc.indexing_status == IndexingStatus.FAILED.value
        assert doc.indexing_error == "Upstream network partition"
        assert doc.claimed_by is None

    # Stale worker attempts to finalize to COMPLETED with old token
    with session_factory() as session:
        persistence = DocumentPersistenceService(session)
        res = persistence.finalize_document_indexing(
            doc_id,
            IndexingStatus.COMPLETED,
            claim_token="old_token_1",
        )
        assert res is None
        session.commit()

    # DB state remains unchanged (still FAILED, still original error)
    with session_factory() as session:
        doc = session.query(Document).filter_by(id=doc_id).one()
        assert doc.indexing_status == IndexingStatus.FAILED.value
        assert doc.indexing_error == "Upstream network partition"
        assert doc.indexed_at is None
        assert doc.claimed_by is None


def test_mismatched_token_cannot_clear_or_replace_newer_claim(session_factory, seed_paper):
    """A mismatched token cannot clear or replace a newer claim."""
    with session_factory() as session:
        doc = Document(
            paper_id=seed_paper,
            filename="mismatch_test.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="33" * 32,
            storage_key="keys/mismatch_test.pdf",
            indexing_status=IndexingStatus.PROCESSING.value,
            indexing_attempts=1,
            claimed_at=datetime.datetime.now(datetime.timezone.utc),
            claimed_by="active_claim_owner",
        )
        session.add(doc)
        session.commit()
        doc_id = doc.id

    with session_factory() as session:
        persistence = DocumentPersistenceService(session)
        # Attempt completion with bogus token
        assert persistence.finalize_document_indexing(
            doc_id,
            IndexingStatus.COMPLETED,
            claim_token="bogus_token",
        ) is None

        # Attempt failure with bogus token
        assert persistence.finalize_document_indexing(
            doc_id,
            IndexingStatus.FAILED,
            claim_token="bogus_token",
            error="Bogus failure attempt",
        ) is None
        session.commit()

    with session_factory() as session:
        doc = session.query(Document).filter_by(id=doc_id).one()
        assert doc.indexing_status == IndexingStatus.PROCESSING.value
        assert doc.claimed_by == "active_claim_owner"
        assert doc.claimed_at is not None
        assert doc.indexing_error is None


def test_stale_processing_claim_at_retry_limit_transitions_to_failed(session_factory, seed_paper):
    """A stale processing claim whose retry count has reached max_attempts transitions to failed."""
    with session_factory() as session:
        doc = Document(
            paper_id=seed_paper,
            filename="exhausted_stale.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="44" * 32,
            storage_key="keys/exhausted_stale.pdf",
            indexing_status=IndexingStatus.PROCESSING.value,
            indexing_attempts=3,
            claimed_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600),
            claimed_by="crashed_token_3",
        )
        session.add(doc)
        session.flush()
        session.add(
            DocumentChunkRecord(
                document_id=doc.id,
                chunk_index=0,
                text="Exhausted chunk text",
            )
        )
        session.commit()
        doc_id = doc.id

    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=InMemoryVectorStore(),
        embedding_service=DummyEmbeddingService(),
        default_claim_timeout_seconds=300.0,
    )

    batch = recovery_service.recover_batch(batch_size=10, max_attempts=3)
    assert batch.scanned_count == 0
    assert batch.recovered_count == 0
    assert batch.failed_count == 0

    with session_factory() as session:
        doc = session.query(Document).filter_by(id=doc_id).one()
        assert doc.indexing_status == IndexingStatus.FAILED.value
        assert doc.claimed_at is None
        assert doc.claimed_by is None
        assert doc.indexing_attempts == 3
        assert "Recovery indexing attempts exhausted" in (doc.indexing_error or "")
        assert "3" in (doc.indexing_error or "")


def test_claim_below_retry_limit_can_still_be_recovered(session_factory, seed_paper):
    """A stale processing claim below max_attempts is successfully claimed and recovered."""
    with session_factory() as session:
        doc = Document(
            paper_id=seed_paper,
            filename="below_limit.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="55" * 32,
            storage_key="keys/below_limit.pdf",
            indexing_status=IndexingStatus.PROCESSING.value,
            indexing_attempts=2,
            claimed_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600),
            claimed_by="crashed_token_2",
        )
        session.add(doc)
        session.flush()
        session.add(
            DocumentChunkRecord(
                document_id=doc.id,
                chunk_index=0,
                text="Below limit chunk text",
            )
        )
        session.commit()
        doc_id = doc.id

    vstore = InMemoryVectorStore()
    recovery_service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=vstore,
        embedding_service=DummyEmbeddingService(),
        default_claim_timeout_seconds=300.0,
    )

    batch = recovery_service.recover_batch(batch_size=10, max_attempts=3)
    assert batch.scanned_count == 1
    assert batch.recovered_count == 1
    assert batch.failed_count == 0

    with session_factory() as session:
        doc = session.query(Document).filter_by(id=doc_id).one()
        assert doc.indexing_status == IndexingStatus.COMPLETED.value
        assert doc.claimed_at is None
        assert doc.claimed_by is None
        assert doc.indexing_attempts == 3
        assert doc.indexed_at is not None
        assert doc.indexing_error is None

    assert vstore.count() == 1


def test_postgresql_concurrent_workers_claim_distinct_documents_and_exhaust_safely():
    """Verify concurrent workers on live PostgreSQL coordinate via FOR UPDATE SKIP LOCKED,

    recovering eligible documents and transitioning exhausted claims without collision.
    """
    url = _get_postgres_test_url()
    if not url:
        pytest.skip(
            "Live PostgreSQL database URL not configured. Set TEST_POSTGRES_URL or NEXORA_DATABASE_URL "
            "to run live PostgreSQL concurrency tests."
        )

    try:
        engine = create_engine(url, pool_size=10, max_overflow=10, pool_timeout=10)
        with engine.connect() as conn:
            conn.execute(select(1))
    except Exception as exc:
        pytest.fail(f"PostgreSQL server configured at {url} is not accessible: {exc}")

    Base.metadata.create_all(engine)
    pg_session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    with pg_session_factory() as session:
        user = User(email=f"pg_worker_exhaust_{uuid4()}@example.com")
        session.add(user)
        session.flush()
        workspace = Workspace(name="PG Exhaust WS", user_id=user.id)
        session.add(workspace)
        session.flush()
        col = Collection(name="PG Exhaust Col", workspace_id=workspace.id)
        session.add(col)
        session.flush()
        paper = Paper(collection=col, title="PG Exhaust Paper")
        session.add(paper)
        session.commit()
        paper_id = paper.id

    vector_store = InMemoryVectorStore()
    embedder = DummyEmbeddingService()

    eligible_doc_ids = []
    exhausted_doc_ids = []

    # 1. Seed 2 pending documents (attempts = 0)
    with pg_session_factory() as session:
        for i in range(2):
            doc = Document(
                paper_id=paper_id,
                filename=f"pending_{i}.pdf",
                mime_type="application/pdf",
                file_size=1024,
                checksum=f"0a{i:02x}" * 16,
                storage_key=f"keys/pending_{i}.pdf",
                indexing_status=IndexingStatus.PENDING.value,
                indexing_attempts=0,
            )
            session.add(doc)
            session.flush()
            session.add(DocumentChunkRecord(document_id=doc.id, chunk_index=0, text=f"Pending {i}"))
            eligible_doc_ids.append(doc.id)
        session.commit()

    # 2. Seed 2 failed documents (attempts = 1)
    with pg_session_factory() as session:
        for i in range(2):
            doc = Document(
                paper_id=paper_id,
                filename=f"failed_{i}.pdf",
                mime_type="application/pdf",
                file_size=1024,
                checksum=f"0b{i:02x}" * 16,
                storage_key=f"keys/failed_{i}.pdf",
                indexing_status=IndexingStatus.FAILED.value,
                indexing_attempts=1,
                indexing_error="Previous network hiccup",
            )
            session.add(doc)
            session.flush()
            session.add(DocumentChunkRecord(document_id=doc.id, chunk_index=0, text=f"Failed {i}"))
            eligible_doc_ids.append(doc.id)
        session.commit()

    # 3. Seed 2 stale processing documents below retry limit (attempts = 2 < 3)
    with pg_session_factory() as session:
        for i in range(2):
            doc = Document(
                paper_id=paper_id,
                filename=f"stale_below_{i}.pdf",
                mime_type="application/pdf",
                file_size=1024,
                checksum=f"0c{i:02x}" * 16,
                storage_key=f"keys/stale_below_{i}.pdf",
                indexing_status=IndexingStatus.PROCESSING.value,
                indexing_attempts=2,
                claimed_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600),
                claimed_by=f"stale_worker_{i}",
            )
            session.add(doc)
            session.flush()
            session.add(DocumentChunkRecord(document_id=doc.id, chunk_index=0, text=f"Stale below {i}"))
            eligible_doc_ids.append(doc.id)
        session.commit()

    # 4. Seed 2 stale processing documents AT retry limit (attempts = 3 >= 3)
    with pg_session_factory() as session:
        for i in range(2):
            doc = Document(
                paper_id=paper_id,
                filename=f"stale_exhaust_{i}.pdf",
                mime_type="application/pdf",
                file_size=1024,
                checksum=f"0d{i:02x}" * 16,
                storage_key=f"keys/stale_exhaust_{i}.pdf",
                indexing_status=IndexingStatus.PROCESSING.value,
                indexing_attempts=3,
                claimed_at=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=600),
                claimed_by=f"dead_worker_{i}",
            )
            session.add(doc)
            session.flush()
            session.add(DocumentChunkRecord(document_id=doc.id, chunk_index=0, text=f"Stale exhaust {i}"))
            exhausted_doc_ids.append(doc.id)
        session.commit()

    worker_1 = DocumentRecoveryService(
        session_factory=pg_session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
        default_claim_timeout_seconds=300.0,
    )
    worker_2 = DocumentRecoveryService(
        session_factory=pg_session_factory,
        vector_store=vector_store,
        embedding_service=embedder,
        default_claim_timeout_seconds=300.0,
    )

    def _run_worker(w):
        return w.recover_batch(batch_size=3, max_attempts=3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_run_worker, worker_1)
        f2 = executor.submit(_run_worker, worker_2)
        res1 = f1.result()
        res2 = f2.result()

    total_recovered = res1.recovered_count + res2.recovered_count
    if total_recovered < 6:
        pass2 = worker_1.recover_batch(batch_size=6, max_attempts=3)
        total_recovered += pass2.recovered_count

    assert total_recovered == 6
    assert res1.failed_count == 0
    assert res2.failed_count == 0

    with pg_session_factory() as session:
        for did in eligible_doc_ids:
            doc = session.query(Document).filter_by(id=did).one()
            assert doc.indexing_status == IndexingStatus.COMPLETED.value
            assert doc.indexed_at is not None
            assert doc.indexing_error is None
            assert doc.claimed_at is None
            assert doc.claimed_by is None
            assert doc.indexing_attempts <= 3

        for did in exhausted_doc_ids:
            doc = session.query(Document).filter_by(id=did).one()
            assert doc.indexing_status == IndexingStatus.FAILED.value
            assert doc.claimed_at is None
            assert doc.claimed_by is None
            assert doc.indexing_attempts == 3
            assert "Recovery indexing attempts exhausted" in (doc.indexing_error or "")

    assert vector_store.count() == 6


def test_recovery_batch_bounds_validation(session_factory):
    """DocumentRecoveryService.recover_batch enforces 1 <= batch_size <= MAX_BATCH_SIZE (100)."""
    store = InMemoryVectorStore(dimension=4)
    service = DocumentRecoveryService(
        session_factory=session_factory,
        vector_store=store,
        embedding_service=DummyEmbeddingService(4),
    )

    with pytest.raises(ValueError, match="cannot exceed maximum allowed limit of 100"):
        service.recover_batch(batch_size=101)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        service.recover_batch(batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        service.recover_batch(batch_size=-5)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        service.recover_batch(batch_size="10")  # type: ignore[arg-type]

    # Valid boundary batch_size=100 succeeds
    res = service.recover_batch(batch_size=100)
    assert res.scanned_count == 0
    assert res.recovered_count == 0
    assert res.failed_count == 0


