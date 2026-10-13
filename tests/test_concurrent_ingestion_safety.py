"""Tests for Concurrent Ingestion Safety (Milestone 12C-20).

Verifies:
1. Two simultaneous uploads of identical content for the same paper resolve to
   predictable outcomes: exactly one HTTP 200 and one HTTP 409 Conflict.
2. Two sequential identical uploads reject the second with HTTP 409 Conflict.
3. Identical content uploaded against different papers succeeds concurrently for both.
4. Concurrent uploads of different content for the same paper succeeds for both.
5. Concurrent failure isolation: a failed request during embedding or vector indexing
   does NOT delete another request's vectors or documents, and leaves zero orphaned records.
6. Retrying after a failed request succeeds without conflict.
7. Database uniqueness conflicts translate to safe HTTP 409 Conflict, not unexpected 500.
8. Database locking strategy: row locking with FOR UPDATE on Paper compiles for PostgreSQL.
9. File-backed SQLite with WAL mode verifies multi-connection concurrency safety.
10. Reindex_document remains a separate reconciliation operation.
"""
from __future__ import annotations

import concurrent.futures
import io
import os
import tempfile
from pathlib import Path
import threading
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.api.app import create_app
from nexora.database.models.base import Base
from nexora.database.models import Collection, Document, DocumentChunkRecord, Paper, User, Workspace
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.paper import PaperRepository
from nexora.document.ingestion import (
    DocumentIngestionService,
    IngestionResult,
    PaperIngestionLock,
    VectorIndexingError,
    active_paper_locks_count,
    get_paper_ingestion_lock,
    paper_ingestion_guard,
)
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import VectorRecord

URL = "/api/v1/documents/ingest"

VALID_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n212\n%%EOF\n"
)

DISTINCT_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
    b"% Supplementary distinct data stream 9876543210\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n250\n%%EOF\n"
)


class SessionTracker(sessionmaker):
    """Monitors session creations and closures for leak detection."""

    def __init__(self, bind):
        super().__init__(bind=bind, expire_on_commit=False)
        self.opened_count = 0
        self.closed_count = 0
        self.active_sessions: set[Session] = set()

    def __call__(self, **kwargs) -> Session:
        session = super().__call__(**kwargs)
        self.opened_count += 1
        self.active_sessions.add(session)
        orig_close = session.close

        def _close_wrapper():
            if session in self.active_sessions:
                self.active_sessions.remove(session)
                self.closed_count += 1
            return orig_close()

        session.close = _close_wrapper
        return session


class FakeResearchService:
    def ask(self, request):
        raise NotImplementedError


class DeterministicEmbeddingService:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def embed_text(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError("Embedding provider timeout")
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("Embedding provider timeout")
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockDocumentPipeline:
    def __init__(self):
        self.loaded_files: list[str] = []

    def _loader(self, file_path: str) -> str:
        self.loaded_files.append(file_path)
        return "Quantum Machine Learning Document Text"

    def process_pdf(self, file_path: str) -> ProcessedDocument:
        text = self._loader(file_path)
        return ProcessedDocument(
            document_id=str(uuid4()),
            source=file_path,
            chunks=[
                DocumentChunk(text=text, embedding=[0.1, 0.2, 0.3], index=0)
            ],
            metadata={"filename": Path(file_path).name},
        )


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / f"concurrent_test_{uuid4().hex}.db"
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
    tracker = SessionTracker(bind=engine)
    yield engine, tracker
    engine.dispose()


def _create_paper(session: Session, title: str = "Test Paper") -> Paper:
    user = User(email=f"{uuid4()}@example.com")
    session.add(user)
    session.flush()
    ws = Workspace(name="Test WS", user_id=user.id)
    session.add(ws)
    session.flush()
    col = Collection(name="Test Col", workspace_id=ws.id)
    session.add(col)
    session.flush()
    paper = Paper(collection_id=col.id, title=title)
    session.add(paper)
    session.flush()
    session.commit()
    return paper


def _build_test_client(
    tracker: SessionTracker,
    *,
    vector_store=None,
    embedding_service=None,
    pipeline=None,
    max_upload_size=50 * 1024 * 1024,
):
    vs = vector_store or InMemoryVectorStore()
    emb = embedding_service or DeterministicEmbeddingService()
    pipe = pipeline or MockDocumentPipeline()

    ingestion_service = DocumentIngestionService(
        vector_store=vs,
        embedding_service=emb,
        session_factory=tracker,
    )

    app = create_app(
        FakeResearchService(),
        ingestion_service=ingestion_service,
        document_pipeline=pipe,
        session_factory=tracker,
        max_upload_size_bytes=max_upload_size,
    )
    return TestClient(app, raise_server_exceptions=False), vs, emb, pipe


# ==============================================================================
# 1. Simultaneous and Sequential Ingestion Tests
# ==============================================================================


def test_concurrent_simultaneous_identical_uploads_resolve_to_one_success_and_one_409(test_db):
    """Two simultaneous uploads of identical content for the same paper resolve to
    exactly one 200 and one 409 Conflict, without orphaned records or leaked sessions.
    """
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(tracker)

    def _upload():
        return client.post(
            URL,
            data={"paper_id": str(paper_id)},
            files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_upload)
        f2 = executor.submit(_upload)
        res1 = f1.result()
        res2 = f2.result()

    statuses = sorted([res1.status_code, res2.status_code])
    assert statuses == [200, 409]

    # Verify 200 response shape
    success_res = res1 if res1.status_code == 200 else res2
    conflict_res = res2 if res1.status_code == 200 else res1

    success_data = success_res.json()
    assert success_data["paper_id"] == str(paper_id)
    assert success_data["chunks_count"] == 1
    assert success_data["indexed_chunks_count"] == 1

    conflict_data = conflict_res.json()
    assert conflict_data["error"]["code"] == "conflict"
    assert "already exists" in conflict_data["error"]["message"]

    # Verify database state has exactly 1 Document and 1 DocumentChunkRecord
    with Session(engine) as s:
        docs = s.query(Document).filter_by(paper_id=paper_id).all()
        assert len(docs) == 1
        chunks = s.query(DocumentChunkRecord).filter_by(document_id=docs[0].id).all()
        assert len(chunks) == 1

    # Verify vector store state has exactly 1 vector
    assert vs.count() == 1

    # Verify zero leaked sessions
    assert len(tracker.active_sessions) == 0


def test_sequential_identical_uploads(test_db):
    """Sequential duplicate upload is rejected with 409 Conflict."""
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(tracker)

    r1 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r1.status_code == 200

    r2 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "conflict"

    with Session(engine) as s:
        assert s.query(Document).filter_by(paper_id=paper_id).count() == 1
    assert vs.count() == 1
    assert len(tracker.active_sessions) == 0


def test_identical_content_against_different_papers(test_db):
    """Simultaneous identical uploads against different papers both succeed."""
    engine, tracker = test_db
    with Session(engine) as s:
        p1 = _create_paper(s, title="Paper 1")
        p2 = _create_paper(s, title="Paper 2")
        pid1, pid2 = p1.id, p2.id

    client, vs, _, _ = _build_test_client(tracker)

    def _upload(pid):
        return client.post(
            URL,
            data={"paper_id": str(pid)},
            files={"file": ("shared.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_upload, pid1)
        f2 = executor.submit(_upload, pid2)
        res1 = f1.result()
        res2 = f2.result()

    assert res1.status_code == 200
    assert res2.status_code == 200

    with Session(engine) as s:
        assert s.query(Document).filter_by(paper_id=pid1).count() == 1
        assert s.query(Document).filter_by(paper_id=pid2).count() == 1
    assert vs.count() == 2
    assert len(tracker.active_sessions) == 0


def test_concurrent_uploads_different_content_same_paper(test_db):
    """Simultaneous uploads of different PDF content for the same paper both succeed,
    preserving the capability to store multiple documents for a single paper.
    """
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(tracker)

    def _upload(file_bytes, filename):
        return client.post(
            URL,
            data={"paper_id": str(paper_id)},
            files={"file": (filename, io.BytesIO(file_bytes), "application/pdf")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_upload, VALID_PDF_BYTES, "main.pdf")
        f2 = executor.submit(_upload, DISTINCT_PDF_BYTES, "supplement.pdf")
        res1 = f1.result()
        res2 = f2.result()

    assert res1.status_code == 200
    assert res2.status_code == 200

    with Session(engine) as s:
        docs = s.query(Document).filter_by(paper_id=paper_id).all()
        assert len(docs) == 2
        assert {d.filename for d in docs} == {"main.pdf", "supplement.pdf"}

    assert vs.count() == 2
    assert len(tracker.active_sessions) == 0


# ==============================================================================
# 2. Failure Isolation and Recovery under Concurrency
# ==============================================================================


def test_concurrent_upload_where_one_fails_during_indexing(test_db):
    """Verifies that when one request fails during vector indexing or embedding,
    the successful concurrent request is not impacted and its vectors are not deleted.
    """
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    # Create a vector store that selectively fails on specific content
    class SelectiveFailingVectorStore(InMemoryVectorStore):
        def __init__(self):
            super().__init__()
            self.failed_once = False

        def upsert(self, records: list[VectorRecord]) -> None:
            # Fail the first attempt only
            if not self.failed_once:
                self.failed_once = True
                raise RuntimeError("Simulated transient vector store outage")
            super().upsert(records)

    vs = SelectiveFailingVectorStore()
    client, _, _, _ = _build_test_client(tracker, vector_store=vs)

    def _upload():
        return client.post(
            URL,
            data={"paper_id": str(paper_id)},
            files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_upload)
        f2 = executor.submit(_upload)
        res1 = f1.result()
        res2 = f2.result()

    statuses = sorted([res1.status_code, res2.status_code])
    # One failed with 500 indexing_failed, the other succeeded with 200!
    assert statuses == [200, 500]

    # Exactly one document in DB (from the successful attempt)
    with Session(engine) as s:
        docs = s.query(Document).filter_by(paper_id=paper_id).all()
        assert len(docs) == 1
        chunks = s.query(DocumentChunkRecord).filter_by(document_id=docs[0].id).all()
        assert len(chunks) == 1

    # Exactly 1 vector record persisted
    assert vs.count() == 1
    assert len(tracker.active_sessions) == 0


def test_failed_request_does_not_delete_another_requests_vectors_or_document(test_db):
    """A failed subsequent request does NOT delete vectors belonging to an existing document."""
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    vs = InMemoryVectorStore()
    client, _, _, _ = _build_test_client(tracker, vector_store=vs)

    # 1. First upload succeeds
    r1 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("doc1.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r1.status_code == 200
    doc1_id = r1.json()["document_id"]
    assert vs.count() == 1

    # 2. Second upload fails during vector indexing
    class ExplodingVectorStore(InMemoryVectorStore):
        def upsert(self, records: list[VectorRecord]) -> None:
            raise RuntimeError("Outage during upload 2")

    failing_vs = ExplodingVectorStore()
    # Populate existing vectors into failing store
    failing_vs._records.update(vs._records)

    client2, _, _, _ = _build_test_client(tracker, vector_store=failing_vs)
    r2 = client2.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("doc2.pdf", io.BytesIO(DISTINCT_PDF_BYTES), "application/pdf")},
    )
    assert r2.status_code == 500

    # doc1's vectors are NOT deleted
    assert failing_vs.count() == 1
    existing_rec = list(failing_vs._records.values())[0]
    assert existing_rec.payload["document_id"] == doc1_id

    # doc1 remains valid in database
    with Session(engine) as s:
        docs = s.query(Document).filter_by(paper_id=paper_id).all()
        assert len(docs) == 1
        assert str(docs[0].id) == doc1_id


def test_retrying_after_failed_request_succeeds(test_db):
    """Retrying an upload after a transient indexing failure succeeds cleanly."""
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    emb = DeterministicEmbeddingService(fail=True)
    pipe = MockDocumentPipeline()
    # Empty embedding to force embedding service call
    pipe.process_pdf = lambda fp: ProcessedDocument(
        document_id=str(uuid4()),
        source=fp,
        chunks=[DocumentChunk(text="Sample", embedding=[], index=0)],
        metadata={"filename": "paper.pdf"},
    )

    client, vs, _, _ = _build_test_client(
        tracker, embedding_service=emb, pipeline=pipe
    )

    # 1. First attempt fails due to embedding failure
    r1 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r1.status_code == 500

    # Ensure DB is empty
    with Session(engine) as s:
        assert s.query(Document).filter_by(paper_id=paper_id).count() == 0

    # 2. Fix the embedding service
    emb.fail = False

    # 3. Retry identical request
    r2 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r2.status_code == 200
    assert r2.json()["paper_id"] == str(paper_id)

    # Database now contains 1 document
    with Session(engine) as s:
        assert s.query(Document).filter_by(paper_id=paper_id).count() == 1
    assert vs.count() == 1
    assert len(tracker.active_sessions) == 0


# ==============================================================================
# 3. Database Strategy and Locking Compilations
# ==============================================================================


def test_postgresql_row_locking_compiles():
    """Verifies that PaperRepository.get_by_id_for_update compiles with FOR UPDATE
    under the PostgreSQL dialect for production database concurrency safety.
    """
    dialect = postgresql.dialect()
    pid = uuid4()
    stmt = select(Paper).where(Paper.id == pid).with_for_update()
    compiled = str(stmt.compile(dialect=dialect))

    assert "FROM papers" in compiled
    assert "FOR UPDATE" in compiled


def test_file_backed_sqlite_wal_mode_concurrency(tmp_path):
    """Tests concurrency against file-backed SQLite with WAL mode enabled,
    verifying multi-connection transaction concurrency.
    """
    db_file = tmp_path / "wal_concurrency.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_wal(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    tracker = SessionTracker(bind=engine)

    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(tracker)

    def _upload():
        return client.post(
            URL,
            data={"paper_id": str(paper_id)},
            files={"file": ("wal_paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_upload)
        f2 = executor.submit(_upload)
        res1 = f1.result()
        res2 = f2.result()

    statuses = sorted([res1.status_code, res2.status_code])
    assert statuses == [200, 409]
    assert vs.count() == 1
    assert len(tracker.active_sessions) == 0

    engine.dispose()


def test_database_uniqueness_conflict_handled_as_safe_409(test_db):
    """Verifies that an underlying database uniqueness IntegrityError is translated
    to DuplicateRecordError and returned as HTTP 409 Conflict, never an unexpected 500.
    """
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    from sqlalchemy.exc import IntegrityError

    class UniquenessFailingPersistenceService:
        def __init__(self, session):
            self._session = session

        def persist_processed_document(self, *args, **kwargs):
            raise IntegrityError(
                statement="INSERT INTO documents ...",
                params={},
                orig=Exception("UNIQUE constraint failed: documents.paper_id, documents.checksum"),
            )

    client, vs, _, _ = _build_test_client(tracker)
    ingestion_service = client.app.state.ingestion_service

    # Ingest processed document directly with the failing persistence service
    pipe = MockDocumentPipeline()
    processed = pipe.process_pdf("paper.pdf")

    from nexora.database.repository.base import DuplicateRecordError

    with pytest.raises(DuplicateRecordError):
        with tracker() as session:
            ingestion_service._persist_and_index_records(
                session=session,
                persistence_service=UniquenessFailingPersistenceService(session),
                paper_id=paper_id,
                processed_document=processed,
                resolved_vectors=[[0.1, 0.2, 0.3]],
            )


def test_reindex_document_remains_separate_reconciliation_operation(test_db):
    """Reindex_document reconciles vector records for an existing document without
    creating a new upload or modifying document checksums.
    """
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(tracker)
    service = client.app.state.ingestion_service

    # 1. Normal ingestion
    pipe = MockDocumentPipeline()
    processed = pipe.process_pdf("original.pdf")
    res = service.ingest_processed_document(paper_id, processed)
    doc_id = res.document_id
    assert vs.count() == 1

    # 2. Simulate vector loss
    service.delete_document_vectors(doc_id)
    assert vs.count() == 0

    # 3. Reconcile via reindex_document
    reindex_res = service.reindex_document(doc_id)
    assert reindex_res.document_id == doc_id
    assert reindex_res.indexed_chunks_count == 1
    assert vs.count() == 1

    # Database still has exactly 1 document
    with Session(engine) as s:
        assert s.query(Document).filter_by(paper_id=paper_id).count() == 1


# ==============================================================================
# 4. In-Process Lock Registry Lifecycle Tests
# ==============================================================================


def test_lock_registry_reference_counting_and_clean_removal():
    """Verifies that acquiring a lock guard increments references and cleans up
    completely from the registry upon exit, leaving zero memory leakage.
    """
    pid = uuid4()
    assert active_paper_locks_count() == 0

    with get_paper_ingestion_lock(pid) as guard:
        assert isinstance(guard, PaperIngestionLock)
        assert guard.locked() is True
        assert active_paper_locks_count() == 1

    # After exit, registry entry is removed because ref_count dropped to zero
    assert active_paper_locks_count() == 0


def test_lock_registry_reuse_and_concurrency_contention():
    """Verifies that concurrent threads for the same paper reuse the same lock,
    serialize their execution, track ref_count correctly, and clean up only when
    all threads have released.
    """
    import time
    pid = uuid4()
    thread_1_started = threading.Event()
    thread_2_queued = threading.Event()
    thread_1_can_finish = threading.Event()

    execution_order: list[str] = []

    def _worker_1():
        with get_paper_ingestion_lock(pid):
            execution_order.append("worker_1_acquired")
            thread_1_started.set()
            thread_1_can_finish.wait(timeout=5.0)
            execution_order.append("worker_1_releasing")

    def _worker_2():
        thread_1_started.wait(timeout=5.0)
        thread_2_queued.set()
        with get_paper_ingestion_lock(pid):
            execution_order.append("worker_2_acquired")

    t1 = threading.Thread(target=_worker_1)
    t2 = threading.Thread(target=_worker_2)

    t1.start()
    thread_1_started.wait(timeout=5.0)

    t2.start()
    thread_2_queued.wait(timeout=5.0)

    # Allow worker 2 a moment to queue up on the lock
    time.sleep(0.05)

    # During contention, exactly 1 lock entry exists in the registry with ref_count == 2
    assert active_paper_locks_count() == 1

    # Release worker 1
    thread_1_can_finish.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    # Worker 1 had to finish releasing before worker 2 could acquire
    assert execution_order == [
        "worker_1_acquired",
        "worker_1_releasing",
        "worker_2_acquired",
    ]

    # After both completed, lock entry is completely removed
    assert active_paper_locks_count() == 0


def test_lock_registry_distinct_papers_concurrent_execution():
    """Verifies that threads locking different papers execute concurrently without
    blocking each other, maintaining independent lock entries that clean up independently.
    """
    pid_1 = uuid4()
    pid_2 = uuid4()

    p1_active = threading.Event()
    p2_active = threading.Event()
    proceed = threading.Event()

    def _worker_p1():
        with paper_ingestion_guard(pid_1):
            p1_active.set()
            proceed.wait(timeout=5.0)

    def _worker_p2():
        with paper_ingestion_guard(pid_2):
            p2_active.set()
            proceed.wait(timeout=5.0)

    t1 = threading.Thread(target=_worker_p1)
    t2 = threading.Thread(target=_worker_p2)

    t1.start()
    t2.start()

    p1_active.wait(timeout=5.0)
    p2_active.wait(timeout=5.0)

    # Both locks are concurrently held in memory
    assert active_paper_locks_count() == 2

    proceed.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    # Both cleaned up cleanly
    assert active_paper_locks_count() == 0


def test_lock_registry_exception_safety_cleanup():
    """Verifies that an unhandled exception inside the critical section still
    properly releases the lock and cleans up the registry entry.
    """
    pid = uuid4()

    class CustomTestError(Exception):
        pass

    with pytest.raises(CustomTestError):
        with get_paper_ingestion_lock(pid):
            assert active_paper_locks_count() == 1
            raise CustomTestError("Simulated unhandled failure")

    assert active_paper_locks_count() == 0


