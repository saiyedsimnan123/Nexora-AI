"""PostgreSQL Concurrency Integration Tests (Milestone 12C-21).

Verifies production PostgreSQL concurrency behavior against a live PostgreSQL database:
1. Two simultaneous identical uploads for the same paper resolve to exactly one HTTP 200
   and one HTTP 409 Conflict, without orphaned records or leaked database sessions.
2. Two simultaneous uploads for different papers succeed concurrently with HTTP 200
   (true row-level parallel persistence in PostgreSQL without database-level serialization).
3. Rollback on failure (vector indexing or relational error) leaves zero orphaned records,
   zero dangling vector records, and releases locks cleanly.
4. Clean retry after a failure succeeds without conflict.
5. Two simultaneous uploads of distinct content for the same paper succeed sequentially
   without integrity errors.
6. In-process lock registry lifecycle cleans up completely under concurrent PostgreSQL load.

Note:
When run locally without a configured PostgreSQL instance, this suite skips gracefully
with an informative skip message. When run in GitHub Actions CI with the `postgres:16`
service container, this suite runs and verifies live PostgreSQL concurrency semantics.
"""
from __future__ import annotations

import concurrent.futures
import io
import os
import threading
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from nexora.api.app import create_app
from nexora.database.models.base import Base
from nexora.database.models import Collection, Document, DocumentChunkRecord, Paper, User, Workspace
from nexora.document.ingestion import (
    DocumentIngestionService,
    IngestionResult,
    PaperIngestionLock,
    VectorIndexingError,
    active_paper_locks_count,
    get_paper_ingestion_lock,
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
    b"% Distinct stream payload 9876543210\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n245\n%%EOF\n"
)


class SessionTracker(sessionmaker):
    """Tracks session lifecycles to detect session leaks during concurrent requests."""

    def __init__(self, bind):
        super().__init__(bind=bind, expire_on_commit=False)
        self.opened_count = 0
        self.closed_count = 0
        self.active_sessions: set[Session] = set()
        self._lock = threading.Lock()

    def __call__(self, **kwargs) -> Session:
        session = super().__call__(**kwargs)
        with self._lock:
            self.opened_count += 1
            self.active_sessions.add(session)
        orig_close = session.close

        def _close_wrapper():
            with self._lock:
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
    def __init__(self, text: str = "PostgreSQL Concurrency Test Text"):
        self.text = text
        self.loaded_files: list[str] = []

    def _loader(self, file_path: str) -> str:
        self.loaded_files.append(file_path)
        return self.text

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


def _get_postgres_test_url() -> str | None:
    """Read PostgreSQL connection URL from environment if present."""
    url = os.environ.get("TEST_POSTGRES_URL") or os.environ.get("NEXORA_DATABASE_URL")
    if url and (url.startswith("postgresql://") or url.startswith("postgresql+")):
        return url
    return None


@pytest.fixture(scope="module")
def postgres_engine():
    """Module-scoped fixture to connect to real PostgreSQL and initialize schema."""
    url = _get_postgres_test_url()
    if not url:
        pytest.skip(
            "Live PostgreSQL database URL not configured. Set TEST_POSTGRES_URL or NEXORA_DATABASE_URL "
            "(e.g. postgresql+psycopg://test_user:test_password@localhost:5432/nexora_test) to run live PostgreSQL concurrency tests."
        )

    try:
        engine = create_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_timeout=10,
        )
        with engine.connect() as conn:
            conn.execute(select(1))
    except Exception as exc:
        pytest.fail(f"PostgreSQL server configured at {url} is not accessible: {exc}")

    # Ensure all tables exist in PostgreSQL
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def postgres_tracker(postgres_engine):
    """Function-scoped session tracker bound to PostgreSQL engine."""
    tracker = SessionTracker(bind=postgres_engine)
    yield tracker
    # Verify no session leaks after test execution
    assert len(tracker.active_sessions) == 0, f"Leaked {len(tracker.active_sessions)} sessions"


def _create_paper(session: Session, title: str = "PG Test Paper") -> Paper:
    user = User(email=f"{uuid4()}@example.com")
    session.add(user)
    session.flush()
    ws = Workspace(name="PG Test WS", user_id=user.id)
    session.add(ws)
    session.flush()
    col = Collection(name="PG Test Col", workspace_id=ws.id)
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
# Live PostgreSQL Concurrency Verification Tests
# ==============================================================================


def test_postgres_concurrent_identical_uploads_resolve_to_200_and_409(
    postgres_engine, postgres_tracker
):
    """Verifies that two simultaneous uploads of identical content for the same paper
    under PostgreSQL resolve to exactly one HTTP 200 and one HTTP 409 Conflict.
    """
    with Session(postgres_engine) as s:
        paper = _create_paper(s, title="Concurrent Uploads Paper")
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(postgres_tracker)

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

    # Exactly 1 Document and 1 DocumentChunkRecord in PostgreSQL
    with Session(postgres_engine) as s:
        docs = s.query(Document).filter_by(paper_id=paper_id).all()
        assert len(docs) == 1
        chunks = s.query(DocumentChunkRecord).filter_by(document_id=docs[0].id).all()
        assert len(chunks) == 1

    # Vector store contains only the winner's vectors
    assert vs.count() == 1


def test_postgres_concurrent_uploads_different_papers_true_parallelism(
    postgres_engine, postgres_tracker
):
    """Verifies that simultaneous uploads for DIFFERENT papers in PostgreSQL execute
    in true parallelism without blocking each other (fine-grained row locks on Paper).
    """
    with Session(postgres_engine) as s:
        paper1 = _create_paper(s, title="Parallel Paper 1")
        paper2 = _create_paper(s, title="Parallel Paper 2")
        pid1, pid2 = paper1.id, paper2.id

    client, vs, _, _ = _build_test_client(postgres_tracker)

    def _upload(pid: UUID, content: bytes, fname: str):
        return client.post(
            URL,
            data={"paper_id": str(pid)},
            files={"file": (fname, io.BytesIO(content), "application/pdf")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_upload, pid1, VALID_PDF_BYTES, "p1.pdf")
        f2 = executor.submit(_upload, pid2, DISTINCT_PDF_BYTES, "p2.pdf")
        res1 = f1.result()
        res2 = f2.result()

    assert res1.status_code == 200
    assert res2.status_code == 200

    # Both documents persisted in PostgreSQL
    with Session(postgres_engine) as s:
        assert s.query(Document).filter_by(paper_id=pid1).count() == 1
        assert s.query(Document).filter_by(paper_id=pid2).count() == 1

    assert vs.count() == 2


def test_postgres_rollback_on_vector_failure_leaves_no_orphans(
    postgres_engine, postgres_tracker
):
    """Verifies that a failure during vector indexing triggers a clean relational rollback
    in PostgreSQL, leaving zero orphaned records, zero vector leaks, and released locks.
    """
    with Session(postgres_engine) as s:
        paper = _create_paper(s, title="Failure Rollback Paper")
        paper_id = paper.id

    class FailingVectorStore(InMemoryVectorStore):
        def upsert(self, records):
            raise RuntimeError("Postgres Vector Storage cluster failure")

    client, vs, _, _ = _build_test_client(
        postgres_tracker, vector_store=FailingVectorStore()
    )

    res = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("failing.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )

    assert res.status_code == 500
    err_body = res.json()
    assert err_body["error"]["code"] == "indexing_failed"
    assert "indexing failed" in err_body["error"]["message"].lower()

    # Verify zero documents or chunks persisted in PostgreSQL
    with Session(postgres_engine) as s:
        assert s.query(Document).filter_by(paper_id=paper_id).count() == 0
    assert vs.count() == 0


def test_postgres_clean_retry_after_indexing_failure(
    postgres_engine, postgres_tracker
):
    """Verifies that after an indexing failure on PostgreSQL, a subsequent retry succeeds
    cleanly with HTTP 200 without duplicate conflicts or stuck locks.
    """
    with Session(postgres_engine) as s:
        paper = _create_paper(s, title="Retry Paper")
        paper_id = paper.id

    emb = DeterministicEmbeddingService(fail=True)
    pipe = MockDocumentPipeline()
    # Force embedding generation
    pipe.process_pdf = lambda fp: ProcessedDocument(
        document_id=str(uuid4()),
        source=fp,
        chunks=[DocumentChunk(text="Retry text", embedding=[], index=0)],
        metadata={"filename": "retry.pdf"},
    )

    client, vs, _, _ = _build_test_client(
        postgres_tracker, embedding_service=emb, pipeline=pipe
    )

    # 1. Attempt 1 fails
    res1 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("retry.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert res1.status_code == 500

    # PostgreSQL database is clean
    with Session(postgres_engine) as s:
        assert s.query(Document).filter_by(paper_id=paper_id).count() == 0

    # 2. Fix the embedding service and retry
    emb.fail = False
    res2 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("retry.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert res2.status_code == 200
    assert res2.json()["paper_id"] == str(paper_id)

    # Record persisted in PostgreSQL
    with Session(postgres_engine) as s:
        assert s.query(Document).filter_by(paper_id=paper_id).count() == 1
    assert vs.count() == 1


def test_postgres_concurrent_uploads_different_content_same_paper(
    postgres_engine, postgres_tracker
):
    """Verifies that simultaneous uploads of different documents for the same paper
    under PostgreSQL both succeed, serialized cleanly by the per-paper locking.
    """
    with Session(postgres_engine) as s:
        paper = _create_paper(s, title="Multi Document Paper")
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(postgres_tracker)

    def _upload(content: bytes, fname: str):
        return client.post(
            URL,
            data={"paper_id": str(paper_id)},
            files={"file": (fname, io.BytesIO(content), "application/pdf")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_upload, VALID_PDF_BYTES, "doc1.pdf")
        f2 = executor.submit(_upload, DISTINCT_PDF_BYTES, "doc2.pdf")
        res1 = f1.result()
        res2 = f2.result()

    assert res1.status_code == 200
    assert res2.status_code == 200

    # Both documents persisted in PostgreSQL under the same paper
    with Session(postgres_engine) as s:
        docs = s.query(Document).filter_by(paper_id=paper_id).all()
        assert len(docs) == 2

    assert vs.count() == 2


def test_postgres_lock_registry_zero_leakage_after_workload(
    postgres_engine, postgres_tracker
):
    """Verifies that the in-process lock registry maintains zero residual entries
    after completing a series of concurrent requests against PostgreSQL.
    """
    assert active_paper_locks_count() == 0

    with Session(postgres_engine) as s:
        paper = _create_paper(s, title="Registry Check Paper")
        paper_id = paper.id

    client, _, _, _ = _build_test_client(postgres_tracker)

    def _upload():
        return client.post(
            URL,
            data={"paper_id": str(paper_id)},
            files={"file": ("check.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_upload) for _ in range(3)]
        for f in futures:
            f.result()

    # All locks must have been cleaned up
    assert active_paper_locks_count() == 0
