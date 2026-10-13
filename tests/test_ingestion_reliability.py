"""Database-backed reliability tests for document ingestion (Milestone 12C-18).

Tests real database repositories, session/transaction lifecycles, PDF validation,
failure atomicity, rollback, and vector store reconciliation.
"""
from __future__ import annotations

import io
from pathlib import Path
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from nexora.api.app import create_app
from nexora.api.service import ResearchApplicationService
from nexora.database.models.base import Base
from nexora.database.models import Collection, Document, DocumentChunkRecord, Paper, User, Workspace
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.database.session import create_session_factory
from nexora.document.ingestion import (
    DocumentIngestionService,
    VectorIndexingError,
)
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import VectorRecord

URL = "/api/v1/documents/ingest"

# Standard minimal valid PDF binary
VALID_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n212\n%%EOF\n"
)


class SessionTracker(sessionmaker):
    """Wraps sessionmaker to monitor session creations and closures."""

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
    def __init__(self, dimension: int = 3, fail: bool = False):
        self.dimension = dimension
        self.fail = fail

    def embed_text(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError("Embedding provider connection timeout")
        return [1.0, 0.0, 0.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("Embedding provider connection timeout")
        return [[1.0, 0.0, 0.0] for _ in texts]


class FailingVectorStore(InMemoryVectorStore):
    def upsert(self, records: list[VectorRecord]) -> None:
        raise RuntimeError("Vector store unreachable or disk full")


class MockDocumentPipeline:
    """Document pipeline with controllable loader for valid and corrupt PDFs."""

    def __init__(self, fail_loader: bool = False, empty_embeddings: bool = False):
        self.fail_loader = fail_loader
        self.empty_embeddings = empty_embeddings
        self.loaded_files: list[str] = []

    def _loader(self, file_path: str) -> str:
        self.loaded_files.append(file_path)
        path = Path(file_path)
        if path.exists():
            with open(path, "rb") as f:
                content = f.read()
            if self.fail_loader or b"corrupt" in content or not content.startswith(b"%PDF"):
                raise ValueError("Could not read file as a PDF: corrupted stream")
        elif self.fail_loader:
            raise ValueError("Could not read file as a PDF: corrupted stream")
        return "Extracted research document text"

    def process_pdf(self, file_path: str) -> ProcessedDocument:
        text = self._loader(file_path)
        emb = [] if self.empty_embeddings else [1.0, 0.0, 0.0]
        return ProcessedDocument(
            document_id=str(uuid4()),
            source=file_path,
            chunks=[
                DocumentChunk(text=text, embedding=emb, index=0)
            ],
            metadata={"filename": Path(file_path).name},
        )


from sqlalchemy.pool import StaticPool


@pytest.fixture
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    tracker = SessionTracker(bind=engine)
    yield engine, tracker
    engine.dispose()


def _create_paper(session: Session, title: str = "Quantum AI Paper") -> Paper:
    user = User(email=f"{uuid4()}@example.com")
    session.add(user)
    session.flush()
    ws = Workspace(name="Workspace", user_id=user.id)
    session.add(ws)
    session.flush()
    col = Collection(name="Collection", workspace_id=ws.id)
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


# --------------------------------------------------------------------------
# 1. Real paper validation (database-backed)
# --------------------------------------------------------------------------


def test_unknown_paper_returns_404_and_leaves_no_orphaned_records(test_db):
    engine, tracker = test_db
    client, vs, emb, pipe = _build_test_client(tracker)

    non_existent_paper_id = uuid4()
    response = client.post(
        URL,
        data={"paper_id": str(non_existent_paper_id)},
        files={"file": ("sample.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"

    # Verify directly against database using a new session
    with Session(engine) as check_session:
        doc_repo = DocumentRepository(check_session)
        assert len(doc_repo.list_by_paper_id(non_existent_paper_id)) == 0
        assert check_session.query(DocumentChunkRecord).count() == 0

    # Verify session was closed and not left open
    assert len(tracker.active_sessions) == 0


def test_known_paper_ingestion_persists_relational_and_vector_records(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, emb, pipe = _build_test_client(tracker)

    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["paper_id"] == str(paper_id)
    assert body["chunks_count"] == 1
    assert body["indexed_chunks_count"] == 1
    doc_id = UUID(body["document_id"])

    # Verify database persistence
    with Session(engine) as check_session:
        doc_repo = DocumentRepository(check_session)
        chunk_repo = DocumentChunkRepository(check_session)
        doc = doc_repo.get_by_id(doc_id)
        assert doc is not None
        assert doc.paper_id == paper_id
        assert doc.filename == "paper.pdf"

        chunks = chunk_repo.list_by_document_id(doc_id)
        assert len(chunks) == 1
        assert chunks[0].text == "Extracted research document text"

    # Verify vector store indexing
    assert vs.count() == 1

    # Verify session lifecycle
    assert len(tracker.active_sessions) == 0


def test_duplicate_document_returns_409_and_preserves_single_record(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, emb, pipe = _build_test_client(tracker)

    # First upload succeeds
    r1 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("unique.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r1.status_code == 200

    # Second upload with same content and filename triggers duplicate detection
    r2 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("unique.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "conflict"

    # Database still contains exactly one document
    with Session(engine) as check_session:
        doc_repo = DocumentRepository(check_session)
        docs = doc_repo.list_by_paper_id(paper_id)
        assert len(docs) == 1

    assert len(tracker.active_sessions) == 0


# --------------------------------------------------------------------------
# 2. PDF validation and cleanup on failure
# --------------------------------------------------------------------------


def test_corrupt_pdf_returns_400_and_cleans_up_temporary_file(test_db, tmp_path):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    pipe = MockDocumentPipeline()
    client, vs, emb, _ = _build_test_client(tracker, pipeline=pipe)

    corrupt_bytes = b"%PDF-corrupt-data-broken-xref"
    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("corrupt.pdf", io.BytesIO(corrupt_bytes), "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"

    # Verify no records added to DB
    with Session(engine) as check_session:
        doc_repo = DocumentRepository(check_session)
        assert len(doc_repo.list_by_paper_id(paper_id)) == 0

    # Verify no active sessions left open
    assert len(tracker.active_sessions) == 0

    # Verify temp file was deleted
    if pipe.loaded_files:
        temp_file = pipe.loaded_files[-1]
        assert not Path(temp_file).exists()


def test_pipeline_loader_failure_returns_400(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    failing_pipe = MockDocumentPipeline(fail_loader=True)
    client, vs, emb, _ = _build_test_client(tracker, pipeline=failing_pipe)

    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("sample.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert len(tracker.active_sessions) == 0


# --------------------------------------------------------------------------
# 3. Partial failures, transaction rollback, and vector cleanup
# --------------------------------------------------------------------------


def test_embedding_failure_rolls_back_relational_persistence(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    failing_emb = DeterministicEmbeddingService(fail=True)
    pipe = MockDocumentPipeline(empty_embeddings=True)
    client, vs, _, _ = _build_test_client(
        tracker, embedding_service=failing_emb, pipeline=pipe
    )

    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "indexing_failed"

    # Verify database was rolled back: no documents or chunks
    with Session(engine) as check_session:
        doc_repo = DocumentRepository(check_session)
        assert len(doc_repo.list_by_paper_id(paper_id)) == 0
        assert check_session.query(DocumentChunkRecord).count() == 0

    assert vs.count() == 0
    assert len(tracker.active_sessions) == 0


def test_vector_store_failure_rolls_back_relational_persistence(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    failing_vs = FailingVectorStore()
    client, _, _, _ = _build_test_client(tracker, vector_store=failing_vs)

    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "indexing_failed"

    # Verify relational savepoint was rolled back
    with Session(engine) as check_session:
        doc_repo = DocumentRepository(check_session)
        assert len(doc_repo.list_by_paper_id(paper_id)) == 0
        assert check_session.query(DocumentChunkRecord).count() == 0

    assert len(tracker.active_sessions) == 0


# --------------------------------------------------------------------------
# 4. Session lifecycle closure across all paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario,paper_id_fn,file_content,expected_status",
    [
        ("success", lambda pid: str(pid), VALID_PDF_BYTES, 200),
        ("not_found", lambda pid: str(uuid4()), VALID_PDF_BYTES, 404),
        ("invalid_uuid", lambda pid: "not-a-uuid", VALID_PDF_BYTES, 400),
        ("empty_file", lambda pid: str(pid), b"", 400),
        ("corrupt_pdf", lambda pid: str(pid), b"corrupt-data", 400),
    ],
)
def test_session_closed_across_all_response_scenarios(
    test_db, scenario, paper_id_fn, file_content, expected_status
):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, _, _, _ = _build_test_client(tracker)

    response = client.post(
        URL,
        data={"paper_id": paper_id_fn(paper_id)},
        files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
    )

    assert response.status_code == expected_status
    # Verify no open sessions leak
    assert len(tracker.active_sessions) == 0


# --------------------------------------------------------------------------
# 5. Reindex and reconciliation with session_factory
# --------------------------------------------------------------------------


def test_reindex_document_with_session_factory(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    vs = InMemoryVectorStore()
    emb = DeterministicEmbeddingService()
    pipe = MockDocumentPipeline()

    service = DocumentIngestionService(
        vector_store=vs,
        embedding_service=emb,
        session_factory=tracker,
    )

    # 1. Ingest document
    processed = pipe.process_pdf("paper.pdf")
    res = service.ingest_processed_document(paper_id, processed)
    assert vs.count() == 1
    assert len(tracker.active_sessions) == 0

    # 2. Clear vector store
    service.delete_document_vectors(res.document_id)
    assert vs.count() == 0
    assert len(tracker.active_sessions) == 0

    # 3. Reconcile and reindex
    reindex_res = service.reindex_document(res.document_id)
    assert reindex_res.indexed_chunks_count == 1
    assert vs.count() == 1
    assert len(tracker.active_sessions) == 0

    # 4. Unknown document reindex raises RecordNotFoundError
    with pytest.raises(Exception):
        service.reindex_document(uuid4())
    assert len(tracker.active_sessions) == 0
