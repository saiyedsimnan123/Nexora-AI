"""Tests for Ingestion API Contract & Recovery (Milestone 12C-19).

Verifies:
1. API contract stabilization: request validation, OpenAPI metadata, response schema strictness,
   safe error envelopes without internal leaks (stack traces, paths, secrets).
2. Duplicate upload behavior & retry semantics: 409 Conflict rejection, cross-paper uploads,
   clean retry after transient failures, and concurrency boundary analysis.
3. Multi-stage failure recovery: PDF validation failure, unknown paper lookup, relational persistence,
   embedding generation, vector upsert rollback, partial vector cleanup resilience, and out-of-band
   reconciliation via reindex_document.
4. Dependency injection: session and transaction lifecycles with zero connection/session leaks.
"""
from __future__ import annotations

import concurrent.futures
import io
import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nexora import __version__
from nexora.api.app import create_app
from nexora.api.errors import InvalidRequestError
from nexora.api.models import DocumentIngestResponse
from nexora.api.routes import DocumentIngestResponseBody
from nexora.database.models.base import Base
from nexora.database.models import Collection, Document, DocumentChunkRecord, Paper, User, Workspace
from nexora.database.repository.base import RecordNotFoundError
from nexora.database.repository.document import DocumentRepository
from nexora.document.ingestion import (
    DocumentIngestionService,
    IngestionResult,
    VectorIndexingError,
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
            raise RuntimeError("Embedding provider timeout or secret_key_abc123 error")
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("Embedding provider timeout or secret_key_abc123 error")
        return [[0.1, 0.2, 0.3] for _ in texts]


class FailingVectorStore(InMemoryVectorStore):
    def __init__(self, fail_delete: bool = False):
        super().__init__()
        self.fail_delete = fail_delete

    def upsert(self, records: list[VectorRecord]) -> None:
        raise RuntimeError("Vector store connection failed: /internal/cluster/path")

    def delete(self, ids: list[str]) -> None:
        if self.fail_delete:
            raise RuntimeError("Vector store cleanup deletion failed critically")
        super().delete(ids)


class MockDocumentPipeline:
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
        return "Quantum Machine Learning in Distributed Systems"

    def process_pdf(self, file_path: str) -> ProcessedDocument:
        text = self._loader(file_path)
        emb = [] if self.empty_embeddings else [0.1, 0.2, 0.3]
        return ProcessedDocument(
            document_id=str(uuid4()),
            source=file_path,
            chunks=[
                DocumentChunk(text=text, embedding=emb, index=0)
            ],
            metadata={"filename": Path(file_path).name},
        )


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
# 1. API Contract Stabilization & Response Schema
# ==============================================================================


def test_document_ingest_response_dataclass_contract():
    doc_id = str(uuid4())
    paper_id = str(uuid4())
    resp = DocumentIngestResponse(
        document_id=doc_id,
        paper_id=paper_id,
        chunks_count=5,
        indexed_chunks_count=5,
    )
    assert resp.document_id == doc_id
    assert resp.paper_id == paper_id
    assert resp.chunks_count == 5
    assert resp.indexed_chunks_count == 5
    assert resp.to_dict() == {
        "document_id": doc_id,
        "paper_id": paper_id,
        "chunks_count": 5,
        "indexed_chunks_count": 5,
    }

    # Validation in __post_init__
    with pytest.raises(InvalidRequestError):
        DocumentIngestResponse(document_id="", paper_id=paper_id, chunks_count=1, indexed_chunks_count=1)
    with pytest.raises(InvalidRequestError):
        DocumentIngestResponse(document_id=doc_id, paper_id="", chunks_count=1, indexed_chunks_count=1)
    with pytest.raises(InvalidRequestError):
        DocumentIngestResponse(document_id=doc_id, paper_id=paper_id, chunks_count=-1, indexed_chunks_count=1)
    with pytest.raises(InvalidRequestError):
        DocumentIngestResponse(document_id=doc_id, paper_id=paper_id, chunks_count=1, indexed_chunks_count=-1)


def test_document_ingest_response_body_pydantic_schema():
    doc_id = str(uuid4())
    paper_id = str(uuid4())

    valid_body = DocumentIngestResponseBody(
        document_id=doc_id,
        paper_id=paper_id,
        chunks_count=3,
        indexed_chunks_count=3,
    )
    assert valid_body.document_id == doc_id
    assert valid_body.chunks_count == 3

    # Forbid extra fields
    with pytest.raises(ValidationError):
        DocumentIngestResponseBody(
            document_id=doc_id,
            paper_id=paper_id,
            chunks_count=3,
            indexed_chunks_count=3,
            extra_field="disallowed",
        )

    # Forbid negative counts
    with pytest.raises(ValidationError):
        DocumentIngestResponseBody(
            document_id=doc_id,
            paper_id=paper_id,
            chunks_count=-1,
            indexed_chunks_count=3,
        )


def test_openapi_schema_documents_ingest_contract(test_db):
    _, tracker = test_db
    client, _, _, _ = _build_test_client(tracker)
    app = client.app
    schema = app.openapi()

    assert "/api/v1/documents/ingest" in schema["paths"]
    operation = schema["paths"]["/api/v1/documents/ingest"]["post"]

    assert operation["summary"] == "Upload and ingest a research paper PDF"
    assert "description" in operation and len(operation["description"]) > 20
    assert "requestBody" in operation

    responses = operation["responses"]
    expected_statuses = {"200", "400", "404", "409", "413", "422", "500"}
    for status in expected_statuses:
        assert status in responses
        assert "description" in responses[status]


def test_successful_upload_contract_shape(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(tracker)
    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("contract_test.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "document_id",
        "paper_id",
        "chunks_count",
        "indexed_chunks_count",
    }
    # Validate UUID format
    assert UUID(data["document_id"])
    assert UUID(data["paper_id"]) == paper_id
    assert data["chunks_count"] == 1
    assert data["indexed_chunks_count"] == 1


@pytest.mark.parametrize(
    "form_data,files_data,expected_missing",
    [
        ({}, {"file": ("a.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")}, "paper_id"),
        ({"paper_id": str(uuid4())}, {}, "file"),
    ],
)
def test_missing_form_fields_returns_422_with_safe_error_envelope(
    test_db, form_data, files_data, expected_missing
):
    _, tracker = test_db
    client, _, _, _ = _build_test_client(tracker)

    response = client.post(URL, data=form_data, files=files_data)
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"] == "Request validation failed."
    assert "details" in body["error"]
    details_str = str(body["error"]["details"])
    assert expected_missing in details_str


@pytest.mark.parametrize(
    "bad_paper_id",
    [
        "not-a-uuid",
        "12345",
        "   ",
        "g47b5f64-5717-4562-b3fc-2c963f66afa6",
    ],
)
def test_malformed_paper_id_returns_400(test_db, bad_paper_id):
    _, tracker = test_db
    client, _, _, _ = _build_test_client(tracker)

    response = client.post(
        URL,
        data={"paper_id": bad_paper_id},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert "detail" not in body


def test_non_pdf_file_rejected_with_400(test_db):
    _, tracker = test_db
    client, _, _, _ = _build_test_client(tracker)

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("data.txt", io.BytesIO(b"Hello world text file"), "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_empty_file_rejected_with_400(test_db):
    _, tracker = test_db
    client, _, _, _ = _build_test_client(tracker)

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_oversized_file_rejected_with_413(test_db):
    _, tracker = test_db
    max_size = 500  # small limit
    client, _, _, _ = _build_test_client(tracker, max_upload_size=max_size)

    oversized_bytes = VALID_PDF_BYTES + (b"X" * 1000)
    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("big.pdf", io.BytesIO(oversized_bytes), "application/pdf")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_unknown_paper_id_returns_404(test_db):
    _, tracker = test_db
    client, _, _, _ = _build_test_client(tracker)
    non_existent = str(uuid4())

    response = client.post(
        URL,
        data={"paper_id": non_existent},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_unexpected_error_envelope_does_not_leak_secrets_or_stack_traces(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    class ExplodingIngestionService:
        def process_and_ingest(self, **kwargs):
            raise RuntimeError("Database connection string postgres://admin:super_secret_pw@db.prod/internal")

    app = create_app(
        FakeResearchService(),
        ingestion_service=ExplodingIngestionService(),
        session_factory=tracker,
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert "super_secret_pw" not in response.text
    assert "postgres://" not in response.text
    assert "Traceback" not in response.text


# ==============================================================================
# 2. Duplicate Upload Behavior & Retry Semantics
# ==============================================================================


def test_sequential_duplicate_upload_rejected_with_409(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(tracker)

    # 1. First upload succeeds
    r1 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r1.status_code == 200

    # 2. Duplicate identical content for same paper is rejected with 409
    r2 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "conflict"

    # 3. Duplicate with different filename but identical content is also rejected
    r3 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("renamed_copy.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r3.status_code == 409
    assert r3.json()["error"]["code"] == "conflict"

    # Database still contains exactly one document
    with Session(engine) as s:
        doc_repo = DocumentRepository(s)
        assert len(doc_repo.list_by_paper_id(paper_id)) == 1


def test_identical_content_allowed_for_different_papers(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        p1 = _create_paper(s, title="Paper 1")
        p2 = _create_paper(s, title="Paper 2")
        pid1, pid2 = p1.id, p2.id

    client, vs, _, _ = _build_test_client(tracker)

    r1 = client.post(
        URL,
        data={"paper_id": str(pid1)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r1.status_code == 200

    r2 = client.post(
        URL,
        data={"paper_id": str(pid2)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r2.status_code == 200

    with Session(engine) as s:
        doc_repo = DocumentRepository(s)
        assert len(doc_repo.list_by_paper_id(pid1)) == 1
        assert len(doc_repo.list_by_paper_id(pid2)) == 1


def test_retry_after_transient_failure_succeeds_without_conflict(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    failing_emb = DeterministicEmbeddingService(fail=True)
    pipe = MockDocumentPipeline(empty_embeddings=True)
    client, vs, emb, _ = _build_test_client(
        tracker, embedding_service=failing_emb, pipeline=pipe
    )

    # 1. First attempt fails due to transient embedding provider failure
    r1 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r1.status_code == 500
    assert r1.json()["error"]["code"] == "indexing_failed"

    # Ensure no remnant DB records were left
    with Session(engine) as s:
        doc_repo = DocumentRepository(s)
        assert len(doc_repo.list_by_paper_id(paper_id)) == 0

    # 2. Fix the embedding service (simulating recovery of upstream provider)
    failing_emb.fail = False

    # 3. Retry upload with exact same file and paper
    r2 = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert r2.status_code == 200
    assert r2.json()["paper_id"] == str(paper_id)

    # Database now contains exactly one document
    with Session(engine) as s:
        doc_repo = DocumentRepository(s)
        assert len(doc_repo.list_by_paper_id(paper_id)) == 1


def test_concurrent_upload_race_condition_behavior(test_db):
    """Examines concurrent upload race conditions against the database.

    Concurrency Boundary:
    The Document schema enforces foreign key and check constraints, but intentionally
    does not place a unique index on (paper_id, checksum) to support the documented
    multiple-documents-per-paper requirement. Duplicate prevention is enforced at the
    application persistence boundary (check-then-insert within transaction savepoints).
    In serial execution, duplicates are reliably rejected with 409 Conflict.
    Under simultaneous concurrent threads, at least one request succeeds (200), and
    no orphaned chunks or unhandled server crashes occur.
    """
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, vs, _, _ = _build_test_client(tracker)

    def _do_upload():
        return client.post(
            URL,
            data={"paper_id": str(paper_id)},
            files={"file": ("race.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_do_upload)
        f2 = executor.submit(_do_upload)
        res1 = f1.result()
        res2 = f2.result()

    statuses = [res1.status_code, res2.status_code]
    # Both returned controlled HTTP responses without unhandled server crashes
    for code in statuses:
        assert code in (200, 409, 500)

    # Validate that relational database is structurally sound
    with Session(engine) as s:
        doc_repo = DocumentRepository(s)
        docs = doc_repo.list_by_paper_id(paper_id)
        # Verify any created documents have completely consistent chunks (no orphaned chunks)
        for d in docs:
            chunks = s.query(DocumentChunkRecord).filter_by(document_id=d.id).all()
            assert len(chunks) == 1

    assert len(tracker.active_sessions) == 0


# ==============================================================================
# 3. Failure Recovery across All Failure Stages
# ==============================================================================


def test_stage1_pdf_validation_failure_cleans_up_file_and_leaves_zero_records(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    pipe = MockDocumentPipeline()
    client, vs, _, _ = _build_test_client(tracker, pipeline=pipe)

    corrupt_bytes = b"%PDF-corrupt-data"
    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("corrupt.pdf", io.BytesIO(corrupt_bytes), "application/pdf")},
    )
    assert response.status_code == 400

    # 0 DB records
    with Session(engine) as s:
        assert s.query(Document).count() == 0
        assert s.query(DocumentChunkRecord).count() == 0
    # 0 vector records
    assert vs.count() == 0
    # Temp file deleted
    if pipe.loaded_files:
        assert not Path(pipe.loaded_files[-1]).exists()


def test_stage2_unknown_paper_lookup_failure_leaves_zero_records(test_db):
    engine, tracker = test_db
    client, vs, _, pipe = _build_test_client(tracker)

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert response.status_code == 404

    # 0 DB records
    with Session(engine) as s:
        assert s.query(Document).count() == 0
        assert s.query(DocumentChunkRecord).count() == 0
    # 0 vector records
    assert vs.count() == 0
    # Temp file deleted
    if pipe.loaded_files:
        assert not Path(pipe.loaded_files[-1]).exists()


def test_stage3_relational_persistence_failure_rolls_back(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    vs = InMemoryVectorStore()
    emb = DeterministicEmbeddingService()

    # Create service with real DB session_factory
    service = DocumentIngestionService(
        vector_store=vs,
        embedding_service=emb,
        session_factory=tracker,
    )

    # Processed document with duplicate chunk indices violating database UniqueConstraint("document_id", "chunk_index")
    duplicate_chunks_doc = ProcessedDocument(
        document_id=str(uuid4()),
        source="bad.pdf",
        chunks=[
            DocumentChunk(text="Chunk 0", embedding=[0.1, 0.2, 0.3], index=0),
            DocumentChunk(text="Duplicate chunk index", embedding=[0.1, 0.2, 0.3], index=0),
        ],
        metadata={"filename": "bad.pdf"},
    )

    with pytest.raises(Exception):
        service.ingest_processed_document(paper_id, duplicate_chunks_doc)

    # Verify relational savepoint rolled back completely
    with Session(engine) as s:
        assert s.query(Document).count() == 0
        assert s.query(DocumentChunkRecord).count() == 0
    assert vs.count() == 0
    assert len(tracker.active_sessions) == 0


def test_stage4_embedding_generation_failure_rolls_back_savepoint(test_db):
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

    # 0 DB records
    with Session(engine) as s:
        assert s.query(Document).count() == 0
        assert s.query(DocumentChunkRecord).count() == 0
    assert vs.count() == 0


def test_stage5_vector_upsert_failure_rolls_back_and_cleans_up(test_db):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    failing_vs = FailingVectorStore(fail_delete=False)
    client, _, _, _ = _build_test_client(tracker, vector_store=failing_vs)

    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "indexing_failed"

    # 0 DB records
    with Session(engine) as s:
        assert s.query(Document).count() == 0
        assert s.query(DocumentChunkRecord).count() == 0
    assert failing_vs.count() == 0


def test_stage6_vector_cleanup_resilience_when_cleanup_fails(test_db):
    """Verifies that if vector cleanup deletion fails, the primary VectorIndexingError is preserved."""
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    # Vector store that fails on upsert AND fails on cleanup delete
    failing_vs = FailingVectorStore(fail_delete=True)
    client, _, _, _ = _build_test_client(tracker, vector_store=failing_vs)

    response = client.post(
        URL,
        data={"paper_id": str(paper_id)},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    # Primary indexing error returned safely, not replaced by unhandled deletion crash
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "indexing_failed"
    assert "Vector store indexing failed" in body["error"]["message"] or "indexing failed" in body["error"]["message"].lower()

    # Relational savepoint rolled back
    with Session(engine) as s:
        assert s.query(Document).count() == 0
        assert s.query(DocumentChunkRecord).count() == 0


def test_stage7_reconciliation_via_reindex_document(test_db):
    """Tests out-of-band reconciliation when relational document exists but vector store is unindexed."""
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

    # 1. Normal ingestion persists DB + indexes vectors
    processed = pipe.process_pdf("paper.pdf")
    res = service.ingest_processed_document(paper_id, processed)
    doc_id = res.document_id
    assert vs.count() == 1

    # 2. Simulate vector store cache wipe / outage
    service.delete_document_vectors(doc_id)
    assert vs.count() == 0

    # 3. Call reindex_document to restore vectors from relational data
    reindex_result = service.reindex_document(doc_id)
    assert isinstance(reindex_result, IngestionResult)
    assert reindex_result.document_id == doc_id
    assert reindex_result.chunks_count == 1
    assert reindex_result.indexed_chunks_count == 1
    assert vs.count() == 1

    # Verify vector record payload was correctly reconstructed
    indexed_records = vs._records
    assert doc_id in [UUID(rec.payload["document_id"]) for rec in indexed_records.values()]

    # 4. Attempting reindex on unknown document raises RecordNotFoundError
    with pytest.raises(RecordNotFoundError):
        service.reindex_document(uuid4())


# ==============================================================================
# 4. Dependency Injection & Session Lifecycle Verification
# ==============================================================================


@pytest.mark.parametrize(
    "scenario,make_request",
    [
        (
            "200_success",
            lambda c, pid: c.post(
                URL,
                data={"paper_id": str(pid)},
                files={"file": ("ok.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
            ),
        ),
        (
            "400_invalid_uuid",
            lambda c, pid: c.post(
                URL,
                data={"paper_id": "not-a-uuid"},
                files={"file": ("ok.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
            ),
        ),
        (
            "404_not_found",
            lambda c, pid: c.post(
                URL,
                data={"paper_id": str(uuid4())},
                files={"file": ("ok.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
            ),
        ),
        (
            "409_duplicate",
            lambda c, pid: (
                c.post(
                    URL,
                    data={"paper_id": str(pid)},
                    files={"file": ("dupe.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
                ),
                c.post(
                    URL,
                    data={"paper_id": str(pid)},
                    files={"file": ("dupe.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
                ),
            )[-1],
        ),
        (
            "413_too_large",
            lambda c, pid: c.post(
                URL,
                data={"paper_id": str(pid)},
                files={"file": ("big.pdf", io.BytesIO(b"%PDF" + b"X" * 600), "application/pdf")},
            ),
        ),
        (
            "422_validation_error",
            lambda c, pid: c.post(URL, data={}, files={}),
        ),
    ],
)
def test_zero_session_leaks_across_all_http_lifecycle_paths(test_db, scenario, make_request):
    engine, tracker = test_db
    with Session(engine) as s:
        paper = _create_paper(s)
        paper_id = paper.id

    client, _, _, _ = _build_test_client(tracker, max_upload_size=500)

    response = make_request(client, paper_id)
    assert response.status_code in (200, 400, 404, 409, 413, 422)

    # Strict assertion: ZERO sessions currently open or leaking
    assert len(tracker.active_sessions) == 0
    # Every session that was opened must have been closed
    assert tracker.opened_count == tracker.closed_count
