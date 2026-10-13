"""Tests for POST /api/v1/documents/ingest and composition root ingestion wiring (Milestone 12C-17)."""
from __future__ import annotations

import io
import os
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from nexora.api.app import create_app
from nexora.api.errors import (
    FileTooLargeError,
    InvalidConfigurationError,
    InvalidRequestError,
)
from nexora.bootstrap.application import (
    build_http_app,
    build_ingestion_service,
)
from nexora.config.settings import AppSettings, ConfigurationError
from nexora.database.repository.base import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from nexora.document.ingestion import (
    DocumentIngestionService,
    IngestionResult,
    VectorIndexingError,
)
from nexora.llm.config import LLMConfig
from nexora.retrieval.models import RetrievedChunk, SearchResult
from nexora.retrieval.quality import RetrievalQualityPolicy


URL = "/api/v1/documents/ingest"
SECRET = "secret-token-key-123"


class FakeResearchService:
    def ask(self, request):
        raise NotImplementedError


class FakeIngestionResult:
    def __init__(self, doc_id: UUID, p_id: UUID, chunks: int = 3, indexed: int = 3):
        self.document_id = doc_id
        self.paper_id = p_id
        self.chunks_count = chunks
        self.indexed_chunks_count = indexed


class RecordingIngestionService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def process_and_ingest(self, paper_id, file_path, pipeline=None, filename=None):
        # Record whether file existed at call time
        file_existed = os.path.exists(file_path)
        content_len = os.path.getsize(file_path) if file_existed else 0
        self.calls.append(
            {
                "paper_id": paper_id,
                "file_path": file_path,
                "file_existed": file_existed,
                "content_len": content_len,
                "filename": filename,
                "pipeline": pipeline,
            }
        )
        if self.error:
            raise self.error
        if self.result:
            return self.result
        return FakeIngestionResult(uuid4(), paper_id, chunks=3, indexed=3)


def make_client(
    ingestion_service=None,
    research_service=None,
    max_upload_size=50 * 1024 * 1024,
    pipeline=None,
):
    ingestion_service = (
        ingestion_service if ingestion_service is not None else RecordingIngestionService()
    )
    research_service = (
        research_service if research_service is not None else FakeResearchService()
    )
    app = create_app(
        research_service,
        ingestion_service=ingestion_service,
        document_pipeline=pipeline,
        max_upload_size_bytes=max_upload_size,
    )
    return TestClient(app, raise_server_exceptions=False), ingestion_service


def test_successful_upload_and_ingestion():
    client, service = make_client()
    paper_id = str(uuid4())
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    response = client.post(
        URL,
        data={"paper_id": paper_id},
        files={"file": ("sample_paper.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["paper_id"] == paper_id
    assert data["chunks_count"] == 3
    assert data["indexed_chunks_count"] == 3

    # Verify service call arguments
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["paper_id"] == UUID(paper_id)
    assert call["filename"] == "sample_paper.pdf"
    assert call["file_existed"] is True
    assert call["content_len"] == len(pdf_content)

    # Verify temporary file cleanup after request finished
    assert not os.path.exists(call["file_path"])


def test_unknown_paper_id_returns_404():
    service = RecordingIngestionService(error=RecordNotFoundError("Paper not found in database"))
    client, _ = make_client(ingestion_service=service)

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("paper.pdf", io.BytesIO(b"%PDF dummy"), "application/pdf")},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert "not found" in body["error"]["message"].lower()


def test_malformed_paper_id_returns_400():
    client, service = make_client()

    response = client.post(
        URL,
        data={"paper_id": "not-a-valid-uuid"},
        files={"file": ("paper.pdf", io.BytesIO(b"%PDF dummy"), "application/pdf")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert len(service.calls) == 0


def test_missing_paper_id_returns_422():
    client, service = make_client()

    response = client.post(
        URL,
        files={"file": ("paper.pdf", io.BytesIO(b"%PDF dummy"), "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert len(service.calls) == 0


def test_non_pdf_file_rejected_with_400():
    client, service = make_client()

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("document.txt", io.BytesIO(b"some plain text"), "text/plain")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert len(service.calls) == 0


def test_empty_file_rejected_with_400():
    client, service = make_client()

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert len(service.calls) == 0


def test_oversized_file_rejected_with_413():
    # Max size = 50 bytes
    client, service = make_client(max_upload_size=50)

    oversized_content = b"X" * 100
    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("large.pdf", io.BytesIO(oversized_content), "application/pdf")},
    )

    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "file_too_large"
    assert len(service.calls) == 0


def test_temporary_file_cleaned_up_on_failure():
    service = RecordingIngestionService(error=RuntimeError(f"Unexpected crash with {SECRET}"))
    client, _ = make_client(ingestion_service=service)

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("sample.pdf", io.BytesIO(b"%PDF crash test"), "application/pdf")},
    )

    assert response.status_code == 500
    assert len(service.calls) == 1
    call = service.calls[0]
    # Check that temporary file was removed even when error was raised
    assert not os.path.exists(call["file_path"])
    # Check that secret and traceback were not leaked
    assert SECRET not in response.text
    assert "Traceback" not in response.text


def test_vector_indexing_failure_returns_500_indexing_failed():
    service = RecordingIngestionService(
        error=VectorIndexingError(f"Vector database unreachable: {SECRET}")
    )
    client, _ = make_client(ingestion_service=service)

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("sample.pdf", io.BytesIO(b"%PDF index test"), "application/pdf")},
    )

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "indexing_failed"
    assert SECRET not in response.text


def test_duplicate_document_returns_409_conflict():
    service = RecordingIngestionService(
        error=DuplicateRecordError("Document with checksum already exists")
    )
    client, _ = make_client(ingestion_service=service)

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("sample.pdf", io.BytesIO(b"%PDF dup test"), "application/pdf")},
    )

    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "conflict"


def test_unconfigured_ingestion_service_returns_500():
    # App created without ingestion_service
    app = create_app(FakeResearchService(), ingestion_service=None)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        URL,
        data={"paper_id": str(uuid4())},
        files={"file": ("sample.pdf", io.BytesIO(b"%PDF test"), "application/pdf")},
    )

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "server_misconfigured"


# --------------------------------------------------------------------------
# Composition Root Ingestion Tests
# --------------------------------------------------------------------------


class FakeEmbeddingService:
    def __init__(self, dimension=3):
        self.dimension = dimension

    def embed_text(self, text):
        return [1.0, 0.0, 0.0]

    def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.records = []

    def upsert(self, records):
        self.records.extend(records)

    def delete(self, ids):
        pass

    def search(self, vector, *, limit=10):
        return SearchResult(query="", results=[])

    def count(self):
        return len(self.records)


def _app_settings(dimension=3):
    return AppSettings(
        llm=LLMConfig(model="test-model", api_key="test-api-key"),
        embedding=SimpleNamespace(dimension=dimension),
        vector_store="memory",
    )


def test_build_ingestion_service_in_composition_root():
    settings = _app_settings(dimension=3)
    emb = FakeEmbeddingService(dimension=3)
    store = FakeVectorStore(dimension=3)

    class FakePersistenceService:
        def persist_processed_document(self, *args, **kwargs):
            pass

    service = build_ingestion_service(
        settings,
        persistence_service=FakePersistenceService(),
        embedding_service=emb,
        vector_store=store,
    )
    assert isinstance(service, DocumentIngestionService)


def test_build_ingestion_service_dimension_mismatch_rejected():
    settings = _app_settings(dimension=3)
    emb = FakeEmbeddingService(dimension=4)  # Mismatch with settings dimension 3
    store = FakeVectorStore(dimension=3)

    with pytest.raises(ConfigurationError, match="dimension 4 does not match"):
        build_ingestion_service(
            settings,
            persistence_service=object(),
            embedding_service=emb,
            vector_store=store,
        )


def test_build_http_app_wires_ingestion_service_to_api():
    settings = _app_settings(dimension=3)
    emb = FakeEmbeddingService(dimension=3)
    store = FakeVectorStore(dimension=3)

    recording_service = RecordingIngestionService()

    app = build_http_app(
        settings,
        embedding_service=emb,
        vector_store=store,
        llm_provider=SimpleNamespace(generate=lambda *a, **k: None),
        ingestion_service=recording_service,
    )
    client = TestClient(app)

    # 1. Health check works
    assert client.get("/health").json() == {"status": "ok"}

    # 2. Document ingestion works
    paper_id = str(uuid4())
    response = client.post(
        URL,
        data={"paper_id": paper_id},
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF content"), "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["paper_id"] == paper_id
    assert len(recording_service.calls) == 1
