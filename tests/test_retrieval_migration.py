"""Tests for Vector Store Migration and Document Chunk Re-Indexing Service."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexora.retrieval.migration import ReindexingReport, ReindexingService
from nexora.retrieval.models import VectorRecord


class FakeEmbeddingService:
    def __init__(self, dimension: int = 768) -> None:
        self.provider = "gemini"
        self.model = "gemini-embedding-001"
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1] * self.dimension for _ in texts]


class FakeVectorStore:
    def __init__(self, dimension: int = 768) -> None:
        self._collection = "research_papers_gemini"
        self.dimension = dimension
        self.upserted_records: list[VectorRecord] = []
        self.ensured = False
        self.validated = False

    def ensure_collection(self) -> None:
        self.ensured = True

    def validate_collection_compatibility(self) -> None:
        self.validated = True

    def upsert(self, records: list[VectorRecord]) -> None:
        self.upserted_records.extend(records)


def test_reindexing_service_successful_migration(monkeypatch):
    mock_chunks = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "text": "Attention Is All You Need introduction.",
            "paper_id": "paper-1",
            "filename": "attention.pdf",
        },
        {
            "id": "chunk-2",
            "document_id": "doc-1",
            "chunk_index": 1,
            "text": "Multi-head attention mechanism description.",
            "paper_id": "paper-1",
            "filename": "attention.pdf",
        },
        {
            "id": "chunk-3",
            "document_id": "doc-2",
            "chunk_index": 0,
            "text": "LoRA: Low-Rank Adaptation of Large Language Models.",
            "paper_id": "paper-2",
            "filename": "lora.pdf",
        },
    ]

    session_factory = MagicMock()
    emb_service = FakeEmbeddingService(dimension=768)
    vector_store = FakeVectorStore(dimension=768)

    service = ReindexingService(
        session_factory=session_factory,
        embedding_service=emb_service,
        vector_store=vector_store,
        batch_size=2,
    )
    # Mock fetching
    monkeypatch.setattr(service, "_fetch_all_chunks", lambda: mock_chunks)

    report = service.reindex_all()

    assert isinstance(report, ReindexingReport)
    assert report.status == "completed"
    assert report.total_chunks == 3
    assert report.total_documents == 2
    assert report.collection_name == "research_papers_gemini"
    assert report.dimension == 768
    assert report.provider == "gemini"
    assert report.model == "gemini-embedding-001"
    assert report.error is None

    assert vector_store.ensured is True
    assert vector_store.validated is True
    assert len(vector_store.upserted_records) == 3

    r1 = vector_store.upserted_records[0]
    assert r1.id == "chunk-1"
    assert len(r1.vector) == 768
    assert r1.payload["document_id"] == "doc-1"
    assert r1.payload["chunk_id"] == "chunk-1"
    assert r1.payload["embedding_provider"] == "gemini"
    assert r1.payload["embedding_model"] == "gemini-embedding-001"
    assert r1.payload["dimension"] == 768


def test_reindexing_empty_database(monkeypatch):
    session_factory = MagicMock()
    emb_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    service = ReindexingService(
        session_factory=session_factory,
        embedding_service=emb_service,
        vector_store=vector_store,
    )
    monkeypatch.setattr(service, "_fetch_all_chunks", lambda: [])

    report = service.reindex_all()
    assert report.status == "empty"
    assert report.total_chunks == 0
    assert report.total_documents == 0
    assert len(vector_store.upserted_records) == 0


def test_reindexing_handles_fetch_error(monkeypatch):
    session_factory = MagicMock()
    emb_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    service = ReindexingService(
        session_factory=session_factory,
        embedding_service=emb_service,
        vector_store=vector_store,
    )
    monkeypatch.setattr(
        service,
        "_fetch_all_chunks",
        MagicMock(side_effect=RuntimeError("DB Connection Dropped")),
    )

    report = service.reindex_all()
    assert report.status == "failed"
    assert "Database fetch failed" in str(report.error)
