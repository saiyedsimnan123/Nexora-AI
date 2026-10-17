"""Comprehensive tests for research document ingestion, provenance, and cross-paper isolation.

Nexora core principle:
"Nexora does not tell users what to believe. It shows what the available evidence justifies believing."

Verifies:
1. Reliable extraction of text and pages from research documents.
2. Clear handling of empty, malformed, unreadable, or text-poor documents.
3. Stable document IDs and chunk IDs.
4. Preservation of document title, source identity, page number, and chunk ordering.
5. Deterministic chunking under the same configuration.
6. Traceability from retrieved evidence chunk and RAG sources back to source document and page.
7. Cross-paper isolation: paper and document filters never leak chunks across papers.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import fitz  # PyMuPDF
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from nexora.database.models.base import Base
from nexora.database.models import Collection, Paper, User, Workspace
from nexora.document.chunker import (
    ProvenanceChunk,
    chunk_pages,
    chunk_text,
)
from nexora.document.loader import (
    DocumentPage,
    ExtractedDocument,
    load_pdf_document,
    load_pdf_pages,
    load_pdf_text,
)
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.pipeline import (
    DocumentPipeline,
    compute_chunk_id,
)
from nexora.document.service import DocumentPersistenceService
from nexora.document.ingestion import (
    DocumentIngestionService,
    IngestionResult,
)
from nexora.rag.models import AttributedSource, RAGRequest, RAGResponse
from nexora.rag.pipeline import RAGPipeline
from nexora.retrieval.bm25 import BM25Config, BM25Index
from nexora.retrieval.dataset import (
    DatasetValidationError,
    _validate_chunk,
)
from nexora.retrieval.evidence import EvidenceItem
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.service import RetrievalService


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------
def _create_sample_pdf(
    path: Path,
    pages_text: list[str],
    *,
    title: str | None = None,
    author: str | None = None,
) -> Path:
    """Create a temporary PDF file with specific page texts and optional metadata."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text.strip():
            page.insert_text((72, 72), text)
    if title or author:
        meta: dict[str, str] = {}
        if title:
            meta["title"] = title
        if author:
            meta["author"] = author
        doc.set_metadata(meta)
    doc.save(str(path))
    doc.close()
    return path


class FakeEmbeddingService:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        val = float(len(text) % 10 + 1)
        return [val * (i + 1) for i in range(self.dimension)]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def sqlite_session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _create_db_paper(session: Session, title: str = "Test Research Paper") -> Paper:
    user = User(email=f"{uuid4()}@example.com")
    session.add(user)
    session.flush()
    ws = Workspace(name="WS", user_id=user.id)
    session.add(ws)
    session.flush()
    col = Collection(name="Col", workspace_id=ws.id)
    session.add(col)
    session.flush()
    paper = Paper(collection_id=col.id, title=title)
    session.add(paper)
    session.flush()
    return paper


# ---------------------------------------------------------------------------
# 1. Extraction and Loader Provenance
# ---------------------------------------------------------------------------
def test_document_page_validation():
    page = DocumentPage(page_number=1, text="Introduction to transformers")
    assert page.page_number == 1
    assert page.text == "Introduction to transformers"

    with pytest.raises(ValueError, match="page_number must be a positive integer"):
        DocumentPage(page_number=0, text="Invalid page")

    with pytest.raises(TypeError, match="page_number must be an int"):
        DocumentPage(page_number="1", text="Invalid type")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="text must be a str"):
        DocumentPage(page_number=1, text=123)  # type: ignore[arg-type]


def test_extracted_document_provenance_and_metadata(tmp_path):
    pdf_path = tmp_path / "attention.pdf"
    _create_sample_pdf(
        pdf_path,
        [
            "Page 1: Attention mechanisms and transformer architecture.",
            "Page 2: Multi-head attention implementation details.",
            "Page 3: Experimental results on WMT 2014 translation.",
        ],
        title="Attention Is All You Need",
        author="Vaswani et al.",
    )

    doc = load_pdf_document(str(pdf_path))
    assert doc.total_pages == 3
    assert doc.title == "Attention Is All You Need"
    assert doc.author == "Vaswani et al."
    assert doc.has_extractable_text is True

    pages = doc.pages
    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert "Page 1" in pages[0].text
    assert pages[1].page_number == 2
    assert "Page 2" in pages[1].text
    assert pages[2].page_number == 3
    assert "Page 3" in pages[2].text

    # Also test load_pdf_pages convenience wrapper
    pages_direct = load_pdf_pages(str(pdf_path))
    assert len(pages_direct) == 3
    assert [p.page_number for p in pages_direct] == [1, 2, 3]


def test_handling_unreadable_or_empty_documents(tmp_path):
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_pdf_document(str(tmp_path / "missing.pdf"))

    # Corrupt / not a PDF
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"not a valid pdf header")
    with pytest.raises(ValueError, match="Could not read file as a PDF"):
        load_pdf_document(str(corrupt_path))

    # Empty file (0 bytes)
    empty_file = tmp_path / "empty_0bytes.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError):
        load_pdf_document(str(empty_file))

    # Scanned image-only / text-poor PDF (pages exist, but text is empty)
    text_poor_path = tmp_path / "scanned_image.pdf"
    _create_sample_pdf(text_poor_path, ["", "   \n\t  ", ""])
    doc = load_pdf_document(str(text_poor_path), require_extractable_text=False)
    assert doc.total_pages == 3
    assert doc.has_extractable_text is False

    # When require_extractable_text=True, it must raise a clear ValueError
    with pytest.raises(ValueError, match="contains no extractable text"):
        load_pdf_document(str(text_poor_path), require_extractable_text=True)


# ---------------------------------------------------------------------------
# 2. Deterministic Page-Aware Chunking
# ---------------------------------------------------------------------------
def test_chunk_pages_preserves_page_numbers_and_ordering():
    pages = [
        DocumentPage(page_number=1, text="Introduction to deep learning. Neural networks."),
        DocumentPage(page_number=2, text="Convolutional layers and feature extraction."),
        DocumentPage(page_number=3, text="Recurrent layers and sequential dependencies."),
    ]

    chunks = chunk_pages(pages, chunk_size=30, chunk_overlap=10)
    assert len(chunks) >= 3

    # Sequential ordering guarantees
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    # Exact page attribution guarantees
    for chunk in chunks:
        assert isinstance(chunk, ProvenanceChunk)
        assert chunk.page_number in (1, 2, 3)

    # First chunks must belong to page 1, middle to page 2, last to page 3
    p1_chunks = [c for c in chunks if c.page_number == 1]
    p2_chunks = [c for c in chunks if c.page_number == 2]
    p3_chunks = [c for c in chunks if c.page_number == 3]

    assert len(p1_chunks) > 0 and len(p2_chunks) > 0 and len(p3_chunks) > 0
    assert max(c.chunk_index for c in p1_chunks) < min(c.chunk_index for c in p2_chunks)
    assert max(c.chunk_index for c in p2_chunks) < min(c.chunk_index for c in p3_chunks)


def test_chunk_pages_is_deterministic():
    pages = [
        DocumentPage(page_number=1, text="Exact deterministic chunking repeatability test text."),
        DocumentPage(page_number=2, text="Second page content for deterministic output verification."),
    ]
    run1 = chunk_pages(pages, chunk_size=25, chunk_overlap=5)
    run2 = chunk_pages(pages, chunk_size=25, chunk_overlap=5)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.text == c2.text
        assert c1.chunk_index == c2.chunk_index
        assert c1.page_number == c2.page_number


def test_chunk_pages_skips_blank_pages():
    pages = [
        DocumentPage(page_number=1, text="First page content."),
        DocumentPage(page_number=2, text="   \n\t   "),  # blank
        DocumentPage(page_number=3, text="Third page content."),
    ]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].page_number == 3
    assert chunks[1].chunk_index == 1


def test_compute_chunk_id_stability():
    doc_id = "550e8400-e29b-41d4-a716-446655440000"
    chunk_0_a = compute_chunk_id(doc_id, 0)
    chunk_0_b = compute_chunk_id(doc_id, 0)
    chunk_1 = compute_chunk_id(doc_id, 1)

    assert chunk_0_a == chunk_0_b
    assert chunk_0_a != chunk_1
    assert isinstance(chunk_0_a, str) and len(chunk_0_a) == 36


# ---------------------------------------------------------------------------
# 3. Cross-Paper Isolation (Preventing Evidence Leakage)
# ---------------------------------------------------------------------------
def test_bm25_search_does_not_leak_chunks_across_papers():
    index = BM25Index()

    # Paper A: LoRA parameter efficient fine-tuning
    index.add_document(
        chunk_id="chunk_a1",
        text="LoRA freezes pre-trained weights and injects trainable rank decomposition matrices.",
        metadata={"paper_id": "paper_lora", "title": "LoRA Paper", "page_number": 1},
    )
    index.add_document(
        chunk_id="chunk_a2",
        text="Low-rank adaptation reduces trainable parameters by 10,000 times.",
        metadata={"paper_id": "paper_lora", "title": "LoRA Paper", "page_number": 2},
    )

    # Paper B: FlashAttention fast memory-efficient exact attention
    index.add_document(
        chunk_id="chunk_b1",
        text="FlashAttention uses tiling to reduce memory reads and writes between GPU HBM and SRAM.",
        metadata={"paper_id": "paper_flash", "title": "FlashAttention Paper", "page_number": 1},
    )
    index.add_document(
        chunk_id="chunk_b2",
        text="Memory-efficient attention speeds up pre-training without approximation.",
        metadata={"paper_id": "paper_flash", "title": "FlashAttention Paper", "page_number": 3},
    )

    # Query with paper filter for LoRA
    lora_result = index.search("attention trainable parameters", paper_id="paper_lora")
    assert len(lora_result.results) > 0
    for chunk in lora_result.results:
        assert chunk.metadata["paper_id"] == "paper_lora"
        assert chunk.metadata["title"] == "LoRA Paper"
        assert chunk.chunk_id in ("chunk_a1", "chunk_a2")

    # Query with paper filter for FlashAttention
    flash_result = index.search("attention trainable parameters", paper_id="paper_flash")
    assert len(flash_result.results) > 0
    for chunk in flash_result.results:
        assert chunk.metadata["paper_id"] == "paper_flash"
        assert chunk.metadata["title"] == "FlashAttention Paper"
        assert chunk.chunk_id in ("chunk_b1", "chunk_b2")


def test_vector_search_does_not_leak_chunks_across_papers():
    store = InMemoryVectorStore(dimension=2)

    # Both papers have identical vector [1.0, 0.0]
    store.upsert(
        [
            VectorRecord(
                id="rec_paper_1",
                vector=[1.0, 0.0],
                payload={"paper_id": "paper_1", "text": "Paper 1 content", "page_number": 1},
            ),
            VectorRecord(
                id="rec_paper_2",
                vector=[1.0, 0.0],
                payload={"paper_id": "paper_2", "text": "Paper 2 content", "page_number": 5},
            ),
        ]
    )

    # Filter to paper_1
    res1 = store.search([1.0, 0.0], limit=10, filter={"paper_id": "paper_1"})
    assert len(res1.results) == 1
    assert res1.results[0].chunk_id == "rec_paper_1"
    assert res1.results[0].metadata["paper_id"] == "paper_1"

    # Filter to paper_2
    res2 = store.search([1.0, 0.0], limit=10, filter={"paper_id": "paper_2"})
    assert len(res2.results) == 1
    assert res2.results[0].chunk_id == "rec_paper_2"
    assert res2.results[0].metadata["paper_id"] == "paper_2"


def test_hybrid_retrieval_service_cross_paper_filtering():
    embedding = FakeEmbeddingService(dimension=2)
    store = InMemoryVectorStore(dimension=2)
    bm25 = BM25Index()

    # Paper A
    v_a = embedding.embed_text("Deep neural networks and residual learning")
    store.upsert(
        [
            VectorRecord(
                id="res_a",
                vector=v_a,
                payload={"paper_id": "resnet_id", "title": "ResNet", "page_number": 2, "text": "Residual networks"},
            )
        ]
    )
    bm25.add_document(
        chunk_id="res_a",
        text="Residual networks and deep learning",
        metadata={"paper_id": "resnet_id", "title": "ResNet", "page_number": 2},
    )

    # Paper B
    v_b = embedding.embed_text("Deep neural networks and transformer learning")
    store.upsert(
        [
            VectorRecord(
                id="trans_b",
                vector=v_b,
                payload={"paper_id": "transformer_id", "title": "Transformer", "page_number": 4, "text": "Attention transformers"},
            )
        ]
    )
    bm25.add_document(
        chunk_id="trans_b",
        text="Attention transformers and deep learning",
        metadata={"paper_id": "transformer_id", "title": "Transformer", "page_number": 4},
    )

    retrieval = RetrievalService(embedding, store, bm25_index=bm25)

    # Hybrid retrieval filtered strictly to resnet_id
    search_res = retrieval.retrieve("deep learning neural", mode="hybrid", paper_id="resnet_id")
    assert len(search_res.results) > 0
    for chunk in search_res.results:
        assert chunk.metadata["paper_id"] == "resnet_id"
        assert chunk.page_number == 2
        assert chunk.title == "ResNet"


# ---------------------------------------------------------------------------
# 4. End-to-End Ingestion and Provenance Flow
# ---------------------------------------------------------------------------
def test_end_to_end_provenance_from_ingestion_to_rag(sqlite_session):
    paper = _create_db_paper(sqlite_session, title="Grounded Research Paper")
    persistence = DocumentPersistenceService(sqlite_session)
    store = InMemoryVectorStore(dimension=4)
    embedding = FakeEmbeddingService(dimension=4)
    bm25 = BM25Index()

    ingestion_service = DocumentIngestionService(
        persistence, store, embedding_service=embedding
    )

    # Create ProcessedDocument with chunk page provenance
    doc = ProcessedDocument(
        document_id="doc_provenance_123",
        source="papers/grounded_research.pdf",
        chunks=[
            DocumentChunk(
                text="Evidence chunk describing empirical methodology and benchmarks.",
                embedding=[1.0, 0.0, 0.0, 0.0],
                index=0,
            ),
            DocumentChunk(
                text="Second chunk outlining statistical significance and p-values.",
                embedding=[0.0, 1.0, 0.0, 0.0],
                index=1,
            ),
        ],
        metadata={
            "title": "Grounded Research Paper",
            "total_pages": 5,
            "chunk_pages": [1, 3],  # Chunk 0 on page 1, chunk 1 on page 3
        },
    )

    ingest_result = ingestion_service.ingest_processed_document(paper.id, doc)
    sqlite_session.commit()

    assert ingest_result.chunks_count == 2
    assert ingest_result.indexed_chunks_count == 2

    # Verify that database chunk IDs are deterministic
    c0_rec = persistence.list_chunks_for_document(ingest_result.document_id)[0]
    expected_chunk_0_uuid = compute_chunk_id(str(ingest_result.document_id), 0)
    assert str(c0_rec.id) == expected_chunk_0_uuid

    # Index into BM25 for hybrid testing
    for rec in ingest_result.chunk_records:
        bm25.add_document(
            chunk_id=str(rec.id),
            text=rec.text,
            document_id=str(ingest_result.document_id),
            metadata={
                "paper_id": str(paper.id),
                "title": "Grounded Research Paper",
                "page_number": 1 if rec.chunk_index == 0 else 3,
                "page": 1 if rec.chunk_index == 0 else 3,
            },
        )

    # Retrieval Service
    retrieval_service = RetrievalService(embedding, store, bm25_index=bm25)
    rag = RAGPipeline(retrieval_service, default_retrieval_mode="hybrid")

    response = rag.answer("empirical methodology", paper_id=str(paper.id))
    assert response.status == "success"
    assert len(response.sources) > 0

    top_source = response.sources[0]
    assert isinstance(top_source, AttributedSource)
    assert top_source.paper_id == str(paper.id)
    assert top_source.paper_title == "Grounded Research Paper"
    assert top_source.page_number == 1
    assert top_source.to_dict()["page_number"] == 1


def test_dataset_chunk_validation_preserves_page_number():
    raw_chunk = {
        "chunk_id": "chunk_pilot_1",
        "paper_id": "paper_vaswani_2017",
        "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        "title": "Attention Is All You Need",
        "page_number": 1,
        "publication_year": 2017,
        "doi": "10.48550/arXiv.1706.03762",
    }
    seen: set[str] = set()
    validated = _validate_chunk(raw_chunk, 0, seen)

    assert validated["chunk_id"] == "chunk_pilot_1"
    assert validated["page_number"] == 1
    assert validated["title"] == "Attention Is All You Need"
    assert validated["doi"] == "10.48550/arXiv.1706.03762"

    # Reject non-positive page_number
    invalid_chunk = dict(raw_chunk)
    invalid_chunk["chunk_id"] = "chunk_pilot_2"
    invalid_chunk["page_number"] = 0
    with pytest.raises(DatasetValidationError, match="must be a positive integer"):
        _validate_chunk(invalid_chunk, 1, seen)
