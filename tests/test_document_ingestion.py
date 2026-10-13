"""Tests for end-to-end DocumentIngestionService (Milestone 12C-16)."""
from __future__ import annotations

from uuid import uuid4
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from nexora.database.models.base import Base
from nexora.database.models import Collection, Paper, User, Workspace
from nexora.database.repository.base import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from nexora.document.ingestion import (
    DocumentIngestionService,
    IngestionResult,
    VectorIndexingError,
)
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.service import DocumentPersistenceService
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import VectorRecord
from nexora.retrieval.service import RetrievalService


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")

    @event.listens_for(eng, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as db:
        yield db


@pytest.fixture
def vector_store():
    return InMemoryVectorStore()


class FakeEmbeddingService:
    """Deterministic embedding service producing unit-like mock vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        # Simple deterministic vector based on text length and char ordinals
        val = float(len(text) % 10 + 1)
        return [val * (i + 1) for i in range(self.dimension)]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


def _create_paper(session: Session, title: str = "Research Paper") -> Paper:
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
    return paper


def _make_processed_document(
    chunks: list[DocumentChunk] | None = None,
    source: str = "papers/quantum_ai.pdf",
    metadata: dict[str, object] | None = None,
) -> ProcessedDocument:
    if chunks is None:
        chunks = [
            DocumentChunk(
                text="Quantum computing principles and qubits",
                embedding=[1.0, 0.0, 0.0, 1.0],
                index=0,
            ),
            DocumentChunk(
                text="Neural network architectures and deep learning",
                embedding=[0.0, 1.0, 1.0, 0.0],
                index=1,
            ),
        ]
    return ProcessedDocument(
        document_id=str(uuid4()),
        source=source,
        chunks=chunks,
        metadata={"category": "ai", "pages": 10} if metadata is None else metadata,
    )


def test_successful_relational_persistence_and_vector_indexing(session, vector_store):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)
    ingestion = DocumentIngestionService(persistence, vector_store)

    doc_in = _make_processed_document()
    result = ingestion.ingest_processed_document(paper.id, doc_in)
    session.commit()

    assert isinstance(result, IngestionResult)
    assert result.paper_id == paper.id
    assert result.chunks_count == 2
    assert result.indexed_chunks_count == 2
    assert len(result.vector_ids) == 2

    # Verify relational database records
    db_doc = persistence.get_document(result.document_id)
    assert db_doc is not None
    assert db_doc.paper_id == paper.id
    db_chunks = persistence.list_chunks_for_document(result.document_id)
    assert len(db_chunks) == 2
    assert [c.chunk_index for c in db_chunks] == [0, 1]

    # Verify vector store records
    assert vector_store.count() == 2
    assert set(result.vector_ids) == {str(c.id) for c in db_chunks}


def test_end_to_end_retrieval_of_persisted_chunks(session, vector_store):
    paper = _create_paper(session, title="Quantum Deep Learning")
    persistence = DocumentPersistenceService(session)
    embedding_service = FakeEmbeddingService(dimension=4)
    ingestion = DocumentIngestionService(
        persistence, vector_store, embedding_service=embedding_service
    )

    doc_in = _make_processed_document(
        chunks=[
            DocumentChunk(
                text="Quantum computing circuits and superposition",
                embedding=[1.0, 0.0, 0.0, 0.0],
                index=0,
            ),
            DocumentChunk(
                text="Gradient descent optimization algorithms",
                embedding=[0.0, 0.0, 1.0, 0.0],
                index=1,
            ),
        ]
    )

    result = ingestion.ingest_processed_document(paper.id, doc_in)
    session.commit()

    # Now verify retrieval through the configured RetrievalService
    retrieval_service = RetrievalService(embedding_service, vector_store)

    # Search with a query vector close to chunk 0
    search_res = vector_store.search([1.0, 0.0, 0.0, 0.0], limit=1)
    assert len(search_res.results) == 1
    top_chunk = search_res.results[0]

    assert top_chunk.text == "Quantum computing circuits and superposition"
    assert top_chunk.document_id == str(result.document_id)
    assert top_chunk.chunk_id == str(result.chunk_records[0].id)
    assert top_chunk.metadata["paper_id"] == str(paper.id)
    assert top_chunk.metadata["chunk_index"] == 0
    assert top_chunk.metadata["category"] == "ai"


def test_chunks_without_embeddings_are_embedded_via_embedding_service(
    session, vector_store
):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)
    embedding_service = FakeEmbeddingService(dimension=4)
    ingestion = DocumentIngestionService(
        persistence, vector_store, embedding_service=embedding_service
    )

    # Chunks without embeddings
    chunks = [
        DocumentChunk(text="Chunk 1 text", embedding=[], index=0),
        DocumentChunk(text="Chunk 2 text", embedding=[], index=1),
    ]
    doc_in = _make_processed_document(chunks=chunks)

    result = ingestion.ingest_processed_document(paper.id, doc_in)
    session.commit()

    assert result.indexed_chunks_count == 2
    assert vector_store.count() == 2


def test_chunks_missing_embeddings_without_service_raises_error(
    session, vector_store
):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)
    ingestion = DocumentIngestionService(persistence, vector_store, embedding_service=None)

    chunks = [DocumentChunk(text="No embedding", embedding=[], index=0)]
    doc_in = _make_processed_document(chunks=chunks)

    with pytest.raises(ValueError, match="no embedding_service was configured"):
        ingestion.ingest_processed_document(paper.id, doc_in)

    # Database remains clean
    assert len(persistence.list_documents_for_paper(paper.id)) == 0


def test_vector_store_failure_rolls_back_relational_persistence(session):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)

    class FailingVectorStore:
        def upsert(self, records):
            raise RuntimeError("Vector store connection timed out")

        def delete(self, ids):
            pass

        def search(self, vector, *, limit=10):
            raise NotImplementedError

        def count(self):
            return 0

    ingestion = DocumentIngestionService(persistence, FailingVectorStore())
    doc_in = _make_processed_document()

    with pytest.raises(VectorIndexingError, match="Vector store indexing failed"):
        ingestion.ingest_processed_document(paper.id, doc_in)

    # Relational persistence was rolled back by savepoint
    assert len(persistence.list_documents_for_paper(paper.id)) == 0


def test_embedding_failure_rolls_back_and_raises_error(session, vector_store):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)

    class FailingEmbeddingService:
        def embed_texts(self, texts):
            raise ConnectionError("Embedding API rate limited")

    ingestion = DocumentIngestionService(
        persistence, vector_store, embedding_service=FailingEmbeddingService()
    )
    chunks = [DocumentChunk(text="Need embedding", embedding=[], index=0)]
    doc_in = _make_processed_document(chunks=chunks)

    with pytest.raises(VectorIndexingError, match="Failed to generate chunk embeddings"):
        ingestion.ingest_processed_document(paper.id, doc_in)

    assert len(persistence.list_documents_for_paper(paper.id)) == 0
    assert vector_store.count() == 0


def test_reindex_document_reconciles_and_retries(session, vector_store):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)
    embedding_service = FakeEmbeddingService(dimension=4)

    # 1. Persist only to database first (database-only workflow)
    doc_in = _make_processed_document()
    db_doc = persistence.persist_processed_document(paper.id, doc_in)
    session.commit()
    assert vector_store.count() == 0

    # 2. Reindex document into vector store
    ingestion = DocumentIngestionService(
        persistence, vector_store, embedding_service=embedding_service
    )
    reindex_res = ingestion.reindex_document(db_doc.id)
    assert reindex_res.indexed_chunks_count == 2
    assert vector_store.count() == 2

    # 3. Repeated reindexing is idempotent and does not create duplicate vectors
    reindex_res2 = ingestion.reindex_document(db_doc.id)
    assert reindex_res2.indexed_chunks_count == 2
    assert vector_store.count() == 2


def test_delete_document_vectors(session, vector_store):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)
    ingestion = DocumentIngestionService(persistence, vector_store)

    doc_in = _make_processed_document()
    res = ingestion.ingest_processed_document(paper.id, doc_in)
    session.commit()
    assert vector_store.count() == 2

    deleted_count = ingestion.delete_document_vectors(res.document_id)
    assert deleted_count == 2
    assert vector_store.count() == 0


def test_repeated_ingestion_duplicate_rejection(session, vector_store):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)
    ingestion = DocumentIngestionService(persistence, vector_store)

    doc_in = _make_processed_document(source="papers/unique_article.pdf")
    res1 = ingestion.ingest_processed_document(paper.id, doc_in)
    session.commit()
    assert res1.indexed_chunks_count == 2

    # Repeated ingestion with allow_duplicate=False raises DuplicateRecordError
    with pytest.raises(DuplicateRecordError):
        ingestion.ingest_processed_document(paper.id, doc_in, allow_duplicate=False)

    # Vector store count remains 2
    assert vector_store.count() == 2


def test_empty_document_ingestion(session, vector_store):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)
    ingestion = DocumentIngestionService(persistence, vector_store)

    empty_doc = _make_processed_document(chunks=[])
    res = ingestion.ingest_processed_document(paper.id, empty_doc)
    session.commit()

    assert res.chunks_count == 0
    assert res.indexed_chunks_count == 0
    assert res.vector_ids == []
    assert vector_store.count() == 0


def test_process_and_ingest_workflow(session, vector_store):
    paper = _create_paper(session)
    persistence = DocumentPersistenceService(session)
    ingestion = DocumentIngestionService(persistence, vector_store)

    class FakePipeline:
        def process_pdf(self, file_path: str) -> ProcessedDocument:
            return ProcessedDocument(
                document_id="doc-p1",
                source=file_path,
                chunks=[
                    DocumentChunk(
                        text="Pipeline extracted text",
                        embedding=[0.2, 0.4, 0.6, 0.8],
                        index=0,
                    )
                ],
                metadata={"filename": "extracted.pdf"},
            )

    result = ingestion.process_and_ingest(
        paper.id,
        "sample.pdf",
        pipeline=FakePipeline(),
    )
    session.commit()

    assert result.chunks_count == 1
    assert result.indexed_chunks_count == 1
    assert vector_store.count() == 1


def test_invalid_constructor_and_method_inputs(session, vector_store):
    persistence = DocumentPersistenceService(session)

    with pytest.raises(TypeError, match="persistence_service must be a DocumentPersistenceService"):
        DocumentIngestionService(object(), vector_store)

    with pytest.raises(TypeError, match="vector_store must implement VectorStore protocol"):
        DocumentIngestionService(persistence, object())

    ingestion = DocumentIngestionService(persistence, vector_store)

    with pytest.raises(TypeError, match="paper_id must be a UUID"):
        ingestion.ingest_processed_document("not-a-uuid", _make_processed_document())

    with pytest.raises(TypeError, match="processed_document must be a ProcessedDocument"):
        ingestion.ingest_processed_document(uuid4(), object())

    with pytest.raises(RecordNotFoundError):
        ingestion.ingest_processed_document(uuid4(), _make_processed_document())
