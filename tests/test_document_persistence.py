"""Tests for DocumentPersistenceService (Milestone 12C-15)."""
from __future__ import annotations

from uuid import uuid4
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from nexora.database.models.base import Base
from nexora.database.models import Collection, Paper, User, Workspace
from nexora.database.repository.base import (
    ConstraintError,
    DuplicateRecordError,
    RecordNotFoundError,
)
from nexora.database.repository.paper import PaperRepository
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.service import DocumentPersistenceService


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
    source: str = "papers/test.pdf",
    metadata: dict[str, object] | None = None,
) -> ProcessedDocument:
    if chunks is None:
        chunks = [
            DocumentChunk(text="First chunk introduction", embedding=[0.1, 0.2], index=0),
            DocumentChunk(text="Second chunk methodology", embedding=[0.3, 0.4], index=1),
        ]
    return ProcessedDocument(
        document_id=str(uuid4()),
        source=source,
        chunks=chunks,
        metadata={} if metadata is None else metadata,
    )


def test_successful_document_and_chunk_persistence(session):
    paper = _create_paper(session)
    service = DocumentPersistenceService(session)

    processed_doc = _make_processed_document()
    doc = service.persist_processed_document(paper.id, processed_doc)
    session.commit()

    assert doc.id is not None
    assert doc.paper_id == paper.id
    assert doc.filename == "test.pdf"
    assert doc.mime_type == "application/pdf"
    assert doc.file_size > 0
    assert len(doc.checksum) == 64
    assert doc.storage_key == "papers/test.pdf"

    # Query back via service
    retrieved = service.get_document(doc.id)
    assert retrieved is not None
    assert retrieved.id == doc.id

    # Verify chunks
    chunks = service.list_chunks_for_document(doc.id)
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "First chunk introduction"
    assert chunks[1].chunk_index == 1
    assert chunks[1].text == "Second chunk methodology"


def test_correct_relationship_and_ordering_of_saved_chunks(session):
    paper = _create_paper(session)
    service = DocumentPersistenceService(session)

    # Chunks created out of natural list order
    chunks = [
        DocumentChunk(text="Chunk 0 intro", embedding=[0.1], index=0),
        DocumentChunk(text="Chunk 1 methods", embedding=[0.2], index=1),
        DocumentChunk(text="Chunk 2 results", embedding=[0.3], index=2),
        DocumentChunk(text="Chunk 3 discussion", embedding=[0.4], index=3),
    ]
    processed_doc = _make_processed_document(chunks=chunks)
    doc = service.persist_processed_document(paper.id, processed_doc)
    session.commit()

    saved_chunks = service.list_chunks_for_document(doc.id)
    assert len(saved_chunks) == 4
    assert [c.chunk_index for c in saved_chunks] == [0, 1, 2, 3]
    assert [c.text for c in saved_chunks] == [
        "Chunk 0 intro",
        "Chunk 1 methods",
        "Chunk 2 results",
        "Chunk 3 discussion",
    ]


def test_empty_chunks_persistence(session):
    paper = _create_paper(session)
    service = DocumentPersistenceService(session)

    processed_doc = _make_processed_document(chunks=[])
    doc = service.persist_processed_document(paper.id, processed_doc)
    session.commit()

    assert doc.id is not None
    chunks = service.list_chunks_for_document(doc.id)
    assert len(chunks) == 0


def test_missing_paper_raises_record_not_found(session):
    service = DocumentPersistenceService(session)
    processed_doc = _make_processed_document()

    with pytest.raises(RecordNotFoundError, match="Paper with id"):
        service.persist_processed_document(uuid4(), processed_doc)


def test_invalid_input_types_rejected(session):
    paper = _create_paper(session)
    service = DocumentPersistenceService(session)

    with pytest.raises(TypeError, match="paper_id must be a UUID"):
        service.persist_processed_document("not-a-uuid", _make_processed_document())

    with pytest.raises(TypeError, match="processed_document must be a ProcessedDocument"):
        service.persist_processed_document(paper.id, {"not": "a ProcessedDocument"})


def test_invalid_chunk_text_raises_error_and_rolls_back(session):
    paper = _create_paper(session)
    service = DocumentPersistenceService(session)

    # Second chunk has blank text (violates model check)
    chunks = [
        DocumentChunk(text="Valid chunk", embedding=[0.1], index=0),
        DocumentChunk(text="   \n  ", embedding=[0.2], index=1),
    ]
    processed_doc = _make_processed_document(chunks=chunks)

    with pytest.raises(ValueError, match="text must not be blank"):
        service.persist_processed_document(paper.id, processed_doc)

    # Verify complete rollback: no document was persisted
    docs = service.list_documents_for_paper(paper.id)
    assert len(docs) == 0


def test_duplicate_chunk_index_in_same_document_rolls_back(session):
    paper = _create_paper(session)
    service = DocumentPersistenceService(session)

    # Duplicate index 0
    chunks = [
        DocumentChunk(text="Chunk index zero", embedding=[0.1], index=0),
        DocumentChunk(text="Another chunk index zero", embedding=[0.2], index=0),
    ]
    processed_doc = _make_processed_document(chunks=chunks)

    with pytest.raises(DuplicateRecordError):
        service.persist_processed_document(paper.id, processed_doc)

    # Verify complete rollback
    docs = service.list_documents_for_paper(paper.id)
    assert len(docs) == 0


def test_repeated_document_ingestion_behavior(session):
    paper = _create_paper(session)
    service = DocumentPersistenceService(session)

    processed_doc = _make_processed_document(source="papers/unique.pdf")
    doc1 = service.persist_processed_document(paper.id, processed_doc)
    session.commit()
    assert doc1.id is not None

    # Repeated ingestion with default allow_duplicate=False raises DuplicateRecordError
    with pytest.raises(DuplicateRecordError):
        service.persist_processed_document(paper.id, processed_doc, allow_duplicate=False)

    # Ingestion with allow_duplicate=True and distinct storage_key succeeds
    doc2 = service.persist_processed_document(
        paper.id,
        processed_doc,
        storage_key="papers/unique_v2.pdf",
        allow_duplicate=True,
    )
    session.commit()
    assert doc2.id != doc1.id

    docs = service.list_documents_for_paper(paper.id)
    assert len(docs) == 2


def test_process_and_persist_workflow(session):
    paper = _create_paper(session)
    service = DocumentPersistenceService(session)

    class FakePipeline:
        def process_pdf(self, file_path: str) -> ProcessedDocument:
            return ProcessedDocument(
                document_id="doc-fake-1",
                source=file_path,
                chunks=[
                    DocumentChunk(text="Extracted text from PDF", embedding=[0.5, 0.5], index=0)
                ],
                metadata={"filename": "custom.pdf", "mime_type": "application/pdf"},
            )

    doc = service.process_and_persist(
        paper.id,
        "sample/file.pdf",
        pipeline=FakePipeline(),
    )
    session.commit()

    assert doc.id is not None
    assert doc.filename == "custom.pdf"
    chunks = service.list_chunks_for_document(doc.id)
    assert len(chunks) == 1
    assert chunks[0].text == "Extracted text from PDF"


def test_process_and_persist_invalid_pipeline_rejected(session):
    paper = _create_paper(session)
    service = DocumentPersistenceService(session)

    with pytest.raises(TypeError, match="pipeline must provide process_pdf"):
        service.process_and_persist(paper.id, "file.pdf", pipeline=object())

    with pytest.raises(TypeError, match="file_path must be a str"):
        service.process_and_persist(paper.id, 12345, pipeline=object())
