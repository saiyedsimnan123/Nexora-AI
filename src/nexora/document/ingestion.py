"""End-to-end document ingestion: relational persistence + embedding + vector store indexing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence
from uuid import UUID

from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.repository.base import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.service import DocumentPersistenceService
from nexora.retrieval.models import VectorRecord
from nexora.retrieval.store import VectorStore


class IngestionError(Exception):
    """Base exception for document ingestion errors."""


class VectorIndexingError(IngestionError):
    """Raised when embedding generation or vector store indexing fails."""


@dataclass(frozen=True)
class IngestionResult:
    """The result of an end-to-end document ingestion operation.

    Attributes:
        document_id: UUID of the persisted Document.
        paper_id: UUID of the owning Paper.
        chunks_count: Total chunks persisted in the relational database.
        indexed_chunks_count: Total chunks indexed in the vector store.
        vector_ids: List of vector store record identifiers (matching chunk UUIDs).
        document: The persisted Document ORM instance.
        chunk_records: List of persisted DocumentChunkRecord ORM instances.
    """

    document_id: UUID
    paper_id: UUID
    chunks_count: int
    indexed_chunks_count: int
    vector_ids: list[str]
    document: Document
    chunk_records: list[DocumentChunkRecord]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector_ids", list(self.vector_ids))
        object.__setattr__(self, "chunk_records", list(self.chunk_records))


class DocumentIngestionService:
    """Orchestrates end-to-end document ingestion:
    persists relational metadata, ensures chunk embeddings exist, and indexes
    vector records in the VectorStore.
    """

    def __init__(
        self,
        persistence_service: DocumentPersistenceService | None = None,
        vector_store: VectorStore | None = None,
        *,
        embedding_service: Any = None,
        session_factory: Any = None,
    ) -> None:
        if persistence_service is None and session_factory is None:
            raise TypeError("Either persistence_service or session_factory must be provided")

        if persistence_service is not None and not isinstance(
            persistence_service, DocumentPersistenceService
        ) and not hasattr(persistence_service, "persist_processed_document"):
            raise TypeError(
                f"persistence_service must be a DocumentPersistenceService, "
                f"got {type(persistence_service).__name__}"
            )
        if session_factory is not None and not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                f"vector_store must implement VectorStore protocol, "
                f"got {type(vector_store).__name__}"
            )
        if embedding_service is not None and not (
            callable(getattr(embedding_service, "embed_texts", None))
            or callable(getattr(embedding_service, "embed_text", None))
        ):
            raise TypeError("embedding_service must provide embed_texts or embed_text")

        self._persistence_service = persistence_service
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._embedding_service = embedding_service

    def _persist_and_index_records(
        self,
        session: Any,
        persistence_service: Any,
        paper_id: UUID,
        processed_document: ProcessedDocument,
        resolved_vectors: Sequence[Sequence[float]],
        *,
        filename: str | None = None,
        mime_type: str | None = None,
        file_size: int | None = None,
        checksum: str | None = None,
        storage_key: str | None = None,
        allow_duplicate: bool = False,
    ) -> tuple[Any, list[Any], list[str]]:
        vector_ids: list[str] = []
        try:
            with session.begin_nested():
                doc = persistence_service.persist_processed_document(
                    paper_id,
                    processed_document,
                    filename=filename,
                    mime_type=mime_type,
                    file_size=file_size,
                    checksum=checksum,
                    storage_key=storage_key,
                    allow_duplicate=allow_duplicate,
                )
                chunk_records = list(
                    persistence_service.list_chunks_for_document(doc.id)
                )

                if chunk_records:
                    records_to_upsert: list[VectorRecord] = []
                    for chunk_rec, vector in zip(chunk_records, resolved_vectors):
                        if not vector:
                            raise ValueError(
                                f"Empty embedding vector for chunk index {chunk_rec.chunk_index}"
                            )
                        rec_id = str(chunk_rec.id)
                        payload: dict[str, object] = {
                            "document_id": str(doc.id),
                            "chunk_id": rec_id,
                            "paper_id": str(paper_id),
                            "chunk_index": chunk_rec.chunk_index,
                            "text": chunk_rec.text,
                            "filename": doc.filename,
                            "source": doc.storage_key,
                        }
                        for k, v in processed_document.metadata.items():
                            if k not in payload and isinstance(k, str):
                                payload[k] = v

                        records_to_upsert.append(
                            VectorRecord(id=rec_id, vector=vector, payload=payload)
                        )
                        vector_ids.append(rec_id)

                    self._vector_store.upsert(records_to_upsert)
            return doc, chunk_records, vector_ids
        except (DuplicateRecordError, RecordNotFoundError, ValueError, TypeError):
            raise
        except Exception as exc:
            if vector_ids:
                try:
                    self._vector_store.delete(vector_ids)
                except Exception:
                    pass
            raise VectorIndexingError(
                f"Vector store indexing failed during ingestion: {exc}"
            ) from exc

    def ingest_processed_document(
        self,
        paper_id: UUID,
        processed_document: ProcessedDocument,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
        file_size: int | None = None,
        checksum: str | None = None,
        storage_key: str | None = None,
        allow_duplicate: bool = False,
        force_reembed: bool = False,
    ) -> IngestionResult:
        """Ingest a ProcessedDocument into both the relational database and the vector store.

        All relational persistence operations run within a managed transaction. If vector
        indexing fails, the relational changes are rolled back and partial vectors are cleaned
        up, ensuring failure atomicity.
        """
        if not isinstance(paper_id, UUID):
            raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")
        if not isinstance(processed_document, ProcessedDocument):
            raise TypeError(
                f"processed_document must be a ProcessedDocument, "
                f"got {type(processed_document).__name__}"
            )

        # 1. Resolve embeddings for all chunks (outside DB transaction to prevent holding DB locks)
        try:
            resolved_vectors = self._resolve_chunk_vectors(
                processed_document.chunks, force_reembed=force_reembed
            )
        except (ValueError, TypeError):
            raise
        except Exception as exc:
            raise VectorIndexingError(
                f"Failed to generate chunk embeddings: {exc}"
            ) from exc

        if self._session_factory is not None:
            from nexora.database.session import database_transaction
            from nexora.document.service import DocumentPersistenceService

            with database_transaction(self._session_factory) as session:
                persistence = DocumentPersistenceService(session)
                doc, chunk_records, vector_ids = self._persist_and_index_records(
                    session,
                    persistence,
                    paper_id,
                    processed_document,
                    resolved_vectors,
                    filename=filename,
                    mime_type=mime_type,
                    file_size=file_size,
                    checksum=checksum,
                    storage_key=storage_key,
                    allow_duplicate=allow_duplicate,
                )
        else:
            session = self._persistence_service._session
            doc, chunk_records, vector_ids = self._persist_and_index_records(
                session,
                self._persistence_service,
                paper_id,
                processed_document,
                resolved_vectors,
                filename=filename,
                mime_type=mime_type,
                file_size=file_size,
                checksum=checksum,
                storage_key=storage_key,
                allow_duplicate=allow_duplicate,
            )

        return IngestionResult(
            document_id=doc.id,
            paper_id=paper_id,
            chunks_count=len(chunk_records),
            indexed_chunks_count=len(vector_ids),
            vector_ids=vector_ids,
            document=doc,
            chunk_records=chunk_records,
        )

    def process_and_ingest(
        self,
        paper_id: UUID,
        file_path: str,
        pipeline: Any,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
        file_size: int | None = None,
        checksum: str | None = None,
        storage_key: str | None = None,
        allow_duplicate: bool = False,
        force_reembed: bool = False,
    ) -> IngestionResult:
        """Process a file via DocumentPipeline and perform end-to-end ingestion."""
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a str, got {type(file_path).__name__}")
        if not hasattr(pipeline, "process_pdf") or not callable(pipeline.process_pdf):
            raise TypeError("pipeline must provide process_pdf(file_path)")

        processed = pipeline.process_pdf(file_path)
        return self.ingest_processed_document(
            paper_id,
            processed,
            filename=filename,
            mime_type=mime_type,
            file_size=file_size,
            checksum=checksum,
            storage_key=storage_key,
            allow_duplicate=allow_duplicate,
            force_reembed=force_reembed,
        )

    def _do_reindex_document(
        self,
        persistence_service: Any,
        document_id: UUID,
        *,
        force_reembed: bool = True,
    ) -> IngestionResult:
        doc = persistence_service.get_document(document_id)
        if doc is None:
            raise RecordNotFoundError(f"Document with id {document_id} not found")

        chunk_records = list(
            persistence_service.list_chunks_for_document(document_id)
        )
        if not chunk_records:
            return IngestionResult(
                document_id=doc.id,
                paper_id=doc.paper_id,
                chunks_count=0,
                indexed_chunks_count=0,
                vector_ids=[],
                document=doc,
                chunk_records=[],
            )

        if self._embedding_service is None:
            raise ValueError(
                "reindex_document requires a configured embedding_service"
            )

        texts = [c.text for c in chunk_records]
        if hasattr(self._embedding_service, "embed_texts"):
            vectors = self._embedding_service.embed_texts(texts)
        else:
            vectors = [self._embedding_service.embed_text(t) for t in texts]

        records_to_upsert: list[VectorRecord] = []
        vector_ids: list[str] = []
        for chunk_rec, vector in zip(chunk_records, vectors):
            rec_id = str(chunk_rec.id)
            payload: dict[str, object] = {
                "document_id": str(doc.id),
                "chunk_id": rec_id,
                "paper_id": str(doc.paper_id),
                "chunk_index": chunk_rec.chunk_index,
                "text": chunk_rec.text,
                "filename": doc.filename,
                "source": doc.storage_key,
            }
            records_to_upsert.append(
                VectorRecord(id=rec_id, vector=vector, payload=payload)
            )
            vector_ids.append(rec_id)

        try:
            self._vector_store.upsert(records_to_upsert)
        except Exception as exc:
            raise VectorIndexingError(
                f"Failed to reindex document {document_id} in vector store: {exc}"
            ) from exc

        return IngestionResult(
            document_id=doc.id,
            paper_id=doc.paper_id,
            chunks_count=len(chunk_records),
            indexed_chunks_count=len(vector_ids),
            vector_ids=vector_ids,
            document=doc,
            chunk_records=chunk_records,
        )

    def reindex_document(
        self,
        document_id: UUID,
        *,
        force_reembed: bool = True,
    ) -> IngestionResult:
        """Reindex an already persisted document in the vector store.

        Reconciles relational chunk records with the vector store or retries
        previously failed indexing.
        """
        if not isinstance(document_id, UUID):
            raise TypeError(f"document_id must be a UUID, got {type(document_id).__name__}")

        if self._session_factory is not None:
            from nexora.database.session import database_transaction
            from nexora.document.service import DocumentPersistenceService

            with database_transaction(self._session_factory) as session:
                persistence = DocumentPersistenceService(session)
                return self._do_reindex_document(
                    persistence, document_id, force_reembed=force_reembed
                )
        return self._do_reindex_document(
            self._persistence_service, document_id, force_reembed=force_reembed
        )

    def delete_document_vectors(self, document_id: UUID) -> int:
        """Delete all indexed vectors for a document from the vector store."""
        if self._session_factory is not None:
            from nexora.database.session import database_transaction
            from nexora.document.service import DocumentPersistenceService

            with database_transaction(self._session_factory) as session:
                persistence = DocumentPersistenceService(session)
                chunk_records = persistence.list_chunks_for_document(document_id)
        else:
            chunk_records = self._persistence_service.list_chunks_for_document(document_id)

        ids = [str(c.id) for c in chunk_records]
        if ids:
            self._vector_store.delete(ids)
        return len(ids)

    def _resolve_chunk_vectors(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        force_reembed: bool = False,
    ) -> list[list[float]]:
        """Resolve embedding vectors for chunks, embedding if necessary."""
        if not chunks:
            return []

        needs_embedding = force_reembed or any(not c.embedding for c in chunks)
        if not needs_embedding:
            return [list(c.embedding) for c in chunks]

        if self._embedding_service is None:
            raise ValueError(
                "Document chunks are missing embeddings and no embedding_service was configured"
            )

        texts = [c.text for c in chunks]
        if hasattr(self._embedding_service, "embed_texts"):
            return self._embedding_service.embed_texts(texts)
        return [self._embedding_service.embed_text(t) for t in texts]
