"""Document persistence service.

Connects document extraction and processing outputs to database persistence,
maintaining transaction boundaries and data integrity.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.repository.base import (
    DuplicateRecordError,
    RecordNotFoundError,
    RepositoryError,
)
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.document.models import DocumentChunk, ProcessedDocument

if TYPE_CHECKING:
    from nexora.document.pipeline import DocumentPipeline


class DocumentPersistenceService:
    """Service coordinating document extraction results and relational database persistence.

    All relational persistence operations execute within a transaction boundary
    (savepoint) so that partial persistence never leaves orphaned documents or
    incomplete chunk sequences.
    """

    def __init__(
        self,
        session: Session,
        *,
        paper_repository: PaperRepository | None = None,
        document_repository: DocumentRepository | None = None,
        chunk_repository: DocumentChunkRepository | None = None,
    ) -> None:
        self._session = session
        self._paper_repo = (
            paper_repository if paper_repository is not None else PaperRepository(session)
        )
        self._doc_repo = (
            document_repository
            if document_repository is not None
            else DocumentRepository(session)
        )
        self._chunk_repo = (
            chunk_repository
            if chunk_repository is not None
            else DocumentChunkRepository(session)
        )

    def persist_processed_document(
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
    ) -> Document:
        """Persist a ProcessedDocument and its chunks under an existing Paper atomically.

        Args:
            paper_id: The UUID of the owning paper.
            processed_document: The ProcessedDocument containing chunks and metadata.
            filename: Optional filename override (falls back to metadata or source name).
            mime_type: Optional MIME type override (defaults to "application/pdf").
            file_size: Optional file size in bytes (defaults to metadata or chunk text length).
            checksum: Optional 64-character SHA-256 hex checksum (computed if omitted).
            storage_key: Optional storage key (defaults to metadata or generated path).
            allow_duplicate: If False, rejects persistence if the checksum or storage_key
                is already registered for this paper.

        Returns:
            The newly created and flushed Document instance.

        Raises:
            TypeError: If input types are invalid.
            RecordNotFoundError: If the owning Paper does not exist.
            DuplicateRecordError: If duplicate document or chunk index is detected.
            ValueError / ConstraintError: If metadata or chunk constraints are violated.
        """
        if not isinstance(paper_id, UUID):
            raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")
        if not isinstance(processed_document, ProcessedDocument):
            raise TypeError(
                f"processed_document must be a ProcessedDocument, got {type(processed_document).__name__}"
            )

        # 1. Verify that owning paper exists (and acquire row lock in supported databases)
        if hasattr(self._paper_repo, "get_by_id_for_update") and callable(self._paper_repo.get_by_id_for_update):
            paper = self._paper_repo.get_by_id_for_update(paper_id)
        else:
            paper = self._paper_repo.get_by_id(paper_id)
        if paper is None:
            raise RecordNotFoundError(f"Paper with id {paper_id} not found")

        # 2. Derive metadata with safe fallbacks
        resolved_filename = (
            filename
            or (
                str(processed_document.metadata["filename"])
                if "filename" in processed_document.metadata
                else None
            )
            or (Path(processed_document.source).name if processed_document.source else None)
            or "document.pdf"
        )

        resolved_mime_type = (
            mime_type
            or (
                str(processed_document.metadata["mime_type"])
                if "mime_type" in processed_document.metadata
                else None
            )
            or "application/pdf"
        )

        if file_size is not None:
            resolved_file_size = file_size
        elif "file_size" in processed_document.metadata and isinstance(
            processed_document.metadata["file_size"], int
        ):
            resolved_file_size = int(processed_document.metadata["file_size"])
        else:
            resolved_file_size = sum(
                len(c.text.encode("utf-8")) for c in processed_document.chunks
            )

        if checksum is not None:
            resolved_checksum = checksum
        elif "checksum" in processed_document.metadata and isinstance(
            processed_document.metadata["checksum"], str
        ):
            resolved_checksum = str(processed_document.metadata["checksum"])
        else:
            hasher = hashlib.sha256()
            hasher.update(processed_document.source.encode("utf-8"))
            for c in processed_document.chunks:
                hasher.update(c.text.encode("utf-8"))
            resolved_checksum = hasher.hexdigest()

        resolved_storage_key = (
            storage_key
            or (
                str(processed_document.metadata["storage_key"])
                if "storage_key" in processed_document.metadata
                else None
            )
            or (processed_document.source if processed_document.source else None)
            or f"papers/{paper_id}/{resolved_filename}"
        )

        # 3. Check duplicate document ingestion
        if not allow_duplicate:
            existing_doc = self._doc_repo.get_by_checksum_and_paper(
                paper_id, resolved_checksum
            )
            if existing_doc is not None:
                raise DuplicateRecordError(
                    f"Document with checksum {resolved_checksum} already exists for paper {paper_id}"
                )
            existing_key = self._doc_repo.get_by_storage_key(resolved_storage_key)
            if existing_key is not None and existing_key.paper_id == paper_id:
                raise DuplicateRecordError(
                    f"Document with storage_key {resolved_storage_key} already exists for paper {paper_id}"
                )

        # 4. Atomic persistence using nested transaction savepoint
        with self._session.begin_nested():
            doc = self._doc_repo.create(
                paper_id=paper_id,
                filename=resolved_filename,
                mime_type=resolved_mime_type,
                file_size=resolved_file_size,
                checksum=resolved_checksum,
                storage_key=resolved_storage_key,
            )

            # Insert chunks in index order
            for chunk in processed_document.chunks:
                if not isinstance(chunk, DocumentChunk):
                    raise TypeError(
                        f"Expected DocumentChunk, got {type(chunk).__name__}"
                    )
                self._chunk_repo.create(
                    document_id=doc.id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                )

        return doc

    def process_and_persist(
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
    ) -> Document:
        """Process a file through a DocumentPipeline and persist it atomically."""
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a str, got {type(file_path).__name__}")
        if not hasattr(pipeline, "process_pdf") or not callable(pipeline.process_pdf):
            raise TypeError("pipeline must provide process_pdf(file_path)")

        processed = pipeline.process_pdf(file_path)
        return self.persist_processed_document(
            paper_id,
            processed,
            filename=filename,
            mime_type=mime_type,
            file_size=file_size,
            checksum=checksum,
            storage_key=storage_key,
            allow_duplicate=allow_duplicate,
        )

    def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID, or None."""
        return self._doc_repo.get_by_id(document_id)

    def list_documents_for_paper(
        self,
        paper_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Document]:
        """List documents for a paper with validated pagination."""
        return self._doc_repo.list_by_paper_id(paper_id, offset=offset, limit=limit)

    def list_chunks_for_document(
        self,
        document_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[DocumentChunkRecord]:
        """List chunks for a document in index order with validated pagination."""
        return self._chunk_repo.list_by_document_id(
            document_id, offset=offset, limit=limit
        )

    def update_document_indexing_status(
        self,
        document_id: UUID,
        status: Any,
        *,
        error: str | None = None,
        indexed_at: Any = None,
        attempts_increment: int = 0,
    ) -> Document:
        """Update a document's durable indexing status."""
        return self._doc_repo.update_indexing_status(
            document_id,
            status,
            error=error,
            indexed_at=indexed_at,
            attempts_increment=attempts_increment,
        )

    def list_unindexed_documents(
        self,
        status: Sequence[str] = ("pending", "failed"),
        *,
        limit: int = 100,
        max_attempts: int | None = None,
        for_update: bool = False,
        skip_locked: bool = False,
    ) -> Sequence[Document]:
        """List unindexed documents (pending or failed) for bounded recovery."""
        return self._doc_repo.list_unindexed(
            status=status,
            limit=limit,
            max_attempts=max_attempts,
            for_update=for_update,
            skip_locked=skip_locked,
        )

    def claim_unindexed_documents(
        self,
        limit: int = 100,
        *,
        max_attempts: int | None = None,
        claim_timeout_seconds: float = 300.0,
        claim_token: str | None = None,
    ) -> Sequence[Document]:
        """Atomically claim a bounded batch of unindexed documents with row-level locking."""
        return self._doc_repo.claim_unindexed(
            limit=limit,
            max_attempts=max_attempts,
            claim_timeout_seconds=claim_timeout_seconds,
            claim_token=claim_token,
        )

    def claim_document_for_recovery(
        self,
        document_id: UUID,
        *,
        claim_timeout_seconds: float = 300.0,
        claim_token: str | None = None,
    ) -> Document | None:
        """Atomically claim a specific document for recovery."""
        return self._doc_repo.claim_document(
            document_id,
            claim_timeout_seconds=claim_timeout_seconds,
            claim_token=claim_token,
        )

    def finalize_document_indexing(
        self,
        document_id: UUID,
        status: Any,
        *,
        claim_token: str | None = None,
        error: str | None = None,
        indexed_at: Any = None,
    ) -> Document | None:
        """Finalize indexing outcome for a claimed document."""
        return self._doc_repo.finalize_indexing(
            document_id,
            status,
            claim_token=claim_token,
            error=error,
            indexed_at=indexed_at,
        )

    def count_indexing_status(self) -> dict[str, int]:
        """Return counts of documents grouped by indexing status."""
        return self._doc_repo.count_by_indexing_status()
