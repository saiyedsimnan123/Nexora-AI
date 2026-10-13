"""DocumentChunk repository."""
from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import delete, func, select

from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.repository.base import BaseRepository


class DocumentChunkRepository(BaseRepository[DocumentChunkRecord]):
    """Repository for DocumentChunkRecord models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, DocumentChunkRecord)

    def create(
        self,
        *,
        document_id: UUID,
        chunk_index: int,
        text: str,
    ) -> DocumentChunkRecord:
        """Create a new document chunk record."""
        return self._create(
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
        )

    def get_by_document_and_index(
        self,
        document_id: UUID,
        chunk_index: int,
    ) -> DocumentChunkRecord | None:
        """Get a chunk by its document ID and zero-based index, or None."""
        stmt = select(DocumentChunkRecord).where(
            DocumentChunkRecord.document_id == document_id,
            DocumentChunkRecord.chunk_index == chunk_index,
        )
        return self._session.scalars(stmt).first()

    def list_by_document_id(
        self,
        document_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[DocumentChunkRecord]:
        """List chunks for a specific document with validated pagination and natural ordering."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = (
            select(DocumentChunkRecord)
            .where(DocumentChunkRecord.document_id == document_id)
            .order_by(DocumentChunkRecord.chunk_index, DocumentChunkRecord.id)
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def count_by_document_id(self, document_id: UUID) -> int:
        """Count total chunks for a document."""
        stmt = (
            select(func.count())
            .select_from(DocumentChunkRecord)
            .where(DocumentChunkRecord.document_id == document_id)
        )
        return self._session.scalar(stmt) or 0

    def delete_by_document_id(self, document_id: UUID) -> int:
        """Delete all chunks for a document and return the count of deleted chunks."""
        count = self.count_by_document_id(document_id)
        stmt = delete(DocumentChunkRecord).where(
            DocumentChunkRecord.document_id == document_id
        )
        self._session.execute(stmt)
        self._session.flush()
        return count
