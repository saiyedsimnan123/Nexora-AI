"""Document repository."""
from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.document import Document
from nexora.database.repository.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document file metadata models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, Document)

    def create(
        self,
        *,
        paper_id: UUID,
        filename: str,
        mime_type: str = "application/pdf",
        file_size: int,
        checksum: str,
        storage_key: str,
    ) -> Document:
        """Create a new document record for a paper."""
        return self._create(
            paper_id=paper_id,
            filename=filename,
            mime_type=mime_type,
            file_size=file_size,
            checksum=checksum,
            storage_key=storage_key,
        )

    def list_by_paper_id(
        self,
        paper_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Document]:
        """List documents belonging to a paper with validated pagination."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = (
            select(Document)
            .where(Document.paper_id == paper_id)
            .order_by(Document.created_at, Document.id)
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def get_by_storage_key(self, storage_key: str) -> Document | None:
        """Get a document by its storage key, or None if not found."""
        stmt = select(Document).where(Document.storage_key == storage_key)
        return self._session.scalars(stmt).first()

    def get_by_checksum_and_paper(
        self, paper_id: UUID, checksum: str
    ) -> Document | None:
        """Get a document by its paper ID and checksum, or None if not found."""
        stmt = select(Document).where(
            Document.paper_id == paper_id,
            Document.checksum == checksum,
        )
        return self._session.scalars(stmt).first()
