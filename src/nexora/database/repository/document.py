"""Document repository."""
from __future__ import annotations

import datetime
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import func, select

from nexora.database.models.document import Document, IndexingStatus
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
        indexing_status: str = IndexingStatus.PENDING.value,
        indexing_error: str | None = None,
        indexed_at: datetime.datetime | None = None,
        indexing_attempts: int = 0,
    ) -> Document:
        """Create a new document record for a paper."""
        return self._create(
            paper_id=paper_id,
            filename=filename,
            mime_type=mime_type,
            file_size=file_size,
            checksum=checksum,
            storage_key=storage_key,
            indexing_status=indexing_status,
            indexing_error=indexing_error,
            indexed_at=indexed_at,
            indexing_attempts=indexing_attempts,
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

    def update_indexing_status(
        self,
        id: UUID,
        status: str | IndexingStatus,
        *,
        error: str | None = None,
        indexed_at: datetime.datetime | None = None,
        attempts_increment: int = 0,
    ) -> Document:
        """Update a document's durable indexing status and error information."""
        doc = self.get_by_id(id)
        if doc is None:
            from nexora.database.repository.base import RecordNotFoundError
            raise RecordNotFoundError(f"Document with id {id} not found")

        status_val = status.value if isinstance(status, IndexingStatus) else status
        doc.indexing_status = status_val
        doc.indexing_error = error
        if indexed_at is not None:
            doc.indexed_at = indexed_at
        elif status_val == IndexingStatus.COMPLETED.value:
            doc.indexed_at = datetime.datetime.now(datetime.timezone.utc)
        if attempts_increment > 0:
            doc.indexing_attempts += attempts_increment

        self._session.flush()
        return doc

    def list_unindexed(
        self,
        status: Sequence[str] | Sequence[IndexingStatus] = (
            IndexingStatus.PENDING.value,
            IndexingStatus.FAILED.value,
        ),
        *,
        limit: int = 100,
        max_attempts: int | None = None,
        for_update: bool = False,
        skip_locked: bool = False,
    ) -> Sequence[Document]:
        """Find pending or failed indexing work ordered deterministically for bounded recovery."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        status_vals = [s.value if isinstance(s, IndexingStatus) else str(s) for s in status]
        stmt = (
            select(Document)
            .where(Document.indexing_status.in_(status_vals))
            .order_by(Document.created_at, Document.id)
            .limit(limit)
        )
        if max_attempts is not None:
            stmt = stmt.where(Document.indexing_attempts < max_attempts)

        if for_update:
            if skip_locked:
                stmt = stmt.with_for_update(skip_locked=True)
            else:
                stmt = stmt.with_for_update()

        return self._session.scalars(stmt).all()

    def count_by_indexing_status(self) -> dict[str, int]:
        """Return counts of documents grouped by indexing_status."""
        stmt = (
            select(Document.indexing_status, func.count(Document.id))
            .group_by(Document.indexing_status)
        )
        counts = {s.value: 0 for s in IndexingStatus}
        for status_val, count in self._session.execute(stmt):
            counts[str(status_val)] = int(count)
        return counts

