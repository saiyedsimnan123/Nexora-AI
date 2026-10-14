"""Document repository."""
from __future__ import annotations

import datetime
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select

from nexora.database.models.document import Document, IndexingStatus
from nexora.database.repository.base import BaseRepository, RecordNotFoundError


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
        claimed_at: datetime.datetime | None = None,
        claimed_by: str | None = None,
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
            claimed_at=claimed_at,
            claimed_by=claimed_by,
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
        claimed_at: datetime.datetime | None = None,
        claimed_by: str | None = None,
    ) -> Document:
        """Update a document's durable indexing status, claim fields, and error information."""
        doc = self.get_by_id(id)
        if doc is None:
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

        if status_val in (IndexingStatus.COMPLETED.value, IndexingStatus.FAILED.value):
            doc.claimed_at = claimed_at
            doc.claimed_by = claimed_by
        else:
            if claimed_at is not None:
                doc.claimed_at = claimed_at
            if claimed_by is not None:
                doc.claimed_by = claimed_by

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

    def fail_exhausted_claims(
        self,
        *,
        max_attempts: int,
        claim_timeout_seconds: float = 300.0,
        limit: int | None = None,
    ) -> Sequence[Document]:
        """Transition stale processing documents that reached max_attempts to failed.

        Clears claim fields and records a durable error without incrementing attempts.
        Uses FOR UPDATE SKIP LOCKED on PostgreSQL so concurrent workers coordinate race-safely.
        """
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        now = datetime.datetime.now(datetime.timezone.utc)
        stale_cutoff = now - datetime.timedelta(seconds=claim_timeout_seconds)

        stmt = (
            select(Document)
            .where(
                Document.indexing_status == IndexingStatus.PROCESSING.value,
                or_(
                    Document.claimed_at.is_(None),
                    Document.claimed_at <= stale_cutoff,
                ),
                Document.indexing_attempts >= max_attempts,
            )
            .order_by(Document.created_at, Document.id)
            .with_for_update(skip_locked=True)
        )
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            stmt = stmt.limit(limit)

        exhausted_docs = list(self._session.scalars(stmt).all())
        for doc in exhausted_docs:
            doc.indexing_status = IndexingStatus.FAILED.value
            doc.indexing_error = (
                f"Recovery indexing attempts exhausted (max {max_attempts} attempts reached)"
            )
            doc.claimed_at = None
            doc.claimed_by = None

        if exhausted_docs:
            self._session.flush()
        return exhausted_docs

    def claim_unindexed(
        self,
        limit: int,
        *,
        max_attempts: int | None = None,
        claim_timeout_seconds: float = 300.0,
        claim_token: str | None = None,
        include_failed: bool = True,
    ) -> Sequence[Document]:
        """Claim a batch of unindexed documents with row-level locking (SKIP LOCKED).

        Atomically transitions exhausted stale processing documents to failed, then
        selects unindexed or expired processing documents under max_attempts, marks them as
        'processing', sets claimed_at and claimed_by, and increments indexing_attempts.
        Uses FOR UPDATE SKIP LOCKED on PostgreSQL so independent workers coordinate safely.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        now = datetime.datetime.now(datetime.timezone.utc)
        stale_cutoff = now - datetime.timedelta(seconds=claim_timeout_seconds)

        if max_attempts is not None:
            self.fail_exhausted_claims(
                max_attempts=max_attempts,
                claim_timeout_seconds=claim_timeout_seconds,
                limit=limit,
            )

        status_list = [IndexingStatus.PENDING.value]
        if include_failed:
            status_list.append(IndexingStatus.FAILED.value)

        status_filter = or_(
            Document.indexing_status.in_(status_list),
            and_(
                Document.indexing_status == IndexingStatus.PROCESSING.value,
                or_(
                    Document.claimed_at.is_(None),
                    Document.claimed_at <= stale_cutoff,
                ),
            ),
        )

        stmt = (
            select(Document)
            .where(status_filter)
            .order_by(Document.created_at, Document.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        if max_attempts is not None:
            stmt = stmt.where(Document.indexing_attempts < max_attempts)

        docs = list(self._session.scalars(stmt).all())
        for doc in docs:
            doc.indexing_status = IndexingStatus.PROCESSING.value
            doc.claimed_at = now
            doc.claimed_by = claim_token
            doc.indexing_attempts += 1

        self._session.flush()
        return docs

    def claim_document(
        self,
        id: UUID,
        *,
        max_attempts: int | None = None,
        claim_timeout_seconds: float = 300.0,
        claim_token: str | None = None,
    ) -> Document | None:
        """Atomically claim a single document for recovery.

        Returns None if not found, already completed, or currently claimed by an
        active unexpired worker, or if max_attempts has been reached.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        stale_cutoff = now - datetime.timedelta(seconds=claim_timeout_seconds)

        stmt = select(Document).where(Document.id == id).with_for_update()
        doc = self._session.scalars(stmt).one_or_none()
        if doc is None:
            return None
        if doc.indexing_status == IndexingStatus.COMPLETED.value:
            return doc

        if doc.indexing_status == IndexingStatus.PROCESSING.value:
            claimed_at = doc.claimed_at
            if claimed_at is not None:
                if claimed_at.tzinfo is None:
                    claimed_at = claimed_at.replace(tzinfo=datetime.timezone.utc)
                if claimed_at > stale_cutoff:
                    return None

        if max_attempts is not None and doc.indexing_attempts >= max_attempts:
            if doc.indexing_status == IndexingStatus.PROCESSING.value:
                doc.indexing_status = IndexingStatus.FAILED.value
                doc.indexing_error = (
                    f"Recovery indexing attempts exhausted (max {max_attempts} attempts reached)"
                )
                doc.claimed_at = None
                doc.claimed_by = None
                self._session.flush()
            return None

        doc.indexing_status = IndexingStatus.PROCESSING.value
        doc.claimed_at = now
        doc.claimed_by = claim_token
        doc.indexing_attempts += 1
        self._session.flush()
        return doc

    def finalize_indexing(
        self,
        id: UUID,
        status: str | IndexingStatus,
        *,
        claim_token: str | None = None,
        error: str | None = None,
        indexed_at: datetime.datetime | None = None,
    ) -> Document | None:
        """Finalize indexing outcome, clearing the claim.

        When claim_token is supplied, finalization is strictly guarded: only allowed if
        claim_token matches the document's active claim (doc.claimed_by == claim_token and
        doc.indexing_status == 'processing').
        Returns None if the claim was lost or superseded by another worker.
        """
        stmt = select(Document).where(Document.id == id).with_for_update()
        doc = self._session.scalars(stmt).one_or_none()
        if doc is None:
            return None

        if claim_token is not None:
            if (
                doc.claimed_by != claim_token
                or doc.indexing_status != IndexingStatus.PROCESSING.value
            ):
                return None

        status_val = status.value if isinstance(status, IndexingStatus) else status
        doc.indexing_status = status_val
        doc.indexing_error = error
        if indexed_at is not None:
            doc.indexed_at = indexed_at
        elif status_val == IndexingStatus.COMPLETED.value:
            doc.indexed_at = datetime.datetime.now(datetime.timezone.utc)
        doc.claimed_at = None
        doc.claimed_by = None
        self._session.flush()
        return doc

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

