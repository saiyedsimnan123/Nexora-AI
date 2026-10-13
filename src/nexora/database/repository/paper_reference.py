"""PaperReference repository."""
from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.paper_reference import PaperReference
from nexora.database.repository.base import BaseRepository, RecordNotFoundError


class PaperReferenceRepository(BaseRepository[PaperReference]):
    """Repository for PaperReference citation models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, PaperReference)

    def create(
        self,
        *,
        source_paper_id: UUID,
        target_paper_id: UUID,
    ) -> PaperReference:
        """Create a new citation reference from source paper to target paper."""
        if source_paper_id == target_paper_id:
            raise ValueError("source_paper_id and target_paper_id must be different")
        return self._create(
            source_paper_id=source_paper_id,
            target_paper_id=target_paper_id,
        )

    def get_by_source_and_target(
        self,
        source_paper_id: UUID,
        target_paper_id: UUID,
    ) -> PaperReference | None:
        """Get a citation reference by source and target paper IDs, or None."""
        stmt = select(PaperReference).where(
            PaperReference.source_paper_id == source_paper_id,
            PaperReference.target_paper_id == target_paper_id,
        )
        return self._session.scalars(stmt).first()

    def list_outgoing(
        self,
        source_paper_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[PaperReference]:
        """List outgoing references cited by a source paper, with validated pagination."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = (
            select(PaperReference)
            .where(PaperReference.source_paper_id == source_paper_id)
            .order_by(PaperReference.created_at, PaperReference.id)
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def list_incoming(
        self,
        target_paper_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[PaperReference]:
        """List incoming references citing a target paper, with validated pagination."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = (
            select(PaperReference)
            .where(PaperReference.target_paper_id == target_paper_id)
            .order_by(PaperReference.created_at, PaperReference.id)
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def delete_by_source_and_target(
        self,
        source_paper_id: UUID,
        target_paper_id: UUID,
    ) -> None:
        """Delete a citation reference by source and target paper IDs."""
        ref = self.get_by_source_and_target(source_paper_id, target_paper_id)
        if ref is None:
            raise RecordNotFoundError(
                f"PaperReference from {source_paper_id} to {target_paper_id} not found"
            )
        self.delete(ref.id)
