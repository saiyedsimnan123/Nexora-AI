"""PaperAuthor repository."""
from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.paper_author import PaperAuthor
from nexora.database.repository.base import BaseRepository

# Explicit sentinel to distinguish between "not provided" and "provided as None"
_UNSET = object()


class PaperAuthorRepository(BaseRepository[PaperAuthor]):
    """Repository for PaperAuthor association models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, PaperAuthor)

    def create(
        self,
        *,
        paper_id: UUID,
        author_id: UUID,
        author_order: int,
    ) -> PaperAuthor:
        """Create a new paper-author association."""
        return self._create(
            paper_id=paper_id,
            author_id=author_id,
            author_order=author_order,
        )

    def get_by_paper_and_author(
        self,
        paper_id: UUID,
        author_id: UUID,
    ) -> PaperAuthor | None:
        """Get an authorship record by paper_id and author_id, or None."""
        stmt = select(PaperAuthor).where(
            PaperAuthor.paper_id == paper_id,
            PaperAuthor.author_id == author_id,
        )
        return self._session.scalars(stmt).first()

    def list_by_paper_id(self, paper_id: UUID) -> Sequence[PaperAuthor]:
        """List all authorship records for a paper, ordered by author_order."""
        stmt = (
            select(PaperAuthor)
            .where(PaperAuthor.paper_id == paper_id)
            .order_by(PaperAuthor.author_order)
        )
        return self._session.scalars(stmt).all()

    def list_by_author_id(
        self,
        author_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[PaperAuthor]:
        """List authorship records for a specific author with validated pagination."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = (
            select(PaperAuthor)
            .where(PaperAuthor.author_id == author_id)
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def update(
        self,
        id: UUID,
        *,
        author_order: int | Any = _UNSET,
    ) -> PaperAuthor:
        """Update an existing authorship record."""
        kwargs: dict[str, Any] = {}
        if author_order is not _UNSET:
            kwargs["author_order"] = author_order
        return self._update(id, **kwargs)
