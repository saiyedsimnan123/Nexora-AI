"""Paper repository."""
from __future__ import annotations

import datetime
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.paper import Paper
from nexora.database.repository.base import BaseRepository

# Explicit sentinel to distinguish between "not provided" and "provided as None"
_UNSET = object()


class PaperRepository(BaseRepository[Paper]):
    """Repository for Paper models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, Paper)

    def create(
        self,
        *,
        collection_id: UUID,
        title: str,
        abstract: str | None = None,
        doi: str | None = None,
        external_id: str | None = None,
        publication_date: datetime.date | None = None,
    ) -> Paper:
        """Create a new paper."""
        return self._create(
            collection_id=collection_id,
            title=title,
            abstract=abstract,
            doi=doi,
            external_id=external_id,
            publication_date=publication_date,
        )

    def list_by_collection_id(
        self,
        collection_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Paper]:
        """List papers for a specific collection with validated pagination."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = (
            select(Paper)
            .where(Paper.collection_id == collection_id)
            .order_by(Paper.created_at, Paper.id)
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def update(
        self,
        id: UUID,
        *,
        title: str | Any = _UNSET,
        abstract: str | None | Any = _UNSET,
        doi: str | None | Any = _UNSET,
        external_id: str | None | Any = _UNSET,
        publication_date: datetime.date | None | Any = _UNSET,
        collection_id: UUID | Any = _UNSET,
    ) -> Paper:
        """Update an existing paper. Only updates provided fields."""
        kwargs: dict[str, Any] = {}
        if title is not _UNSET:
            kwargs["title"] = title
        if abstract is not _UNSET:
            kwargs["abstract"] = abstract
        if doi is not _UNSET:
            kwargs["doi"] = doi
        if external_id is not _UNSET:
            kwargs["external_id"] = external_id
        if publication_date is not _UNSET:
            kwargs["publication_date"] = publication_date
        if collection_id is not _UNSET:
            kwargs["collection_id"] = collection_id
        return self._update(id, **kwargs)

    def get_by_id_for_update(self, id: UUID) -> Paper | None:
        """Get a paper by ID with a database row lock (FOR UPDATE), or None if not found."""
        stmt = select(Paper).where(Paper.id == id).with_for_update()
        return self._session.scalars(stmt).first()
