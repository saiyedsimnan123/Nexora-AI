"""Collection repository."""
from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.collection import Collection
from nexora.database.repository.base import BaseRepository


class CollectionRepository(BaseRepository[Collection]):
    """Repository for Collection models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, Collection)

    def create(
        self,
        *,
        workspace_id: UUID,
        name: str,
        description: str | None = None,
    ) -> Collection:
        """Create a new collection."""
        return self._create(
            workspace_id=workspace_id,
            name=name,
            description=description,
        )

    def list_by_workspace_id(
        self,
        workspace_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Collection]:
        """List collections belonging to a workspace with validated pagination."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = (
            select(Collection)
            .where(Collection.workspace_id == workspace_id)
            .order_by(Collection.created_at, Collection.id)
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def update(
        self,
        id: UUID,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Collection:
        """Update an existing collection."""
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if description is not None:
            kwargs["description"] = description
        return self._update(id, **kwargs)
