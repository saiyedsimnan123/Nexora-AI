"""Collection repository."""
from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.collection import Collection
from nexora.database.repository.base import BaseRepository

# Use an explicit sentinel to distinguish between "not provided" and "provided as None"
_UNSET = object()

class CollectionRepository(BaseRepository[Collection]):
    """Repository for Collection models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, Collection)

    def create(self, *, workspace_id: UUID, name: str, description: str | None = None) -> Collection:
        """Create a new collection."""
        return self._create(workspace_id=workspace_id, name=name, description=description)

    def list_by_workspace_id(self, workspace_id: UUID, *, offset: int = 0, limit: int = 100) -> Sequence[Collection]:
        """List collections for a specific workspace."""
        stmt = select(Collection).where(Collection.workspace_id == workspace_id).offset(offset).limit(limit)
        return self._session.scalars(stmt).all()

    def update(self, id: UUID, *, name: str | Any = _UNSET, description: str | None | Any = _UNSET) -> Collection:
        """Update an existing collection. Only updates provided fields."""
        kwargs = {}
        if name is not _UNSET:
            kwargs["name"] = name
        if description is not _UNSET:
            kwargs["description"] = description
        return self._update(id, **kwargs)
