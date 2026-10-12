"""Workspace repository."""
from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.workspace import Workspace
from nexora.database.repository.base import BaseRepository


class WorkspaceRepository(BaseRepository[Workspace]):
    """Repository for Workspace models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, Workspace)

    def create(self, *, user_id: UUID, name: str, description: str = "") -> Workspace:
        """Create a new workspace."""
        return self._create(user_id=user_id, name=name, description=description)

    def list_by_user_id(self, user_id: UUID, *, offset: int = 0, limit: int = 100) -> Sequence[Workspace]:
        """List workspaces for a specific user."""
        stmt = select(Workspace).where(Workspace.user_id == user_id).offset(offset).limit(limit)
        return self._session.scalars(stmt).all()
