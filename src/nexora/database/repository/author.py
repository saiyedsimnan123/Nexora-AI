"""Author repository."""
from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.author import Author
from nexora.database.repository.base import BaseRepository

# Explicit sentinel to distinguish between "not provided" and "provided as None"
_UNSET = object()


class AuthorRepository(BaseRepository[Author]):
    """Repository for Author models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, Author)

    def create(self, *, name: str) -> Author:
        """Create a new author."""
        return self._create(name=name)

    def get_by_name(self, name: str) -> Author | None:
        """Get an author by name, or None if not found."""
        stmt = select(Author).where(Author.name == name)
        return self._session.scalars(stmt).first()

    def update(self, id: UUID, *, name: str | Any = _UNSET) -> Author:
        """Update an existing author. Only updates provided fields."""
        kwargs: dict[str, Any] = {}
        if name is not _UNSET:
            kwargs["name"] = name
        return self._update(id, **kwargs)
