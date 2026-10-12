"""User repository."""
from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.user import User
from nexora.database.repository.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Repository for User models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, User)

    def create(self, *, email: str) -> User:
        """Create a new user."""
        return self._create(email=email)

    def get_by_email(self, email: str) -> User | None:
        """Get a user by email, or None if not found."""
        stmt = select(User).where(User.email == email)
        return self._session.scalars(stmt).first()

    def update(self, id: UUID, *, email: str) -> User:
        """Update an existing user."""
        return self._update(id, email=email)
