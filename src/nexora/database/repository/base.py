"""Base repository interface and generic implementations."""

from typing import Generic, TypeVar, Protocol
import uuid
from sqlalchemy.orm import Session

T = TypeVar("T")


class RepositoryError(Exception):
    """Base exception for all repository errors."""
    pass


class RecordNotFoundError(RepositoryError):
    """Raised when a requested record is not found."""
    pass


class Repository(Protocol[T]):
    """Base interface for all repositories."""

    def get_by_id(self, id: uuid.UUID) -> T:
        """Get a record by ID or raise RecordNotFoundError."""
        ...

    def add(self, entity: T) -> T:
        """Add a new record."""
        ...

    def update(self, entity: T) -> T:
        """Update an existing record."""
        ...

    def delete(self, entity: T) -> None:
        """Delete an existing record."""
        ...
