"""Base generic repository."""
from __future__ import annotations

from typing import Generic, TypeVar, Any, Sequence, cast
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy import select

from nexora.database.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class RepositoryError(Exception):
    """Base exception for repository errors."""


class RecordNotFoundError(RepositoryError):
    """Raised when a record is not found."""


class ConstraintError(RepositoryError):
    """Raised when an operation violates a database constraint (e.g. foreign key, check)."""

class DuplicateRecordError(RepositoryError):
    """Raised when creating or updating a record would cause a uniqueness violation."""


class BaseRepository(Generic[ModelT]):
    """Generic base repository for SQLAlchemy models.

    Repositories rely on the caller to provide and commit the session.
    They do not open sessions or call session.commit().
    """

    def __init__(self, session: Session, model_cls: type[ModelT]) -> None:
        """Initialize the repository.

        Args:
            session: SQLAlchemy session to use for database operations.
            model_cls: The SQLAlchemy model class this repository manages.
        """
        self._session = session
        self._model_cls = model_cls

    def get_by_id(self, id: UUID) -> ModelT | None:
        """Get a record by ID, or None if not found."""
        return self._session.get(self._model_cls, id)

    def get_by_id_or_raise(self, id: UUID) -> ModelT:
        """Get a record by ID, or raise RecordNotFoundError."""
        record = self.get_by_id(id)
        if record is None:
            raise RecordNotFoundError(f"{self._model_cls.__name__} with id {id} not found")
        return record

    def list_all(self, *, offset: int = 0, limit: int = 100) -> Sequence[ModelT]:
        """List records with pagination."""
        stmt = select(self._model_cls).offset(offset).limit(limit)
        return self._session.scalars(stmt).all()

    def _create(self, **kwargs: Any) -> ModelT:
        """Internal helper to create a record and flush."""
        record = self._model_cls(**kwargs)

        # Use a nested transaction (savepoint) so an IntegrityError does not invalidate
        # the caller's entire transaction.
        try:
            with self._session.begin_nested():
                self._session.add(record)
                self._session.flush()
        except IntegrityError as e:
            error_str = str(e.orig).lower() if e.orig else str(e).lower()
            if "unique" in error_str or "duplicate" in error_str:
                raise DuplicateRecordError(str(e)) from e
            raise ConstraintError(str(e)) from e
        return record

    def _update(self, id: UUID, **kwargs: Any) -> ModelT:
        """Internal helper to update a record."""
        record = self.get_by_id_or_raise(id)
        for key, value in kwargs.items():
            setattr(record, key, value)

        try:
            with self._session.begin_nested():
                self._session.flush()
        except IntegrityError as e:
            # Refresh to revert attributes or expunge if it somehow got removed
            if record in self._session:
                self._session.refresh(record)
            error_str = str(e.orig).lower() if e.orig else str(e).lower()
            if "unique" in error_str or "duplicate" in error_str:
                raise DuplicateRecordError(str(e)) from e
            raise ConstraintError(str(e)) from e
        return record

    def delete(self, id: UUID) -> None:
        """Delete a record by ID."""
        record = self.get_by_id_or_raise(id)
        self._session.delete(record)
        self._session.flush()
