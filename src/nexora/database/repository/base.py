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
        self._session.add(record)
        try:
            self._session.flush()
        except IntegrityError as e:
            # We don't rollback the entire transaction here automatically because
            # SQLAlchemy handles savepoints, but the caller should usually handle
            # this, or rollback if needed. We translate the exception.
            # Usually IntegrityError due to unique constraint or foreign key.
            # For this base implementation we map to DuplicateRecordError for simplicity
            # if it looks like a duplicate.
            raise DuplicateRecordError(str(e)) from e
        return record

    def delete(self, id: UUID) -> None:
        """Delete a record by ID."""
        record = self.get_by_id_or_raise(id)
        self._session.delete(record)
        self._session.flush()
