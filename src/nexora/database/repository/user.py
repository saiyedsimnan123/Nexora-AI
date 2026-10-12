"""User repository."""

import uuid
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from nexora.database.models.user import User
from nexora.database.repository.base import Repository, RecordNotFoundError, RepositoryError


class UserRepository(Repository[User]):
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, id: uuid.UUID) -> User:
        user = self._session.get(User, id)
        if user is None:
            raise RecordNotFoundError(f"User with id {id} not found")
        return user

    def get_by_email(self, email: str) -> User:
        user = self._session.scalars(select(User).where(User.email == email)).first()
        if user is None:
            raise RecordNotFoundError(f"User with email {email} not found")
        return user

    def add(self, entity: User) -> User:
        self._session.add(entity)
        try:
            self._session.flush()
        except IntegrityError as e:
            self._session.rollback()
            raise RepositoryError("Failed to add User: constraint violation") from e
        return entity

    def update(self, entity: User) -> User:
        # Assuming the caller has modified the entity attached to the session
        try:
            self._session.flush()
        except IntegrityError as e:
            self._session.rollback()
            raise RepositoryError("Failed to update User: constraint violation") from e
        return entity

    def delete(self, entity: User) -> None:
        self._session.delete(entity)
        try:
            self._session.flush()
        except IntegrityError as e:
            self._session.rollback()
            raise RepositoryError("Failed to delete User: constraint violation") from e
