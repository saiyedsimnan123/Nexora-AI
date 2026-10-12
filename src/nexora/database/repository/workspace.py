"""Workspace repository."""

import uuid
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from nexora.database.models.workspace import Workspace
from nexora.database.repository.base import Repository, RecordNotFoundError, RepositoryError


class WorkspaceRepository(Repository[Workspace]):
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, id: uuid.UUID) -> Workspace:
        workspace = self._session.get(Workspace, id)
        if workspace is None:
            raise RecordNotFoundError(f"Workspace with id {id} not found")
        return workspace

    def get_by_user_id(self, user_id: uuid.UUID) -> list[Workspace]:
        return list(self._session.scalars(select(Workspace).where(Workspace.user_id == user_id)))

    def add(self, entity: Workspace) -> Workspace:
        self._session.add(entity)
        try:
            self._session.flush()
        except IntegrityError as e:
            raise RepositoryError(f"Failed to add Workspace: constraint violation") from e
        return entity

    def update(self, entity: Workspace) -> Workspace:
        try:
            self._session.flush()
        except IntegrityError as e:
            raise RepositoryError(f"Failed to update Workspace: constraint violation") from e
        return entity

    def delete(self, entity: Workspace) -> None:
        self._session.delete(entity)
        self._session.flush()
