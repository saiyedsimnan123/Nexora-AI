"""Tests for the database repository layer."""

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from nexora.database.models.base import Base
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository import (
    RecordNotFoundError,
    RepositoryError,
    UserRepository,
    WorkspaceRepository,
)


@pytest.fixture
def engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Enable foreign keys for SQLite
    def enable_fk(dbapi_con, con_record):
        dbapi_con.execute('pragma foreign_keys=ON')

    import sqlalchemy.event
    sqlalchemy.event.listen(engine, 'connect', enable_fk)

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(engine):
    with Session(engine) as session:
        yield session
        session.rollback()


def test_user_repository_crud(db_session: Session):
    repo = UserRepository(db_session)

    # Add
    user = User(email="test@example.com")
    repo.add(user)

    # Needs a real flush to set the ID in tests, although `add` does `flush`
    assert user.id is not None

    # Read by ID
    found_user = repo.get_by_id(user.id)
    assert found_user == user

    # Read by email
    found_by_email = repo.get_by_email("test@example.com")
    assert found_by_email == user

    # Update
    user.email = "updated@example.com"
    repo.update(user)
    assert repo.get_by_id(user.id).email == "updated@example.com"

    # Delete
    repo.delete(user)
    with pytest.raises(RecordNotFoundError):
        repo.get_by_id(user.id)


def test_user_repository_not_found(db_session: Session):
    repo = UserRepository(db_session)

    with pytest.raises(RecordNotFoundError, match="User with id .* not found"):
        repo.get_by_id(uuid.uuid4())

    with pytest.raises(RecordNotFoundError, match="User with email missing@example.com not found"):
        repo.get_by_email("missing@example.com")


def test_user_repository_constraint_violation(db_session: Session):
    repo = UserRepository(db_session)
    user1 = User(email="duplicate@example.com")
    repo.add(user1)

    user2 = User(email="duplicate@example.com")
    with pytest.raises(RepositoryError, match="Failed to add User: constraint violation"):
        repo.add(user2)

    db_session.rollback()


def test_workspace_repository_crud(db_session: Session):
    user_repo = UserRepository(db_session)
    user = User(email="owner@example.com")
    user_repo.add(user)

    repo = WorkspaceRepository(db_session)

    # Add
    ws = Workspace(user_id=user.id, name="Test Workspace", description="Desc")
    repo.add(ws)
    assert ws.id is not None

    # Read
    found_ws = repo.get_by_id(ws.id)
    assert found_ws == ws

    # Read by user
    user_workspaces = repo.get_by_user_id(user.id)
    assert len(user_workspaces) == 1
    assert user_workspaces[0] == ws

    # Update
    ws.name = "Updated Workspace"
    repo.update(ws)
    assert repo.get_by_id(ws.id).name == "Updated Workspace"

    # Delete
    repo.delete(ws)
    with pytest.raises(RecordNotFoundError):
        repo.get_by_id(ws.id)


def test_workspace_repository_foreign_key_violation(db_session: Session):
    repo = WorkspaceRepository(db_session)
    # Trying to add workspace with non-existent user_id
    ws = Workspace(user_id=uuid.uuid4(), name="Invalid", description="Desc")

    with pytest.raises(RepositoryError, match="Failed to add Workspace: constraint violation"):
        repo.add(ws)

    db_session.rollback()
