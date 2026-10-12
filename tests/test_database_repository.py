import pytest
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from nexora.database.models.base import Base

from nexora.database.repository.base import BaseRepository, RecordNotFoundError, DuplicateRecordError
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository

@pytest.fixture
def engine():
    # Enforce foreign key constraints in sqlite
    engine = create_engine("sqlite:///:memory:")
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session

def test_user_repository(session):
    repo = UserRepository(session)

    # Create
    user = repo.create(email="test@example.com")
    session.commit()

    # Read
    found = repo.get_by_id(user.id)
    assert found is not None
    assert found.email == "test@example.com"

    found_by_email = repo.get_by_email("test@example.com")
    assert found_by_email is not None
    assert found_by_email.id == user.id

    # Missing
    assert repo.get_by_id(uuid4()) is None
    assert repo.get_by_email("missing@example.com") is None

    with pytest.raises(RecordNotFoundError):
        repo.get_by_id_or_raise(uuid4())

    # Duplicate
    with pytest.raises(DuplicateRecordError):
        repo.create(email="test@example.com")

def test_workspace_repository(session):
    user_repo = UserRepository(session)
    repo = WorkspaceRepository(session)

    user = user_repo.create(email="w@example.com")
    session.commit()

    # Create
    ws = repo.create(user_id=user.id, name="Test WS")
    session.commit()

    # Read
    found = repo.get_by_id(ws.id)
    assert found is not None
    assert found.name == "Test WS"

    # List by user
    workspaces = repo.list_by_user_id(user.id)
    assert len(workspaces) == 1
    assert workspaces[0].id == ws.id

    # Delete
    repo.delete(ws.id)
    session.commit()

    assert repo.get_by_id(ws.id) is None

    # Missing parent FK error handling
    with pytest.raises(DuplicateRecordError): # Using this mapped error for all IntegrityErrors for now
        repo.create(user_id=uuid4(), name="Invalid WS")
