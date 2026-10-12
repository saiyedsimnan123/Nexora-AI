import pytest
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from nexora.database.models.base import Base

from nexora.database.repository.base import BaseRepository, RecordNotFoundError, DuplicateRecordError, ConstraintError
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository
from nexora.database.models.user import User

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

    # Duplicate mapping test (should be DuplicateRecordError)
    with pytest.raises(DuplicateRecordError):
        repo.create(email="test@example.com")

    # Test update
    updated_user = repo.update(user.id, email="updated@example.com")
    assert updated_user.email == "updated@example.com"
    assert repo.get_by_id(user.id).email == "updated@example.com"

    # Update duplicate
    user2 = repo.create(email="other@example.com")
    with pytest.raises(DuplicateRecordError):
        repo.update(user2.id, email="updated@example.com")

def test_workspace_repository(session):
    user_repo = UserRepository(session)
    repo = WorkspaceRepository(session)

    user = user_repo.create(email="w@example.com")

    # Create
    ws = repo.create(user_id=user.id, name="Test WS")

    # Read
    found = repo.get_by_id(ws.id)
    assert found is not None
    assert found.name == "Test WS"

    # List by user (and pagination)
    workspaces = repo.list_by_user_id(user.id)
    assert len(workspaces) == 1
    assert workspaces[0].id == ws.id

    # Add a second workspace to test limit
    ws2 = repo.create(user_id=user.id, name="Test WS 2")
    ws_page_1 = repo.list_by_user_id(user.id, offset=0, limit=1)
    assert len(ws_page_1) == 1

    # Update
    ws = repo.update(ws.id, name="Renamed WS", description="A description")
    assert ws.name == "Renamed WS"
    assert ws.description == "A description"

    # Delete
    repo.delete(ws.id)
    assert repo.get_by_id(ws.id) is None

    # Constraint mapping test for Foreign Key (should be ConstraintError now)
    with pytest.raises(ConstraintError):
        repo.create(user_id=uuid4(), name="Invalid WS")

def test_transaction_recovery_on_constraint_error(session):
    repo = UserRepository(session)
    # The session starts empty. We create one valid record
    repo.create(email="ok@example.com")

    # Now trigger an error (duplicate)
    with pytest.raises(DuplicateRecordError):
        repo.create(email="ok@example.com")

    # Thanks to nested transactions, the parent session is NOT invalidated
    # We should still be able to create a new user in the same session/transaction
    repo.create(email="another@example.com")
    session.commit() # The user manually commits

    # Ensure both valid records exist
    assert repo.get_by_email("ok@example.com") is not None
    assert repo.get_by_email("another@example.com") is not None

def test_list_all(session):
    repo = UserRepository(session)
    repo.create(email="1@example.com")
    repo.create(email="2@example.com")

    res = repo.list_all()
    assert len(res) == 2

    res = repo.list_all(offset=1, limit=1)
    assert len(res) == 1
