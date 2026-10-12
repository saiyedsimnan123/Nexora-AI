import pytest
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from nexora.database.models.base import Base

from nexora.database.repository.base import BaseRepository, RecordNotFoundError, DuplicateRecordError, ConstraintError
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository
from nexora.database.repository.collection import CollectionRepository
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

def test_unrelated_pending_changes_are_safely_persisted_on_error(session):
    repo = UserRepository(session)
    ws_repo = WorkspaceRepository(session)

    # 1. Start a transaction by creating a valid record without flushing immediately in the caller
    valid_user = User(email="pending@example.com")
    session.add(valid_user)

    # 2. Try to create an invalid record using the repository (e.g., missing foreign key constraint)
    # The repository will flush the pending user FIRST successfully, then fail on the invalid workspace
    with pytest.raises(ConstraintError):
        ws_repo.create(user_id=uuid4(), name="Invalid WS")

    # 3. The invalid workspace creation failed (and rolled back its savepoint).
    # But the pending valid user should have been flushed beforehand and persisted.
    session.commit() # The user manually commits

    # Ensure the valid record exists
    assert repo.get_by_email("pending@example.com") is not None

def test_delete_constraint_failure(session):
    user_repo = UserRepository(session)
    ws_repo = WorkspaceRepository(session)

    # Create user and workspace
    user = user_repo.create(email="delete_test@example.com")
    ws = ws_repo.create(user_id=user.id, name="WS for delete")
    session.commit()

    # Manually delete user without using repository to bypass Python-level cascades,
    # OR set up a scenario where deletion fails due to DB constraints.
    # For a User, foreign keys on Workspaces are set to ON DELETE CASCADE, so deleting a User
    # will actually succeed and cascade to the Workspace.
    user_id = user.id

    # We want to mock session.flush ONLY when it's called from inside the `begin_nested()` block
    # so the first flush (for pending changes) succeeds, and the second one fails.
    import unittest.mock
    original_flush = session.flush
    flush_call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal flush_call_count
        flush_call_count += 1
        if flush_call_count == 2:
            raise IntegrityError("mock", "mock", "mock")
        return original_flush(*args, **kwargs)

    with unittest.mock.patch.object(session, 'flush', side_effect=side_effect):
        with pytest.raises(ConstraintError):
            user_repo.delete(user_id)

def test_transaction_recovery_on_update_constraint_error(session):
    repo = UserRepository(session)
    user1 = repo.create(email="user1@example.com")
    user2 = repo.create(email="user2@example.com")

    # Trigger duplicate error on update
    with pytest.raises(DuplicateRecordError):
        repo.update(user2.id, email="user1@example.com")

    # Test session is still usable
    repo.create(email="user3@example.com")
    session.commit()
    assert repo.get_by_email("user3@example.com") is not None


def test_collection_repository(session):
    user_repo = UserRepository(session)
    ws_repo = WorkspaceRepository(session)
    repo = CollectionRepository(session)

    user = user_repo.create(email="c@example.com")
    ws = ws_repo.create(user_id=user.id, name="Workspace for Collection")

    # Create
    col = repo.create(workspace_id=ws.id, name="Test Collection", description="Collection Description")

    # Read
    found = repo.get_by_id(col.id)
    assert found is not None
    assert found.name == "Test Collection"
    assert found.description == "Collection Description"

    # List by workspace
    collections = repo.list_by_workspace_id(ws.id)
    assert len(collections) == 1
    assert collections[0].id == col.id

    # Pagination
    col2 = repo.create(workspace_id=ws.id, name="Test Collection 2")
    col_page_1 = repo.list_by_workspace_id(ws.id, offset=0, limit=1)
    assert len(col_page_1) == 1

    # Update
    col = repo.update(col.id, name="Renamed Collection", description="New Description")
    assert col.name == "Renamed Collection"
    assert col.description == "New Description"

    # Update partial
    col = repo.update(col.id, name="Renamed Again")
    assert col.name == "Renamed Again"
    assert col.description == "New Description"

    # Delete
    repo.delete(col.id)
    assert repo.get_by_id(col.id) is None

    # Constraint mapping test for Foreign Key
    with pytest.raises(ConstraintError):
        repo.create(workspace_id=uuid4(), name="Invalid Collection")

    # Constraint mapping test for check constraint (blank name)
    # The check constraint CK_collections_name_not_blank will trigger
    with pytest.raises(ConstraintError):
        repo.create(workspace_id=ws.id, name="   ")


def test_collection_repository_update_description_to_none(session):
    user_repo = UserRepository(session)
    ws_repo = WorkspaceRepository(session)
    repo = CollectionRepository(session)

    user = user_repo.create(email="d@example.com")
    ws = ws_repo.create(user_id=user.id, name="Workspace")

    # Create with description
    col = repo.create(workspace_id=ws.id, name="Test Collection", description="Desc")
    assert col.description == "Desc"

    # Explicitly clear description
    col = repo.update(col.id, description=None)
    assert col.description is None
