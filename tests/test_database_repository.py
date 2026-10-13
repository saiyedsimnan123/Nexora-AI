import pytest
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from nexora.database.models.base import Base

from datetime import date

from nexora.database.repository.base import BaseRepository, RecordNotFoundError, DuplicateRecordError, ConstraintError
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository
from nexora.database.repository.paper import PaperRepository
from nexora.database.repository.author import AuthorRepository
from nexora.database.models.user import User
from nexora.database.models.collection import Collection
from nexora.database.models.paper import Paper
from nexora.database.models.author import Author

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


def _make_collection(session, name="Collection"):
    user_repo = UserRepository(session)
    ws_repo = WorkspaceRepository(session)
    user = user_repo.create(email=f"user_{uuid4()}@example.com")
    ws = ws_repo.create(user_id=user.id, name=f"WS_{uuid4()}")
    col = Collection(workspace_id=ws.id, name=name)
    session.add(col)
    session.flush()
    return col


def test_paper_repository_create_and_read(session):
    col = _make_collection(session)
    repo = PaperRepository(session)

    # Create with required fields
    paper = repo.create(collection_id=col.id, title="Attention Is All You Need")
    assert paper.id is not None
    assert paper.collection_id == col.id
    assert paper.title == "Attention Is All You Need"
    assert paper.abstract is None
    assert paper.doi is None
    assert paper.external_id is None
    assert paper.publication_date is None
    assert paper.created_at is not None
    assert paper.updated_at is not None

    # Read by ID
    found = repo.get_by_id(paper.id)
    assert found is not None
    assert found.id == paper.id
    assert found.title == "Attention Is All You Need"

    # Missing lookup
    missing_id = uuid4()
    assert repo.get_by_id(missing_id) is None

    with pytest.raises(RecordNotFoundError):
        repo.get_by_id_or_raise(missing_id)

    assert repo.get_by_id_or_raise(paper.id).id == paper.id


def test_paper_repository_create_all_fields(session):
    col = _make_collection(session)
    repo = PaperRepository(session)
    pub_date = date(2017, 6, 12)

    paper = repo.create(
        collection_id=col.id,
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        doi="10.48550/arXiv.1706.03762",
        external_id="arXiv:1706.03762",
        publication_date=pub_date,
    )

    found = repo.get_by_id_or_raise(paper.id)
    assert found.title == "Attention Is All You Need"
    assert found.abstract.startswith("The dominant sequence")
    assert found.doi == "10.48550/arXiv.1706.03762"
    assert found.external_id == "arXiv:1706.03762"
    assert found.publication_date == pub_date


def test_paper_repository_list_by_collection_id(session):
    col1 = _make_collection(session, name="Collection 1")
    col2 = _make_collection(session, name="Collection 2")
    repo = PaperRepository(session)

    p1 = repo.create(collection_id=col1.id, title="Paper 1")
    p2 = repo.create(collection_id=col1.id, title="Paper 2")
    p3 = repo.create(collection_id=col1.id, title="Paper 3")
    p_other = repo.create(collection_id=col2.id, title="Other Collection Paper")

    # List col1 papers
    col1_papers = repo.list_by_collection_id(col1.id)
    assert len(col1_papers) == 3
    col1_ids = {p.id for p in col1_papers}
    assert col1_ids == {p1.id, p2.id, p3.id}
    assert p_other.id not in col1_ids

    # List col2 papers
    col2_papers = repo.list_by_collection_id(col2.id)
    assert len(col2_papers) == 1
    assert col2_papers[0].id == p_other.id

    # List non-existent collection returns empty list
    assert repo.list_by_collection_id(uuid4()) == []

    # Pagination
    page_1 = repo.list_by_collection_id(col1.id, offset=0, limit=2)
    assert len(page_1) == 2
    page_2 = repo.list_by_collection_id(col1.id, offset=2, limit=2)
    assert len(page_2) == 1


def test_paper_repository_pagination_validation(session):
    col = _make_collection(session)
    repo = PaperRepository(session)

    with pytest.raises(ValueError, match="offset must be a non-negative integer"):
        repo.list_by_collection_id(col.id, offset=-1)

    with pytest.raises(ValueError, match="offset must be a non-negative integer"):
        repo.list_by_collection_id(col.id, offset=True)

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        repo.list_by_collection_id(col.id, limit=0)

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        repo.list_by_collection_id(col.id, limit=-5)

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        repo.list_by_collection_id(col.id, limit=False)


def test_paper_repository_update_full_and_partial(session):
    col1 = _make_collection(session, name="Collection 1")
    col2 = _make_collection(session, name="Collection 2")
    repo = PaperRepository(session)

    paper = repo.create(
        collection_id=col1.id,
        title="Initial Title",
        abstract="Initial Abstract",
        doi="10.1000/1",
        external_id="ext-1",
        publication_date=date(2020, 1, 1),
    )

    # Partial update: title only (other fields must remain untouched)
    updated = repo.update(paper.id, title="Updated Title Only")
    assert updated.title == "Updated Title Only"
    assert updated.abstract == "Initial Abstract"
    assert updated.doi == "10.1000/1"
    assert updated.external_id == "ext-1"
    assert updated.publication_date == date(2020, 1, 1)

    # Partial update: clearing nullable fields to None
    updated = repo.update(paper.id, abstract=None, doi=None)
    assert updated.title == "Updated Title Only"
    assert updated.abstract is None
    assert updated.doi is None
    assert updated.external_id == "ext-1"

    # Moving to a different collection
    updated = repo.update(paper.id, collection_id=col2.id)
    assert updated.collection_id == col2.id

    # Update non-existent paper
    with pytest.raises(RecordNotFoundError):
        repo.update(uuid4(), title="Nonexistent")


def test_paper_repository_delete(session):
    col = _make_collection(session)
    repo = PaperRepository(session)

    paper = repo.create(collection_id=col.id, title="To be deleted")
    paper_id = paper.id

    repo.delete(paper_id)
    assert repo.get_by_id(paper_id) is None

    # Deleting missing paper raises RecordNotFoundError
    with pytest.raises(RecordNotFoundError):
        repo.delete(uuid4())


def test_paper_repository_constraints_and_validation(session):
    col = _make_collection(session)
    repo = PaperRepository(session)

    # Foreign key violation (collection does not exist)
    with pytest.raises(ConstraintError):
        repo.create(collection_id=uuid4(), title="Invalid Collection")

    # In-memory validation violations from Paper model validators
    with pytest.raises(ValueError, match="title must not be blank"):
        repo.create(collection_id=col.id, title="")

    with pytest.raises(ValueError, match="title must not be blank"):
        repo.create(collection_id=col.id, title="   ")

    with pytest.raises(ValueError, match="title must be at most 500 characters"):
        repo.create(collection_id=col.id, title="T" * 501)

    with pytest.raises(ValueError, match="abstract must be at most 20000 characters"):
        repo.create(collection_id=col.id, title="Valid Title", abstract="A" * 20001)

    with pytest.raises(ValueError, match="doi must be at most 255 characters"):
        repo.create(collection_id=col.id, title="Valid Title", doi="D" * 256)

    with pytest.raises(ValueError, match="external_id must be at most 255 characters"):
        repo.create(collection_id=col.id, title="Valid Title", external_id="E" * 256)

    # Invalid collection in update triggers ConstraintError
    valid_paper = repo.create(collection_id=col.id, title="Valid Paper")
    with pytest.raises(ConstraintError):
        repo.update(valid_paper.id, collection_id=uuid4())


def test_paper_repository_transaction_recovery(session):
    col = _make_collection(session)
    repo = PaperRepository(session)

    # Initial valid paper
    p1 = repo.create(collection_id=col.id, title="First Valid Paper")

    # Trigger a ConstraintError (invalid foreign key)
    with pytest.raises(ConstraintError):
        repo.create(collection_id=uuid4(), title="Failing Paper")

    # Session is preserved and functional due to begin_nested()
    p2 = repo.create(collection_id=col.id, title="Second Valid Paper")
    session.commit()

    assert repo.get_by_id(p1.id) is not None
    assert repo.get_by_id(p2.id) is not None


def test_author_repository_create_and_read(session):
    repo = AuthorRepository(session)

    # Create
    author = repo.create(name="Ashish Vaswani")
    assert author.id is not None
    assert author.name == "Ashish Vaswani"
    assert author.created_at is not None
    assert author.updated_at is not None

    # Read by ID
    found = repo.get_by_id(author.id)
    assert found is not None
    assert found.id == author.id
    assert found.name == "Ashish Vaswani"

    # Missing ID
    missing_id = uuid4()
    assert repo.get_by_id(missing_id) is None

    with pytest.raises(RecordNotFoundError):
        repo.get_by_id_or_raise(missing_id)

    assert repo.get_by_id_or_raise(author.id).id == author.id


def test_author_repository_get_by_name(session):
    repo = AuthorRepository(session)

    author = repo.create(name="Noam Shazeer")
    found = repo.get_by_name("Noam Shazeer")
    assert found is not None
    assert found.id == author.id
    assert found.name == "Noam Shazeer"

    # Missing name
    assert repo.get_by_name("Nonexistent Author") is None


def test_author_repository_list_all(session):
    repo = AuthorRepository(session)

    a1 = repo.create(name="Author 1")
    a2 = repo.create(name="Author 2")

    all_authors = repo.list_all()
    assert len(all_authors) == 2

    page_1 = repo.list_all(offset=0, limit=1)
    assert len(page_1) == 1

    page_2 = repo.list_all(offset=1, limit=1)
    assert len(page_2) == 1


def test_author_repository_update(session):
    repo = AuthorRepository(session)

    author = repo.create(name="Original Author Name")

    # Update name
    updated = repo.update(author.id, name="Updated Author Name")
    assert updated.name == "Updated Author Name"
    assert repo.get_by_id(author.id).name == "Updated Author Name"

    # Update with unset name does not change name
    updated_same = repo.update(author.id)
    assert updated_same.name == "Updated Author Name"

    # Update non-existent author
    with pytest.raises(RecordNotFoundError):
        repo.update(uuid4(), name="Missing")


def test_author_repository_delete(session):
    repo = AuthorRepository(session)

    author = repo.create(name="Author To Delete")
    author_id = author.id

    repo.delete(author_id)
    assert repo.get_by_id(author_id) is None

    with pytest.raises(RecordNotFoundError):
        repo.delete(uuid4())


def test_author_repository_validation_and_constraints(session):
    repo = AuthorRepository(session)

    # Blank name
    with pytest.raises(ValueError, match="name must not be blank"):
        repo.create(name="")

    with pytest.raises(ValueError, match="name must not be blank"):
        repo.create(name="   ")

    # Overlength name
    with pytest.raises(ValueError, match="name must be at most 500 characters"):
        repo.create(name="A" * 501)

    # Non-string name
    with pytest.raises(TypeError, match="name must be a str"):
        repo.create(name=123)

    # Update with invalid name
    author = repo.create(name="Valid Author")
    with pytest.raises(ValueError, match="name must not be blank"):
        repo.update(author.id, name="")


def test_author_repository_transaction_recovery(session):
    repo = AuthorRepository(session)

    a1 = repo.create(name="First Valid Author")

    # Invalid creation should not corrupt transaction
    with pytest.raises(ValueError):
        repo.create(name="")

    a2 = repo.create(name="Second Valid Author")
    session.commit()

    assert repo.get_by_id(a1.id) is not None
    assert repo.get_by_id(a2.id) is not None

