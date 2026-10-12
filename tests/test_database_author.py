"""Tests for the Author and PaperAuthor ORM models (milestone 12C-6)."""

import uuid
from typing import Generator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from nexora.database.models import Author, Base, Collection, Paper, PaperAuthor, User, Workspace

# Enable sqlite foreign key checks
@pytest.fixture(scope="session")
def engine() -> Generator:
    # Use in-memory sqlite
    engine = create_engine("sqlite:///:memory:", echo=False)

    @pytest.hookimpl(tryfirst=True)
    def enable_fk(dbapi_con, con_record):
        dbapi_con.execute('pragma foreign_keys=ON')

    import sqlalchemy.event
    sqlalchemy.event.listen(engine, 'connect', enable_fk)

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

@pytest.fixture
def db_session(engine) -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
        session.rollback()

@pytest.fixture
def user(db_session: Session) -> User:
    user = User(email=f"test_{uuid.uuid4()}@example.com")
    db_session.add(user)
    db_session.commit()
    return user

@pytest.fixture
def workspace(db_session: Session, user: User) -> Workspace:
    ws = Workspace(user_id=user.id, name="Test WS")
    db_session.add(ws)
    db_session.commit()
    return ws

@pytest.fixture
def collection(db_session: Session, workspace: Workspace) -> Collection:
    col = Collection(workspace_id=workspace.id, name="Test Collection")
    db_session.add(col)
    db_session.commit()
    return col

@pytest.fixture
def paper(db_session: Session, collection: Collection) -> Paper:
    p = Paper(collection_id=collection.id, title="Test Paper")
    db_session.add(p)
    db_session.commit()
    return p

# --- Author Tests ---

def test_author_creation_persistence(db_session: Session):
    # 1. Author creation and persistence.
    author = Author(name="John Doe")
    db_session.add(author)
    db_session.commit()
    assert author.id is not None
    assert author.name == "John Doe"

def test_author_uuid_primary_key(db_session: Session):
    # 2. UUID primary key.
    author = Author(name="Jane Doe")
    db_session.add(author)
    db_session.commit()
    assert isinstance(author.id, uuid.UUID)

def test_author_name_persistence(db_session: Session):
    # 3. Name persistence.
    author = Author(name="Alice Bob")
    db_session.add(author)
    db_session.commit()
    db_session.refresh(author)
    assert author.name == "Alice Bob"

def test_author_missing_name(db_session: Session):
    # 4. Missing name handling.
    with pytest.raises((TypeError, IntegrityError)):
        # Depending on when the error is caught, could be early or db level
        author = Author()
        db_session.add(author)
        db_session.commit()

def test_author_empty_name_rejection():
    # 5. Empty name rejection.
    with pytest.raises(ValueError, match="name must not be blank"):
        Author(name="")

def test_author_whitespace_only_name_rejection():
    # 6. Whitespace-only name rejection.
    with pytest.raises(ValueError, match="name must not be blank"):
        Author(name="   ")

def test_author_name_length_limit():
    # 7. Name length limit.
    long_name = "A" * 501
    with pytest.raises(ValueError, match="name must be at most 500 characters"):
        Author(name=long_name)

def test_author_non_string_name_rejection():
    # 8. Non-string name rejection.
    with pytest.raises(TypeError, match="name must be a str"):
        Author(name=123)

def test_author_repr():
    # 9. "__repr__".
    author = Author(id=uuid.uuid4(), name="Test Name")
    rep = repr(author)
    assert "Author(id=" in rep
    assert "name='Test Name'" in rep
    assert "created_at" not in rep

def test_author_db_blank_name_constraint(db_session: Session, engine):
    # 10. Database-level blank-name constraint tested independently of ORM validation.
    with pytest.raises(IntegrityError):
        # Insert directly to bypass ORM validates
        db_session.execute(Author.__table__.insert().values(id=uuid.uuid4(), name="   "))
        db_session.commit()
    db_session.rollback()


# --- PaperAuthor Tests ---

def test_paper_author_creation_persistence(db_session: Session, paper: Paper):
    # 11. Valid PaperAuthor creation and persistence.
    author = Author(name="Dr. Smith")
    db_session.add(author)
    db_session.commit()

    pa = PaperAuthor(paper_id=paper.id, author_id=author.id, author_order=1)
    db_session.add(pa)
    db_session.commit()
    assert pa.id is not None
    assert pa.paper_id == paper.id
    assert pa.author_id == author.id

def test_paper_author_correct_paper_id(db_session: Session):
    # 12. Correct "paper_id".
    author = Author(name="Dr. Smith")
    db_session.add(author)
    db_session.commit()

    pa = PaperAuthor(paper_id=uuid.uuid4(), author_id=author.id, author_order=1)
    db_session.add(pa)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

def test_paper_author_correct_author_id(db_session: Session, paper: Paper):
    # 13. Correct "author_id".
    pa = PaperAuthor(paper_id=paper.id, author_id=uuid.uuid4(), author_order=1)
    db_session.add(pa)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

def test_paper_author_correct_author_order(db_session: Session, paper: Paper):
    # 14. Correct "author_order".
    author = Author(name="A")
    db_session.add(author)
    pa = PaperAuthor(paper=paper, author=author, author_order=2)
    db_session.add(pa)
    db_session.commit()
    assert pa.author_order == 2

def test_paper_author_accepts_valid_positive_integers(db_session: Session, paper: Paper):
    # 15. "author_order" accepts valid positive integers.
    author = Author(name="B")
    db_session.add(author)
    pa = PaperAuthor(paper=paper, author=author, author_order=100)
    db_session.add(pa)
    db_session.commit()
    assert pa.author_order == 100

def test_paper_author_zero_order_rejected():
    # 16. Zero order rejected.
    with pytest.raises(ValueError, match="author_order must be >= 1"):
        PaperAuthor(author_order=0)

def test_paper_author_negative_order_rejected():
    # 17. Negative order rejected.
    with pytest.raises(ValueError, match="author_order must be >= 1"):
        PaperAuthor(author_order=-1)

def test_paper_author_boolean_invalid_order_types_rejected():
    # 18. Boolean and other invalid order types rejected.
    with pytest.raises(TypeError, match="author_order must be an integer, got bool"):
        PaperAuthor(author_order=True)
    with pytest.raises(TypeError, match="author_order must be an int"):
        PaperAuthor(author_order="1")

def test_paper_author_duplicate_pair_rejected(db_session: Session, paper: Paper):
    # 19. Duplicate "(paper_id, author_id)" rejected.
    author = Author(name="C")
    db_session.add(author)
    db_session.commit()

    pa1 = PaperAuthor(paper_id=paper.id, author_id=author.id, author_order=1)
    db_session.add(pa1)
    db_session.commit()

    pa2 = PaperAuthor(paper_id=paper.id, author_id=author.id, author_order=2)
    db_session.add(pa2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

def test_paper_authorship_relationships(db_session: Session, paper: Paper):
    # 20. Paper-to-authorship relationship works.
    # 21. Author-to-authorship relationship works.
    author = Author(name="D")
    pa = PaperAuthor(paper=paper, author=author, author_order=1)
    db_session.add(pa)
    db_session.commit()

    db_session.refresh(paper)
    db_session.refresh(author)

    assert len(paper.authorships) == 1
    assert paper.authorships[0] == pa

    assert len(author.paper_authorships) == 1
    assert author.paper_authorships[0] == pa

def test_paper_author_cascade_delete_paper(db_session: Session, paper: Paper):
    # 22. Deleting a Paper cascades to its PaperAuthor rows.
    author = Author(name="E")
    pa = PaperAuthor(paper=paper, author=author, author_order=1)
    db_session.add(pa)
    db_session.commit()

    pa_id = pa.id
    db_session.delete(paper)
    db_session.commit()

    # Verify paper_author is gone
    result = db_session.scalars(select(PaperAuthor).where(PaperAuthor.id == pa_id)).first()
    assert result is None

    # Author should still exist
    result_author = db_session.scalars(select(Author).where(Author.id == author.id)).first()
    assert result_author is not None

def test_paper_author_cascade_delete_author(db_session: Session, paper: Paper):
    # 23. Deleting an Author cascades to its PaperAuthor rows.
    author = Author(name="F")
    pa = PaperAuthor(paper=paper, author=author, author_order=1)
    db_session.add(pa)
    db_session.commit()

    pa_id = pa.id
    db_session.delete(author)
    db_session.commit()

    result = db_session.scalars(select(PaperAuthor).where(PaperAuthor.id == pa_id)).first()
    assert result is None

def test_one_paper_multiple_authors(db_session: Session, paper: Paper):
    # 24. One Paper supports multiple Authors.
    a1 = Author(name="Author 1")
    a2 = Author(name="Author 2")
    pa1 = PaperAuthor(paper=paper, author=a1, author_order=1)
    pa2 = PaperAuthor(paper=paper, author=a2, author_order=2)
    db_session.add_all([pa1, pa2])
    db_session.commit()

    db_session.refresh(paper)
    assert len(paper.authorships) == 2

def test_one_author_multiple_papers(db_session: Session, collection: Collection):
    # 25. One Author can be associated with multiple Papers.
    p1 = Paper(collection_id=collection.id, title="Paper 1")
    p2 = Paper(collection_id=collection.id, title="Paper 2")
    author = Author(name="Author A")

    pa1 = PaperAuthor(paper=p1, author=author, author_order=1)
    pa2 = PaperAuthor(paper=p2, author=author, author_order=1)
    db_session.add_all([p1, p2, pa1, pa2])
    db_session.commit()

    db_session.refresh(author)
    assert len(author.paper_authorships) == 2

def test_author_order_preserved(db_session: Session, paper: Paper):
    # 26. Author order is preserved when associations are queried in the appropriate order.
    a1 = Author(name="First")
    a2 = Author(name="Second")
    a3 = Author(name="Third")

    pa2 = PaperAuthor(paper=paper, author=a2, author_order=2)
    pa1 = PaperAuthor(paper=paper, author=a1, author_order=1)
    pa3 = PaperAuthor(paper=paper, author=a3, author_order=3)
    db_session.add_all([pa2, pa1, pa3])
    db_session.commit()

    # Query ordered by author_order
    results = db_session.scalars(
        select(PaperAuthor)
        .where(PaperAuthor.paper_id == paper.id)
        .order_by(PaperAuthor.author_order)
    ).all()

    assert len(results) == 3
    assert results[0].author_order == 1
    assert results[0].author.name == "First"
    assert results[1].author_order == 2
    assert results[1].author.name == "Second"
    assert results[2].author_order == 3
    assert results[2].author.name == "Third"

def test_paper_author_db_author_order_constraint(db_session: Session, paper: Paper):
    # Test DB level constraint author_order >= 1
    author = Author(name="G")
    db_session.add(author)
    db_session.commit()

    with pytest.raises(IntegrityError):
        db_session.execute(
            PaperAuthor.__table__.insert().values(
                id=uuid.uuid4(),
                paper_id=paper.id,
                author_id=author.id,
                author_order=0
            )
        )
        db_session.commit()
    db_session.rollback()
