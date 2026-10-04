"""Tests for the Paper ORM model (milestone 12C-3).

ADAPT POINT: ``_create_collection`` builds the User -> Workspace ->
Collection chain. Adjust the constructor arguments there to match the real
12C-2 models; nothing else in this file depends on those details.
"""

from __future__ import annotations

import datetime
import inspect
import json
import os
import subprocess
import sys
import textwrap
import uuid

import pytest
from sqlalchemy import Date, String, create_engine, event, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex, CreateTable

from nexora.database.models.base import Base
from nexora.database.models import Collection, Paper, User, Workspace
from nexora.database.models import paper as paper_module


def _create_collection(session: Session, name: str = "Collection") -> Collection:
    user = User(email=f"{uuid.uuid4()}@example.com")
    session.add(user)
    session.flush()
    workspace = Workspace(name="Workspace", user_id=user.id)
    session.add(workspace)
    session.flush()
    collection = Collection(name=name, workspace_id=workspace.id)
    session.add(collection)
    session.flush()
    return collection


@pytest.fixture
def engine():
    eng = create_engine("sqlite://")

    @event.listens_for(eng, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as db:
        yield db


@pytest.fixture
def collection(session):
    return _create_collection(session)


def _count(session, model):
    return session.execute(select(func.count()).select_from(model)).scalar_one()


# 1. UUID primary key
def test_primary_key_is_uuid(session, collection):
    paper = Paper(collection_id=collection.id, title="A")
    session.add(paper)
    session.flush()
    assert isinstance(paper.id, uuid.UUID)
    assert Paper.__table__.c.id.primary_key


def test_each_paper_gets_a_distinct_id(session, collection):
    first = Paper(collection_id=collection.id, title="A")
    second = Paper(collection_id=collection.id, title="B")
    session.add_all([first, second])
    session.flush()
    assert first.id != second.id


# 2. Timestamps
def test_timestamps_are_set(session, collection):
    paper = Paper(collection_id=collection.id, title="A")
    session.add(paper)
    session.flush()
    assert paper.created_at is not None
    assert paper.updated_at is not None


# 3. Required collection
def test_collection_id_is_required(session):
    assert Paper.__table__.c.collection_id.nullable is False
    session.add(Paper(title="Orphan"))
    with pytest.raises(IntegrityError):
        session.flush()


# 4. Title maximum length
def test_title_at_maximum_length_is_accepted(session, collection):
    title = "t" * 500
    paper = Paper(collection_id=collection.id, title=title)
    session.add(paper)
    session.flush()
    assert paper.title == title


def test_title_over_maximum_length_is_rejected(collection):
    with pytest.raises(ValueError, match="500"):
        Paper(collection_id=collection.id, title="t" * 501)


def test_title_column_length_is_500():
    column = Paper.__table__.c.title
    assert isinstance(column.type, String)
    assert column.type.length == 500
    assert column.nullable is False


# 5. Blank title rejection
@pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\n", " \t\n "])
def test_blank_title_is_rejected(collection, blank):
    with pytest.raises(ValueError, match="blank"):
        Paper(collection_id=collection.id, title=blank)


def test_title_must_be_a_string(collection):
    with pytest.raises(TypeError):
        Paper(collection_id=collection.id, title=123)


def test_title_text_is_preserved_unchanged(session, collection):
    paper = Paper(collection_id=collection.id, title="  Padded Title  ")
    session.add(paper)
    session.flush()
    assert paper.title == "  Padded Title  "


def test_blank_title_check_constraint_exists():
    names = {c.name for c in Paper.__table__.constraints if c.name}
    assert any("title_not_blank" in name for name in names)


# 6. Optional abstract
def test_abstract_is_optional(session, collection):
    paper = Paper(collection_id=collection.id, title="A")
    session.add(paper)
    session.flush()
    assert paper.abstract is None
    assert Paper.__table__.c.abstract.nullable is True


def test_abstract_is_stored(session, collection):
    paper = Paper(collection_id=collection.id, title="A", abstract="Summary.")
    session.add(paper)
    session.flush()
    assert paper.abstract == "Summary."


def test_abstract_is_bounded():
    limit = Paper.__table__.c.abstract.type.length
    assert limit is not None and limit >= 5000
    with pytest.raises(ValueError):
        Paper(title="A", abstract="x" * (limit + 1))


# 7. DOI
def test_doi_is_optional_and_stored_as_given(session, collection):
    plain = Paper(collection_id=collection.id, title="A")
    with_doi = Paper(collection_id=collection.id, title="B", doi="10.1000/ABC.1")
    session.add_all([plain, with_doi])
    session.flush()
    assert plain.doi is None
    assert with_doi.doi == "10.1000/ABC.1"


def test_doi_is_not_unique(session, collection):
    session.add_all(
        [
            Paper(collection_id=collection.id, title="A", doi="10.1000/x"),
            Paper(collection_id=collection.id, title="B", doi="10.1000/x"),
        ]
    )
    session.flush()
    assert _count(session, Paper) == 2
    assert Paper.__table__.c.doi.unique in (None, False)


# 8. External id
def test_external_id_is_optional_and_not_unique(session, collection):
    session.add_all(
        [
            Paper(collection_id=collection.id, title="A", external_id="arxiv:1"),
            Paper(collection_id=collection.id, title="B", external_id="arxiv:1"),
            Paper(collection_id=collection.id, title="C"),
        ]
    )
    session.flush()
    assert _count(session, Paper) == 3
    assert Paper.__table__.c.external_id.nullable is True
    assert Paper.__table__.c.external_id.unique in (None, False)


# 9. Publication date
def test_publication_date_round_trip(session, collection):
    paper = Paper(
        collection_id=collection.id,
        title="A",
        publication_date=datetime.date(2024, 5, 17),
    )
    session.add(paper)
    session.commit()
    session.expire_all()
    stored = session.get(Paper, paper.id)
    assert stored.publication_date == datetime.date(2024, 5, 17)
    assert isinstance(Paper.__table__.c.publication_date.type, Date)


def test_publication_date_is_optional(session, collection):
    paper = Paper(collection_id=collection.id, title="A")
    session.add(paper)
    session.flush()
    assert paper.publication_date is None


# 10. Collection foreign key
def test_collection_foreign_key_definition():
    (fk,) = Paper.__table__.c.collection_id.foreign_keys
    assert fk.target_fullname == "collections.id"
    assert fk.ondelete == "CASCADE"


def test_foreign_key_rejects_unknown_collection(session):
    session.add(Paper(collection_id=uuid.uuid4(), title="A"))
    with pytest.raises(IntegrityError):
        session.flush()


# 11. collection_id index
def test_collection_id_is_indexed():
    indexed = [
        [c.name for c in index.columns] for index in Paper.__table__.indexes
    ]
    assert ["collection_id"] in indexed


# 12. Relationship
def test_collection_papers_relationship(session, collection):
    paper = Paper(collection=collection, title="A")
    session.add(paper)
    session.flush()
    assert paper.collection is collection
    assert paper in collection.papers
    assert paper.collection_id == collection.id


def test_deletion_is_left_to_the_database():
    papers = Collection.__mapper__.relationships["papers"]
    documents = Paper.__mapper__.relationships["documents"]
    for relationship in (papers, documents):
        assert relationship.passive_deletes == "all"
        assert not relationship.cascade.delete
        assert not relationship.cascade.delete_orphan


# 13. Multiple papers
def test_multiple_papers_in_one_collection(session, collection):
    papers = [Paper(collection=collection, title=f"P{i}") for i in range(3)]
    session.add_all(papers)
    session.flush()
    assert len(collection.papers) == 3
    assert {p.title for p in collection.papers} == {"P0", "P1", "P2"}


def test_papers_in_different_collections_are_separate(session):
    first = _create_collection(session, "One")
    second = _create_collection(session, "Two")
    session.add_all(
        [Paper(collection=first, title="A"), Paper(collection=second, title="B")]
    )
    session.flush()
    assert [p.title for p in first.papers] == ["A"]
    assert [p.title for p in second.papers] == ["B"]


# 14. PostgreSQL DDL
def test_postgresql_ddl_compiles():
    dialect = postgresql.dialect()
    ddl = str(CreateTable(Paper.__table__).compile(dialect=dialect))
    assert "CREATE TABLE papers" in ddl
    assert "REFERENCES collections (id) ON DELETE CASCADE" in ddl
    assert "length(trim(title)) > 0" in ddl
    for index in Paper.__table__.indexes:
        assert "CREATE INDEX" in str(CreateIndex(index).compile(dialect=dialect))


# 15. Cascade deletion
def test_deleting_collection_cascades_to_papers(session, collection):
    session.add_all([Paper(collection=collection, title=f"P{i}") for i in range(2)])
    session.commit()
    assert _count(session, Paper) == 2
    session.delete(collection)
    session.commit()
    session.expire_all()
    assert _count(session, Paper) == 0


def test_deleting_one_collection_keeps_other_papers(session):
    first = _create_collection(session, "One")
    second = _create_collection(session, "Two")
    session.add_all(
        [Paper(collection=first, title="A"), Paper(collection=second, title="B")]
    )
    session.commit()
    session.delete(first)
    session.commit()
    session.expire_all()
    assert [p.title for p in session.scalars(select(Paper))] == ["B"]


# repr
def test_repr_exposes_only_the_id(collection):
    paper = Paper(
        collection_id=collection.id,
        title="Secret Title",
        abstract="Secret abstract",
        doi="10.1000/secret",
    )
    paper.id = uuid.uuid4()
    text = repr(paper)
    assert text == f"Paper(id={paper.id!r})"
    for hidden in ("Secret Title", "Secret abstract", "10.1000/secret"):
        assert hidden not in text


# 16-18. Import safety
_PROBE = textwrap.dedent(
    """
    import json, os, socket

    accessed = []

    class Tracking(dict):
        def get(self, key, default=None):
            accessed.append(str(key))
            return super().get(key, default)

        def __getitem__(self, key):
            accessed.append(str(key))
            return super().__getitem__(key)

        def __contains__(self, key):
            accessed.append(str(key))
            return super().__contains__(key)

    os.environ = Tracking(os.environ)
    network = []

    def blocked(*args, **kwargs):
        network.append(1)
        raise RuntimeError("network blocked")

    socket.socket.connect = blocked
    socket.getaddrinfo = blocked

    import sqlalchemy
    engines = []
    original = sqlalchemy.create_engine

    def tracking_create_engine(*args, **kwargs):
        engines.append(1)
        return original(*args, **kwargs)

    sqlalchemy.create_engine = tracking_create_engine

    import nexora.database.models.paper  # noqa: F401

    print(json.dumps({"accessed": accessed, "network": len(network), "engines": len(engines)}))
    """
)


def _run_probe():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["DATABASE_URL"] = "postgresql://user:secret@localhost/db"
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_import_is_safe_no_network_no_engine():
    probe = _run_probe()
    assert probe["network"] == 0
    assert probe["engines"] == 0


def test_import_does_not_read_database_environment_values():
    probe = _run_probe()
    touched = [
        key
        for key in probe["accessed"]
        if any(word in key.upper() for word in ("DATABASE", "DB_", "SECRET", "PASSWORD"))
    ]
    assert touched == []


def test_no_engine_or_session_globals():
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session as OrmSession, sessionmaker

    for name, value in vars(paper_module).items():
        assert not isinstance(value, (Engine, OrmSession, sessionmaker)), name


def test_source_has_no_environment_filesystem_or_network_access():
    source = inspect.getsource(paper_module)
    for forbidden in (
        "environ",
        "getenv",
        "create_engine",
        "sessionmaker",
        "Session(",
        "open(",
        "pathlib",
        "socket",
        "requests",
        "httpx",
        "urllib",
    ):
        assert forbidden not in source, forbidden
