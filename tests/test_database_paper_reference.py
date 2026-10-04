"""Tests for the PaperReference ORM model (milestone 12C-5).

Every test uses the single ``session`` fixture below (one ``Session(engine)``
on a SQLite engine with foreign keys enforced). Records are always created
in that same session; sessions are never built from model objects.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
import uuid

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy import create_engine, event, func, insert, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex, CreateTable

import nexora.database.models as models_package
from nexora.database.models import (
    Collection,
    Paper,
    PaperReference,
    User,
    Workspace,
)
from nexora.database.models import paper_reference as reference_module
from nexora.database.models.base import Base


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


def create_papers(session: Session, count: int = 2) -> list[Paper]:
    """Create User -> Workspace -> Collection and ``count`` papers, all in ``session``."""
    user = User(email=f"{uuid.uuid4()}@example.com")
    session.add(user)
    session.flush()

    workspace = Workspace(user_id=user.id, name="Test Workspace")
    session.add(workspace)
    session.flush()

    collection = Collection(workspace_id=workspace.id, name="Test Collection")
    session.add(collection)
    session.flush()

    papers = [
        Paper(collection_id=collection.id, title=f"Paper {index}")
        for index in range(count)
    ]
    session.add_all(papers)
    session.flush()
    return papers


def add_reference(session: Session, source: Paper, target: Paper) -> PaperReference:
    reference = PaperReference(source_paper_id=source.id, target_paper_id=target.id)
    session.add(reference)
    session.flush()
    return reference


def count_references(session: Session) -> int:
    return session.execute(
        select(func.count()).select_from(PaperReference)
    ).scalar_one()


def raw_reference_row(source_id, target_id):
    return {"source_paper_id": source_id, "target_paper_id": target_id}


# 1. Table registration
def test_model_is_exported_and_registered():
    assert models_package.PaperReference is PaperReference
    assert PaperReference.__tablename__ == "paper_references"
    assert "paper_references" in Base.metadata.tables
    assert Base.metadata.tables["paper_references"] is PaperReference.__table__


def test_existing_paper_api_is_preserved():
    relationships = set(Paper.__mapper__.relationships.keys())
    assert {
        "collection",
        "documents",
        "outgoing_references",
        "incoming_references",
    } <= relationships


# 2. Required columns
def test_table_has_exactly_the_expected_columns():
    assert set(PaperReference.__table__.c.keys()) == {
        "id",
        "source_paper_id",
        "target_paper_id",
        "created_at",
        "updated_at",
    }


# 3. UUID primary key
def test_primary_key_is_uuid(session):
    source, target = create_papers(session)
    reference = add_reference(session, source, target)
    assert isinstance(reference.id, uuid.UUID)
    assert PaperReference.__table__.c.id.primary_key
    assert [c.name for c in PaperReference.__table__.primary_key.columns] == ["id"]


def test_each_reference_gets_a_distinct_id(session):
    a, b, c = create_papers(session, 3)
    first, second = add_reference(session, a, b), add_reference(session, a, c)
    assert first.id != second.id


def test_timestamps_are_set(session):
    source, target = create_papers(session)
    reference = add_reference(session, source, target)
    assert reference.created_at is not None
    assert reference.updated_at is not None


# 4-5. NOT NULL
def test_source_paper_id_is_not_nullable(session):
    assert PaperReference.__table__.c.source_paper_id.nullable is False
    _, target = create_papers(session)
    session.add(PaperReference(target_paper_id=target.id))
    with pytest.raises(IntegrityError):
        session.flush()


def test_target_paper_id_is_not_nullable(session):
    source, _ = create_papers(session)
    assert PaperReference.__table__.c.target_paper_id.nullable is False
    session.add(PaperReference(source_paper_id=source.id))
    with pytest.raises(IntegrityError):
        session.flush()


# 6. PostgreSQL CREATE TABLE compilation
def test_postgresql_create_table_compiles():
    dialect = postgresql.dialect()
    ddl = str(CreateTable(PaperReference.__table__).compile(dialect=dialect))
    assert "CREATE TABLE paper_references" in ddl
    assert ddl.count("REFERENCES papers (id) ON DELETE CASCADE") == 2
    assert "source_paper_id <> target_paper_id" in ddl
    assert "UNIQUE (source_paper_id, target_paper_id)" in ddl
    assert "source_target_unique" in ddl
    assert "no_self_reference" in ddl
    for index in PaperReference.__table__.indexes:
        assert "CREATE INDEX" in str(CreateIndex(index).compile(dialect=dialect))


# 7-8. Foreign keys and cascade
@pytest.mark.parametrize("column", ["source_paper_id", "target_paper_id"])
def test_foreign_key_references_papers_id_with_cascade(column):
    (fk,) = PaperReference.__table__.c[column].foreign_keys
    assert fk.target_fullname == "papers.id"
    assert fk.ondelete == "CASCADE"


def test_foreign_key_rejects_unknown_paper(session):
    (source,) = create_papers(session, 1)
    session.add(
        PaperReference(source_paper_id=source.id, target_paper_id=uuid.uuid4())
    )
    with pytest.raises(IntegrityError):
        session.flush()


# 9. Unique constraint
def test_unique_constraint_covers_source_and_target():
    uniques = [
        c
        for c in PaperReference.__table__.constraints
        if isinstance(c, UniqueConstraint)
    ]
    assert len(uniques) == 1
    assert [col.name for col in uniques[0].columns] == [
        "source_paper_id",
        "target_paper_id",
    ]
    assert "source_target_unique" in str(uniques[0].name)


# 10. Self-reference check constraint
def test_no_self_reference_check_constraint_is_defined():
    checks = [
        c
        for c in PaperReference.__table__.constraints
        if isinstance(c, CheckConstraint)
    ]
    assert len(checks) == 1
    assert "no_self_reference" in str(checks[0].name)
    assert str(checks[0].sqltext) == "source_paper_id <> target_paper_id"


# 11-12. Indexes
@pytest.mark.parametrize("column", ["source_paper_id", "target_paper_id"])
def test_foreign_key_column_is_indexed(column):
    indexed = [[c.name for c in index.columns] for index in PaperReference.__table__.indexes]
    assert [column] in indexed


# 13. Valid A -> B
def test_valid_reference_is_stored(session):
    a, b = create_papers(session)
    reference = add_reference(session, a, b)
    assert reference.source_paper_id == a.id
    assert reference.target_paper_id == b.id
    assert count_references(session) == 1


# 14. Duplicate A -> B
def test_duplicate_reference_is_rejected(session):
    a, b = create_papers(session)
    add_reference(session, a, b)
    session.add(PaperReference(source_paper_id=a.id, target_paper_id=b.id))
    with pytest.raises(IntegrityError):
        session.flush()


def test_duplicate_is_rejected_by_raw_insert(session):
    a, b = create_papers(session)
    row = raw_reference_row(a.id, b.id)
    session.execute(insert(PaperReference.__table__).values(**row))
    with pytest.raises(IntegrityError):
        session.execute(insert(PaperReference.__table__).values(**row))


# 15. Self-reference rejected by model validation
def test_self_reference_is_rejected_by_model_validation(session):
    (paper,) = create_papers(session, 1)
    with pytest.raises(ValueError, match="different"):
        PaperReference(source_paper_id=paper.id, target_paper_id=paper.id)


def test_self_reference_is_rejected_whichever_field_is_set_last(session):
    (paper,) = create_papers(session, 1)
    with pytest.raises(ValueError, match="different"):
        PaperReference(target_paper_id=paper.id, source_paper_id=paper.id)


def test_self_reference_is_rejected_on_later_assignment(session):
    a, b = create_papers(session)
    reference = PaperReference(source_paper_id=a.id, target_paper_id=b.id)
    with pytest.raises(ValueError, match="different"):
        reference.target_paper_id = a.id
    with pytest.raises(ValueError, match="different"):
        reference.source_paper_id = b.id


# 16. Self-reference rejected by the database (validators bypassed)
def test_self_reference_is_rejected_by_database_constraint(session):
    (paper,) = create_papers(session, 1)
    with pytest.raises(IntegrityError):
        session.execute(
            insert(PaperReference.__table__).values(
                **raw_reference_row(paper.id, paper.id)
            )
        )


# 17. Reverse direction
def test_reverse_direction_is_allowed(session):
    a, b = create_papers(session)
    add_reference(session, a, b)
    add_reference(session, b, a)
    assert count_references(session) == 2


def test_one_paper_may_cite_many_and_be_cited_by_many(session):
    a, b, c = create_papers(session, 3)
    add_reference(session, a, b)
    add_reference(session, a, c)
    add_reference(session, c, b)
    assert count_references(session) == 3


# 18-21. Relationships
def test_outgoing_references_relationship(session):
    a, b = create_papers(session)
    reference = add_reference(session, a, b)
    session.refresh(a)
    assert a.outgoing_references == [reference]
    assert b.outgoing_references == []


def test_incoming_references_relationship(session):
    a, b = create_papers(session)
    reference = add_reference(session, a, b)
    session.refresh(b)
    assert b.incoming_references == [reference]
    assert a.incoming_references == []


def test_source_paper_relationship(session):
    a, b = create_papers(session)
    reference = add_reference(session, a, b)
    session.refresh(reference)
    assert reference.source_paper is a


def test_target_paper_relationship(session):
    a, b = create_papers(session)
    reference = add_reference(session, a, b)
    session.refresh(reference)
    assert reference.target_paper is b


def test_relationships_can_be_used_to_build_a_reference(session):
    a, b = create_papers(session)
    reference = PaperReference(source_paper=a, target_paper=b)
    session.add(reference)
    session.flush()
    assert reference.source_paper_id == a.id
    assert reference.target_paper_id == b.id
    assert reference in a.outgoing_references
    assert reference in b.incoming_references


# 22-24. Database cascade on deletion
def test_deleting_source_paper_removes_its_outgoing_reference(session):
    a, b = create_papers(session)
    add_reference(session, a, b)
    session.commit()
    session.delete(a)
    session.commit()
    session.expire_all()
    assert count_references(session) == 0
    assert session.get(Paper, b.id) is not None


def test_deleting_target_paper_removes_its_incoming_reference(session):
    a, b = create_papers(session)
    add_reference(session, a, b)
    session.commit()
    session.delete(b)
    session.commit()
    session.expire_all()
    assert count_references(session) == 0
    assert session.get(Paper, a.id) is not None


def test_deleting_a_middle_paper_removes_both_of_its_edges(session):
    a, b, c = create_papers(session, 3)
    add_reference(session, a, b)
    add_reference(session, b, c)
    session.commit()
    session.delete(b)
    session.commit()
    session.expire_all()
    assert count_references(session) == 0


def test_unrelated_references_survive_when_another_paper_is_deleted(session):
    a, b, c, d = create_papers(session, 4)
    add_reference(session, a, b)
    survivor = add_reference(session, c, d)
    survivor_id = survivor.id
    session.commit()
    session.delete(a)
    session.commit()
    session.expire_all()
    remaining = session.scalars(select(PaperReference)).all()
    assert [r.id for r in remaining] == [survivor_id]


def test_deleting_the_collection_removes_all_references(session):
    a, b = create_papers(session)
    add_reference(session, a, b)
    session.commit()
    collection = session.get(Collection, a.collection_id)
    session.delete(collection)
    session.commit()
    session.expire_all()
    assert count_references(session) == 0


# 25-29. UUID validation
@pytest.mark.parametrize("bad", ["not-a-uuid", str(uuid.uuid4()), 1.5, None, b"id"])
def test_invalid_source_value_is_rejected(bad):
    with pytest.raises(TypeError, match="source_paper_id"):
        PaperReference(source_paper_id=bad, target_paper_id=uuid.uuid4())


@pytest.mark.parametrize("bad", ["not-a-uuid", str(uuid.uuid4()), 1.5, None, b"id"])
def test_invalid_target_value_is_rejected(bad):
    with pytest.raises(TypeError, match="target_paper_id"):
        PaperReference(source_paper_id=uuid.uuid4(), target_paper_id=bad)


def test_integer_source_value_is_rejected():
    with pytest.raises(TypeError, match="source_paper_id"):
        PaperReference(source_paper_id=12345, target_paper_id=uuid.uuid4())


def test_integer_target_value_is_rejected():
    with pytest.raises(TypeError, match="target_paper_id"):
        PaperReference(source_paper_id=uuid.uuid4(), target_paper_id=12345)


def test_identical_uuid_values_are_rejected_without_a_database():
    shared = uuid.uuid4()
    with pytest.raises(ValueError, match="different"):
        PaperReference(source_paper_id=shared, target_paper_id=shared)


def test_distinct_uuid_values_are_accepted_without_a_database():
    reference = PaperReference(
        source_paper_id=uuid.uuid4(), target_paper_id=uuid.uuid4()
    )
    assert reference.source_paper_id != reference.target_paper_id


# 30-31. Relationship configuration
@pytest.mark.parametrize("name", ["outgoing_references", "incoming_references"])
def test_paper_relationships_use_passive_deletes_all(name):
    relationship = Paper.__mapper__.relationships[name]
    assert relationship.passive_deletes == "all"


@pytest.mark.parametrize("name", ["outgoing_references", "incoming_references"])
def test_paper_relationships_have_no_orm_delete_cascade(name):
    relationship = Paper.__mapper__.relationships[name]
    assert not relationship.cascade.delete
    assert not relationship.cascade.delete_orphan


def test_paper_relationships_use_distinct_foreign_keys():
    outgoing = Paper.__mapper__.relationships["outgoing_references"]
    incoming = Paper.__mapper__.relationships["incoming_references"]
    assert [remote.name for _, remote in outgoing.local_remote_pairs] == [
        "source_paper_id"
    ]
    assert [remote.name for _, remote in incoming.local_remote_pairs] == [
        "target_paper_id"
    ]


def test_reference_side_relationships_are_paired_with_paper():
    mapper = PaperReference.__mapper__
    assert mapper.relationships["source_paper"].back_populates == "outgoing_references"
    assert mapper.relationships["target_paper"].back_populates == "incoming_references"
    for name in ("source_paper", "target_paper"):
        relationship = mapper.relationships[name]
        assert not relationship.cascade.delete
        assert not relationship.cascade.delete_orphan


# 32. repr
def test_repr_exposes_only_the_id(session):
    a, b = create_papers(session)
    reference = add_reference(session, a, b)
    text = repr(reference)
    assert text == f"PaperReference(id={reference.id!r})"
    assert str(a.id) not in text
    assert str(b.id) not in text


# 33-36. Import safety
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

    import nexora.database.models.paper_reference  # noqa: F401

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

    for name, value in vars(reference_module).items():
        assert not isinstance(value, (Engine, OrmSession, sessionmaker)), name


def test_source_has_no_environment_filesystem_or_network_access():
    source = inspect.getsource(reference_module)
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
