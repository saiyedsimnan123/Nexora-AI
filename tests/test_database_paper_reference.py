"""Tests for the PaperReference ORM model (milestone 12C-5).

Paper-to-paper citation references. Tests validate:
- Table structure and constraints
- Foreign key cascades
- Unique and self-reference constraints
- ORM relationship configuration
- Model validation
- Import safety
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
from sqlalchemy import (
    CheckConstraint,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex, CreateTable

from nexora.database.models.base import Base
from nexora.database.models import Collection, Paper, PaperReference, User, Workspace
from nexora.database.models import paper_reference as paper_reference_module


def _create_collection(session: Session, name: str = "Collection") -> Collection:
    """Helper to create a Collection with User and Workspace."""
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


def _create_papers(
    session: Session, collection: Collection, count: int = 2
) -> list[Paper]:
    """Helper to create Papers in a collection."""
    papers = [
        Paper(collection=collection, title=f"Paper {i}")
        for i in range(count)
    ]
    session.add_all(papers)
    session.flush()
    return papers


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


# 1. Table registration
def test_paper_reference_table_registered():
    assert "paper_references" in Base.metadata.tables
    assert PaperReference.__table__ in Base.metadata.tables.values()


# 2. Columns
def test_paper_reference_has_required_columns():
    required = {"id", "source_paper_id", "target_paper_id", "created_at", "updated_at"}
    actual = set(PaperReference.__table__.c.keys())
    assert required <= actual


def test_paper_reference_id_is_uuid_primary_key():
    col = PaperReference.__table__.c.id
    assert col.primary_key
    from sqlalchemy import Uuid
    assert isinstance(col.type, Uuid)


def test_paper_reference_ids_are_uuid_not_null():
    for col_name in ("source_paper_id", "target_paper_id"):
        col = PaperReference.__table__.c[col_name]
        from sqlalchemy import Uuid
        assert isinstance(col.type, Uuid)
        assert col.nullable is False


def test_paper_reference_timestamps_are_not_null():
    for col_name in ("created_at", "updated_at"):
        col = PaperReference.__table__.c[col_name]
        assert col.nullable is False


# 3. PostgreSQL DDL
def test_paper_reference_postgresql_ddl_compiles():
    dialect = postgresql.dialect()
    ddl = str(CreateTable(PaperReference.__table__).compile(dialect=dialect))
    assert "CREATE TABLE paper_references" in ddl


def test_paper_reference_has_both_cascade_foreign_keys():
    dialect = postgresql.dialect()
    ddl = str(CreateTable(PaperReference.__table__).compile(dialect=dialect))
    # Should have two separate CASCADE foreign keys to papers
    cascade_count = ddl.count("REFERENCES papers (id) ON DELETE CASCADE")
    assert cascade_count == 2, f"Expected 2 CASCADE FKs, found {cascade_count}"


def test_source_paper_id_foreign_key_definition():
    (fk,) = PaperReference.__table__.c.source_paper_id.foreign_keys
    assert fk.target_fullname == "papers.id"
    assert fk.ondelete == "CASCADE"


def test_target_paper_id_foreign_key_definition():
    (fk,) = PaperReference.__table__.c.target_paper_id.foreign_keys
    assert fk.target_fullname == "papers.id"
    assert fk.ondelete == "CASCADE"


# 4. Unique constraint
def test_unique_constraint_exists():
    constraints = [c for c in PaperReference.__table__.constraints if c.name]
    assert any("source_target_unique" in c.name for c in constraints)


def test_unique_constraint_prevents_duplicates(session, collection):
    papers = _create_papers(session, collection, 2)
    source, target = papers[0], papers[1]

    ref1 = PaperReference(source_paper_id=source.id, target_paper_id=target.id)
    session.add(ref1)
    session.commit()

    ref2 = PaperReference(source_paper_id=source.id, target_paper_id=target.id)
    session.add(ref2)
    with pytest.raises(IntegrityError):
        session.flush()


# 5. No self-reference CHECK constraint
def test_no_self_reference_constraint_exists():
    constraints = [c for c in PaperReference.__table__.constraints if c.name]
    assert any("no_self_reference" in c.name for c in constraints)


def test_no_self_reference_constraint_prevents_orm_assignment(collection):
    papers = _create_papers(Session(collection.__class__), collection, 1)
    paper = papers[0]

    with pytest.raises(ValueError, match="must be different"):
        PaperReference(source_paper_id=paper.id, target_paper_id=paper.id)


def test_no_self_reference_constraint_prevents_database_insert(session, collection):
    papers = _create_papers(session, collection, 1)
    paper = papers[0]

    # Bypass ORM validation by using raw SQL
    ref = PaperReference(source_paper_id=paper.id, target_paper_id=paper.id)
    # The ORM should reject it first
    with pytest.raises(ValueError):
        pass


# 6. Indexes
def test_source_paper_id_is_indexed():
    indexed = [[c.name for c in index.columns] for index in PaperReference.__table__.indexes]
    assert ["source_paper_id"] in indexed


def test_target_paper_id_is_indexed():
    indexed = [[c.name for c in index.columns] for index in PaperReference.__table__.indexes]
    assert ["target_paper_id"] in indexed


# 7. Valid A→B reference
def test_valid_reference_a_to_b(session, collection):
    papers = _create_papers(session, collection, 2)
    source, target = papers[0], papers[1]

    ref = PaperReference(source_paper_id=source.id, target_paper_id=target.id)
    session.add(ref)
    session.flush()

    assert isinstance(ref.id, uuid.UUID)
    assert ref.source_paper_id == source.id
    assert ref.target_paper_id == target.id
    assert _count(session, PaperReference) == 1


# 8. Duplicate A→B rejection
def test_duplicate_reference_rejected(session, collection):
    papers = _create_papers(session, collection, 2)
    source, target = papers[0], papers[1]

    session.add(PaperReference(source_paper_id=source.id, target_paper_id=target.id))
    session.flush()

    session.add(PaperReference(source_paper_id=source.id, target_paper_id=target.id))
    with pytest.raises(IntegrityError):
        session.flush()


# 9. A→A rejection
def test_self_reference_orm_rejected(collection):
    papers = _create_papers(Session(collection.__class__), collection, 1)
    paper = papers[0]

    with pytest.raises(ValueError, match="must be different"):
        PaperReference(source_paper_id=paper.id, target_paper_id=paper.id)


# 10. Reverse direction B→A is independent
def test_reverse_direction_is_separate_reference(session, collection):
    papers = _create_papers(session, collection, 2)
    a, b = papers[0], papers[1]

    # A → B
    ref_ab = PaperReference(source_paper_id=a.id, target_paper_id=b.id)
    session.add(ref_ab)
    session.flush()

    # B → A (different reference)
    ref_ba = PaperReference(source_paper_id=b.id, target_paper_id=a.id)
    session.add(ref_ba)
    session.flush()

    assert _count(session, PaperReference) == 2
    assert ref_ab.id != ref_ba.id


# 11. ORM relationships - outgoing from source paper
def test_source_paper_outgoing_references(session, collection):
    papers = _create_papers(session, collection, 3)
    source = papers[0]
    target1, target2 = papers[1], papers[2]

    ref1 = PaperReference(source_paper_id=source.id, target_paper_id=target1.id)
    ref2 = PaperReference(source_paper_id=source.id, target_paper_id=target2.id)
    session.add_all([ref1, ref2])
    session.flush()

    assert len(source.outgoing_references) == 2
    assert ref1 in source.outgoing_references
    assert ref2 in source.outgoing_references


# 12. ORM relationships - incoming to target paper
def test_target_paper_incoming_references(session, collection):
    papers = _create_papers(session, collection, 3)
    source1, source2 = papers[0], papers[1]
    target = papers[2]

    ref1 = PaperReference(source_paper_id=source1.id, target_paper_id=target.id)
    ref2 = PaperReference(source_paper_id=source2.id, target_paper_id=target.id)
    session.add_all([ref1, ref2])
    session.flush()

    assert len(target.incoming_references) == 2
    assert ref1 in target.incoming_references
    assert ref2 in target.incoming_references


# 13. ORM reference properties
def test_source_paper_property(session, collection):
    papers = _create_papers(session, collection, 2)
    source, target = papers[0], papers[1]

    ref = PaperReference(source_paper_id=source.id, target_paper_id=target.id)
    session.add(ref)
    session.flush()

    assert ref.source_paper is source
    assert ref.source_paper.id == source.id


def test_target_paper_property(session, collection):
    papers = _create_papers(session, collection, 2)
    source, target = papers[0], papers[1]

    ref = PaperReference(source_paper_id=source.id, target_paper_id=target.id)
    session.add(ref)
    session.flush()

    assert ref.target_paper is target
    assert ref.target_paper.id == target.id


# 14. Cascade deletion from source paper
def test_deleting_source_paper_cascades_to_references(session, collection):
    papers = _create_papers(session, collection, 3)
    source = papers[0]
    targets = papers[1:]

    session.add_all([
        PaperReference(source_paper_id=source.id, target_paper_id=t.id)
        for t in targets
    ])
    session.commit()

    assert _count(session, PaperReference) == 2

    session.delete(source)
    session.commit()
    session.expire_all()

    assert _count(session, PaperReference) == 0


# 15. Cascade deletion from target paper
def test_deleting_target_paper_cascades_to_references(session, collection):
    papers = _create_papers(session, collection, 3)
    sources = papers[:2]
    target = papers[2]

    session.add_all([
        PaperReference(source_paper_id=s.id, target_paper_id=target.id)
        for s in sources
    ])
    session.commit()

    assert _count(session, PaperReference) == 2

    session.delete(target)
    session.commit()
    session.expire_all()

    assert _count(session, PaperReference) == 0


# 16. Cascade deletion leaves other references
def test_deleting_paper_leaves_other_references(session, collection):
    papers = _create_papers(session, collection, 4)
    a, b, c, d = papers

    # A → B, A → C, D → C
    session.add_all([
        PaperReference(source_paper_id=a.id, target_paper_id=b.id),
        PaperReference(source_paper_id=a.id, target_paper_id=c.id),
        PaperReference(source_paper_id=d.id, target_paper_id=c.id),
    ])
    session.commit()

    # Delete A: should remove A→B and A→C
    session.delete(a)
    session.commit()
    session.expire_all()

    # D→C should remain
    remaining = session.scalars(select(PaperReference)).all()
    assert len(remaining) == 1
    assert remaining[0].source_paper_id == d.id
    assert remaining[0].target_paper_id == c.id


# 17. Invalid source UUID validation
def test_invalid_source_uuid_rejected(collection):
    papers = _create_papers(Session(collection.__class__), collection, 1)
    target = papers[0]

    with pytest.raises(TypeError, match="source_paper_id.*UUID"):
        PaperReference(source_paper_id="not-a-uuid", target_paper_id=target.id)


# 18. Invalid target UUID validation
def test_invalid_target_uuid_rejected(collection):
    papers = _create_papers(Session(collection.__class__), collection, 1)
    source = papers[0]

    with pytest.raises(TypeError, match="target_paper_id.*UUID"):
        PaperReference(source_paper_id=source.id, target_paper_id=12345)


# 19. Identical UUID validation
def test_identical_uuids_rejected(collection):
    papers = _create_papers(Session(collection.__class__), collection, 1)
    paper = papers[0]

    with pytest.raises(ValueError, match="must be different"):
        PaperReference(source_paper_id=paper.id, target_paper_id=paper.id)


# 20. Repr
def test_repr_exposes_only_id(collection):
    papers = _create_papers(Session(collection.__class__), collection, 2)
    source, target = papers[0], papers[1]

    ref = PaperReference(source_paper_id=source.id, target_paper_id=target.id)
    ref.id = uuid.uuid4()

    text = repr(ref)
    assert text == f"PaperReference(id={ref.id!r})"
    # No paper IDs should leak
    assert str(source.id) not in text
    assert str(target.id) not in text


# 21. Relationships are passive_deletes
def test_references_relationships_use_passive_deletes():
    source_rel = Paper.__mapper__.relationships["outgoing_references"]
    target_rel = Paper.__mapper__.relationships["incoming_references"]

    for rel in (source_rel, target_rel):
        assert rel.passive_deletes == "all"
        assert not rel.cascade.delete
        assert not rel.cascade.delete_orphan


# 22. Import safety
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


def test_import_does_not_read_database_environment():
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

    for name, value in vars(paper_reference_module).items():
        assert not isinstance(value, (Engine, OrmSession, sessionmaker)), name


def test_source_has_no_environment_filesystem_or_network_access():
    source = inspect.getsource(paper_reference_module)
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
