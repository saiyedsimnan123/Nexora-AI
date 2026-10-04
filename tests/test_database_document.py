"""Tests for the Document ORM model (milestone 12C-3).

ADAPT POINT: ``_create_collection`` builds the User -> Workspace ->
Collection chain. Adjust the constructor arguments there to match the real
12C-2 models; nothing else in this file depends on those details.
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
from sqlalchemy import BigInteger, String, create_engine, event, func, insert, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex, CreateTable

from nexora.database.models.base import Base
from nexora.database.models import Collection, Document, Paper, User, Workspace
from nexora.database.models import document as document_module

VALID_CHECKSUM = "0123456789abcdef" * 4


def _create_collection(session: Session) -> Collection:
    user = User(email=f"{uuid.uuid4()}@example.com")
    session.add(user)
    session.flush()
    workspace = Workspace(name="Workspace", user_id=user.id)
    session.add(workspace)
    session.flush()
    collection = Collection(name="Collection", workspace_id=workspace.id)
    session.add(collection)
    session.flush()
    return collection


def _create_paper(session: Session, title: str = "Paper") -> Paper:
    paper = Paper(collection=_create_collection(session), title=title)
    session.add(paper)
    session.flush()
    return paper


def make_document(paper_id, **overrides):
    values = {
        "paper_id": paper_id,
        "filename": "paper.pdf",
        "mime_type": "application/pdf",
        "file_size": 1024,
        "checksum": VALID_CHECKSUM,
        "storage_key": "papers/abc/paper.pdf",
    }
    values.update(overrides)
    return Document(**values)


def raw_row(paper_id, **overrides):
    """A row for Core inserts, which bypass the ORM validators."""
    values = {
        "paper_id": paper_id,
        "filename": "paper.pdf",
        "mime_type": "application/pdf",
        "file_size": 1024,
        "checksum": VALID_CHECKSUM,
        "storage_key": "papers/abc/paper.pdf",
    }
    values.update(overrides)
    return values


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
def paper(session):
    return _create_paper(session)


def _count(session, model):
    return session.execute(select(func.count()).select_from(model)).scalar_one()


# 1. UUID primary key
def test_primary_key_is_uuid(session, paper):
    document = make_document(paper.id)
    session.add(document)
    session.flush()
    assert isinstance(document.id, uuid.UUID)
    assert Document.__table__.c.id.primary_key


def test_each_document_gets_a_distinct_id(session, paper):
    first, second = make_document(paper.id), make_document(paper.id)
    session.add_all([first, second])
    session.flush()
    assert first.id != second.id


# 2. Timestamps
def test_timestamps_are_set(session, paper):
    document = make_document(paper.id)
    session.add(document)
    session.flush()
    assert document.created_at is not None
    assert document.updated_at is not None


# 3. Required paper
def test_paper_id_is_required(session):
    assert Document.__table__.c.paper_id.nullable is False
    session.add(
        Document(
            filename="a.pdf",
            mime_type="application/pdf",
            file_size=1,
            checksum=VALID_CHECKSUM,
            storage_key="k",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


# 4. Filename
def test_filename_is_stored_unchanged(session, paper):
    document = make_document(paper.id, filename="My Paper (v2).pdf")
    session.add(document)
    session.flush()
    assert document.filename == "My Paper (v2).pdf"


def test_filename_length_is_bounded(paper):
    limit = Document.__table__.c.filename.type.length
    assert limit == 255
    make_document(paper.id, filename="f" * limit)
    with pytest.raises(ValueError):
        make_document(paper.id, filename="f" * (limit + 1))


# 5. MIME type
@pytest.mark.parametrize(
    "mime", ["application/pdf", "text/plain", "image/png", "application/x-custom"]
)
def test_mime_type_is_not_restricted_to_pdf(session, paper, mime):
    document = make_document(paper.id, mime_type=mime)
    session.add(document)
    session.flush()
    assert document.mime_type == mime


def test_mime_type_is_required_and_bounded(paper):
    column = Document.__table__.c.mime_type
    assert column.nullable is False
    assert isinstance(column.type, String) and column.type.length == 255
    with pytest.raises(ValueError):
        make_document(paper.id, mime_type="m" * 256)


# 6. File size
def test_file_size_is_stored(session, paper):
    document = make_document(paper.id, file_size=5 * 1024**3)
    session.add(document)
    session.commit()
    session.expire_all()
    assert session.get(Document, document.id).file_size == 5 * 1024**3
    assert isinstance(Document.__table__.c.file_size.type, BigInteger)


def test_zero_file_size_is_allowed(session, paper):
    document = make_document(paper.id, file_size=0)
    session.add(document)
    session.flush()
    assert document.file_size == 0


@pytest.mark.parametrize("bad", ["10", 1.5, None, True])
def test_file_size_must_be_an_int(paper, bad):
    with pytest.raises(TypeError):
        make_document(paper.id, file_size=bad)


# 7. Checksum
def test_checksum_is_stored(session, paper):
    document = make_document(paper.id)
    session.add(document)
    session.flush()
    assert document.checksum == VALID_CHECKSUM
    column = Document.__table__.c.checksum
    assert column.type.length == 64 and column.nullable is False


def test_uppercase_hex_checksum_is_accepted(paper):
    make_document(paper.id, checksum="ABCDEF0123456789" * 4)


# 8. Storage key
def test_storage_key_is_stored_and_bounded(session, paper):
    document = make_document(paper.id, storage_key="a/b/c/" + "k" * 100)
    session.add(document)
    session.flush()
    assert document.storage_key.startswith("a/b/c/")
    limit = Document.__table__.c.storage_key.type.length
    assert limit >= 512
    with pytest.raises(ValueError):
        make_document(paper.id, storage_key="k" * (limit + 1))


# 9. Paper foreign key
def test_paper_foreign_key_definition():
    (fk,) = Document.__table__.c.paper_id.foreign_keys
    assert fk.target_fullname == "papers.id"
    assert fk.ondelete == "CASCADE"


def test_foreign_key_rejects_unknown_paper(session):
    session.add(make_document(uuid.uuid4()))
    with pytest.raises(IntegrityError):
        session.flush()


# 10. paper_id index
def test_paper_id_is_indexed():
    indexed = [
        [c.name for c in index.columns] for index in Document.__table__.indexes
    ]
    assert ["paper_id"] in indexed


# 11. Relationship
def test_paper_documents_relationship(session, paper):
    document = Document(
        paper=paper,
        filename="a.pdf",
        mime_type="application/pdf",
        file_size=1,
        checksum=VALID_CHECKSUM,
        storage_key="k",
    )
    session.add(document)
    session.flush()
    assert document.paper is paper
    assert document in paper.documents
    assert document.paper_id == paper.id


def test_document_deletion_is_left_to_the_database():
    relationship = Paper.__mapper__.relationships["documents"]
    assert relationship.passive_deletes == "all"
    assert not relationship.cascade.delete
    assert not relationship.cascade.delete_orphan


# 12. Multiple documents per paper
def test_multiple_documents_per_paper(session, paper):
    session.add_all(
        [make_document(paper.id, filename=f"v{i}.pdf", storage_key=f"k{i}") for i in range(3)]
    )
    session.flush()
    session.refresh(paper)
    assert len(paper.documents) == 3
    assert {d.filename for d in paper.documents} == {"v0.pdf", "v1.pdf", "v2.pdf"}


# 13-16. Validation (model level)
@pytest.mark.parametrize("size", [-1, -1024])
def test_negative_file_size_is_rejected(paper, size):
    with pytest.raises(ValueError, match="non-negative"):
        make_document(paper.id, file_size=size)


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n", "  \t  "])
def test_blank_filename_is_rejected(paper, blank):
    with pytest.raises(ValueError, match="filename"):
        make_document(paper.id, filename=blank)


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_mime_type_is_rejected(paper, blank):
    with pytest.raises(ValueError, match="mime_type"):
        make_document(paper.id, mime_type=blank)


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_blank_storage_key_is_rejected(paper, blank):
    with pytest.raises(ValueError, match="storage_key"):
        make_document(paper.id, storage_key=blank)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "abc",
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "z" * 64,
        " " + "a" * 63,
        "a" * 63 + "\n",
    ],
)
def test_invalid_checksum_is_rejected(paper, bad):
    with pytest.raises(ValueError, match="checksum"):
        make_document(paper.id, checksum=bad)


def test_checksum_must_be_a_string(paper):
    with pytest.raises(TypeError):
        make_document(paper.id, checksum=123)


# 13-16. Validation (database level, bypassing validators)
@pytest.mark.parametrize(
    "overrides",
    [
        {"file_size": -1},
        {"checksum": "abc"},
        {"checksum": "a" * 65},
        {"filename": "   "},
        {"mime_type": ""},
        {"storage_key": "  "},
    ],
)
def test_database_constraints_reject_invalid_rows(session, paper, overrides):
    with pytest.raises(IntegrityError):
        session.execute(
            insert(Document.__table__).values(**raw_row(paper.id, **overrides))
        )


def test_database_accepts_valid_raw_row(session, paper):
    session.execute(insert(Document.__table__).values(**raw_row(paper.id)))
    session.flush()
    assert _count(session, Document) == 1


# 17. PostgreSQL DDL
def test_postgresql_ddl_compiles():
    dialect = postgresql.dialect()
    ddl = str(CreateTable(Document.__table__).compile(dialect=dialect))
    assert "CREATE TABLE documents" in ddl
    assert "REFERENCES papers (id) ON DELETE CASCADE" in ddl
    assert "file_size >= 0" in ddl
    assert "length(checksum) = 64" in ddl
    assert "length(trim(filename)) > 0" in ddl
    for index in Document.__table__.indexes:
        assert "CREATE INDEX" in str(CreateIndex(index).compile(dialect=dialect))


# 18. Cascade deletion
def test_deleting_paper_cascades_to_documents(session, paper):
    session.add_all([make_document(paper.id, storage_key=f"k{i}") for i in range(2)])
    session.commit()
    assert _count(session, Document) == 2
    session.delete(paper)
    session.commit()
    session.expire_all()
    assert _count(session, Document) == 0


def test_deleting_collection_cascades_through_papers_to_documents(session, paper):
    session.add(make_document(paper.id))
    session.commit()
    session.delete(paper.collection)
    session.commit()
    session.expire_all()
    assert _count(session, Paper) == 0
    assert _count(session, Document) == 0


def test_deleting_one_paper_keeps_other_documents(session):
    first = _create_paper(session, "One")
    second = _create_paper(session, "Two")
    session.add_all([make_document(first.id), make_document(second.id)])
    session.commit()
    session.delete(first)
    session.commit()
    session.expire_all()
    remaining = session.scalars(select(Document)).all()
    assert [d.paper_id for d in remaining] == [second.id]


# repr
def test_repr_exposes_only_the_id(paper):
    document = make_document(
        paper.id, filename="secret-name.pdf", storage_key="secret/key"
    )
    document.id = uuid.uuid4()
    text = repr(document)
    assert text == f"Document(id={document.id!r})"
    for hidden in ("secret-name.pdf", "secret/key", VALID_CHECKSUM):
        assert hidden not in text


# 19-21. Import safety
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

    import nexora.database.models.document  # noqa: F401

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

    for name, value in vars(document_module).items():
        assert not isinstance(value, (Engine, OrmSession, sessionmaker)), name


def test_source_has_no_environment_filesystem_or_network_access():
    source = inspect.getsource(document_module)
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
