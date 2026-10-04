"""Tests for the DocumentChunkRecord ORM model (milestone 12C-4).

ADAPT POINT: ``_create_document`` builds the User -> Workspace ->
Collection -> Paper -> Document chain. Adjust the constructor arguments there
to match the real models; nothing else in this file depends on those details.
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
from sqlalchemy import Integer, String, create_engine, event, func, insert, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex, CreateTable

import nexora.database.models as models_package
from nexora.database.models import (
    Collection,
    Document,
    DocumentChunkRecord,
    Paper,
    User,
    Workspace,
)
from nexora.database.models import document_chunk as chunk_module
from nexora.database.models.base import Base
from nexora.document.models import DocumentChunk

TEXT_LIMIT = 100_000


def _create_document(session: Session, filename: str = "paper.pdf") -> Document:
    user = User(email=f"{uuid.uuid4()}@example.com")
    session.add(user)
    session.flush()
    workspace = Workspace(name="Workspace", user_id=user.id)
    session.add(workspace)
    session.flush()
    collection = Collection(name="Collection", workspace_id=workspace.id)
    session.add(collection)
    session.flush()
    paper = Paper(collection=collection, title="Paper")
    session.add(paper)
    session.flush()
    document = Document(
        paper_id=paper.id,
        filename=filename,
        mime_type="application/pdf",
        file_size=1024,
        checksum="0123456789abcdef" * 4,
        storage_key=f"papers/{uuid.uuid4()}/{filename}",
    )
    session.add(document)
    session.flush()
    return document


def make_chunk(document_id, chunk_index=0, text="chunk text"):
    return DocumentChunkRecord(
        document_id=document_id, chunk_index=chunk_index, text=text
    )


def raw_row(document_id, **overrides):
    """A row for Core inserts, which bypass the ORM validators."""
    values = {"document_id": document_id, "chunk_index": 0, "text": "chunk text"}
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
def document(session):
    return _create_document(session)


def _count(session, model):
    return session.execute(select(func.count()).select_from(model)).scalar_one()


# 1-2. Import and metadata registration
def test_model_is_exported_from_models_package():
    assert models_package.DocumentChunkRecord is DocumentChunkRecord
    assert DocumentChunkRecord.__module__ == "nexora.database.models.document_chunk"


def test_table_is_registered_in_metadata():
    assert DocumentChunkRecord.__tablename__ == "document_chunks"
    assert "document_chunks" in Base.metadata.tables
    assert {
        "users",
        "workspaces",
        "collections",
        "papers",
        "documents",
        "document_chunks",
    } <= set(Base.metadata.tables)


def test_record_is_distinct_from_the_processing_dataclass():
    assert DocumentChunkRecord is not DocumentChunk
    assert DocumentChunkRecord.__name__ != DocumentChunk.__name__


def test_schema_has_exactly_the_expected_columns():
    assert set(DocumentChunkRecord.__table__.c.keys()) == {
        "id",
        "document_id",
        "chunk_index",
        "text",
        "created_at",
        "updated_at",
    }


def test_no_embedding_is_stored_in_the_database():
    for name in DocumentChunkRecord.__table__.c.keys():
        assert "embed" not in name.lower()
        assert "vector" not in name.lower()


# 3. UUID primary key
def test_primary_key_is_uuid(session, document):
    chunk = make_chunk(document.id)
    session.add(chunk)
    session.flush()
    assert isinstance(chunk.id, uuid.UUID)
    assert DocumentChunkRecord.__table__.c.id.primary_key


def test_each_chunk_gets_a_distinct_id(session, document):
    first, second = make_chunk(document.id, 0), make_chunk(document.id, 1)
    session.add_all([first, second])
    session.flush()
    assert first.id != second.id


def test_timestamps_are_set(session, document):
    chunk = make_chunk(document.id)
    session.add(chunk)
    session.flush()
    assert chunk.created_at is not None
    assert chunk.updated_at is not None


# 4-5. document_id foreign key and cascade definition
def test_document_foreign_key_definition():
    column = DocumentChunkRecord.__table__.c.document_id
    (fk,) = column.foreign_keys
    assert fk.target_fullname == "documents.id"
    assert fk.ondelete == "CASCADE"
    assert column.nullable is False


def test_document_id_is_required(session):
    session.add(DocumentChunkRecord(chunk_index=0, text="orphan"))
    with pytest.raises(IntegrityError):
        session.flush()


def test_foreign_key_rejects_unknown_document(session):
    session.add(make_chunk(uuid.uuid4()))
    with pytest.raises(IntegrityError):
        session.flush()


# 6. Index on document_id
def test_document_id_is_indexed():
    indexed = [
        [c.name for c in index.columns]
        for index in DocumentChunkRecord.__table__.indexes
    ]
    assert ["document_id"] in indexed


# 7-8. chunk_index accepts zero and positive values
@pytest.mark.parametrize("value", [0, 1, 7, 10_000])
def test_valid_chunk_index_is_accepted(session, document, value):
    chunk = make_chunk(document.id, value)
    session.add(chunk)
    session.flush()
    assert chunk.chunk_index == value


def test_chunk_index_column_is_a_required_integer():
    column = DocumentChunkRecord.__table__.c.chunk_index
    assert isinstance(column.type, Integer)
    assert column.nullable is False


# 9-10. Invalid chunk_index
@pytest.mark.parametrize("value", [-1, -100])
def test_negative_chunk_index_is_rejected(document, value):
    with pytest.raises(ValueError, match="non-negative"):
        make_chunk(document.id, value)


@pytest.mark.parametrize("value", ["0", 1.5, None, True, False, [0]])
def test_non_integer_chunk_index_is_rejected(document, value):
    with pytest.raises(TypeError, match="chunk_index"):
        make_chunk(document.id, value)


# 11. Blank text
@pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\n", " \t\n "])
def test_blank_text_is_rejected(document, blank):
    with pytest.raises(ValueError, match="blank"):
        make_chunk(document.id, text=blank)


@pytest.mark.parametrize("bad", [None, 5, b"bytes", ["a"]])
def test_text_must_be_a_string(document, bad):
    with pytest.raises(TypeError, match="text"):
        make_chunk(document.id, text=bad)


def test_text_is_stored_unchanged(session, document):
    text = "  Leading and trailing  \nmultiline \u00dcnic\u00f6de  "
    chunk = make_chunk(document.id, text=text)
    session.add(chunk)
    session.flush()
    assert chunk.text == text


# 12. Text length boundary
def test_text_at_maximum_length_is_accepted(session, document):
    chunk = make_chunk(document.id, text="x" * TEXT_LIMIT)
    session.add(chunk)
    session.flush()
    assert len(chunk.text) == TEXT_LIMIT


def test_text_over_maximum_length_is_rejected(document):
    with pytest.raises(ValueError, match=str(TEXT_LIMIT)):
        make_chunk(document.id, text="x" * (TEXT_LIMIT + 1))


def test_text_column_is_a_bounded_required_string():
    column = DocumentChunkRecord.__table__.c.text
    assert isinstance(column.type, String)
    assert column.type.length == TEXT_LIMIT
    assert column.nullable is False


def test_default_sized_chunks_fit_comfortably():
    # The document chunker's default chunk_size is 1000 characters.
    assert chunk_module.CHUNK_TEXT_MAX_LENGTH >= 50 * 1000


# 13. Duplicate (document_id, chunk_index) rejected by the database
def test_duplicate_chunk_index_in_a_document_is_rejected(session, document):
    session.add_all([make_chunk(document.id, 3, "a"), make_chunk(document.id, 3, "b")])
    with pytest.raises(IntegrityError):
        session.flush()


def test_duplicate_is_rejected_by_raw_insert(session, document):
    session.execute(insert(DocumentChunkRecord.__table__).values(**raw_row(document.id)))
    with pytest.raises(IntegrityError):
        session.execute(
            insert(DocumentChunkRecord.__table__).values(**raw_row(document.id))
        )


def test_identical_text_at_different_indexes_is_allowed(session, document):
    session.add_all(
        [make_chunk(document.id, 0, "same"), make_chunk(document.id, 1, "same")]
    )
    session.flush()
    assert _count(session, DocumentChunkRecord) == 2


def test_unique_constraint_is_named_and_covers_both_columns():
    uniques = [
        c
        for c in DocumentChunkRecord.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    ]
    assert len(uniques) == 1
    assert [col.name for col in uniques[0].columns] == ["document_id", "chunk_index"]
    assert "document_id_chunk_index" in str(uniques[0].name)


# 14. Different documents may share chunk_index 0
def test_different_documents_may_both_have_chunk_index_zero(session):
    first = _create_document(session, "a.pdf")
    second = _create_document(session, "b.pdf")
    session.add_all([make_chunk(first.id, 0, "A"), make_chunk(second.id, 0, "B")])
    session.flush()
    assert _count(session, DocumentChunkRecord) == 2


# 15-16. Multiple chunks, relationship and ordering
def test_multiple_chunks_for_one_document(session, document):
    session.add_all([make_chunk(document.id, i, f"chunk {i}") for i in range(4)])
    session.flush()
    session.refresh(document)
    assert len(document.chunks) == 4


def test_relationship_both_directions(session, document):
    chunk = DocumentChunkRecord(document=document, chunk_index=0, text="hello")
    session.add(chunk)
    session.flush()
    assert chunk.document is document
    assert chunk in document.chunks
    assert chunk.document_id == document.id


def test_document_chunks_are_ordered_by_chunk_index(session, document):
    for index in (2, 0, 3, 1):
        session.add(make_chunk(document.id, index, f"chunk {index}"))
    session.commit()
    session.refresh(document)
    assert [c.chunk_index for c in document.chunks] == [0, 1, 2, 3]
    assert [c.text for c in document.chunks] == [f"chunk {i}" for i in range(4)]


def test_chunks_belong_to_their_own_document(session):
    first = _create_document(session, "a.pdf")
    second = _create_document(session, "b.pdf")
    session.add_all([make_chunk(first.id, 0, "A0"), make_chunk(second.id, 0, "B0")])
    session.commit()
    session.refresh(first)
    session.refresh(second)
    assert [c.text for c in first.chunks] == ["A0"]
    assert [c.text for c in second.chunks] == ["B0"]


def test_chunk_deletion_is_left_to_the_database():
    relationship = Document.__mapper__.relationships["chunks"]
    assert relationship.passive_deletes == "all"
    assert not relationship.cascade.delete
    assert not relationship.cascade.delete_orphan


# Database-level constraints (bypassing the ORM validators)
@pytest.mark.parametrize(
    "overrides",
    [{"chunk_index": -1}, {"text": ""}, {"text": "   "}, {"text": "\n\t"}],
)
def test_database_constraints_reject_invalid_rows(session, document, overrides):
    with pytest.raises(IntegrityError):
        session.execute(
            insert(DocumentChunkRecord.__table__).values(
                **raw_row(document.id, **overrides)
            )
        )


def test_database_accepts_valid_raw_row(session, document):
    session.execute(insert(DocumentChunkRecord.__table__).values(**raw_row(document.id)))
    session.flush()
    assert _count(session, DocumentChunkRecord) == 1


# PostgreSQL DDL
def test_postgresql_ddl_compiles():
    dialect = postgresql.dialect()
    ddl = str(CreateTable(DocumentChunkRecord.__table__).compile(dialect=dialect))
    assert "CREATE TABLE document_chunks" in ddl
    assert "REFERENCES documents (id) ON DELETE CASCADE" in ddl
    assert "chunk_index >= 0" in ddl
    assert "length(trim(text)) > 0" in ddl
    assert "UNIQUE (document_id, chunk_index)" in ddl
    for name in ("chunk_index_non_negative", "text_not_blank", "document_id_chunk_index"):
        assert name in ddl
    for index in DocumentChunkRecord.__table__.indexes:
        assert "CREATE INDEX" in str(CreateIndex(index).compile(dialect=dialect))


# 17. Cascade deletion
def test_deleting_document_removes_its_chunks(session, document):
    session.add_all([make_chunk(document.id, i) for i in range(3)])
    session.commit()
    assert _count(session, DocumentChunkRecord) == 3
    session.delete(document)
    session.commit()
    session.expire_all()
    assert _count(session, DocumentChunkRecord) == 0


def test_deleting_one_document_keeps_other_documents_chunks(session):
    first = _create_document(session, "a.pdf")
    second = _create_document(session, "b.pdf")
    session.add_all([make_chunk(first.id, 0, "A"), make_chunk(second.id, 0, "B")])
    session.commit()
    session.delete(first)
    session.commit()
    session.expire_all()
    remaining = session.scalars(select(DocumentChunkRecord)).all()
    assert [c.text for c in remaining] == ["B"]


def test_deleting_paper_cascades_through_documents_to_chunks(session, document):
    session.add(make_chunk(document.id))
    session.commit()
    paper = session.get(Paper, document.paper_id)
    session.delete(paper)
    session.commit()
    session.expire_all()
    assert _count(session, Document) == 0
    assert _count(session, DocumentChunkRecord) == 0


def test_deleting_collection_cascades_down_to_chunks(session, document):
    session.add(make_chunk(document.id))
    session.commit()
    paper = session.get(Paper, document.paper_id)
    collection = session.get(Collection, paper.collection_id)
    session.delete(collection)
    session.commit()
    session.expire_all()
    assert _count(session, Paper) == 0
    assert _count(session, DocumentChunkRecord) == 0


# repr
def test_repr_exposes_only_the_id(document):
    chunk = make_chunk(document.id, 0, "Secret chunk text")
    chunk.id = uuid.uuid4()
    text = repr(chunk)
    assert text == f"DocumentChunkRecord(id={chunk.id!r})"
    assert "Secret chunk text" not in text


# Import safety
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

    import nexora.database.models.document_chunk  # noqa: F401

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

    for name, value in vars(chunk_module).items():
        assert not isinstance(value, (Engine, OrmSession, sessionmaker)), name


def test_source_has_no_environment_filesystem_or_network_access():
    source = inspect.getsource(chunk_module)
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
