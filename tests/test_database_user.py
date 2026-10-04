import ast
import inspect
import uuid

import pytest
from sqlalchemy import DateTime, String, Uuid, create_engine, event, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import RelationshipDirection, Session, sessionmaker
from sqlalchemy import inspect as orm_inspect
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateTable, UniqueConstraint

import nexora.database.models.user as user_module
from nexora.database.models import Base, TimestampMixin, UUIDPrimaryKeyMixin, User, Workspace

PG = postgresql.dialect()
TABLES = [User.__table__, Workspace.__table__]


@pytest.fixture
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)

    @event.listens_for(eng, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):  # SQLite ignores FKs unless enabled
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng, tables=[t for t in Base.metadata.sorted_tables if t.name in {"users", "workspaces", "collections"}])
    yield eng
    eng.dispose()


def count(session, model):
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def test_table_name_and_mixins():
    assert User.__tablename__ == "users"
    assert issubclass(User, UUIDPrimaryKeyMixin) and issubclass(User, TimestampMixin) and issubclass(User, Base)


def test_columns_are_exactly_identity_and_timestamps_with_no_secrets():
    columns = set(User.__table__.c.keys())
    assert columns == {"id", "email", "created_at", "updated_at"}
    assert not any(word in name for name in columns for word in ("password", "secret", "token", "hash"))


def test_uuid_primary_key_and_timezone_aware_timestamps():
    table = User.__table__
    assert table.c.id.primary_key and isinstance(table.c.id.type, Uuid)
    for name in ("created_at", "updated_at"):
        assert isinstance(table.c[name].type, DateTime) and table.c[name].type.timezone is True


def test_email_is_required_unique_and_length_limited():
    email = User.__table__.c.email
    assert email.nullable is False
    assert isinstance(email.type, String) and email.type.length == 320
    uniques = [c for c in User.__table__.constraints if isinstance(c, UniqueConstraint)]
    assert [[col.name for col in c.columns] for c in uniques] == [["email"]]
    assert list(User.__table__.indexes) == []  # the unique constraint already provides the index


def test_postgresql_ddl():
    ddl = str(CreateTable(User.__table__).compile(dialect=PG))
    assert "id UUID NOT NULL" in ddl
    assert "email VARCHAR(320) NOT NULL" in ddl
    assert "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL" in ddl
    assert "CONSTRAINT pk_users PRIMARY KEY (id)" in ddl
    assert "CONSTRAINT uq_users_email UNIQUE (email)" in ddl


def test_relationship_declaration():
    rel = orm_inspect(User).relationships["workspaces"]
    assert rel.mapper.class_ is Workspace
    assert rel.direction is RelationshipDirection.ONETOMANY and rel.uselist is True
    assert rel.back_populates == "user"
    assert rel.passive_deletes == "all"
    assert not rel.cascade.delete and not rel.cascade.delete_orphan  # ownership delete is the database's job


def test_user_persists_with_generated_id_and_timestamps(engine):
    with Session(engine) as session:
        user = User(email="Researcher@Example.org")
        session.add(user)
        session.flush()
        assert isinstance(user.id, uuid.UUID)
        assert user.created_at is not None and user.updated_at is not None
        session.commit()
        assert session.get(User, user.id).email == "Researcher@Example.org"  # stored unchanged


def test_duplicate_email_rejected(engine):
    with Session(engine) as session:
        session.add(User(email="dup@example.org"))
        session.commit()
        session.add(User(email="dup@example.org"))
        with pytest.raises(IntegrityError):
            session.flush()


def test_missing_email_rejected(engine):
    with Session(engine) as session:
        session.add(User())
        with pytest.raises(IntegrityError):
            session.flush()


def test_user_to_workspaces_relationship(engine):
    with Session(engine) as session:
        user = User(email="owner@example.org")
        user.workspaces.extend([Workspace(name="one"), Workspace(name="two")])
        session.add(user)
        session.commit()
        assert sorted(w.name for w in user.workspaces) == ["one", "two"]
        assert all(w.user is user for w in user.workspaces)


def test_repr_does_not_expose_email_or_relationships():
    user = User(email="private@example.org")
    assert "private@example.org" not in repr(user) and "workspaces" not in repr(user)


def test_module_is_side_effect_free():
    for value in vars(user_module).values():
        assert not isinstance(value, (Engine, Session, sessionmaker))
    source = inspect.getsource(user_module)
    for forbidden in ("os.environ", "getenv", "create_engine", "create_all", ".connect("):
        assert forbidden not in source
    tree = ast.parse(source)
    assert not [n for n in tree.body if isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant)]
