import inspect
import uuid

import pytest
from sqlalchemy import DateTime, ForeignKey, String, Uuid, create_engine, event, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import RelationshipDirection, Session, sessionmaker
from sqlalchemy.orm import inspect as orm_inspect
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateIndex, CreateTable

import nexora.database.models.workspace as workspace_module
from nexora.database.models import Base, Collection, TimestampMixin, UUIDPrimaryKeyMixin, User, Workspace

PG = postgresql.dialect()


@pytest.fixture
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)

    @event.listens_for(eng, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng, tables=[t for t in Base.metadata.sorted_tables if t.name in {"users", "workspaces", "collections"}])
    yield eng
    eng.dispose()


def count(session, model):
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def test_sqlite_foreign_key_enforcement_is_actually_on(engine):
    with Session(engine) as session:
        assert session.connection().exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        session.add(Workspace(user_id=uuid.uuid4(), name="orphan"))
        with pytest.raises(IntegrityError):
            session.flush()


def test_table_name_mixins_and_columns():
    assert Workspace.__tablename__ == "workspaces"
    assert issubclass(Workspace, UUIDPrimaryKeyMixin) and issubclass(Workspace, TimestampMixin)
    assert set(Workspace.__table__.c.keys()) == {"id", "user_id", "name", "description", "created_at", "updated_at"}
    table = Workspace.__table__
    assert table.c.id.primary_key and isinstance(table.c.id.type, Uuid)
    assert all(isinstance(table.c[n].type, DateTime) and table.c[n].type.timezone for n in ("created_at", "updated_at"))


def test_foreign_key_to_users_with_database_cascade():
    column = Workspace.__table__.c.user_id
    assert column.nullable is False and isinstance(column.type, Uuid)
    (fk,) = column.foreign_keys
    assert fk.target_fullname == "users.id" and fk.ondelete == "CASCADE"


def test_name_and_description_schema():
    table = Workspace.__table__
    assert table.c.name.nullable is False and isinstance(table.c.name.type, String) and table.c.name.type.length == 255
    assert table.c.description.nullable is True and table.c.description.type.length == 2000


def test_index_on_user_id():
    (index,) = Workspace.__table__.indexes
    assert index.name == "ix_workspaces_user_id" and [c.name for c in index.columns] == ["user_id"]


def test_postgresql_ddl():
    ddl = str(CreateTable(Workspace.__table__).compile(dialect=PG))
    assert "user_id UUID NOT NULL" in ddl and "name VARCHAR(255) NOT NULL" in ddl
    assert "description VARCHAR(2000)" in ddl and "description VARCHAR(2000) NOT NULL" not in ddl
    assert "CONSTRAINT pk_workspaces PRIMARY KEY (id)" in ddl
    assert "CONSTRAINT fk_workspaces_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE" in ddl
    index_ddl = str(CreateIndex(next(iter(Workspace.__table__.indexes))).compile(dialect=PG))
    assert "CREATE INDEX ix_workspaces_user_id ON workspaces (user_id)" in index_ddl


def test_postgresql_ddl_contains_name_not_blank_check():
    ddl = str(CreateTable(Workspace.__table__).compile(dialect=PG))
    assert "CONSTRAINT ck_workspaces_name_not_blank CHECK (length(trim(name)) > 0)" in ddl


@pytest.mark.parametrize("blank", ["", " ", "     "])
def test_blank_name_rejected(engine, blank):
    with Session(engine) as session:
        user = User(email="blank@example.org")
        session.add(user)
        session.commit()
        session.add(Workspace(user_id=user.id, name=blank))
        with pytest.raises(IntegrityError):
            session.flush()


@pytest.mark.parametrize("name", ["Research", " padded ", "a"])
def test_valid_name_succeeds_and_is_stored_unchanged(engine, name):
    with Session(engine) as session:
        user = User(email="valid@example.org")
        session.add(Workspace(user=user, name=name))
        session.commit()
        assert session.execute(select(Workspace.name)).scalar_one() == name


def test_relationship_declarations():
    mapper = orm_inspect(Workspace)
    user_rel, collections_rel = mapper.relationships["user"], mapper.relationships["collections"]
    assert user_rel.mapper.class_ is User and user_rel.direction is RelationshipDirection.MANYTOONE
    assert user_rel.back_populates == "workspaces"
    assert collections_rel.mapper.class_ is Collection and collections_rel.uselist is True
    assert collections_rel.back_populates == "workspace" and collections_rel.passive_deletes == "all"
    assert not collections_rel.cascade.delete and not collections_rel.cascade.delete_orphan


def test_workspace_user_navigation_and_optional_description(engine):
    with Session(engine) as session:
        user = User(email="owner@example.org")
        workspace = Workspace(name="Climate papers", user=user)
        session.add(workspace)
        session.commit()
        assert workspace.user is user and workspace.description is None
        assert workspace.user_id == user.id and workspace in user.workspaces
        assert isinstance(workspace.id, uuid.UUID)


def test_required_columns_enforced(engine):
    with Session(engine) as session:
        user = User(email="o@example.org")
        session.add(user)
        session.commit()
        session.add(Workspace(user_id=user.id))  # no name
        with pytest.raises(IntegrityError):
            session.flush()
    with Session(engine) as session:
        session.add(Workspace(name="no owner"))  # no user_id
        with pytest.raises(IntegrityError):
            session.flush()


def test_multiple_workspaces_per_user(engine):
    with Session(engine) as session:
        user = User(email="many@example.org")
        session.add_all([Workspace(name=f"w{i}", user=user) for i in range(3)])
        session.commit()
        assert count(session, Workspace) == 3 and len(user.workspaces) == 3


@pytest.mark.parametrize("load_children_first", [False, True])
def test_deleting_user_deletes_workspaces_in_the_database(engine, load_children_first):
    with Session(engine) as session:
        user = User(email="gone@example.org")
        user.workspaces.extend([Workspace(name="a"), Workspace(name="b")])
        keeper = User(email="stays@example.org")
        keeper.workspaces.append(Workspace(name="kept"))
        session.add_all([user, keeper])
        session.commit()
        if load_children_first:
            assert len(user.workspaces) == 2  # ORM has the children loaded; DB still does the delete
        session.delete(user)
        session.commit()
        assert count(session, User) == 1
        assert [w.name for w in session.execute(select(Workspace)).scalars()] == ["kept"]


def test_module_is_side_effect_free():
    for value in vars(workspace_module).values():
        assert not isinstance(value, (Engine, Session, sessionmaker))
    source = inspect.getsource(workspace_module)
    for forbidden in ("os.environ", "getenv", "create_engine", "create_all", ".connect("):
        assert forbidden not in source
