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
from sqlalchemy.schema import CreateIndex, CreateTable

import nexora.database.models as models_package
import nexora.database.models.collection as collection_module
from nexora.database.models import Base, Collection, TimestampMixin, UUIDPrimaryKeyMixin, User, Workspace

PG = postgresql.dialect()
OWNED = {"users", "workspaces", "collections"}


@pytest.fixture
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)

    @event.listens_for(eng, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng, tables=[t for t in Base.metadata.sorted_tables if t.name in OWNED])
    yield eng
    eng.dispose()


def count(session, model):
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def seed(session, n_workspaces=1, n_collections=2, email="o@example.org"):
    user = User(email=email)
    for i in range(n_workspaces):
        workspace = Workspace(name=f"w{i}")
        workspace.collections.extend(Collection(name=f"c{i}-{j}") for j in range(n_collections))
        user.workspaces.append(workspace)
    session.add(user)
    session.commit()
    return user


def test_table_name_mixins_and_columns():
    assert Collection.__tablename__ == "collections"
    assert issubclass(Collection, UUIDPrimaryKeyMixin) and issubclass(Collection, TimestampMixin)
    assert set(Collection.__table__.c.keys()) == {"id", "workspace_id", "name", "description", "created_at", "updated_at"}
    table = Collection.__table__
    assert table.c.id.primary_key and isinstance(table.c.id.type, Uuid)
    assert all(isinstance(table.c[n].type, DateTime) and table.c[n].type.timezone for n in ("created_at", "updated_at"))


def test_foreign_key_to_workspaces_with_database_cascade():
    column = Collection.__table__.c.workspace_id
    assert column.nullable is False and isinstance(column.type, Uuid)
    (fk,) = column.foreign_keys
    assert fk.target_fullname == "workspaces.id" and fk.ondelete == "CASCADE"


def test_name_and_description_schema_and_index():
    table = Collection.__table__
    assert table.c.name.nullable is False and isinstance(table.c.name.type, String) and table.c.name.type.length == 255
    assert table.c.description.nullable is True and table.c.description.type.length == 2000
    (index,) = table.indexes
    assert index.name == "ix_collections_workspace_id" and [c.name for c in index.columns] == ["workspace_id"]


def test_postgresql_ddl():
    ddl = str(CreateTable(Collection.__table__).compile(dialect=PG))
    assert "workspace_id UUID NOT NULL" in ddl and "name VARCHAR(255) NOT NULL" in ddl
    assert "CONSTRAINT pk_collections PRIMARY KEY (id)" in ddl
    assert ("CONSTRAINT fk_collections_workspace_id_workspaces FOREIGN KEY(workspace_id) "
            "REFERENCES workspaces (id) ON DELETE CASCADE") in ddl
    index_ddl = str(CreateIndex(next(iter(Collection.__table__.indexes))).compile(dialect=PG))
    assert "CREATE INDEX ix_collections_workspace_id ON collections (workspace_id)" in index_ddl


def test_postgresql_ddl_contains_name_not_blank_check():
    ddl = str(CreateTable(Collection.__table__).compile(dialect=PG))
    assert "CONSTRAINT ck_collections_name_not_blank CHECK (length(trim(name)) > 0)" in ddl


@pytest.mark.parametrize("blank", ["", " ", "     "])
def test_blank_name_rejected(engine, blank):
    with Session(engine) as session:
        user = seed(session, n_collections=0)
        session.add(Collection(workspace_id=user.workspaces[0].id, name=blank))
        with pytest.raises(IntegrityError):
            session.flush()


@pytest.mark.parametrize("name", ["Transformers", " padded ", "a"])
def test_valid_name_succeeds_and_is_stored_unchanged(engine, name):
    with Session(engine) as session:
        user = seed(session, n_collections=0)
        session.add(Collection(workspace_id=user.workspaces[0].id, name=name))
        session.commit()
        assert session.execute(select(Collection.name)).scalar_one() == name


def test_relationship_declaration():
    rel = orm_inspect(Collection).relationships["workspace"]
    assert rel.mapper.class_ is Workspace and rel.direction is RelationshipDirection.MANYTOONE
    assert rel.back_populates == "collections"


def test_collection_workspace_navigation_and_optional_description(engine):
    with Session(engine) as session:
        user = seed(session, n_workspaces=1, n_collections=1)
        workspace = user.workspaces[0]
        collection = workspace.collections[0]
        assert collection.workspace is workspace and collection.workspace_id == workspace.id
        assert collection.description is None and isinstance(collection.id, uuid.UUID)


def test_required_columns_enforced(engine):
    with Session(engine) as session:
        user = seed(session, n_collections=0)
        session.add(Collection(workspace_id=user.workspaces[0].id))
        with pytest.raises(IntegrityError):
            session.flush()
    with Session(engine) as session:
        session.add(Collection(name="no workspace"))
        with pytest.raises(IntegrityError):
            session.flush()
    with Session(engine) as session:
        session.add(Collection(workspace_id=uuid.uuid4(), name="dangling"))
        with pytest.raises(IntegrityError):
            session.flush()


def test_multiple_collections_per_workspace_and_workspaces_per_user(engine):
    with Session(engine) as session:
        user = seed(session, n_workspaces=2, n_collections=3)
        assert len(user.workspaces) == 2 and all(len(w.collections) == 3 for w in user.workspaces)
        assert count(session, Collection) == 6


@pytest.mark.parametrize("load_children_first", [False, True])
def test_deleting_workspace_deletes_its_collections_only(engine, load_children_first):
    with Session(engine) as session:
        user = seed(session, n_workspaces=2, n_collections=2)
        doomed, kept = user.workspaces
        if load_children_first:
            assert len(doomed.collections) == 2
        session.delete(doomed)
        session.commit()
        assert count(session, Workspace) == 1 and count(session, Collection) == 2
        assert {c.workspace_id for c in session.execute(select(Collection)).scalars()} == {kept.id}


@pytest.mark.parametrize("load_children_first", [False, True])
def test_deleting_user_cascades_through_workspaces_to_collections(engine, load_children_first):
    with Session(engine) as session:
        user = seed(session, n_workspaces=2, n_collections=2)
        other = seed(session, n_workspaces=1, n_collections=1, email="other@example.org")
        if load_children_first:
            _ = [len(w.collections) for w in user.workspaces]
        session.delete(user)
        session.commit()
        assert count(session, User) == 1 and count(session, Workspace) == 1 and count(session, Collection) == 1
        assert session.get(User, other.id) is not None


def test_models_register_on_base_metadata_and_exports():
    assert OWNED <= set(Base.metadata.tables)
    for name in ("User", "Workspace", "Collection"):
        assert name in models_package.__all__ and hasattr(models_package, name)


def test_module_is_side_effect_free():
    for value in vars(collection_module).values():
        assert not isinstance(value, (Engine, Session, sessionmaker))
    source = inspect.getsource(collection_module)
    for forbidden in ("os.environ", "getenv", "create_engine", "create_all", ".connect("):
        assert forbidden not in source

    # 12C-3: Collection now owns Papers.
    assert hasattr(models_package, "Paper")
    assert "papers" in Collection.__mapper__.relationships

    papers = Collection.__mapper__.relationships["papers"]
    assert papers.passive_deletes == "all"
    assert not papers.cascade.delete
    assert not papers.cascade.delete_orphan
