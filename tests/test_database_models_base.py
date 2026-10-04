import ast
import inspect
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, MetaData, String, Uuid, create_engine, inspect as sa_inspect, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateIndex, CreateTable

import nexora.database.models.base as base_module
from nexora.database.models import NAMING_CONVENTION, Base, TimestampMixin, UUIDPrimaryKeyMixin, utc_now

PG = postgresql.dialect()


class Widget(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "test_widgets"
    name: Mapped[str] = mapped_column(String(50), unique=True)
    last_seen: Mapped[datetime | None]


class Gadget(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "test_gadgets"
    __table_args__ = (CheckConstraint("length(name) > 0", name="name_not_empty"),)
    name: Mapped[str] = mapped_column(String(50))
    widget_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("test_widgets.id"), index=True)


@pytest.fixture(scope="module", autouse=True)
def _remove_test_tables_from_shared_metadata():
    yield
    Base.metadata.remove(Gadget.__table__)
    Base.metadata.remove(Widget.__table__)


@pytest.fixture
def engine():
    # In-memory SQLite exists only in tests; it is never part of the production code.
    eng = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    Base.metadata.create_all(eng, tables=[Widget.__table__, Gadget.__table__])
    yield eng
    eng.dispose()


def ddl(table):
    return str(CreateTable(table).compile(dialect=PG))


# ---- base, metadata, naming ----

def test_base_importable_and_is_declarative_base():
    assert issubclass(Base, DeclarativeBase)
    assert isinstance(Base.metadata, MetaData)
    assert not hasattr(Base, "__table__")  # Base itself maps no table


def test_naming_convention_is_complete_and_deterministic():
    assert NAMING_CONVENTION == {
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
    assert dict(Base.metadata.naming_convention) == NAMING_CONVENTION


def test_constraint_and_index_names_follow_convention_in_postgresql_ddl():
    widgets, gadgets = ddl(Widget.__table__), ddl(Gadget.__table__)
    assert "CONSTRAINT pk_test_widgets PRIMARY KEY (id)" in widgets
    assert "CONSTRAINT uq_test_widgets_name UNIQUE (name)" in widgets
    assert "CONSTRAINT pk_test_gadgets PRIMARY KEY (id)" in gadgets
    assert "CONSTRAINT ck_test_gadgets_name_not_empty CHECK" in gadgets
    assert "CONSTRAINT fk_test_gadgets_widget_id_test_widgets FOREIGN KEY(widget_id)" in gadgets
    index_ddl = [str(CreateIndex(i).compile(dialect=PG)) for i in Gadget.__table__.indexes]
    assert any("CREATE INDEX ix_test_gadgets_widget_id ON test_gadgets (widget_id)" in s for s in index_ddl)


def test_same_convention_gives_identical_names_on_a_fresh_metadata():
    from sqlalchemy import Column, Table

    fresh = MetaData(naming_convention=NAMING_CONVENTION)
    table = Table("test_widgets", fresh, Column("id", Uuid, primary_key=True), Column("name", String(50), unique=True))
    text = str(CreateTable(table).compile(dialect=PG))
    assert "pk_test_widgets" in text and "uq_test_widgets_name" in text


# ---- declarative mapping and typing ----

def test_models_inherit_base_and_register_tables():
    assert issubclass(Widget, Base) and issubclass(Gadget, Base)
    assert Widget.__table__ in Base.metadata.tables.values()
    assert {"id", "name", "last_seen", "created_at", "updated_at"} <= set(Widget.__table__.c.keys())


def test_mapped_annotations_resolve_to_expected_column_types():
    c = Widget.__table__.c
    assert c.name.nullable is False and c.name.type.length == 50
    assert isinstance(c.last_seen.type, DateTime) and c.last_seen.type.timezone is True
    assert c.last_seen.nullable is True  # Mapped[datetime | None]


# ---- UUID strategy ----

def test_uuid_primary_key_definition_is_postgresql_native():
    id_col = Widget.__table__.c.id
    assert id_col.primary_key and isinstance(id_col.type, Uuid) and id_col.type.as_uuid is True
    assert "id UUID NOT NULL" in ddl(Widget.__table__)
    assert isinstance(Gadget.__table__.c.widget_id.type, Uuid)
    assert "widget_id UUID NOT NULL" in ddl(Gadget.__table__)


def test_uuid_generated_on_flush_and_round_trips(engine):
    with Session(engine) as session:
        first, second = Widget(name="a"), Widget(name="b")
        assert first.id is None
        session.add_all([first, second])
        session.flush()
        assert isinstance(first.id, uuid.UUID) and first.id.version == 4
        assert first.id != second.id
        session.commit()
        first_id = first.id
    with Session(engine) as session:
        loaded = session.get(Widget, first_id)
        assert loaded is not None and loaded.id == first_id and isinstance(loaded.id, uuid.UUID)


def test_explicit_uuid_is_respected(engine):
    explicit = uuid.uuid4()
    with Session(engine) as session:
        session.add(Widget(id=explicit, name="explicit"))
        session.flush()
        assert session.get(Widget, explicit) is not None


# ---- timestamp strategy ----

def test_timestamp_columns_are_timezone_aware_with_server_defaults():
    for name in ("created_at", "updated_at"):
        col = Widget.__table__.c[name]
        assert isinstance(col.type, DateTime) and col.type.timezone is True
        assert col.nullable is False and col.server_default is not None
    text = ddl(Widget.__table__)
    assert "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL" in text
    assert "updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL" in text
    assert "created_at" not in Gadget.__table__.c  # timestamps are opt-in


def test_utc_now_is_aware_utc():
    now = utc_now()
    assert now.tzinfo is not None and now.utcoffset() == timedelta(0)
    assert abs(datetime.now(timezone.utc) - now) < timedelta(seconds=5)


def test_timestamps_set_on_insert_and_updated_at_moves_on_update(engine):
    # SQLite does not preserve time zones for DateTime(timezone=True), so values read back
    # from it are naive. Timezone-awareness is verified through the column type (see the
    # timestamp-column test above) and utc_now(); here every comparison uses values that
    # were all read back from SQLite, and no aware/naive mix is ever ordered.
    past = datetime(2000, 1, 1)
    table = Widget.__table__
    with Session(engine) as session:
        widget = Widget(name="w")
        session.add(widget)
        session.flush()
        assert widget.created_at is not None
        assert widget.updated_at is not None

        # Pin updated_at to a known old value (explicit values override onupdate).
        session.execute(update(table).where(table.c.id == widget.id).values(updated_at=past))
        session.refresh(widget)
        assert widget.updated_at == past
        created_before = widget.created_at

        widget.name = "renamed"
        session.flush()
        session.refresh(widget)

        assert widget.created_at == created_before  # unchanged by the update
        assert widget.updated_at != past
        assert widget.updated_at > past  # refreshed by onupdate


# ---- safety: no side effects, no globals, no env ----

def test_nothing_is_created_automatically():
    eng = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    assert sa_inspect(eng).get_table_names() == []
    eng.dispose()


def test_module_has_no_engine_session_or_global_state():
    for value in vars(base_module).values():
        assert not isinstance(value, (Engine, Session, sessionmaker))
    tree = ast.parse(inspect.getsource(base_module))
    assigned = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assigned |= {n.target.id for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    assert assigned == {"NAMING_CONVENTION"}
    source = inspect.getsource(base_module)
    for forbidden in ("os.environ", "getenv", "create_engine", "create_all", ".connect(", "print("):
        assert forbidden not in source


def test_fresh_interpreter_import_is_safe_offline_and_creates_no_tables():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NEXORA", "DATABASE", "PG"))}
    code = (
        f"import sys; sys.path[:0] = {sys.path!r}\n"
        "import gc\n"
        "import nexora.database.models as m\n"
        "from sqlalchemy.engine import Engine\n"
        "print(len(m.Base.metadata.tables))\n"
        "print(sum(isinstance(o, Engine) for o in gc.get_objects()))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["0", "0"]


# ---- public exports ----

def test_public_exports():
    import nexora.database as database
    import nexora.database.models as models

    assert models.__all__ == ["NAMING_CONVENTION", "Base", "TimestampMixin", "UUIDPrimaryKeyMixin", "utc_now"]
    assert database.Base is Base and "Base" in database.__all__
    assert models.UUIDPrimaryKeyMixin is UUIDPrimaryKeyMixin and models.TimestampMixin is TimestampMixin
