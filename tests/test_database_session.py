import ast
import inspect

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, event, func, insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import nexora.database.session as session_module
from nexora.database import (
    DatabaseConfig,
    DatabaseConfigurationError,
    create_database_session,
    create_session_factory,
    database_transaction,
)

PASSWORD = "placeholder-password"
PG_URL = f"postgresql://nexora:{PASSWORD}@localhost:5432/nexora"

metadata = MetaData()
items = Table("items", metadata, Column("id", Integer, primary_key=True), Column("name", String))


class TrackingSession(Session):
    """Session that records lifecycle calls (offline test double)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def commit(self):
        self.calls.append("commit")
        super().commit()

    def rollback(self):
        self.calls.append("rollback")
        super().rollback()

    def close(self):
        self.calls.append("close")
        super().close()


@pytest.fixture
def engine():
    # In-memory SQLite is used only here, in tests. DatabaseConfig stays PostgreSQL-only.
    eng = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def tracking_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, class_=TrackingSession)


def row_count(engine):
    with create_session_factory(engine)() as s:
        return s.execute(select(func.count()).select_from(items)).scalar_one()


# ---- A. session factory ----

def test_valid_engine_creates_session_factory(engine):
    factory = create_session_factory(engine)
    assert isinstance(factory, sessionmaker)


@pytest.mark.parametrize("bad", [None, object(), "engine", 5, {"url": PG_URL}])
def test_invalid_engine_rejected(bad):
    with pytest.raises(DatabaseConfigurationError):
        create_session_factory(bad)


def test_factory_creates_independent_sessions_with_expected_settings(engine):
    factory = create_session_factory(engine)
    first, second = factory(), factory()
    try:
        assert first is not second
        assert first.expire_on_commit is False and second.expire_on_commit is False
        assert getattr(first, "autocommit", False) is False
        assert not first.in_transaction()
    finally:
        first.close()
        second.close()


def test_factory_and_session_creation_do_not_connect():
    eng = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    connects = []
    event.listen(eng, "connect", lambda *args: connects.append(args))
    factory = create_session_factory(eng)
    session = create_database_session(factory)
    session.close()
    assert connects == []
    eng.dispose()


# ---- B. session creation ----

def test_create_database_session_returns_new_sessions(engine):
    factory = create_session_factory(engine)
    first, second = create_database_session(factory), create_database_session(factory)
    try:
        assert isinstance(first, Session) and isinstance(second, Session)
        assert first is not second
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("bad", [None, object(), "factory", 5])
def test_invalid_factory_rejected_for_session_and_transaction(bad):
    with pytest.raises(DatabaseConfigurationError):
        create_database_session(bad)
    with pytest.raises(DatabaseConfigurationError):
        with database_transaction(bad):
            pytest.fail("block must not run")


# ---- C/F. successful transaction ----

def test_transaction_commits_and_closes(engine, tracking_factory):
    with database_transaction(tracking_factory) as session:
        session.execute(insert(items).values(name="kept"))
    assert session.calls == ["commit", "close"]
    assert not session.in_transaction()
    assert row_count(engine) == 1


# ---- D/E/F. rollback, propagation, closure ----

def test_transaction_rolls_back_reraises_original_and_closes(engine, tracking_factory):
    failure = RuntimeError("expected failure")
    captured = []
    with pytest.raises(RuntimeError) as info:
        with database_transaction(tracking_factory) as session:
            captured.append(session)
            session.execute(insert(items).values(name="discarded"))
            raise failure
    assert info.value is failure
    assert captured[0].calls == ["rollback", "close"]
    assert not captured[0].in_transaction()
    assert row_count(engine) == 0


@pytest.mark.parametrize("error", [ValueError("v"), KeyError("k"), ConnectionError("c")])
def test_original_exception_type_and_object_propagate(tracking_factory, error):
    with pytest.raises(type(error)) as info:
        with database_transaction(tracking_factory):
            raise error
    assert info.value is error


def test_failed_commit_rolls_back_and_closes(engine):
    class FailingCommit(TrackingSession):
        def commit(self):
            self.calls.append("commit")
            raise RuntimeError("commit failed")

    factory = sessionmaker(bind=engine, expire_on_commit=False, class_=FailingCommit)
    captured = []
    with pytest.raises(RuntimeError) as info:
        with database_transaction(factory) as session:
            captured.append(session)
    assert str(info.value) == "commit failed"
    assert captured[0].calls == ["commit", "rollback", "close"]


def test_each_transaction_uses_a_new_session(tracking_factory):
    seen = []
    for _ in range(2):
        with database_transaction(tracking_factory) as session:
            seen.append(session)
    assert seen[0] is not seen[1]


# ---- G. independence ----

def test_committed_work_is_visible_to_a_new_session(engine):
    factory = create_session_factory(engine)
    with database_transaction(factory) as session:
        session.execute(insert(items).values(name="a"))
    assert row_count(engine) == 1


# ---- H/I. import safety and global state ----

def test_module_has_no_module_level_engine_session_or_factory():
    for value in vars(session_module).values():
        assert not isinstance(value, (Engine, Session, sessionmaker))
    tree = ast.parse(inspect.getsource(session_module))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            pytest.fail("module-level assignment found")
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant)  # docstring only


def test_session_module_imports_no_driver_or_environment():
    tree = ast.parse(inspect.getsource(session_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"psycopg", "psycopg2", "asyncpg", "os", "dotenv"}
    source = inspect.getsource(session_module)
    for forbidden in (".connect(", "create_all", "create_engine", "print("):
        assert forbidden not in source


# ---- J. secret safety ----

@pytest.mark.parametrize("bad", [DatabaseConfig(url=PG_URL), PG_URL, {"url": PG_URL}])
def test_errors_do_not_contain_password(bad):
    for call in (create_session_factory, create_database_session):
        with pytest.raises(DatabaseConfigurationError) as info:
            call(bad)
        assert PASSWORD not in str(info.value)
    with pytest.raises(DatabaseConfigurationError) as info:
        with database_transaction(bad):
            pass
    assert PASSWORD not in str(info.value)


# ---- K. public API ----

def test_public_api_exports():
    import nexora.database as pkg

    for name in ("DatabaseConfig", "DatabaseConfigurationError", "create_database_engine",
                 "create_session_factory", "create_database_session", "database_transaction"):
        assert hasattr(pkg, name) and name in pkg.__all__
    assert pkg.database_transaction is database_transaction
