import ast
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy.dialects import registry
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.engine import Engine

import nexora.database.connection as connection_module
from nexora.database import DatabaseConfig, DatabaseConfigurationError, create_database_engine

PASSWORD = "placeholder-password"
PG_URL = f"postgresql://nexora:{PASSWORD}@localhost:5432/nexora"
FAKE_URL = f"postgresql+nexorafake://nexora:{PASSWORD}@127.0.0.1:1/nexora"  # port 1: unreachable

CONNECT_CALLS = []


def _forbidden_connect(*args, **kwargs):
    CONNECT_CALLS.append((args, kwargs))
    raise AssertionError("a database connection was attempted")


class OfflineDialect(PGDialect):
    """PostgreSQL dialect with a fake DBAPI, so Engines can be built with no driver or server."""

    driver = "nexorafake"
    supports_statement_cache = True

    @classmethod
    def import_dbapi(cls):
        return SimpleNamespace(paramstyle="named", connect=_forbidden_connect, Error=Exception)


registry.register("postgresql.nexorafake", __name__, "OfflineDialect")


@pytest.fixture(autouse=True)
def _reset_connect_calls():
    CONNECT_CALLS.clear()


def make_engine(**kw):
    return create_database_engine(DatabaseConfig(url=FAKE_URL, **kw))


# ---- real Engine objects (fake dialect, never connected) ----

def test_creates_sqlalchemy_engine():
    engine = make_engine()
    try:
        assert isinstance(engine, Engine)
    finally:
        engine.dispose()


def test_engine_uses_configured_url_and_hides_password_in_str():
    engine = make_engine()
    try:
        assert engine.url.render_as_string(hide_password=False) == FAKE_URL
        assert PASSWORD not in str(engine.url) and PASSWORD not in repr(engine)
    finally:
        engine.dispose()


@pytest.mark.parametrize("echo", [False, True])
def test_echo_setting_applied(echo):
    engine = make_engine(echo=echo)
    try:
        assert bool(engine.echo) is echo
    finally:
        engine.dispose()


def test_pool_settings_applied_via_public_pool_api():
    engine = make_engine(pool_size=3, pool_timeout=12.5)
    try:
        assert engine.pool.size() == 3
        assert engine.pool.timeout() == 12.5
    finally:
        engine.dispose()


def test_creating_and_disposing_engine_never_connects():
    engine = make_engine()
    engine.dispose()
    engine.dispose()  # disposing needs no server and is repeatable
    assert CONNECT_CALLS == []


# ---- exact arguments passed to SQLAlchemy ----

def test_all_config_values_passed_to_create_engine():
    config = DatabaseConfig(url=PG_URL, pool_size=7, max_overflow=2, pool_timeout=9.5,
                            pool_recycle=-1, echo=True)
    with patch("nexora.database.connection.create_engine") as fake:
        result = create_database_engine(config)
    assert result is fake.return_value
    fake.assert_called_once_with(PG_URL, pool_size=7, max_overflow=2, pool_timeout=9.5,
                                 pool_recycle=-1, echo=True)


def test_defaults_passed_to_create_engine():
    with patch("nexora.database.connection.create_engine") as fake:
        create_database_engine(DatabaseConfig(url=PG_URL))
    fake.assert_called_once_with(PG_URL, pool_size=5, max_overflow=10, pool_timeout=30.0,
                                 pool_recycle=1800, echo=False)


def test_surrounding_whitespace_in_url_is_stripped():
    with patch("nexora.database.connection.create_engine") as fake:
        create_database_engine(DatabaseConfig(url=f"  {PG_URL}  "))
    assert fake.call_args[0][0] == PG_URL


@pytest.mark.parametrize("bad", [None, object(), PG_URL, {"url": PG_URL}, 5])
def test_invalid_argument_raises_without_calling_sqlalchemy(bad):
    with patch("nexora.database.connection.create_engine") as fake:
        with pytest.raises(DatabaseConfigurationError) as info:
            create_database_engine(bad)
    assert PASSWORD not in str(info.value)
    fake.assert_not_called()


def test_sqlalchemy_errors_are_not_wrapped():
    boom = RuntimeError("engine failure")
    with patch("nexora.database.connection.create_engine", side_effect=boom):
        with pytest.raises(RuntimeError) as info:
            create_database_engine(DatabaseConfig(url=PG_URL))
    assert info.value is boom


# ---- security, structure, public API ----

def test_no_password_in_logs_or_output(capsys, caplog):
    caplog.set_level("DEBUG")
    engine = make_engine()
    engine.dispose()
    captured = capsys.readouterr()
    assert PASSWORD not in captured.out + captured.err + caplog.text


def test_public_api_exports():
    import nexora.database as pkg

    assert pkg.create_database_engine is create_database_engine
    assert pkg.DatabaseConfig is DatabaseConfig
    assert pkg.DatabaseConfigurationError is DatabaseConfigurationError


def test_no_module_level_engine_or_import_time_calls():
    for value in vars(connection_module).values():
        assert not isinstance(value, Engine)
    tree = ast.parse(inspect.getsource(connection_module))
    for node in tree.body:  # module-level statements
        assert not isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)  # docstring
        )


def test_connection_module_imports_only_sqlalchemy_and_nexora_config():
    tree = ast.parse(inspect.getsource(connection_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"psycopg", "psycopg2", "asyncpg", "pg8000", "dotenv", "os"}
    source = inspect.getsource(connection_module)
    for forbidden in (".connect(", ".begin(", ".execute(", "create_all", "print("):
        assert forbidden not in source
