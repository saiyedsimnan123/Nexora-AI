import dataclasses
import inspect

import pytest

import nexora.database.config as config_module
from nexora.database.config import DatabaseConfig, DatabaseConfigurationError

PASSWORD = "placeholder-password"
URL = f"postgresql://nexora:{PASSWORD}@localhost:5432/nexora"


def test_defaults():
    c = DatabaseConfig(url=URL)
    assert (c.pool_size, c.max_overflow, c.pool_timeout, c.pool_recycle, c.echo) == (5, 10, 30.0, 1800, False)


def test_explicit_values_and_boundaries():
    c = DatabaseConfig(url=URL, pool_size=1, max_overflow=0, pool_timeout=0.5, pool_recycle=-1, echo=True)
    assert (c.pool_size, c.max_overflow, c.pool_timeout, c.pool_recycle, c.echo) == (1, 0, 0.5, -1, True)
    assert DatabaseConfig(url=URL, pool_timeout=10).pool_timeout == 10


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DatabaseConfig(url=URL).pool_size = 9


@pytest.mark.parametrize("url", [
    "postgresql://localhost/db", "postgresql+psycopg://u:p@host:5432/db",
    "postgresql+psycopg2://u@host/db", "postgresql+asyncpg://u:p@host/db",
    "postgresql:///db", "  postgresql://localhost/db  ",
])
def test_valid_urls(url):
    assert DatabaseConfig(url=url).url == url


@pytest.mark.parametrize("url", [
    "", "   ", None, 5, ["postgresql://x"], "localhost/db", "mysql://u:p@host/db",
    "sqlite:///file.db", "postgres://u:p@host/db", "postgresql", "postgresql+://u@h/db",
    "postgresqlx://u@h/db", f"postgresql://u:{PASSWORD}@host:notaport/db",
])
def test_invalid_urls(url):
    with pytest.raises(DatabaseConfigurationError) as info:
        DatabaseConfig(url=url)
    assert PASSWORD not in str(info.value)


@pytest.mark.parametrize("kw", [
    {"pool_size": 0}, {"pool_size": -1}, {"pool_size": True}, {"pool_size": 1.5}, {"pool_size": "5"},
    {"max_overflow": -1}, {"max_overflow": True}, {"max_overflow": 2.5}, {"max_overflow": None},
    {"pool_timeout": 0}, {"pool_timeout": -1.0}, {"pool_timeout": True}, {"pool_timeout": "30"},
    {"pool_timeout": float("nan")}, {"pool_timeout": float("inf")},
    {"pool_recycle": -2}, {"pool_recycle": True}, {"pool_recycle": 1.5},
    {"echo": 1}, {"echo": "true"}, {"echo": None},
])
def test_invalid_numeric_and_bool_fields(kw):
    with pytest.raises(DatabaseConfigurationError) as info:
        DatabaseConfig(url=URL, **kw)
    assert PASSWORD not in str(info.value)


def test_error_is_a_value_error():
    assert issubclass(DatabaseConfigurationError, ValueError)


def test_repr_and_str_never_show_password():
    c = DatabaseConfig(url=URL)
    for text in (repr(c), str(c)):
        assert PASSWORD not in text and "nexora" in text and "***" in text


def test_redacted_url():
    assert DatabaseConfig(url=URL).redacted_url() == "postgresql://nexora:***@localhost:5432/nexora"
    assert DatabaseConfig(url="postgresql://localhost/db").redacted_url() == "postgresql://localhost/db"
    assert DatabaseConfig(url="postgresql://user@host/db").redacted_url() == "postgresql://user@host/db"


def test_from_env_url_only_uses_defaults():
    c = DatabaseConfig.from_env({"NEXORA_DATABASE_URL": URL})
    assert c == DatabaseConfig(url=URL)


def test_from_env_all_variables():
    env = {"NEXORA_DATABASE_URL": f"  {URL}  ", "NEXORA_DATABASE_POOL_SIZE": " 8 ",
           "NEXORA_DATABASE_MAX_OVERFLOW": "0", "NEXORA_DATABASE_POOL_TIMEOUT": "12.5",
           "NEXORA_DATABASE_POOL_RECYCLE": "-1", "NEXORA_DATABASE_ECHO": "Yes"}
    c = DatabaseConfig.from_env(env)
    assert c.url == URL
    assert (c.pool_size, c.max_overflow, c.pool_timeout, c.pool_recycle, c.echo) == (8, 0, 12.5, -1, True)


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("False", False), ("0", False), ("no", False), ("OFF", False),
])
def test_echo_boolean_parsing(raw, expected):
    assert DatabaseConfig.from_env({"NEXORA_DATABASE_URL": URL, "NEXORA_DATABASE_ECHO": raw}).echo is expected


@pytest.mark.parametrize("env", [{}, {"NEXORA_DATABASE_URL": ""}, {"NEXORA_DATABASE_URL": "   "}])
def test_missing_url_raises(env):
    with pytest.raises(DatabaseConfigurationError):
        DatabaseConfig.from_env(env)


@pytest.mark.parametrize("var,value", [
    ("NEXORA_DATABASE_POOL_SIZE", "abc"), ("NEXORA_DATABASE_POOL_SIZE", "2.5"),
    ("NEXORA_DATABASE_POOL_SIZE", "0"), ("NEXORA_DATABASE_MAX_OVERFLOW", "-1"),
    ("NEXORA_DATABASE_MAX_OVERFLOW", "many"), ("NEXORA_DATABASE_POOL_TIMEOUT", "soon"),
    ("NEXORA_DATABASE_POOL_TIMEOUT", "0"), ("NEXORA_DATABASE_POOL_TIMEOUT", "nan"),
    ("NEXORA_DATABASE_POOL_RECYCLE", "-5"), ("NEXORA_DATABASE_POOL_RECYCLE", "x"),
    ("NEXORA_DATABASE_ECHO", "maybe"), ("NEXORA_DATABASE_ECHO", "2"),
])
def test_invalid_env_values_raise_without_leaking_url(var, value):
    with pytest.raises(DatabaseConfigurationError) as info:
        DatabaseConfig.from_env({"NEXORA_DATABASE_URL": URL, var: value})
    assert PASSWORD not in str(info.value) and URL not in str(info.value)


def test_invalid_url_from_env_does_not_leak_it():
    bad = f"mysql://u:{PASSWORD}@host/db"
    with pytest.raises(DatabaseConfigurationError) as info:
        DatabaseConfig.from_env({"NEXORA_DATABASE_URL": bad})
    assert PASSWORD not in str(info.value)


def test_blank_optional_variables_use_defaults():
    env = {"NEXORA_DATABASE_URL": URL, "NEXORA_DATABASE_POOL_SIZE": "  ", "NEXORA_DATABASE_ECHO": ""}
    c = DatabaseConfig.from_env(env)
    assert c.pool_size == 5 and c.echo is False


def test_from_env_uses_supplied_mapping_not_process_environment(monkeypatch):
    monkeypatch.setenv("NEXORA_DATABASE_URL", URL)

    with pytest.raises(DatabaseConfigurationError):
        DatabaseConfig.from_env({})


def test_from_env_prefers_supplied_mapping_over_process_environment(monkeypatch):
    monkeypatch.setenv("NEXORA_DATABASE_URL", "postgresql://process-env-host/db")
    monkeypatch.setenv("NEXORA_DATABASE_POOL_SIZE", "99")

    config = DatabaseConfig.from_env({"NEXORA_DATABASE_URL": URL})

    assert config.url == URL
    assert config.pool_size == 5


def test_from_env_reads_process_environment_when_no_mapping_given(monkeypatch):
    monkeypatch.setenv("NEXORA_DATABASE_URL", URL)
    monkeypatch.setenv("NEXORA_DATABASE_POOL_SIZE", "7")

    config = DatabaseConfig.from_env()

    assert config.url == URL and config.pool_size == 7


def test_package_exports():
    import nexora.database as pkg

    assert pkg.DatabaseConfig is DatabaseConfig
    assert pkg.DatabaseConfigurationError is DatabaseConfigurationError


def test_module_has_no_driver_imports_or_connection_code():
    import ast

    tree = ast.parse(inspect.getsource(config_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"sqlalchemy", "psycopg", "psycopg2", "asyncpg", "dotenv"}
    source = inspect.getsource(config_module)
    assert "create_engine" not in source and ".connect(" not in source
