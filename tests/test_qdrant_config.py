"""Tests for QdrantConfig (Step 11C): validation, env loading, security, isolation."""

import ast
import dataclasses
import importlib.util
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from nexora.retrieval.qdrant_config import (
    ENV_QDRANT_API_KEY,
    ENV_QDRANT_COLLECTION,
    ENV_QDRANT_TIMEOUT,
    ENV_QDRANT_URL,
    QdrantConfig,
)

SECRET = "test-secret-key"
URL = "http://localhost:6333"

ALL_ENV_VARS = (
    ENV_QDRANT_URL,
    ENV_QDRANT_API_KEY,
    ENV_QDRANT_COLLECTION,
    ENV_QDRANT_TIMEOUT,
)

FORBIDDEN_IMPORTS = (
    # SDKs and third-party infrastructure
    "qdrant_client",
    "openai",
    "fastapi",
    "psycopg",
    "psycopg2",
    "sqlalchemy",
    "redis",
    "requests",
    "httpx",
    # Standard-library network access (urllib.parse is allowed and used)
    "socket",
    "ssl",
    "http",
    "urllib.request",
    "urllib.error",
    "logging",
)


@pytest.fixture(autouse=True)
def _clean_qdrant_env(monkeypatch):
    """Ensure ambient NEXORA_QDRANT_* variables never leak into tests."""
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _source_path() -> Path:
    spec = importlib.util.find_spec("nexora.retrieval.qdrant_config")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin)


def _source_tree() -> ast.Module:
    return ast.parse(_source_path().read_text(encoding="utf-8"))


def _imported_names() -> list[str]:
    names: list[str] = []
    for node in ast.walk(_source_tree()):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None and node.level == 0
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


# --- Construction ----------------------------------------------------------


def test_valid_url_with_defaults():
    config = QdrantConfig(url=URL)
    assert config.url == URL
    assert config.api_key is None
    assert config.collection_name == "nexora"
    assert config.timeout == 30.0


def test_custom_values():
    config = QdrantConfig(
        url="https://example.qdrant.io",
        api_key=SECRET,
        collection_name="papers",
        timeout=60.5,
    )
    assert config.url == "https://example.qdrant.io"
    assert config.api_key == SECRET
    assert config.collection_name == "papers"
    assert config.timeout == 60.5


def test_config_is_immutable():
    config = QdrantConfig(url=URL, api_key=SECRET)
    for field_name, value in [
        ("url", "http://other:6333"),
        ("api_key", "other"),
        ("collection_name", "other"),
        ("timeout", 1.0),
    ]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(config, field_name, value)


# --- URL validation --------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:6333",
        "https://example.qdrant.io",
        "HTTPS://Example.qdrant.io",
        "http://127.0.0.1:6333",
        "http://[::1]:6333",
    ],
)
def test_valid_urls_accepted(url):
    assert QdrantConfig(url=url).url == url


@pytest.mark.parametrize("url", [None, 123, b"http://localhost:6333", ["http://localhost"]])
def test_non_string_url_rejected(url):
    with pytest.raises(TypeError):
        QdrantConfig(url=url)


@pytest.mark.parametrize("url", ["", "   "])
def test_empty_url_rejected(url):
    with pytest.raises(ValueError, match="empty"):
        QdrantConfig(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "localhost:6333",
        "not-a-url",
        "//localhost:6333",
        "ftp://example.com",
        "file:///etc/passwd",
        "http://",
        "https:///collections",
        "http://localhost:notaport",
        "http://localhost:99999",
        "http://[::1",
    ],
)
def test_invalid_urls_rejected(url):
    with pytest.raises(ValueError):
        QdrantConfig(url=url)


# --- API key validation ----------------------------------------------------


def test_api_key_none_accepted():
    assert QdrantConfig(url=URL, api_key=None).api_key is None


def test_api_key_string_accepted():
    assert QdrantConfig(url=URL, api_key=SECRET).api_key == SECRET


@pytest.mark.parametrize("api_key", ["", "   "])
def test_empty_api_key_rejected(api_key):
    with pytest.raises(ValueError):
        QdrantConfig(url=URL, api_key=api_key)


@pytest.mark.parametrize("api_key", [123, b"secret", ["secret"], True])
def test_non_string_api_key_rejected(api_key):
    with pytest.raises(TypeError):
        QdrantConfig(url=URL, api_key=api_key)


# --- Collection name validation -------------------------------------------


def test_collection_default():
    assert QdrantConfig(url=URL).collection_name == "nexora"


def test_collection_custom():
    assert QdrantConfig(url=URL, collection_name="my-papers_v2").collection_name == "my-papers_v2"


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_collection_rejected(name):
    with pytest.raises(ValueError):
        QdrantConfig(url=URL, collection_name=name)


@pytest.mark.parametrize("name", [None, 123, b"nexora"])
def test_non_string_collection_rejected(name):
    with pytest.raises(TypeError):
        QdrantConfig(url=URL, collection_name=name)


@pytest.mark.parametrize("name", ["a/b", "bad\nname", "bad\x00name"])
def test_unsafe_collection_characters_rejected(name):
    with pytest.raises(ValueError):
        QdrantConfig(url=URL, collection_name=name)


# --- Timeout validation ----------------------------------------------------


def test_timeout_default():
    assert QdrantConfig(url=URL).timeout == 30.0


@pytest.mark.parametrize("timeout", [1, 5, 30, 60.5, 0.001])
def test_positive_timeouts_accepted(timeout):
    assert QdrantConfig(url=URL, timeout=timeout).timeout == timeout


@pytest.mark.parametrize("timeout", [0, 0.0, -1, -0.5])
def test_non_positive_timeout_rejected(timeout):
    with pytest.raises(ValueError):
        QdrantConfig(url=URL, timeout=timeout)


@pytest.mark.parametrize("timeout", [True, False])
def test_bool_timeout_rejected(timeout):
    with pytest.raises(TypeError):
        QdrantConfig(url=URL, timeout=timeout)


@pytest.mark.parametrize("timeout", [math.nan, math.inf, -math.inf])
def test_non_finite_timeout_rejected(timeout):
    with pytest.raises(ValueError):
        QdrantConfig(url=URL, timeout=timeout)


@pytest.mark.parametrize("timeout", ["30", "abc", None, [30]])
def test_non_numeric_timeout_rejected(timeout):
    with pytest.raises(TypeError):
        QdrantConfig(url=URL, timeout=timeout)


# --- Environment loading ---------------------------------------------------


def test_from_env_complete(monkeypatch):
    monkeypatch.setenv(ENV_QDRANT_URL, "https://example.qdrant.io")
    monkeypatch.setenv(ENV_QDRANT_API_KEY, SECRET)
    monkeypatch.setenv(ENV_QDRANT_COLLECTION, "papers")
    monkeypatch.setenv(ENV_QDRANT_TIMEOUT, "12.5")

    assert QdrantConfig.from_env() == QdrantConfig(
        url="https://example.qdrant.io",
        api_key=SECRET,
        collection_name="papers",
        timeout=12.5,
    )


def test_from_env_only_url_uses_defaults(monkeypatch):
    monkeypatch.setenv(ENV_QDRANT_URL, URL)

    config = QdrantConfig.from_env()

    assert config.url == URL
    assert config.api_key is None
    assert config.collection_name == "nexora"
    assert config.timeout == 30.0


def test_from_env_integer_timeout_is_parsed_as_number(monkeypatch):
    monkeypatch.setenv(ENV_QDRANT_URL, URL)
    monkeypatch.setenv(ENV_QDRANT_TIMEOUT, "45")

    assert QdrantConfig.from_env().timeout == 45.0


def test_from_env_blank_optional_values_are_treated_as_absent(monkeypatch):
    monkeypatch.setenv(ENV_QDRANT_URL, URL)
    monkeypatch.setenv(ENV_QDRANT_API_KEY, "  ")
    monkeypatch.setenv(ENV_QDRANT_COLLECTION, "")
    monkeypatch.setenv(ENV_QDRANT_TIMEOUT, " ")

    config = QdrantConfig.from_env()

    assert config.api_key is None
    assert config.collection_name == "nexora"
    assert config.timeout == 30.0


def test_from_env_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv(ENV_QDRANT_URL, f"  {URL}\n")
    monkeypatch.setenv(ENV_QDRANT_API_KEY, f"{SECRET}\n")

    config = QdrantConfig.from_env()

    assert config.url == URL
    assert config.api_key == SECRET


def test_from_env_missing_url_raises_clear_error():
    with pytest.raises(ValueError, match=ENV_QDRANT_URL):
        QdrantConfig.from_env()


@pytest.mark.parametrize("value", ["", "   "])
def test_from_env_empty_url_raises_clear_error(monkeypatch, value):
    monkeypatch.setenv(ENV_QDRANT_URL, value)
    with pytest.raises(ValueError, match=ENV_QDRANT_URL):
        QdrantConfig.from_env()


@pytest.mark.parametrize("value", ["abc", "nan", "inf", "-inf", "0", "-5", "1,5"])
def test_from_env_invalid_timeout_raises_clear_error(monkeypatch, value):
    monkeypatch.setenv(ENV_QDRANT_URL, URL)
    monkeypatch.setenv(ENV_QDRANT_TIMEOUT, value)
    with pytest.raises(ValueError, match="(?i)timeout"):
        QdrantConfig.from_env()


def test_from_env_invalid_url_is_rejected(monkeypatch):
    monkeypatch.setenv(ENV_QDRANT_URL, "localhost:6333")
    with pytest.raises(ValueError):
        QdrantConfig.from_env()


def test_from_env_works_without_api_key(monkeypatch):
    monkeypatch.setenv(ENV_QDRANT_URL, URL)
    assert QdrantConfig.from_env().api_key is None


# --- Security --------------------------------------------------------------


def test_api_key_not_in_repr_or_str():
    config = QdrantConfig(url=URL, api_key=SECRET)
    assert SECRET not in repr(config)
    assert SECRET not in str(config)
    assert "api_key" not in repr(config) or "***" in repr(config)


def test_api_key_not_in_validation_errors():
    bad_inputs = [
        {"url": "not-a-url"},
        {"url": ""},
        {"url": URL, "collection_name": ""},
        {"url": URL, "collection_name": "a/b"},
        {"url": URL, "timeout": 0},
        {"url": URL, "timeout": True},
        {"url": URL, "timeout": "30"},
    ]
    for kwargs in bad_inputs:
        with pytest.raises((TypeError, ValueError)) as excinfo:
            QdrantConfig(api_key=SECRET, **kwargs)
        assert SECRET not in str(excinfo.value)
        assert SECRET not in repr(excinfo.value)


def test_api_key_not_in_from_env_errors(monkeypatch):
    monkeypatch.setenv(ENV_QDRANT_API_KEY, SECRET)

    with pytest.raises(ValueError) as missing_url:
        QdrantConfig.from_env()
    assert SECRET not in str(missing_url.value)

    monkeypatch.setenv(ENV_QDRANT_URL, URL)
    monkeypatch.setenv(ENV_QDRANT_TIMEOUT, "not-a-number")
    with pytest.raises(ValueError) as bad_timeout:
        QdrantConfig.from_env()
    assert SECRET not in str(bad_timeout.value)


def test_source_contains_no_hardcoded_credentials():
    tree = _source_tree()
    suspicious = re.compile(
        r"(sk-[A-Za-z0-9]{8,}|[A-Za-z0-9_\-]{32,}|https?://[^/\s:@]+:[^/\s@]+@)"
    )
    for node in ast.walk(tree):
        # Docstrings are exempt; every other string literal is inspected.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.strip().startswith(("Validated, offline", "Immutable")):
                continue
            assert not suspicious.search(node.value), node.value
    # The only permitted default for api_key is None.
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "api_key":
            assert node.value is not None
            assert "None" in ast.unparse(node.value)


def test_source_never_prints_or_logs():
    tree = _source_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "print"
    assert not any(n.split(".")[0] == "logging" for n in _imported_names())


# --- Dependency and environment isolation ---------------------------------


def test_module_does_not_import_forbidden_packages():
    for imported in _imported_names():
        for forbidden in FORBIDDEN_IMPORTS:
            assert not (
                imported == forbidden or imported.startswith(forbidden + ".")
            ), f"{imported} is not allowed in qdrant_config.py"


def test_module_imports_only_the_standard_library():
    for imported in _imported_names():
        assert imported.split(".")[0] in sys.stdlib_module_names, imported


def test_environment_is_read_only_inside_from_env():
    tree = _source_tree()
    allowed_function = "from_env"
    helper_function = "_read_env"

    def reads_env(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Attribute)
            and n.attr in {"environ", "getenv", "environb"}
            for n in ast.walk(node)
        )

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == helper_function:
            continue  # only called from from_env
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == allowed_function:
                    continue
                assert not reads_env(item), f"environment read in {item}"
            continue
        assert not reads_env(node), "environment read at import time"


def test_import_requires_no_configuration_and_creates_no_files(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ALL_ENV_VARS}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", "import nexora.retrieval.qdrant_config"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_construction_does_not_read_environment(monkeypatch):
    monkeypatch.setenv(ENV_QDRANT_URL, "http://from-env:6333")
    config = QdrantConfig(url=URL)
    assert config.url == URL
    assert config.collection_name == "nexora"
