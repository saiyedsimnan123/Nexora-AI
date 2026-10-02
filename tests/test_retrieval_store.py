"""Contract and architecture tests for the VectorStore interface (Step 11B)."""

import ast
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nexora.retrieval.models import SearchResult, VectorRecord
from nexora.retrieval.store import VectorStore

REQUIRED_METHODS = {"upsert", "delete", "search", "count"}

FORBIDDEN_IMPORTS = {
    # Infrastructure and third-party services
    "qdrant_client",
    "openai",
    "psycopg",
    "psycopg2",
    "sqlalchemy",
    "fastapi",
    "redis",
    "requests",
    "httpx",
    "dotenv",
    # Standard-library modules that would imply network, filesystem,
    # database or environment access
    "os",
    "pathlib",
    "shutil",
    "socket",
    "sqlite3",
    "subprocess",
    "urllib",
    "http",
}

CONFIG_ENV_VARS = {
    "OPENAI_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "DATABASE_URL",
}


def _store_source_path() -> Path:
    spec = importlib.util.find_spec("nexora.retrieval.store")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin)


def _imported_modules() -> list[str]:
    """Return the fully qualified module names imported by store.py."""
    tree = ast.parse(_store_source_path().read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not allowed in store.py"
            assert node.module is not None
            modules.append(node.module)
    return modules


class FakeVectorStore:
    """Minimal structural implementation used only to exercise the protocol."""

    def upsert(self, records):
        raise NotImplementedError

    def delete(self, ids):
        raise NotImplementedError

    def search(self, vector, *, limit=10):
        raise NotImplementedError

    def count(self):
        raise NotImplementedError


class IncompleteVectorStore:
    """Missing ``count`` so it must not satisfy the protocol."""

    def upsert(self, records):
        raise NotImplementedError

    def delete(self, ids):
        raise NotImplementedError

    def search(self, vector, *, limit=10):
        raise NotImplementedError


# --- Protocol properties ---------------------------------------------------


def test_vector_store_is_importable():
    from nexora.retrieval.store import VectorStore as Imported

    assert Imported is VectorStore


def test_vector_store_is_a_protocol():
    assert getattr(VectorStore, "_is_protocol", False) is True
    with pytest.raises(TypeError):
        VectorStore()  # protocols cannot be instantiated


def test_vector_store_is_runtime_checkable():
    # A non-runtime-checkable protocol raises TypeError on isinstance().
    assert isinstance(FakeVectorStore(), VectorStore)
    assert not isinstance(object(), VectorStore)


# --- Required methods and signatures --------------------------------------


def test_protocol_defines_exactly_the_required_methods():
    public_callables = {
        name
        for name, member in vars(VectorStore).items()
        if callable(member) and not name.startswith("_")
    }
    assert public_callables == REQUIRED_METHODS


def test_upsert_signature():
    sig = inspect.signature(VectorStore.upsert)
    assert list(sig.parameters) == ["self", "records"]
    assert sig.parameters["records"].annotation == list[VectorRecord]
    assert sig.return_annotation is None


def test_delete_signature():
    sig = inspect.signature(VectorStore.delete)
    assert list(sig.parameters) == ["self", "ids"]
    assert sig.parameters["ids"].annotation == list[str]
    assert sig.return_annotation is None


def test_search_signature():
    sig = inspect.signature(VectorStore.search)
    assert list(sig.parameters) == ["self", "vector", "limit"]
    assert sig.parameters["vector"].annotation == list[float]
    limit = sig.parameters["limit"]
    assert limit.kind is inspect.Parameter.KEYWORD_ONLY
    assert limit.annotation is int
    assert limit.default == 10
    assert sig.return_annotation is SearchResult


def test_count_signature():
    sig = inspect.signature(VectorStore.count)
    assert list(sig.parameters) == ["self"]
    assert sig.return_annotation is int


# --- Structural compatibility ---------------------------------------------


def test_complete_fake_satisfies_protocol():
    assert isinstance(FakeVectorStore(), VectorStore)


@pytest.mark.parametrize("missing", sorted(REQUIRED_METHODS))
def test_implementation_missing_a_method_does_not_satisfy_protocol(missing):
    namespace = {
        name: (lambda self, *args, **kwargs: None)
        for name in REQUIRED_METHODS - {missing}
    }
    partial_cls = type("PartialStore", (), namespace)
    assert not isinstance(partial_cls(), VectorStore)


def test_incomplete_fake_does_not_satisfy_protocol():
    assert not isinstance(IncompleteVectorStore(), VectorStore)


# --- Architectural constraints --------------------------------------------


def test_store_module_defines_only_the_protocol():
    tree = ast.parse(_store_source_path().read_text(encoding="utf-8"))
    class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    function_names = [
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert class_names == ["VectorStore"]
    assert function_names == []


def test_store_has_no_async_methods():
    assert not any(
        inspect.iscoroutinefunction(member) for member in vars(VectorStore).values()
    )


def test_store_does_not_import_forbidden_packages():
    top_level = {name.split(".")[0] for name in _imported_modules()}
    assert top_level.isdisjoint(FORBIDDEN_IMPORTS), top_level & FORBIDDEN_IMPORTS


def test_store_imports_only_stdlib_and_retrieval_models():
    for name in _imported_modules():
        top_level = name.split(".")[0]
        if top_level == "nexora":
            assert name == "nexora.retrieval.models", name
        else:
            assert top_level in sys.stdlib_module_names, name


# --- Environment isolation -------------------------------------------------


def test_importing_store_requires_no_environment_configuration():
    env = {k: v for k, v in os.environ.items() if k not in CONFIG_ENV_VARS}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    result = subprocess.run(
        [sys.executable, "-c", "import nexora.retrieval.store"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
