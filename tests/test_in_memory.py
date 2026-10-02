"""Tests for the in-memory VectorStore implementation."""

from future import annotations

import ast
import importlib.util
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.store import VectorStore

---------------------------------------------------------------------------

Helpers

---------------------------------------------------------------------------

def make_record(
record_id: str,
vector: list[float],
*,
document_id: str = "doc-1",
chunk_id: str | None = None,
text: str = "sample text",
metadata: dict[str, object] | None = None,
) -> VectorRecord:
payload: dict[str, object] = {
"document_id": document_id,
"chunk_id": chunk_id or record_id,
"text": text,
}

if metadata:
    payload.update(metadata)

return VectorRecord(
    id=record_id,
    vector=vector,
    payload=payload,
)

---------------------------------------------------------------------------

Construction and protocol

---------------------------------------------------------------------------

def test_empty_store_constructs() -> None:
store = InMemoryVectorStore()

assert store.count() == 0
assert store.dimension is None

def test_explicit_dimension() -> None:
store = InMemoryVectorStore(dimension=3)

assert store.dimension == 3
assert store.count() == 0

def test_runtime_protocol_compatibility() -> None:
store = InMemoryVectorStore()

assert isinstance(store, VectorStore)

def test_public_method_signatures_match_contract() -> None:
assert inspect.signature(InMemoryVectorStore.upsert) == inspect.Signature(
[
inspect.Parameter(
"self",
inspect.Parameter.POSITIONAL_OR_KEYWORD,
),
inspect.Parameter(
"records",
inspect.Parameter.POSITIONAL_OR_KEYWORD,
annotation="list[VectorRecord]",
),
],
return_annotation="None",
)

def test_search_signature_contains_keyword_only_limit() -> None:
signature = inspect.signature(InMemoryVectorStore.search)

assert list(signature.parameters) == ["self", "vector", "limit"]
assert signature.parameters["limit"].kind is inspect.Parameter.KEYWORD_ONLY

---------------------------------------------------------------------------

Constructor validation

---------------------------------------------------------------------------

@pytest.mark.parametrize("dimension", [0, -1, -5])
def test_constructor_rejects_non_positive_dimension(dimension: int) -> None:
with pytest.raises(ValueError):
InMemoryVectorStore(dimension=dimension)

@pytest.mark.parametrize("dimension", [True, False, 1.5, "3", []])
def test_constructor_rejects_invalid_dimension(dimension: object) -> None:
with pytest.raises(TypeError):
InMemoryVectorStore(dimension=dimension)  # type: ignore[arg-type]

---------------------------------------------------------------------------

Upsert

---------------------------------------------------------------------------

def test_upsert_single_record() -> None:
store = InMemoryVectorStore()

record = make_record("chunk-1", [1.0, 0.0])

store.upsert([record])

assert store.count() == 1
assert store.dimension == 2

def test_upsert_multiple_records() -> None:
store = InMemoryVectorStore()

store.upsert(
    [
        make_record("a", [1.0, 0.0]),
        make_record("b", [0.0, 1.0]),
    ]
)

assert store.count() == 2
assert store.dimension == 2

"""Tests for the in-memory VectorStore implementation."""

from future import annotations

import ast
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.store import VectorStore

def make_record(
record_id: str,
vector: list[float],
*,
document_id: str = "doc-1",
chunk_id: str | None = None,
text: str = "sample text",
metadata: dict[str, object] | None = None,
) -> VectorRecord:
payload: dict[str, object] = {
"document_id": document_id,
"chunk_id": chunk_id or record_id,
"text": text,
}

if metadata:
    payload.update(metadata)

return VectorRecord(
    id=record_id,
    vector=vector,
    payload=payload,
)

---------------------------------------------------------------------------

Construction

---------------------------------------------------------------------------

def test_empty_store_constructs() -> None:
store = InMemoryVectorStore()

assert store.count() == 0
assert store.dimension is None

def test_explicit_dimension() -> None:
store = InMemoryVectorStore(dimension=3)

assert store.dimension == 3
assert store.count() == 0

def test_runtime_protocol_compatibility() -> None:
assert isinstance(InMemoryVectorStore(), VectorStore)

@pytest.mark.parametrize("dimension", [0, -1, -5])
def test_constructor_rejects_non_positive_dimension(dimension: int) -> None:
with pytest.raises(ValueError):
InMemoryVectorStore(dimension=dimension)

@pytest.mark.parametrize("dimension", [True, False, 1.5, "3", []])
def test_constructor_rejects_invalid_dimension(dimension: object) -> None:
with pytest.raises(TypeError):
InMemoryVectorStore(dimension=dimension)  # type: ignore[arg-type]

---------------------------------------------------------------------------

Upsert

---------------------------------------------------------------------------

def test_upsert_single_record() -> None:
store = InMemoryVectorStore()

store.upsert([make_record("a", [1.0, 0.0])])

assert store.count() == 1
assert store.dimension == 2

def test_upsert_multiple_records() -> None:
store = InMemoryVectorStore()

store.upsert(
    [
        make_record("a", [1.0, 0.0]),
        make_record("b", [0.0, 1.0]),
    ]
)

assert store.count() == 2
assert store.dimension == 2

def test_empty_upsert_does_nothing() -> None:
store = InMemoryVectorStore()

store.upsert([])

assert store.count() == 0
assert store.dimension is None

def test_upsert_requires_list() -> None:
store = InMemoryVectorStore()

with pytest.raises(TypeError):
    store.upsert(("a",))  # type: ignore[arg-type]

def test_upsert_requires_vector_records() -> None:
store = InMemoryVectorStore()

with pytest.raises(TypeError):
    store.upsert([{"id": "bad"}])  # type: ignore[list-item]

def test_upsert_replaces_existing_id() -> None:
store = InMemoryVectorStore()

store.upsert([make_record("same", [1.0, 0.0], text="old")])
store.upsert([make_record("same", [0.0, 1.0], text="new")])

assert store.count() == 1

result = store.search([0.0, 1.0])

assert result.results[0].text == "new"
assert result.results[0].score == pytest.approx(1.0)

def test_duplicate_ids_last_wins() -> None:
store = InMemoryVectorStore()

store.upsert(
    [
        make_record("same", [1.0, 0.0], text="first"),
        make_record("same", [0.0, 1.0], text="second"),
    ]
)

result = store.search([0.0, 1.0])

assert store.count() == 1
assert result.results[0].text == "second"

def test_first_upsert_establishes_dimension() -> None:
store = InMemoryVectorStore()

store.upsert([make_record("a", [1.0, 2.0, 3.0])])

assert store.dimension == 3

def test_mismatched_dimension_rejected() -> None:
store = InMemoryVectorStore(dimension=2)

with pytest.raises(ValueError):
    store.upsert([make_record("bad", [1.0, 2.0, 3.0])])

assert store.count() == 0

def test_invalid_batch_does_not_partially_mutate_store() -> None:
store = InMemoryVectorStore(dimension=2)

store.upsert([make_record("existing", [1.0, 0.0])])

with pytest.raises(ValueError):
    store.upsert(
        [
            make_record("new", [0.0, 1.0]),
            make_record("bad", [1.0, 2.0, 3.0]),
        ]
    )

assert store.count() == 1

---------------------------------------------------------------------------

Delete

---------------------------------------------------------------------------

def test_delete_existing_record() -> None:
store = InMemoryVectorStore(dimension=2)

store.upsert(
    [
        make_record("a", [1.0, 0.0]),
        make_record("b", [0.0, 1.0]),
    ]
)

store.delete(["a"])

assert store.count() == 1

def test_delete_multiple_records() -> None:
store = InMemoryVectorStore(dimension=2)

store.upsert(
    [
        make_record("a", [1.0, 0.0]),
        make_record("b", [0.0, 1.0]),
        make_record("c", [1.0, 1.0]),
    ]
)

store.delete(["a", "c"])

assert store.count() == 1

def test_delete_missing_id_is_noop() -> None:
store = InMemoryVectorStore(dimension=2)

store.upsert([make_record("a", [1.0, 0.0])])
store.delete(["missing"])

assert store.count() == 1

def test_empty_delete_is_valid() -> None:
store = InMemoryVectorStore()

store.delete([])

assert store.count() == 0

def test_delete_requires_list() -> None:
store = InMemoryVectorStore()

with pytest.raises(TypeError):
    store.delete(("a",))  # type: ignore[arg-type]

@pytest.mark.parametrize("value", [1, True, None, [], {}])
def test_delete_rejects_invalid_ids(value: object) -> None:
store = InMemoryVectorStore()

with pytest.raises(TypeError):
    store.delete([value])  # type: ignore[list-item]

def test_delete_rejects_empty_id() -> None:
store = InMemoryVectorStore()

with pytest.raises(ValueError):
    store.delete([""])

def test_delete_all_preserves_dimension() -> None:
store = InMemoryVectorStore()

store.upsert([make_record("a", [1.0, 0.0])])
store.delete(["a"])

assert store.count() == 0
assert store.dimension == 2

---------------------------------------------------------------------------

Search

---------------------------------------------------------------------------

def test_empty_store_search_returns_empty_result() -> None:
store = InMemoryVectorStore()

result = store.search([1.0, 0.0])

assert isinstance(result, SearchResult)
assert result.query == ""
assert result.results == []

def test_exact_vector_scores_one() -> None:
store = InMemoryVectorStore()

store.upsert([make_record("a", [1.0, 0.0])])

result = store.search([1.0, 0.0])

assert result.results[0].score == pytest.approx(1.0)

def test_search_ranking() -> None:
store = InMemoryVectorStore()

store.upsert(
    [
        make_record("wrong", [0.0, 1.0]),
        make_record("best", [1.0, 0.0]),
        make_record("middle", [1.0, 1.0]),
    ]
)

result = store.search([1.0, 0.0])

assert [item.chunk_id for item in result.results] == [
    "best",
    "middle",
    "wrong",
]

assert result.results[0].score == pytest.approx(1.0)
assert result.results[1].score == pytest.approx(math.sqrt(0.5))
assert result.results[2].score == pytest.approx(0.0)

def test_search_respects_limit() -> None:
store = InMemoryVectorStore(dimension=2)

store.upsert(
    [
        make_record("a", [1.0, 0.0]),
        make_record("b", [0.9, 0.1]),
        make_record("c", [0.8, 0.2]),
    ]
)

result = store.search([1.0, 0.0], limit=2)

assert len(result.results) == 2
assert [item.chunk_id for item in result.results] == ["a", "b"]

@pytest.mark.parametrize("limit", [0, -1, -10])
def test_search_rejects_non_positive_limit(limit: int) -> None:
store = InMemoryVectorStore(dimension=2)

with pytest.raises(ValueError):
    store.search([1.0, 0.0], limit=limit)

@pytest.mark.parametrize("limit", [True, False, 1.5, "10", None])
def test_search_rejects_invalid_limit(limit: object) -> None:
store = InMemoryVectorStore(dimension=2)

with pytest.raises(TypeError):
    store.search([1.0, 0.0], limit=limit)  # type: ignore[arg-type]

def test_search_dimension_mismatch() -> None:
store = InMemoryVectorStore(dimension=3)

store.upsert([make_record("a", [1.0, 0.0, 0.0])])

with pytest.raises(ValueError):
    store.search([1.0, 0.0])

def test_zero_query_vector_returns_zero_scores() -> None:
store = InMemoryVectorStore(dimension=2)

store.upsert(
    [
        make_record("a", [1.0, 0.0]),
        make_record("b", [0.0, 1.0]),
    ]
)

result = store.search([0.0, 0.0])

assert all(item.score == pytest.approx(0.0) for item in result.results)

def test_zero_stored_vector_returns_zero_similarity() -> None:
store = InMemoryVectorStore(dimension=2)

store.upsert(
    [
        make_record("zero", [0.0, 0.0]),
        make_record("normal", [1.0, 0.0]),
    ]
)

result = store.search([1.0, 0.0])

assert result.results[0].chunk_id == "normal"
assert result.results[0].score == pytest.approx(1.0)
assert result.results[1].chunk_id == "zero"
assert result.results[1].score == pytest.approx(0.0)

---------------------------------------------------------------------------

Deterministic ranking

---------------------------------------------------------------------------

def test_equal_scores_are_sorted_by_id() -> None:
store = InMemoryVectorStore(dimension=2)

store.upsert(
    [
        make_record("z", [0.0, 1.0]),
        make_record("a", [0.0, 1.0]),
        make_record("m", [0.0, 1.0]),
    ]
)

result = store.search([1.0, 0.0])

assert [item.chunk_id for item in result.results] == ["a", "m", "z"]

---------------------------------------------------------------------------

Payload mapping

---------------------------------------------------------------------------

def test_payload_maps_to_retrieved_chunk() -> None:
store = InMemoryVectorStore(dimension=2)

record = make_record(
    "record-id",
    [1.0, 0.0],
    document_id="document-123",
    chunk_id="chunk-456",
    text="Research paper text",
    metadata={"page": 3, "section": "Methods"},
)

store.upsert([record])

chunk = store.search([1.0, 0.0]).results[0]

assert isinstance(chunk, RetrievedChunk)
assert chunk.document_id == "document-123"
assert chunk.chunk_id == "chunk-456"
assert chunk.text == "Research paper text"
assert chunk.metadata["page"] == 3
assert chunk.metadata["section"] == "Methods"

def test_missing_payload_fields_use_defaults() -> None:
store = InMemoryVectorStore(dimension=2)

store.upsert(
    [
        VectorRecord(
            id="record-id",
            vector=[1.0, 0.0],
            payload={},
        )
    ]
)

chunk = store.search([1.0, 0.0]).results[0]

assert chunk.document_id == "record-id"
assert chunk.chunk_id == "record-id"
assert chunk.text == ""
assert chunk.metadata == {}

---------------------------------------------------------------------------

Defensive copying

---------------------------------------------------------------------------

def test_caller_mutation_does_not_change_store() -> None:
payload = {
"document_id": "doc-1",
"text": "original",
"nested": {"value": 1},
}

record = VectorRecord(
    id="a",
    vector=[1.0, 0.0],
    payload=payload,
)

store = InMemoryVectorStore()
store.upsert([record])

payload["text"] = "changed"
payload["nested"]["value"] = 999

result = store.search([1.0, 0.0])

assert result.results[0].text == "original"
assert result.results[0].metadata["nested"] == {"value": 1}

def test_returned_metadata_does_not_change_store() -> None:
store = InMemoryVectorStore()

store.upsert(
    [
        make_record(
            "a",
            [1.0, 0.0],
            metadata={"nested": {"value": 1}},
        )
    ]
)

result = store.search([1.0, 0.0])
result.results[0].metadata["nested"]["value"] = 999

second = store.search([1.0, 0.0])

assert second.results[0].metadata["nested"] == {"value": 1}

---------------------------------------------------------------------------

Dependency isolation

---------------------------------------------------------------------------

def _module_path() -> Path:
return (
Path(file).resolve().parents[1]
/ "src"
/ "nexora"
/ "retrieval"
/ "in_memory.py"
)

def test_only_allowed_imports() -> None:
tree = ast.parse(_module_path().read_text(encoding="utf-8"))

allowed_nexora = {
    "nexora.retrieval.models",
    "nexora.retrieval.store",
}

stdlib = set(sys.stdlib_module_names)

for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            assert alias.name.split(".")[0] in stdlib

    elif isinstance(node, ast.ImportFrom):
        if node.module is None:
            continue

        if node.module.startswith("nexora."):
            assert node.module in allowed_nexora
        else:
            assert node.module.split(".")[0] in stdlib

def test_no_forbidden_infrastructure_imports() -> None:
source = _module_path().read_text(encoding="utf-8")

forbidden = [
    "qdrant_client",
    "openai",
    "fastapi",
    "psycopg",
    "sqlalchemy",
    "redis",
    "requests",
    "httpx",
    "numpy",
    "pandas",
    "sklearn",
    "socket",
    "ssl",
    "urllib.request",
    "urllib.error",
]

for name in forbidden:
    assert name not in source

def test_no_environment_access() -> None:
tree = ast.parse(_module_path().read_text(encoding="utf-8"))

for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        assert node.func.attr not in {"getenv", "environ"}

def test_import_works_without_configuration(
monkeypatch: pytest.MonkeyPatch,
) -> None:
for name in [
"NEXORA_QDRANT_URL",
"NEXORA_QDRANT_API_KEY",
"NEXORA_QDRANT_COLLECTION",
"NEXORA_QDRANT_TIMEOUT",
"NEXORA_EMBEDDING_API_KEY",
"OPENAI_API_KEY",
]:
monkeypatch.delenv(name, raising=False)

project_root = _module_path().parents[3]

result = subprocess.run(
    [
        sys.executable,
        "-c",
        (
            "from nexora.retrieval.in_memory import InMemoryVectorStore; "
            "s = InMemoryVectorStore(); "
            "assert s.count() == 0"
        ),
    ],
    capture_output=True,
    text=True,
    env={
        **os.environ,
        "PYTHONPATH": str(project_root),
    },
    check=False,
)

assert result.returncode == 0, result.stderr
