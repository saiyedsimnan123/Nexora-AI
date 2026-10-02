"""Tests for the Step 11A retrieval data models."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import socket
import sys

import pytest

import nexora.retrieval.models as models_module
from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord

NAN = float("nan")
INF = float("inf")


def make_record(**overrides) -> VectorRecord:
    values = {
        "id": "rec-1",
        "vector": [0.1, 0.2, 0.3],
        "payload": {"source": "paper.pdf"},
    }
    values.update(overrides)
    return VectorRecord(**values)


def make_chunk(**overrides) -> RetrievedChunk:
    values = {
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "text": "some text",
        "score": 0.75,
        "metadata": {"page": 1},
    }
    values.update(overrides)
    return RetrievedChunk(**values)


# --------------------------------------------------------------------------
# VectorRecord
# --------------------------------------------------------------------------


def test_vector_record_valid_construction():
    record = make_record()
    assert record.id == "rec-1"
    assert record.vector == [0.1, 0.2, 0.3]
    assert record.payload == {"source": "paper.pdf"}


def test_vector_record_accepts_int_values_and_empty_payload():
    record = make_record(vector=[1, 2, 3], payload={})
    assert record.vector == [1, 2, 3]
    assert record.payload == {}


def test_vector_record_id_is_opaque_string():
    assert make_record(id="42").id == "42"
    assert make_record(id="not-a-uuid/with spaces").id == "not-a-uuid/with spaces"


def test_vector_record_empty_vector_rejected():
    with pytest.raises(ValueError):
        make_record(vector=[])


@pytest.mark.parametrize("bad_id", [None, 123, b"rec", ["rec"]])
def test_vector_record_non_string_id_rejected(bad_id):
    with pytest.raises(TypeError):
        make_record(id=bad_id)


def test_vector_record_empty_id_rejected():
    with pytest.raises(ValueError):
        make_record(id="")


@pytest.mark.parametrize("bad_vector", [(0.1, 0.2), "abc", None, 5, {"a": 1}])
def test_vector_record_non_list_vector_rejected(bad_vector):
    with pytest.raises(TypeError):
        make_record(vector=bad_vector)


@pytest.mark.parametrize("bad_value", ["1.0", None, [1.0], 1 + 2j])
def test_vector_record_invalid_vector_element_rejected(bad_value):
    with pytest.raises(TypeError):
        make_record(vector=[0.1, bad_value])


@pytest.mark.parametrize("bad_value", [True, False])
def test_vector_record_bool_vector_element_rejected(bad_value):
    with pytest.raises(TypeError):
        make_record(vector=[0.1, bad_value])


@pytest.mark.parametrize("bad_value", [NAN, INF, -INF, 10**400])
def test_vector_record_non_finite_vector_element_rejected(bad_value):
    with pytest.raises(ValueError):
        make_record(vector=[0.1, bad_value])


@pytest.mark.parametrize("bad_payload", [None, [], "x", [("a", 1)]])
def test_vector_record_non_dict_payload_rejected(bad_payload):
    with pytest.raises(TypeError):
        make_record(payload=bad_payload)


@pytest.mark.parametrize("bad_payload", [{1: "a"}, {None: 1}, {"ok": 1, 2: 2}])
def test_vector_record_non_string_payload_key_rejected(bad_payload):
    with pytest.raises(TypeError):
        make_record(payload=bad_payload)


def test_vector_record_copies_vector_defensively():
    vector = [0.1, 0.2]
    record = make_record(vector=vector)
    vector.append(0.3)
    vector[0] = 9.9
    assert record.vector == [0.1, 0.2]
    assert record.vector is not vector


def test_vector_record_copies_payload_defensively():
    payload = {"a": 1}
    record = make_record(payload=payload)
    payload["b"] = 2
    payload["a"] = 99
    assert record.payload == {"a": 1}
    assert record.payload is not payload


@pytest.mark.parametrize("field", ["id", "vector", "payload"])
def test_vector_record_is_frozen(field):
    record = make_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, field, getattr(record, field))


def test_vector_record_equality():
    assert make_record() == make_record()
    assert make_record() != make_record(id="rec-2")
    assert make_record() != make_record(vector=[0.1, 0.2, 0.4])
    assert make_record() != make_record(payload={})


def test_vector_record_instances_are_independent():
    shared_vector = [0.1, 0.2]
    first = make_record(vector=shared_vector)
    second = make_record(vector=shared_vector)
    first.vector.append(0.3)
    first.payload["extra"] = True
    assert second.vector == [0.1, 0.2]
    assert second.payload == {"source": "paper.pdf"}


# --------------------------------------------------------------------------
# RetrievedChunk
# --------------------------------------------------------------------------


def test_retrieved_chunk_valid_construction():
    chunk = make_chunk()
    assert chunk.document_id == "doc-1"
    assert chunk.chunk_id == "chunk-1"
    assert chunk.text == "some text"
    assert chunk.score == 0.75
    assert chunk.metadata == {"page": 1}


def test_retrieved_chunk_allows_empty_text_and_any_finite_score():
    assert make_chunk(text="").text == ""
    assert make_chunk(score=-12.5).score == -12.5
    assert make_chunk(score=1e9).score == 1e9
    assert make_chunk(score=3).score == 3
    assert make_chunk(metadata={}).metadata == {}


@pytest.mark.parametrize("field", ["document_id", "chunk_id"])
@pytest.mark.parametrize("bad_value", [None, 7, b"x"])
def test_retrieved_chunk_non_string_ids_rejected(field, bad_value):
    with pytest.raises(TypeError):
        make_chunk(**{field: bad_value})


@pytest.mark.parametrize("field", ["document_id", "chunk_id"])
def test_retrieved_chunk_empty_ids_rejected(field):
    with pytest.raises(ValueError):
        make_chunk(**{field: ""})


@pytest.mark.parametrize("bad_text", [None, 1, b"text", ["text"]])
def test_retrieved_chunk_invalid_text_rejected(bad_text):
    with pytest.raises(TypeError):
        make_chunk(text=bad_text)


@pytest.mark.parametrize("bad_score", ["0.5", None, [0.5], 1 + 0j])
def test_retrieved_chunk_invalid_score_rejected(bad_score):
    with pytest.raises(TypeError):
        make_chunk(score=bad_score)


@pytest.mark.parametrize("bad_score", [True, False])
def test_retrieved_chunk_bool_score_rejected(bad_score):
    with pytest.raises(TypeError):
        make_chunk(score=bad_score)


@pytest.mark.parametrize("bad_score", [NAN, INF, -INF, 10**400])
def test_retrieved_chunk_non_finite_score_rejected(bad_score):
    with pytest.raises(ValueError):
        make_chunk(score=bad_score)


@pytest.mark.parametrize("bad_metadata", [None, [], "x", [("a", 1)]])
def test_retrieved_chunk_non_dict_metadata_rejected(bad_metadata):
    with pytest.raises(TypeError):
        make_chunk(metadata=bad_metadata)


@pytest.mark.parametrize("bad_metadata", [{1: "a"}, {None: 1}, {"ok": 1, 2: 2}])
def test_retrieved_chunk_non_string_metadata_key_rejected(bad_metadata):
    with pytest.raises(TypeError):
        make_chunk(metadata=bad_metadata)


def test_retrieved_chunk_copies_metadata_defensively():
    metadata = {"page": 1}
    chunk = make_chunk(metadata=metadata)
    metadata["page"] = 2
    metadata["extra"] = True
    assert chunk.metadata == {"page": 1}
    assert chunk.metadata is not metadata


@pytest.mark.parametrize(
    "field", ["document_id", "chunk_id", "text", "score", "metadata"]
)
def test_retrieved_chunk_is_frozen(field):
    chunk = make_chunk()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(chunk, field, getattr(chunk, field))


def test_retrieved_chunk_equality():
    assert make_chunk() == make_chunk()
    assert make_chunk() != make_chunk(chunk_id="chunk-2")
    assert make_chunk() != make_chunk(score=0.5)
    assert make_chunk() != make_chunk(metadata={})


# --------------------------------------------------------------------------
# SearchResult
# --------------------------------------------------------------------------


def test_search_result_valid_construction():
    chunk = make_chunk()
    result = SearchResult(query="what is nexora?", results=[chunk])
    assert result.query == "what is nexora?"
    assert result.results == [chunk]


def test_search_result_allows_empty_results():
    result = SearchResult(query="nothing matches", results=[])
    assert result.results == []


def test_search_result_multiple_results_and_ordering_preserved():
    chunks = [
        make_chunk(chunk_id="c1", score=0.9),
        make_chunk(chunk_id="c2", score=0.1),
        make_chunk(chunk_id="c3", score=0.5),
    ]
    result = SearchResult(query="q", results=chunks)
    assert [c.chunk_id for c in result.results] == ["c1", "c2", "c3"]
    assert [c.score for c in result.results] == [0.9, 0.1, 0.5]


@pytest.mark.parametrize("bad_query", [None, 1, b"q", ["q"]])
def test_search_result_invalid_query_rejected(bad_query):
    with pytest.raises(TypeError):
        SearchResult(query=bad_query, results=[])


@pytest.mark.parametrize(
    "bad_results", [None, "abc", {"a": 1}, 5, (make_chunk(),), iter([make_chunk()])]
)
def test_search_result_invalid_results_container_rejected(bad_results):
    with pytest.raises(TypeError):
        SearchResult(query="q", results=bad_results)


@pytest.mark.parametrize(
    "bad_item",
    [
        "chunk",
        None,
        {"document_id": "doc-1"},
        make_record(),
    ],
)
def test_search_result_non_retrieved_chunk_item_rejected(bad_item):
    with pytest.raises(TypeError):
        SearchResult(query="q", results=[make_chunk(), bad_item])


def test_search_result_copies_results_defensively():
    first = make_chunk(chunk_id="c1")
    results = [first]
    result = SearchResult(query="q", results=results)
    results.append(make_chunk(chunk_id="c2"))
    results[0] = make_chunk(chunk_id="other")
    assert result.results == [first]
    assert result.results is not results


@pytest.mark.parametrize("field", ["query", "results"])
def test_search_result_is_frozen(field):
    result = SearchResult(query="q", results=[make_chunk()])
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, field, getattr(result, field))


def test_search_result_equality():
    chunk = make_chunk()
    assert SearchResult("q", [chunk]) == SearchResult("q", [make_chunk()])
    assert SearchResult("q", [chunk]) != SearchResult("other", [chunk])
    assert SearchResult("q", [chunk]) != SearchResult("q", [])


# --------------------------------------------------------------------------
# Safety: no network, no environment access, no extra dependencies
# --------------------------------------------------------------------------


def _build_all_models() -> SearchResult:
    record = make_record()
    chunk = make_chunk()
    assert record.vector
    return SearchResult(query="q", results=[chunk])


def test_models_make_no_network_calls(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    class BlockedSocket:
        def __init__(self, *args, **kwargs):
            blocked()

    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "gethostbyname", blocked)

    assert len(_build_all_models().results) == 1


def test_models_do_not_read_environment(monkeypatch):
    class GuardedEnv(dict):
        def _fail(self, *args, **kwargs):
            raise AssertionError("environment accessed")

        __getitem__ = get = __contains__ = __iter__ = _fail
        keys = values = items = copy = setdefault = pop = _fail

    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", GuardedEnv())
        assert len(_build_all_models().results) == 1


def _imported_modules() -> set[str]:
    tree = ast.parse(inspect.getsource(models_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not expected"
            imported.add(node.module)
    return imported


def test_models_import_only_standard_library():
    imported = _imported_modules()
    assert imported, "expected the models module to have imports"
    for module in imported:
        top_level = module.split(".")[0]
        assert top_level in sys.stdlib_module_names, f"non-stdlib import: {module}"


def test_models_do_not_import_other_nexora_modules():
    for module in _imported_modules():
        assert module.split(".")[0] != "nexora", f"Nexora import: {module}"
