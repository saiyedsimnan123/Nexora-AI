import ast
import inspect
import math
import os
import socket
from unittest import mock

import pytest

from nexora.retrieval import in_memory
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import VectorRecord
from nexora.retrieval.store import VectorStore


def rec(id, vector, payload=None):
    return VectorRecord(id=id, vector=vector, payload={} if payload is None else payload)


def raw(id, vector):
    record = rec(id, [1.0])  # bypasses any model-level validation
    object.__setattr__(record, "vector", vector)
    return record


def ids(result):
    return [chunk.chunk_id for chunk in result.results]


def filled(*records, dimension=None):
    store = InMemoryVectorStore(dimension)
    store.upsert(list(records))
    return store


def test_constructor_and_protocol():
    store = InMemoryVectorStore()
    assert store.dimension is None and store.count() == 0
    assert InMemoryVectorStore(3).dimension == 3
    assert isinstance(store, VectorStore)


@pytest.mark.parametrize("bad, error", [(0, ValueError), (-2, ValueError), (True, TypeError), (False, TypeError), ("3", TypeError), (2.5, TypeError)])
def test_invalid_dimension(bad, error):
    with pytest.raises(error):
        InMemoryVectorStore(bad)


def test_upsert_count_and_last_duplicate_wins():
    store = filled(rec("a", [1, 0], {"v": 1}), rec("b", [0, 1]), rec("a", [1, 0], {"v": 2}))
    assert store.count() == 2
    found = {c.chunk_id: c for c in store.search([1, 0]).results}
    assert found["a"].metadata == {"v": 2}
    store.upsert([rec("a", [0, 1], {"v": 3})])  # existing id is replaced
    assert store.count() == 2 and store.search([0, 1]).results[0].metadata == {"v": 3}


def test_dimension_established_by_first_upsert_and_enforced():
    store = InMemoryVectorStore()
    with pytest.raises(TypeError):
        store.upsert([raw("a", None)])
    assert store.dimension is None
    store.upsert([rec("a", [1, 2, 3])])
    assert store.dimension == 3
    with pytest.raises(ValueError, match="expected 3"):
        store.upsert([rec("b", [1, 2])])
    with pytest.raises(ValueError):
        InMemoryVectorStore(2).upsert([rec("a", [1, 2, 3])])
    with pytest.raises(ValueError):
        InMemoryVectorStore().upsert([rec("a", [1, 2]), rec("b", [1, 2, 3])])


@pytest.mark.parametrize("bad", [[], "ab", None, True, [True, 1.0], ["1", 2], [math.nan, 1], [math.inf, 1], [1, -math.inf], [None, 1]])
def test_invalid_record_vectors_are_rejected(bad):
    store = InMemoryVectorStore()
    with pytest.raises((TypeError, ValueError)):
        store.upsert([raw("a", bad)])
    assert store.count() == 0


def test_upsert_is_atomic():
    store = filled(rec("a", [1, 0]))
    for bad_batch in ([rec("b", [0, 1]), raw("c", [math.nan, 0])], [rec("b", [0, 1]), "x"], [rec("b", [0, 1]), rec("d", [1])]):
        with pytest.raises((TypeError, ValueError)):
            store.upsert(bad_batch)
        assert store.count() == 1
    with pytest.raises(TypeError):
        store.upsert(rec("b", [0, 1]))


def test_delete_behaviour():
    store = filled(rec("a", [1, 0]), rec("b", [0, 1]))
    store.delete(["a", "missing", "a"])
    store.delete(["a"])
    assert store.count() == 1
    for bad, error in ((None, TypeError), ("a", TypeError), ([5], TypeError), ([""], ValueError), (["b", " "], ValueError)):
        with pytest.raises(error):
            store.delete(bad)
    assert store.count() == 1
    store.delete(["b"])
    assert store.count() == 0 and store.dimension == 2
    with pytest.raises(ValueError):
        store.upsert([rec("c", [1, 2, 3])])


def test_cosine_ranking_scores_and_limit():
    store = filled(rec("d", [-1, 0]), rec("c", [0, 1]), rec("b", [1, 1]), rec("a", [2, 0]))
    result = store.search([1, 0])
    assert ids(result) == ["a", "b", "c", "d"] and result.query == ""
    assert [c.score for c in result.results] == pytest.approx([1.0, math.sqrt(0.5), 0.0, -1.0])
    assert ids(store.search([1, 0], limit=2)) == ["a", "b"]
    assert len(store.search([1, 0], limit=99).results) == 4


def test_equal_scores_are_ordered_by_id():
    store = filled(rec("b", [1, 0]), rec("c", [3, 0]), rec("a", [1, 0]), rec("z", [0, 5]))
    assert ids(store.search([1, 0])) == ["a", "b", "c", "z"]
    assert ids(store.search([1, 0], limit=2)) == ["a", "b"]


@pytest.mark.parametrize("bad, error", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError), ("2", TypeError), (None, TypeError)])
def test_invalid_search_limit(bad, error):
    with pytest.raises(error):
        filled(rec("a", [1, 0])).search([1, 0], limit=bad)


def test_invalid_query_vectors():
    store = filled(rec("a", [1, 0]))
    for bad, error in (([1, 0, 0], ValueError), ([1], ValueError), ([], ValueError), ((1, 0), TypeError), (None, TypeError), ([True, 0], TypeError), ([math.nan, 0], ValueError), ([math.inf, 0], ValueError)):
        with pytest.raises(error):
            store.search(bad)


def test_zero_vectors_score_zero():
    store = filled(rec("a", [0, 0]), rec("b", [1, 0]))
    assert [(c.chunk_id, c.score) for c in store.search([1, 0]).results] == [("b", 1.0), ("a", 0.0)]
    assert [(c.chunk_id, c.score) for c in store.search([0, 0]).results] == [("a", 0.0), ("b", 0.0)]


def test_extreme_magnitudes_stay_finite_and_clamped():
    store = filled(rec("big", [1e308, 1e308]), rec("tiny", [5e-324, 5e-324]), rec("opp", [-1e308, -1e308]))
    scores = {c.chunk_id: c.score for c in store.search([1e308, 1e308]).results}
    assert scores == pytest.approx({"big": 1.0, "tiny": 1.0, "opp": -1.0})
    assert all(-1.0 <= s <= 1.0 and math.isfinite(s) for s in scores.values())


def test_empty_store_search():
    result = InMemoryVectorStore().search([1.0, 2.0])
    assert result.query == "" and result.results == []
    store = filled(rec("a", [1, 0]))
    store.delete(["a"])
    assert store.search([1, 0]).results == []


def test_payload_mapping_and_fallbacks():
    payload = {"document_id": "d1", "chunk_id": "c1", "text": "hello", "page": 4}
    chunk = filled(rec("r1", [1, 0], payload)).search([1, 0]).results[0]
    assert (chunk.document_id, chunk.chunk_id, chunk.text) == ("d1", "c1", "hello")
    assert chunk.metadata == payload
    for fallback_payload in ({}, {"document_id": "", "chunk_id": "", "text": 5}, {"document_id": 7, "chunk_id": None, "text": None}):
        chunk = filled(rec("r2", [1, 0], fallback_payload)).search([1, 0]).results[0]
        assert (chunk.document_id, chunk.chunk_id, chunk.text) == ("r2", "r2", "")
        assert chunk.metadata == fallback_payload


def test_defensive_copies():
    payload, vector = {"text": "t", "nested": {"tags": ["a"]}}, [1.0, 0.0]
    store = filled(rec("a", vector, payload))
    payload["nested"]["tags"].append("MUTATED")
    vector[0], vector[1] = 0.0, 1.0
    first = store.search([1, 0]).results[0]
    assert first.metadata["nested"]["tags"] == ["a"] and first.score == pytest.approx(1.0)
    first.metadata["nested"]["tags"].append("OTHER")
    first.metadata["new"] = 1
    assert store.search([1, 0]).results[0].metadata == {"text": "t", "nested": {"tags": ["a"]}}


def test_implementation_has_no_forbidden_imports_or_io():
    tree = ast.parse(inspect.getsource(in_memory))
    imported = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert {m.split(".")[0] for m in imported} <= {"copy", "dataclasses", "heapq", "math", "nexora"}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"os", "environ", "getenv", "open", "socket", "urllib", "subprocess", "__import__", "eval", "exec"}


def test_operations_use_no_network_files_or_environment():
    with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network")), mock.patch("builtins.open", side_effect=AssertionError("file")), mock.patch.object(os, "environ", {}):
        store = filled(rec("a", [1, 0]), rec("b", [0, 1]))
        assert ids(store.search([1, 0])) == ["a", "b"]
        store.delete(["a"])
        assert store.count() == 1
