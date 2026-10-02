import ast
import inspect
import math
import socket
import sys
import threading
from contextlib import contextmanager
from unittest import mock

import pytest

from nexora.embedding import cache as cache_module
from nexora.embedding.cache import EmbeddingCache, InMemoryEmbeddingCache


@contextmanager
def no_network():
    with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network used")):
        yield


# ------------------------------------------------------------ construction

def test_cache_can_be_created():
    assert isinstance(InMemoryEmbeddingCache(), InMemoryEmbeddingCache)


def test_cache_satisfies_embedding_cache_protocol():
    assert isinstance(InMemoryEmbeddingCache(), EmbeddingCache)


def test_object_missing_methods_is_not_an_embedding_cache():
    class Partial:
        def get(self, key):
            return None

    assert not isinstance(Partial(), EmbeddingCache)
    assert not isinstance(object(), EmbeddingCache)


def test_empty_cache_works_correctly():
    cache = InMemoryEmbeddingCache()
    assert len(cache) == 0
    assert cache.get("anything") is None
    cache.delete("anything")
    cache.clear()
    assert len(cache) == 0


# --------------------------------------------------------- basic behaviour

def test_cache_miss_returns_none():
    assert InMemoryEmbeddingCache().get("missing") is None


def test_set_then_get_returns_the_vector():
    cache = InMemoryEmbeddingCache()
    cache.set("abc", [0.1, 0.2, 0.3])
    assert cache.get("abc") == [0.1, 0.2, 0.3]


def test_exact_float_values_are_returned():
    cache = InMemoryEmbeddingCache()
    vector = [0.28432901419429385, -0.478667309717539, 1e-300, 1e300, -0.0]
    cache.set("exact", vector)
    assert cache.get("exact") == vector


def test_integer_values_are_stored_as_floats():
    cache = InMemoryEmbeddingCache()
    cache.set("ints", [1, 2, 3])
    result = cache.get("ints")
    assert result == [1.0, 2.0, 3.0]
    assert all(type(value) is float for value in result)


def test_multiple_keys_work_independently():
    cache = InMemoryEmbeddingCache()
    cache.set("a", [1.0])
    cache.set("b", [2.0, 2.0])
    cache.set("c", [3.0, 3.0, 3.0])
    assert cache.get("a") == [1.0]
    assert cache.get("b") == [2.0, 2.0]
    assert cache.get("c") == [3.0, 3.0, 3.0]
    assert len(cache) == 3


def test_existing_key_is_overwritten():
    cache = InMemoryEmbeddingCache()
    cache.set("k", [1.0, 1.0])
    cache.set("k", [9.0, 8.0, 7.0])
    assert cache.get("k") == [9.0, 8.0, 7.0]
    assert len(cache) == 1


def test_keys_are_used_exactly_as_given():
    cache = InMemoryEmbeddingCache()
    cache.set("Key", [1.0])
    cache.set("key", [2.0])
    cache.set(" key", [3.0])
    assert cache.get("Key") == [1.0]
    assert cache.get("key") == [2.0]
    assert cache.get(" key") == [3.0]
    assert cache.get("key ") is None


def test_unicode_and_long_keys_work():
    cache = InMemoryEmbeddingCache()
    long_key = "k" * 10_000
    cache.set("α ≥ β — 日本語", [1.0])
    cache.set(long_key, [2.0])
    assert cache.get("α ≥ β — 日本語") == [1.0]
    assert cache.get(long_key) == [2.0]


def test_delete_removes_a_key():
    cache = InMemoryEmbeddingCache()
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    cache.delete("a")
    assert cache.get("a") is None
    assert cache.get("b") == [2.0]
    assert len(cache) == 1


def test_deleting_a_missing_key_is_safe():
    cache = InMemoryEmbeddingCache()
    cache.set("a", [1.0])
    cache.delete("not-there")
    cache.delete("a")
    cache.delete("a")
    assert len(cache) == 0


def test_clear_removes_all_entries():
    cache = InMemoryEmbeddingCache()
    for i in range(5):
        cache.set(f"k{i}", [float(i)])
    cache.clear()
    assert len(cache) == 0
    assert all(cache.get(f"k{i}") is None for i in range(5))


def test_clear_followed_by_get_returns_none_and_cache_is_reusable():
    cache = InMemoryEmbeddingCache()
    cache.set("a", [1.0])
    cache.clear()
    assert cache.get("a") is None
    cache.set("a", [2.0])
    assert cache.get("a") == [2.0]


# ---------------------------------------------------------- key validation

@pytest.mark.parametrize("bad_key", ["", "   ", "\n", "\t", " \t\n "])
def test_empty_or_whitespace_key_raises_value_error(bad_key):
    cache = InMemoryEmbeddingCache()
    with pytest.raises(ValueError, match="key must not be empty"):
        cache.set(bad_key, [1.0])
    with pytest.raises(ValueError, match="key must not be empty"):
        cache.get(bad_key)
    with pytest.raises(ValueError, match="key must not be empty"):
        cache.delete(bad_key)
    assert len(cache) == 0


@pytest.mark.parametrize("bad_key", [None, 123, 4.5, b"key", ["k"], ("k",), {"k": 1}])
def test_non_string_key_raises_type_error(bad_key):
    cache = InMemoryEmbeddingCache()
    with pytest.raises(TypeError, match="key must be a str"):
        cache.set(bad_key, [1.0])
    with pytest.raises(TypeError, match="key must be a str"):
        cache.get(bad_key)
    with pytest.raises(TypeError, match="key must be a str"):
        cache.delete(bad_key)
    assert len(cache) == 0


# -------------------------------------------------------- vector validation

@pytest.mark.parametrize("bad_vector", [None, "abc", 5, 1.5, (1.0, 2.0), {1.0}, {"a": 1.0}, b"ab"])
def test_non_list_vector_is_rejected(bad_vector):
    cache = InMemoryEmbeddingCache()
    with pytest.raises(TypeError, match="vector must be a list"):
        cache.set("k", bad_vector)
    assert cache.get("k") is None


@pytest.mark.parametrize("bad_value", ["0.5", None, [0.5], b"1", 1 + 2j, {"a": 1}, object()])
def test_non_numeric_vector_values_are_rejected(bad_value):
    cache = InMemoryEmbeddingCache()
    with pytest.raises(TypeError, match=r"vector\[1\] must be a number"):
        cache.set("k", [0.1, bad_value, 0.3])
    assert cache.get("k") is None


@pytest.mark.parametrize("bad_value", [True, False])
def test_bool_vector_values_are_rejected(bad_value):
    cache = InMemoryEmbeddingCache()
    with pytest.raises(TypeError, match="must be a number, got bool"):
        cache.set("k", [0.1, bad_value])
    assert cache.get("k") is None


def test_nan_is_rejected():
    cache = InMemoryEmbeddingCache()
    with pytest.raises(ValueError, match="finite"):
        cache.set("k", [0.1, math.nan])
    assert cache.get("k") is None


def test_positive_infinity_is_rejected():
    cache = InMemoryEmbeddingCache()
    with pytest.raises(ValueError, match="finite"):
        cache.set("k", [math.inf, 0.1])
    assert cache.get("k") is None


def test_negative_infinity_is_rejected():
    cache = InMemoryEmbeddingCache()
    with pytest.raises(ValueError, match="finite"):
        cache.set("k", [0.1, -math.inf])
    assert cache.get("k") is None


def test_empty_vector_is_rejected():
    cache = InMemoryEmbeddingCache()
    with pytest.raises(ValueError, match="vector must not be empty"):
        cache.set("k", [])
    assert cache.get("k") is None


def test_invalid_vector_does_not_overwrite_existing_entry():
    cache = InMemoryEmbeddingCache()
    cache.set("k", [1.0, 2.0])
    with pytest.raises(ValueError):
        cache.set("k", [1.0, math.nan])
    with pytest.raises(TypeError):
        cache.set("k", [1.0, "x"])
    assert cache.get("k") == [1.0, 2.0]


def test_error_messages_do_not_contain_vector_values_or_key():
    cache = InMemoryEmbeddingCache()
    secret_key = "sk-test-secret-not-real-123"
    with pytest.raises(TypeError) as info:
        cache.set(secret_key, [0.123456789, "leaky-value-xyz"])
    message = str(info.value)
    assert "leaky-value-xyz" not in message
    assert secret_key not in message
    assert "0.123456789" not in message


# -------------------------------------------------------- defensive copying

def test_mutating_original_vector_after_set_does_not_change_cache():
    cache = InMemoryEmbeddingCache()
    vector = [1.0, 2.0]
    cache.set("abc", vector)
    vector[0] = 999.0
    vector.append(5.0)
    assert cache.get("abc") == [1.0, 2.0]


def test_mutating_returned_vector_does_not_change_cache():
    cache = InMemoryEmbeddingCache()
    cache.set("abc", [1.0, 2.0])
    result = cache.get("abc")
    result[0] = 999.0
    result.append(5.0)
    assert cache.get("abc") == [1.0, 2.0]


def test_each_get_returns_a_new_list():
    cache = InMemoryEmbeddingCache()
    cache.set("abc", [1.0, 2.0])
    first = cache.get("abc")
    second = cache.get("abc")
    assert first == second
    assert first is not second


def test_stored_list_is_a_different_object_than_the_input():
    cache = InMemoryEmbeddingCache()
    vector = [1.0, 2.0]
    cache.set("abc", vector)
    assert cache.get("abc") is not vector


def test_separate_keys_with_equal_vectors_remain_independent():
    cache = InMemoryEmbeddingCache()
    vector = [1.0, 2.0]
    cache.set("a", vector)
    cache.set("b", vector)
    cache.get("a")[0] = 42.0
    assert cache.get("b") == [1.0, 2.0]
    assert cache.get("a") == [1.0, 2.0]


# ------------------------------------------------------------ thread safety

def _run_threads(target, thread_count):
    barrier = threading.Barrier(thread_count)
    errors = []

    def runner(thread_id):
        try:
            barrier.wait()
            target(thread_id)
        except BaseException as error:  # noqa: BLE001 - collected and asserted below
            errors.append(error)

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "a thread did not finish"
    assert errors == []


def test_concurrent_set_and_get_on_separate_keys_stays_consistent():
    cache = InMemoryEmbeddingCache()
    threads, per_thread = 8, 200

    def work(thread_id):
        for i in range(per_thread):
            key = f"t{thread_id}-{i}"
            vector = [float(thread_id), float(i), 1.0]
            cache.set(key, vector)
            assert cache.get(key) == vector

    _run_threads(work, threads)

    assert len(cache) == threads * per_thread
    for thread_id in range(threads):
        for i in range(per_thread):
            assert cache.get(f"t{thread_id}-{i}") == [float(thread_id), float(i), 1.0]


def test_concurrent_writes_to_one_key_never_produce_torn_values():
    cache = InMemoryEmbeddingCache()
    cache.set("shared", [0.0] * 64)
    threads, iterations = 6, 300

    def work(thread_id):
        for i in range(iterations):
            if thread_id % 2 == 0:
                cache.set("shared", [float(thread_id * 1000 + i)] * 64)
            else:
                value = cache.get("shared")
                assert len(value) == 64
                assert len(set(value)) == 1  # every element from the same write

    _run_threads(work, threads)

    final = cache.get("shared")
    assert len(final) == 64 and len(set(final)) == 1
    assert len(cache) == 1


def test_concurrent_set_delete_and_clear_do_not_break_the_cache():
    cache = InMemoryEmbeddingCache()
    threads, iterations = 6, 200

    def work(thread_id):
        for i in range(iterations):
            key = f"k{i % 10}"
            if thread_id % 3 == 0:
                cache.set(key, [float(i)])
            elif thread_id % 3 == 1:
                cache.delete(key)
            else:
                value = cache.get(key)
                assert value is None or len(value) == 1
                if i % 50 == 0:
                    cache.clear()

    _run_threads(work, threads)

    assert 0 <= len(cache) <= 10
    cache.set("after", [1.0])
    assert cache.get("after") == [1.0]


# ------------------------------------------------- determinism / no network

def test_behavior_is_deterministic_across_instances():
    def run():
        cache = InMemoryEmbeddingCache()
        cache.set("a", [1.0, 2.0])
        cache.set("b", [3.0])
        cache.set("a", [4.0, 5.0])
        cache.delete("b")
        return cache.get("a"), cache.get("b"), len(cache)

    assert run() == run() == ([4.0, 5.0], None, 1)


def test_no_network_is_used():
    with no_network():
        cache = InMemoryEmbeddingCache()
        cache.set("a", [1.0])
        assert cache.get("a") == [1.0]
        cache.delete("a")
        cache.clear()


def test_module_imports_only_the_standard_library():
    tree = ast.parse(inspect.getsource(cache_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported, "expected the module to import something"
    assert imported <= set(sys.stdlib_module_names)
