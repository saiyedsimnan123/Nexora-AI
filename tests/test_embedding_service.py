import ast
import hashlib
import inspect
import socket
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from nexora.embedding import (
    EmbeddingConfig,
    EmbeddingProvider,
)
from nexora.embedding import service as service_module
from nexora.embedding.cache import EmbeddingCache, InMemoryEmbeddingCache
from nexora.embedding.openai_provider import OpenAIEmbeddingProvider
from nexora.embedding.service import EmbeddingService

FAKE_SECRET = "test-secret-not-a-real-key-123"
GOLDEN_KEY = (
    "nexora:embedding:v1:default:"
    "426fa55d661313b216d0f7b6b14dc27ef5f2795007c77484c723f8dad61faa98"
)


def vec(text, tag=0.0):
    """Deterministic fake embedding; distinct for distinct texts."""
    return [
        float(len(text)),
        float(ord(text[0])),
        float(sum(map(ord, text)) % 1000) + tag,
    ]


class FakeProvider:
    """Records every call; returns deterministic vectors."""

    def __init__(self, tag=0.0, error=None, batch_result=None):
        self.tag = tag
        self.error = error
        self.batch_result = batch_result
        self.single_calls = []
        self.batch_calls = []
        self.last_returned = None

    def embed_text(self, text):
        self.single_calls.append(text)
        if self.error is not None:
            raise self.error
        self.last_returned = vec(text, self.tag)
        return self.last_returned

    def embed_texts(self, texts):
        self.batch_calls.append(list(texts))
        if self.error is not None:
            raise self.error
        if self.batch_result is not None:
            return self.batch_result(texts)
        return [vec(t, self.tag) for t in texts]

    @property
    def total_calls(self):
        return len(self.single_calls) + len(self.batch_calls)


class RecordingCache:
    """Wraps InMemoryEmbeddingCache and records cache operations."""

    def __init__(self):
        self.inner = InMemoryEmbeddingCache()
        self.get_keys = []
        self.set_keys = []
        self.delete_keys = []
        self.clears = 0

    def get(self, key):
        self.get_keys.append(key)
        return self.inner.get(key)

    def set(self, key, vector):
        self.set_keys.append(key)
        self.inner.set(key, vector)

    def delete(self, key):
        self.delete_keys.append(key)
        self.inner.delete(key)

    def clear(self):
        self.clears += 1
        self.inner.clear()

    @property
    def total_calls(self):
        return len(self.get_keys) + len(self.set_keys) + len(self.delete_keys) + self.clears

    def __len__(self):
        return len(self.inner)


@contextmanager
def no_network():
    with mock.patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("network used"),
    ):
        yield


def make(provider=None, cache="new", namespace="default"):
    provider = FakeProvider() if provider is None else provider
    cache = RecordingCache() if cache == "new" else cache
    return EmbeddingService(provider, cache, namespace), provider, cache


def test_service_can_be_constructed():
    service, _, _ = make()
    assert isinstance(service, EmbeddingService)


def test_service_can_be_constructed_with_defaults_and_no_cache():
    service = EmbeddingService(FakeProvider())
    assert isinstance(service, EmbeddingService)


def test_service_satisfies_embedding_provider_interface():
    service, _, _ = make()
    assert isinstance(service, EmbeddingProvider)


def test_fake_cache_wrapper_satisfies_cache_interface():
    assert isinstance(RecordingCache(), EmbeddingCache)


@pytest.mark.parametrize("bad_provider", [None, object(), "provider", 5])
def test_invalid_provider_raises_type_error(bad_provider):
    with pytest.raises(TypeError, match="provider must implement EmbeddingProvider"):
        EmbeddingService(bad_provider)


def test_provider_with_only_embed_text_is_rejected():
    class SingleOnly:
        def embed_text(self, text):
            return [1.0]

    with pytest.raises(TypeError, match="provider must implement"):
        EmbeddingService(SingleOnly())


@pytest.mark.parametrize("bad_cache", [object(), "cache", {}, 5])
def test_invalid_cache_raises_type_error(bad_cache):
    with pytest.raises(TypeError, match="cache must implement EmbeddingCache"):
        EmbeddingService(FakeProvider(), bad_cache)


@pytest.mark.parametrize("bad_namespace", [None, 5, b"ns", ["ns"]])
def test_non_string_namespace_raises_type_error(bad_namespace):
    with pytest.raises(TypeError, match="cache_namespace must be a str"):
        EmbeddingService(FakeProvider(), None, bad_namespace)


@pytest.mark.parametrize("bad_namespace", ["", "   ", "\n\t"])
def test_blank_namespace_raises_value_error(bad_namespace):
    with pytest.raises(ValueError, match="cache_namespace must not be empty"):
        EmbeddingService(FakeProvider(), None, bad_namespace)


def test_service_works_with_cache_none_single():
    provider = FakeProvider()
    service = EmbeddingService(provider, cache=None)
    assert service.embed_text("paper A") == vec("paper A")
    assert provider.single_calls == ["paper A"]


def test_cache_none_never_caches_single():
    provider = FakeProvider()
    service = EmbeddingService(provider, cache=None)
    service.embed_text("paper A")
    service.embed_text("paper A")
    assert provider.single_calls == ["paper A", "paper A"]


def test_batch_with_no_cache_delegates_to_provider():
    provider = FakeProvider()
    service = EmbeddingService(provider, cache=None)
    texts = ["A x", "B x", "A x"]
    assert service.embed_texts(texts) == [vec(t) for t in texts]
    assert provider.batch_calls == [texts]
    assert provider.single_calls == []


def test_cache_is_never_used_when_cache_is_none():
    spy = RecordingCache()
    provider = FakeProvider()
    service = EmbeddingService(provider, cache=None)
    service.embed_text("paper A")
    service.embed_texts(["paper A", "paper B"])
    assert spy.total_calls == 0
    assert provider.total_calls == 2


def test_single_miss_calls_provider_once():
    service, provider, _ = make()
    assert service.embed_text("paper A") == vec("paper A")
    assert provider.single_calls == ["paper A"]
    assert provider.batch_calls == []


def test_single_hit_does_not_call_provider():
    service, provider, cache = make()
    cache.inner.set(service.cache_key("paper A"), [7.0, 8.0, 9.0])
    assert service.embed_text("paper A") == [7.0, 8.0, 9.0]
    assert provider.total_calls == 0


def test_single_miss_stores_result_in_cache():
    service, _, cache = make()
    service.embed_text("paper A")
    assert cache.set_keys == [service.cache_key("paper A")]
    assert cache.inner.get(service.cache_key("paper A")) == vec("paper A")


def test_repeating_the_same_text_uses_the_cache():
    service, provider, cache = make()
    first = service.embed_text("paper A")
    second = service.embed_text("paper A")
    third = service.embed_text("paper A")
    assert first == second == third
    assert provider.single_calls == ["paper A"]
    assert len(cache.set_keys) == 1


def test_single_texts_are_passed_unmodified():
    service, provider, _ = make()
    text = "  Eq. (3): y = wx + b \n α ≥ β  "
    service.embed_text(text)
    assert provider.single_calls == [text]


def test_different_texts_produce_different_keys():
    service, _, _ = make()
    texts = ["a", "b", "A", "a ", " a", "a.", "ab", "α", "paper A", "paper B"]
    keys = [service.cache_key(t) for t in texts]
    assert len(set(keys)) == len(texts)


def test_same_text_and_namespace_produce_the_same_key():
    first, _, _ = make(namespace="ns1")
    second, _, _ = make(namespace="ns1")
    assert first.cache_key("same text") == second.cache_key("same text")


def test_key_format_is_versioned_and_matches_known_value():
    service, _, _ = make(namespace="default")
    key = service.cache_key("nexora")
    assert key == GOLDEN_KEY
    payload = len(b"default").to_bytes(8, "big") + b"default" + b"nexora"
    assert key == "nexora:embedding:v1:default:" + hashlib.sha256(payload).hexdigest()
    assert key.startswith("nexora:embedding:v1:")


def test_namespace_and_text_boundaries_cannot_collide():
    a, _, _ = make(namespace="a")
    ab, _, _ = make(namespace="ab")
    assert a.cache_key("bc") != ab.cache_key("c")
    colon_a, _, _ = make(namespace="x:y")
    plain, _, _ = make(namespace="x")
    assert colon_a.cache_key("z") != plain.cache_key("y:z")


def test_keys_with_unusual_text_work():
    service, provider, _ = make()
    text = "bad surrogate \ud800 here"
    assert service.cache_key(text) == service.cache_key(text)
    service.embed_text(text)
    service.embed_text(text)
    assert len(provider.single_calls) == 1


def test_cache_key_rejects_non_string():
    service, _, _ = make()
    with pytest.raises(TypeError, match="text must be a str"):
        service.cache_key(123)


def test_cache_key_generation_does_not_use_builtin_hash():
    tree = ast.parse(inspect.getsource(service_module))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "hash" not in called_names
    with mock.patch("builtins.hash", side_effect=AssertionError("hash() used")):
        service, _, _ = make()
        service.cache_key("some text")
        service.embed_texts(["x1", "y2", "x1"])


def test_keys_do_not_contain_the_input_text():
    service, _, cache = make()
    text = "Highly Distinctive Research Sentence About Quantum Widgets " * 20
    key = service.cache_key(text)
    assert "Quantum" not in key
    assert "Distinctive" not in key
    assert text not in key
    assert len(key) < 200
    service.embed_text(text)
    assert all("Quantum" not in stored_key for stored_key in cache.set_keys)


def test_keys_do_not_contain_api_keys_or_secrets():
    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimension=3,
        api_key=FAKE_SECRET,
    )
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                data=[
                    SimpleNamespace(index=i, embedding=[0.1, 0.2, 0.3])
                    for i in range(
                        len(kw["input"]) if isinstance(kw["input"], list) else 1
                    )
                ]
            )
        )
    )
    provider = OpenAIEmbeddingProvider(config, client=client)
    cache = RecordingCache()
    service = EmbeddingService(
        provider, cache, "openai:text-embedding-3-small:3"
    )
    service.embed_text("paper A")
    service.embed_texts(["paper B", "paper C"])
    assert cache.set_keys
    assert all(FAKE_SECRET not in key for key in cache.set_keys)
    assert FAKE_SECRET not in service.cache_key("paper A")


def test_different_namespaces_isolate_cached_embeddings():
    cache = RecordingCache()
    provider_a, provider_b = FakeProvider(tag=0.0), FakeProvider(tag=0.5)
    service_a = EmbeddingService(provider_a, cache, "model-a")
    service_b = EmbeddingService(provider_b, cache, "model-b")

    assert service_a.embed_text("paper A") == vec("paper A", 0.0)
    assert service_b.embed_text("paper A") == vec("paper A", 0.5)
    assert service_a.embed_text("paper A") == vec("paper A", 0.0)
    assert provider_a.single_calls == ["paper A"]
    assert provider_b.single_calls == ["paper A"]
    assert len(cache) == 2


def test_changing_namespace_prevents_cross_configuration_collisions_in_batches():
    cache = RecordingCache()
    old = EmbeddingService(FakeProvider(tag=0.0), cache, "model-v1")
    new_provider = FakeProvider(tag=0.5)
    new = EmbeddingService(new_provider, cache, "model-v2")
    texts = ["paper A", "paper B"]

    assert old.embed_texts(texts) == [vec(t, 0.0) for t in texts]
    assert new.embed_texts(texts) == [vec(t, 0.5) for t in texts]
    assert new_provider.batch_calls == [texts]
    assert old.cache_key("paper A") != new.cache_key("paper A")


def test_same_namespace_shares_the_cache_between_services():
    cache = RecordingCache()
    first_provider, second_provider = FakeProvider(), FakeProvider()
    EmbeddingService(first_provider, cache, "shared").embed_texts(
        ["paper A", "paper B"]
    )
    result = EmbeddingService(second_provider, cache, "shared").embed_texts(
        ["paper B", "paper A"]
    )
    assert result == [vec("paper B"), vec("paper A")]
    assert second_provider.total_calls == 0


def test_batch_with_all_cache_hits_does_not_call_provider():
    service, provider, cache = make()
    texts = ["paper A", "paper B", "paper C"]
    for text in texts:
        cache.inner.set(service.cache_key(text), vec(text, 0.25))
    assert service.embed_texts(texts) == [vec(t, 0.25) for t in texts]
    assert provider.total_calls == 0


def test_batch_with_no_cache_hits_calls_provider_exactly_once():
    service, provider, cache = make()
    texts = ["paper A", "paper B", "paper C"]
    assert service.embed_texts(texts) == [vec(t) for t in texts]
    assert provider.batch_calls == [texts]
    assert provider.single_calls == []
    assert len(cache.set_keys) == 3


def test_batch_with_partial_cache_hits_calls_provider_only_for_misses():
    service, provider, cache = make()
    cache.inner.set(service.cache_key("paper B"), vec("paper B", 0.25))
    texts = ["paper A", "paper B", "paper C", "paper D"]
    result = service.embed_texts(texts)
    assert provider.batch_calls == [["paper A", "paper C", "paper D"]]
    assert provider.single_calls == []
    assert result == [
        vec("paper A"),
        vec("paper B", 0.25),
        vec("paper C"),
        vec("paper D"),
    ]


def test_batch_misses_are_cached_under_their_own_keys():
    service, _, cache = make()
    texts = ["paper A", "paper B", "paper C"]
    service.embed_texts(texts)
    for text in texts:
        assert cache.inner.get(service.cache_key(text)) == vec(text)


@pytest.mark.parametrize(
    "cached_indexes",
    [[], [0], [1], [4], [0, 2, 4], [1, 3], [0, 1, 2, 3, 4]],
)
def test_batch_preserves_original_order(cached_indexes):
    service, provider, cache = make()
    texts = ["paper E", "paper D", "paper C", "paper B", "paper A"]
    for i in cached_indexes:
        cache.inner.set(service.cache_key(texts[i]), vec(texts[i], 0.25))
    result = service.embed_texts(texts)
    expected = [
        vec(t, 0.25 if i in cached_indexes else 0.0)
        for i, t in enumerate(texts)
    ]
    assert result == expected
    expected_misses = [
        t for i, t in enumerate(texts) if i not in cached_indexes
    ]
    assert provider.batch_calls == ([expected_misses] if expected_misses else [])


def test_batch_deduplicates_identical_missing_texts():
    service, provider, cache = make()
    texts = ["A x", "B x", "A x", "C x", "B x"]
    service.embed_texts(texts)
    assert provider.batch_calls == [["A x", "B x", "C x"]]
    assert len(cache.set_keys) == 3


def test_duplicate_texts_return_duplicate_vectors_in_correct_positions():
    service, _, _ = make()
    texts = ["A x", "B x", "A x", "C x", "B x"]
    assert service.embed_texts(texts) == [vec(t) for t in texts]


def test_duplicate_results_are_independent_lists():
    service, _, _ = make()
    result = service.embed_texts(["A x", "A x"])
    result[0][0] = 999.0
    assert result[1] == vec("A x")
    assert service.embed_text("A x") == vec("A x")


def test_cache_lookup_happens_once_per_unique_text():
    service, _, cache = make()
    service.embed_texts(["A x", "B x", "A x", "A x", "B x"])
    assert len(cache.get_keys) == 2


def test_second_batch_call_is_served_entirely_from_cache():
    service, provider, _ = make()
    texts = ["A x", "B x", "C x"]
    first = service.embed_texts(texts)
    second = service.embed_texts(texts)
    assert first == second
    assert len(provider.batch_calls) == 1


def test_batch_and_single_share_the_same_cache_entries():
    service, provider, _ = make()
    service.embed_texts(["A x", "B x"])
    assert service.embed_text("A x") == vec("A x")
    assert provider.single_calls == []
    service.embed_text("C x")
    assert service.embed_texts(["C x"]) == [vec("C x")]
    assert len(provider.batch_calls) == 1


def test_batch_does_not_modify_the_input_list_or_share_it_with_the_provider():
    service, provider, _ = make()
    texts = ["A x", "B x", "A x"]
    snapshot = list(texts)
    service.embed_texts(texts)
    assert texts == snapshot
    assert provider.batch_calls[0] is not texts


def test_empty_batch_returns_empty_list_without_any_calls():
    service, provider, cache = make()
    assert service.embed_texts([]) == []
    no_cache_service = EmbeddingService(provider, None)
    assert no_cache_service.embed_texts([]) == []
    assert provider.total_calls == 0
    assert cache.total_calls == 0


@pytest.mark.parametrize("returned", [0, 1, 2, 4])
def test_wrong_number_of_vectors_raises_clear_error(returned):
    provider = FakeProvider(
        batch_result=lambda texts: [[1.0, 2.0, 3.0]] * returned
    )
    service, _, cache = make(provider)
    with pytest.raises(
        RuntimeError,
        match=f"returned {returned} vectors for 3 unique texts",
    ):
        service.embed_texts(["A x", "B x", "C x"])
    assert cache.set_keys == []
    assert len(cache) == 0


@pytest.mark.parametrize("returned", [None, ([1.0, 2.0, 3.0],) * 3, "abc"])
def test_non_list_provider_result_raises_clear_error(returned):
    provider = FakeProvider(batch_result=lambda texts: returned)
    service, _, cache = make(provider)
    with pytest.raises(RuntimeError, match="instead of a list"):
        service.embed_texts(["A x", "B x", "C x"])
    assert len(cache) == 0


def test_count_check_uses_unique_misses_not_input_length():
    provider = FakeProvider(batch_result=lambda texts: [vec(t) for t in texts])
    service, _, _ = make(provider)
    texts = ["A x", "B x", "A x", "A x"]
    assert service.embed_texts(texts) == [vec(t) for t in texts]

    wrong = FakeProvider(
        batch_result=lambda texts: [vec(t) for t in texts] * 2
    )
    service2, _, cache2 = make(wrong)
    with pytest.raises(RuntimeError):
        service2.embed_texts(texts)
    assert len(cache2) == 0


def test_single_provider_failure_is_not_swallowed():
    original = ConnectionError("provider down")
    service, _, _ = make(FakeProvider(error=original))
    with pytest.raises(ConnectionError) as info:
        service.embed_text("paper A")
    assert info.value is original


def test_batch_provider_failure_is_not_swallowed():
    original = TimeoutError("timed out")
    service, _, _ = make(FakeProvider(error=original))
    with pytest.raises(TimeoutError) as info:
        service.embed_texts(["A x", "B x"])
    assert info.value is original


def test_failed_provider_results_are_not_cached_and_retry_works():
    provider = FakeProvider(error=ConnectionError("down"))
    service, _, cache = make(provider)
    with pytest.raises(ConnectionError):
        service.embed_text("paper A")
    with pytest.raises(ConnectionError):
        service.embed_texts(["A x", "B x"])
    assert cache.set_keys == []
    assert len(cache) == 0

    provider.error = None
    assert service.embed_text("paper A") == vec("paper A")
    assert service.embed_texts(["A x", "B x"]) == [
        vec("A x"),
        vec("B x"),
    ]


def test_hits_are_returned_even_if_provider_would_fail_when_all_cached():
    provider = FakeProvider(error=ConnectionError("provider down"))
    service, _, cache = make(provider)

    texts = ["A x", "B x"]
    for text in texts:
        cache.inner.set(service.cache_key(text), vec(text, 0.25))

    assert service.embed_texts(texts) == [
        vec("A x", 0.25),
        vec("B x", 0.25),
    ]
    assert provider.total_calls == 0


def test_partial_cache_hits_still_propagate_provider_failure():
    provider = FakeProvider(error=ConnectionError("provider down"))
    service, _, cache = make(provider)

    cache.inner.set(service.cache_key("A x"), vec("A x", 0.25))

    with pytest.raises(ConnectionError, match="provider down"):
        service.embed_texts(["A x", "B x"])

    assert cache.inner.get(service.cache_key("A x")) == vec("A x", 0.25)
    assert cache.inner.get(service.cache_key("B x")) is None


def test_no_network_is_used():
    provider = FakeProvider()
    service, _, _ = make(provider)

    with no_network():
        assert service.embed_text("paper A") == vec("paper A")
        assert service.embed_texts(["A x", "B x"]) == [
            vec("A x"),
            vec("B x"),
        ]


def test_invalid_single_text_follows_validation_contract():
    service, provider, cache = make()

    with pytest.raises(TypeError, match="text must be a str"):
        service.embed_text(123)

    with pytest.raises(ValueError, match="text must not be empty"):
        service.embed_text("   ")

    assert provider.total_calls == 0
    assert cache.total_calls == 0


@pytest.mark.parametrize("bad_texts", [None, "not-a-list", [1], [""] , ["   "]])
def test_invalid_batch_input_follows_validation_contract(bad_texts):
    service, provider, cache = make()

    with pytest.raises((TypeError, ValueError)):
        service.embed_texts(bad_texts)

    assert provider.total_calls == 0
    assert cache.total_calls == 0
