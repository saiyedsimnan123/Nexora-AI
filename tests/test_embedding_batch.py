import math
import socket
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from nexora.embedding import EmbeddingConfig, EmbeddingProvider, HashEmbeddingProvider
from nexora.embedding.openai_provider import EmbeddingProviderError, OpenAIEmbeddingProvider

FAKE_SECRET = "test-secret-not-a-real-key-123"
DIM = 3


def vec(text):
    """Deterministic fake embedding for a text (distinct for distinct texts)."""
    return [float(len(text)), float(ord(text[0])), 1.0]


def item(index, vector):
    return SimpleNamespace(index=index, embedding=vector)


class FakeEmbeddings:
    """Fake ``client.embeddings``; records calls and builds deterministic replies."""

    def __init__(self, reverse=False, response=None, fail_on_call=None, error=None):
        self.calls = []
        self.reverse = reverse
        self.response = response
        self.fail_on_call = fail_on_call
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise self.error
        if self.response is not None:
            return self.response
        texts = kwargs["input"]
        items = [item(i, vec(t)) for i, t in enumerate(texts)]
        if self.reverse:
            items.reverse()
        return SimpleNamespace(data=items)


class FakeClient:
    def __init__(self, **kwargs):
        self.embeddings = FakeEmbeddings(**kwargs)


def make_provider(
    model="text-embedding-3-small", dimension=DIM, batch_size=None, api_key=None, **client_kwargs
):
    client = FakeClient(**client_kwargs)
    config = EmbeddingConfig(
        provider="openai", model=model, dimension=dimension,
        batch_size=batch_size, api_key=api_key,
    )
    return OpenAIEmbeddingProvider(config, client=client), client


def texts_of(n):
    return [f"paper {chr(65 + i % 26)}{i}" for i in range(n)]


def response_with(*items):
    return SimpleNamespace(data=list(items))


@contextmanager
def no_network():
    with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network used")):
        yield


# ---------------------------------------------------------------- interface

def test_both_providers_satisfy_the_batch_interface():
    assert isinstance(HashEmbeddingProvider(), EmbeddingProvider)
    assert isinstance(make_provider()[0], EmbeddingProvider)


def test_object_with_only_embed_text_is_no_longer_a_full_provider():
    class SingleOnly:
        def embed_text(self, text):
            return [1.0]

    assert not isinstance(SingleOnly(), EmbeddingProvider)


# ------------------------------------------------------------ hash provider

def test_hash_empty_list_returns_empty_list():
    assert HashEmbeddingProvider().embed_texts([]) == []


def test_hash_one_text_works():
    provider = HashEmbeddingProvider(dimension=16)
    assert provider.embed_texts(["only"]) == [provider.embed_text("only")]


def test_hash_batch_matches_single_embeddings_in_order():
    provider = HashEmbeddingProvider(dimension=16)
    texts = ["paper A", "paper B", "paper C", "paper A"]
    result = provider.embed_texts(texts)
    assert result == [provider.embed_text(t) for t in texts]
    assert len(result) == len(texts)
    assert result[0] == result[3]
    assert result[0] != result[1]


def test_hash_batch_is_deterministic_across_instances():
    texts = ["x1", "x2", "x3"]
    assert HashEmbeddingProvider(8).embed_texts(texts) == HashEmbeddingProvider(8).embed_texts(texts)


def test_hash_single_text_behavior_is_unchanged():
    expected = [0.28432901419429385, 0.8295271989233097, -0.478667309717539, 0.04380918330124561]
    vector = HashEmbeddingProvider(dimension=4).embed_text("nexora")
    assert vector == pytest.approx(expected, abs=1e-12)


def test_hash_batch_does_not_modify_input_list():
    texts = ["a b", "c d"]
    snapshot = list(texts)
    HashEmbeddingProvider().embed_texts(texts)
    assert texts == snapshot


# ---------------------------------------------------- shared input validation

def _providers():
    return [HashEmbeddingProvider(dimension=DIM), make_provider()[0]]


@pytest.mark.parametrize("bad_input", [None, "paper", 123, ("a", "b"), {"a"}, {"a": 1}, b"ab"])
def test_non_list_input_raises_type_error(bad_input):
    for provider in _providers():
        with pytest.raises(TypeError, match="texts must be a list"):
            provider.embed_texts(bad_input)


@pytest.mark.parametrize("bad_element", [None, 123, 4.5, ["x"], b"x"])
def test_non_string_element_raises_type_error(bad_element):
    for provider in _providers():
        with pytest.raises(TypeError, match=r"texts\[1\] must be a str"):
            provider.embed_texts(["ok", bad_element])


@pytest.mark.parametrize("bad_text", ["", "   ", "\n", "\t", " \t\n "])
def test_empty_or_whitespace_element_raises_value_error(bad_text):
    for provider in _providers():
        with pytest.raises(ValueError, match=r"texts\[2\] must not be empty"):
            provider.embed_texts(["ok", "fine", bad_text])


def test_invalid_input_makes_no_api_request():
    provider, client = make_provider()
    for bad in [None, ["ok", 5], ["ok", "  "]]:
        with pytest.raises((TypeError, ValueError)):
            provider.embed_texts(bad)
    assert client.embeddings.calls == []


# ---------------------------------------------------------- openai: basics

def test_openai_empty_list_returns_empty_list_without_request():
    provider, client = make_provider()
    assert provider.embed_texts([]) == []
    assert client.embeddings.calls == []


def test_openai_one_text_works():
    provider, client = make_provider()
    assert provider.embed_texts(["paper A"]) == [vec("paper A")]
    assert client.embeddings.calls[0]["input"] == ["paper A"]


def test_openai_multiple_texts_work_in_one_request():
    provider, client = make_provider()
    texts = ["paper A", "paper B", "paper C"]
    assert provider.embed_texts(texts) == [vec(t) for t in texts]
    assert len(client.embeddings.calls) == 1
    call = client.embeddings.calls[0]
    assert call["input"] == texts
    assert call["model"] == "text-embedding-3-small"
    assert call["encoding_format"] == "float"
    assert call["dimensions"] == DIM


def test_openai_does_not_call_embed_text_per_item():
    provider, client = make_provider()
    provider.embed_texts(texts_of(5))
    assert len(client.embeddings.calls) == 1
    assert isinstance(client.embeddings.calls[0]["input"], list)


def test_openai_texts_are_sent_unmodified_and_input_list_not_mutated():
    provider, client = make_provider()
    texts = ["  padded A \n", "Eq. (3): y = wx + b", "α ≥ β"]
    snapshot = list(texts)
    provider.embed_texts(texts)
    assert client.embeddings.calls[0]["input"] == snapshot
    assert texts == snapshot


def test_openai_batch_dimensions_not_sent_for_unsupported_model():
    provider, client = make_provider(model="text-embedding-ada-002")
    provider.embed_texts(["paper A", "paper B"])
    assert "dimensions" not in client.embeddings.calls[0]


def test_openai_results_are_float_lists():
    provider, _ = make_provider(response=response_with(item(0, (1, 2, 3)), item(1, [4, 5, 6])))
    result = provider.embed_texts(["a", "b"])
    assert result == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert all(isinstance(v, list) and all(type(x) is float for x in v) for v in result)


def test_existing_single_text_embedding_still_works():
    provider, client = make_provider(response=response_with(item(0, [0.1, 0.2, 0.3])))
    assert provider.embed_text("hello") == [0.1, 0.2, 0.3]
    assert client.embeddings.calls[0]["input"] == "hello"


# -------------------------------------------------------------- batch_size

def test_batch_size_splits_requests_20_by_8():
    provider, client = make_provider(batch_size=8)
    texts = texts_of(20)
    result = provider.embed_texts(texts)
    assert [len(c["input"]) for c in client.embeddings.calls] == [8, 8, 4]
    assert result == [vec(t) for t in texts]
    assert len(result) == 20


def test_none_batch_size_sends_one_request():
    provider, client = make_provider(batch_size=None)
    provider.embed_texts(texts_of(50))
    assert len(client.embeddings.calls) == 1
    assert len(client.embeddings.calls[0]["input"]) == 50


@pytest.mark.parametrize(
    "count, batch_size, expected_sizes",
    [(5, 1, [1, 1, 1, 1, 1]), (6, 3, [3, 3]), (3, 10, [3]), (7, 7, [7]), (1, 4, [1])],
)
def test_batch_size_edge_cases(count, batch_size, expected_sizes):
    provider, client = make_provider(batch_size=batch_size)
    texts = texts_of(count)
    assert provider.embed_texts(texts) == [vec(t) for t in texts]
    assert [len(c["input"]) for c in client.embeddings.calls] == expected_sizes


def test_batches_cover_every_text_exactly_once_in_order():
    provider, client = make_provider(batch_size=4)
    texts = texts_of(10)
    provider.embed_texts(texts)
    sent = [t for call in client.embeddings.calls for t in call["input"]]
    assert sent == texts


def test_global_order_preserved_across_requests_with_reversed_responses():
    provider, client = make_provider(batch_size=3, reverse=True)
    texts = texts_of(8)
    assert provider.embed_texts(texts) == [vec(t) for t in texts]
    assert len(client.embeddings.calls) == 3


def test_batch_size_one_does_not_trigger_per_text_string_input():
    provider, client = make_provider(batch_size=1)
    provider.embed_texts(["a1", "b2"])
    assert all(isinstance(c["input"], list) for c in client.embeddings.calls)


def test_failure_in_later_batch_is_wrapped_and_returns_no_partial_result():
    original = TimeoutError("timed out")
    provider, client = make_provider(batch_size=2, fail_on_call=2, error=original)
    with pytest.raises(EmbeddingProviderError, match="request failed") as info:
        provider.embed_texts(texts_of(5))
    assert info.value.__cause__ is original
    assert len(client.embeddings.calls) == 2


# ---------------------------------------------------------- order / indexes

def test_returned_indexes_are_used_to_restore_input_order():
    a, b, c = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    response = response_with(item(2, c), item(0, a), item(1, b))
    provider, _ = make_provider(response=response)
    assert provider.embed_texts(["paper A", "paper B", "paper C"]) == [a, b, c]


def test_response_without_indexes_uses_response_order():
    response = response_with(
        SimpleNamespace(embedding=[1.0, 0.0, 0.0]), SimpleNamespace(embedding=[0.0, 1.0, 0.0])
    )
    provider, _ = make_provider(response=response)
    assert provider.embed_texts(["a", "b"]) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_response_with_all_none_indexes_uses_response_order():
    response = response_with(item(None, [1.0, 0.0, 0.0]), item(None, [0.0, 1.0, 0.0]))
    provider, _ = make_provider(response=response)
    assert provider.embed_texts(["a", "b"]) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


@pytest.mark.parametrize(
    "indexes",
    [[0, 0, 2], [0, 1, 1], [0, 1, 5], [1, 2, 3], [-1, 0, 1], [0, 1, 3]],
)
def test_duplicate_missing_or_out_of_range_indexes_are_rejected(indexes):
    response = response_with(*[item(i, [0.1, 0.2, 0.3]) for i in indexes])
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="duplicated, missing or out of range"):
        provider.embed_texts(["a", "b", "c"])


def test_partially_missing_indexes_are_rejected():
    response = response_with(item(0, [0.1, 0.2, 0.3]), item(None, [0.1, 0.2, 0.3]))
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="incomplete"):
        provider.embed_texts(["a", "b"])


@pytest.mark.parametrize("bad_index", ["0", 1.0, True])
def test_non_integer_indexes_are_rejected(bad_index):
    response = response_with(item(bad_index, [0.1, 0.2, 0.3]), item(1, [0.1, 0.2, 0.3]))
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="non-integer index"):
        provider.embed_texts(["a", "b"])


# ------------------------------------------------------ response validation

@pytest.mark.parametrize("returned", [0, 1, 2, 4])
def test_wrong_number_of_returned_embeddings_is_rejected(returned):
    response = response_with(*[item(i, [0.1, 0.2, 0.3]) for i in range(returned)])
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="embeddings for 3 inputs"):
        provider.embed_texts(["a", "b", "c"])


@pytest.mark.parametrize(
    "response",
    [SimpleNamespace(), SimpleNamespace(data=None), SimpleNamespace(data=5)],
)
def test_response_without_embeddings_is_rejected(response):
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="did not contain embeddings"):
        provider.embed_texts(["a", "b"])


def test_item_without_embedding_is_rejected():
    response = response_with(item(0, [0.1, 0.2, 0.3]), SimpleNamespace(index=1))
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="input 1 did not contain an embedding"):
        provider.embed_texts(["a", "b"])


@pytest.mark.parametrize("embedding", ["abc", b"abc", 5, None])
def test_non_sequence_embedding_is_rejected(embedding):
    response = response_with(item(0, [0.1, 0.2, 0.3]), item(1, embedding))
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="must be a sequence"):
        provider.embed_texts(["a", "b"])


def test_empty_embedding_is_rejected():
    response = response_with(item(0, []), item(1, [0.1, 0.2, 0.3]))
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="empty embedding"):
        provider.embed_texts(["a", "b"])


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(bad_value):
    response = response_with(item(0, [0.1, 0.2, 0.3]), item(1, [0.1, bad_value, 0.3]))
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="non-finite") as info:
        provider.embed_texts(["a", "b"])
    assert "input 1" in str(info.value)


@pytest.mark.parametrize("bad_value", ["0.5", None, [0.5], True, b"1"])
def test_non_numeric_values_are_rejected(bad_value):
    response = response_with(item(0, [0.1, 0.2, 0.3]), item(1, [0.1, bad_value, 0.3]))
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="non-numeric"):
        provider.embed_texts(["a", "b"])


@pytest.mark.parametrize("vector", [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4]])
def test_wrong_embedding_dimension_is_rejected(vector):
    response = response_with(item(0, [0.1, 0.2, 0.3]), item(1, vector))
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="expected 3"):
        provider.embed_texts(["a", "b"])


def test_wrong_dimension_in_second_batch_is_rejected():
    provider, client = make_provider(batch_size=2)
    original_create = client.embeddings.create

    def create(**kwargs):
        response = original_create(**kwargs)
        if len(client.embeddings.calls) == 2:
            response.data[0].embedding = [0.1]
        return response

    client.embeddings.create = create
    with pytest.raises(EmbeddingProviderError, match="expected 3"):
        provider.embed_texts(texts_of(4))


# ---------------------------------------------------------- safety / network

def test_client_failure_is_wrapped_with_cause():
    original = ConnectionError("boom")
    provider, _ = make_provider(fail_on_call=1, error=original)
    with pytest.raises(EmbeddingProviderError, match="request failed") as info:
        provider.embed_texts(["a", "b"])
    assert info.value.__cause__ is original


def test_api_key_never_appears_in_errors():
    leaky = RuntimeError(f"401 Unauthorized: bad key {FAKE_SECRET}")
    bad_responses = [
        response_with(item(0, [0.1, 0.2, 0.3])),
        response_with(item(0, [math.nan, 0.1, 0.2]), item(1, [0.1, 0.2, 0.3])),
        response_with(item(0, ["x", 0.1, 0.2]), item(1, [0.1, 0.2, 0.3])),
        response_with(item(0, [0.1]), item(1, [0.1, 0.2, 0.3])),
        response_with(item(0, [0.1, 0.2, 0.3]), item(0, [0.1, 0.2, 0.3])),
        SimpleNamespace(),
    ]
    providers = [make_provider(api_key=FAKE_SECRET, response=r)[0] for r in bad_responses]
    providers.append(make_provider(api_key=FAKE_SECRET, fail_on_call=1, error=leaky)[0])
    for provider in providers:
        with pytest.raises(EmbeddingProviderError) as info:
            provider.embed_texts(["a", "b"])
        assert FAKE_SECRET not in str(info.value)
        assert FAKE_SECRET not in repr(info.value)


def test_no_network_is_used_with_injected_client():
    with no_network():
        provider, client = make_provider(batch_size=2)
        texts = texts_of(5)
        assert provider.embed_texts(texts) == [vec(t) for t in texts]
        assert HashEmbeddingProvider().embed_texts(["a b", "c d"])
    assert len(client.embeddings.calls) == 3


def test_injected_client_is_reused_and_no_real_client_created():
    config = EmbeddingConfig(
        provider="openai", model="text-embedding-3-small", dimension=DIM,
        batch_size=2, api_key=FAKE_SECRET,
    )
    client = FakeClient()
    with mock.patch("nexora.embedding.openai_provider._create_openai_client") as create_client:
        provider = OpenAIEmbeddingProvider(config, client=client)
        provider.embed_texts(texts_of(5))
    create_client.assert_not_called()
    assert len(client.embeddings.calls) == 3


def test_separate_instances_behave_consistently():
    texts = texts_of(7)
    first, _ = make_provider(batch_size=3)
    second, _ = make_provider(batch_size=3)
    assert first.embed_texts(texts) == second.embed_texts(texts)
