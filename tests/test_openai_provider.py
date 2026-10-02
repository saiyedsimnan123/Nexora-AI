import math
import socket
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from nexora.embedding import EmbeddingConfig, EmbeddingProvider
from nexora.embedding.config import ENV_API_KEY
from nexora.embedding.openai_provider import (
    EmbeddingProviderError,
    OpenAIEmbeddingProvider,
    _create_openai_client,
)

FAKE_SECRET = "test-secret-not-a-real-key-123"
CREATE_CLIENT = "nexora.embedding.openai_provider._create_openai_client"


class FakeEmbeddings:
    """Stands in for ``client.embeddings``; records every create() call."""

    def __init__(self, vector=None, error=None, response=None):
        self.vector = [0.1, 0.2, 0.3] if vector is None else vector
        self.error = error
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.vector)])


class FakeClient:
    def __init__(self, **kwargs):
        self.embeddings = FakeEmbeddings(**kwargs)


def make_provider(vector=None, model="text-embedding-3-small", dimension=3, api_key=None, **client_kwargs):
    client = FakeClient(vector=vector, **client_kwargs)
    config = EmbeddingConfig(provider="openai", model=model, dimension=dimension, api_key=api_key)
    return OpenAIEmbeddingProvider(config, client=client), client


@contextmanager
def no_network():
    with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network used")):
        yield


def test_provider_can_be_created_with_injected_client():
    provider, _ = make_provider()
    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_provider_satisfies_embedding_provider_abstraction():
    provider, _ = make_provider()
    assert isinstance(provider, EmbeddingProvider)


def test_configured_model_is_used():
    provider, client = make_provider(model="text-embedding-3-large")
    provider.embed_text("hello")
    assert client.embeddings.calls[0]["model"] == "text-embedding-3-large"


def test_input_text_is_sent_unmodified():
    provider, client = make_provider()
    text = "  Eq. (3): y = wx + b \n α ≥ β  "
    provider.embed_text(text)
    assert client.embeddings.calls[0]["input"] == text


def test_float_encoding_is_requested():
    provider, client = make_provider()
    provider.embed_text("hello")
    assert client.embeddings.calls[0]["encoding_format"] == "float"


def test_dimension_is_sent_for_models_that_support_it():
    provider, client = make_provider(model="text-embedding-3-small", dimension=3)
    provider.embed_text("hello")
    assert client.embeddings.calls[0]["dimensions"] == 3


def test_dimension_is_not_sent_for_models_that_do_not_support_it():
    provider, client = make_provider(model="text-embedding-ada-002", dimension=3)
    provider.embed_text("hello")
    assert "dimensions" not in client.embeddings.calls[0]


def test_dimension_is_not_sent_when_not_configured():
    provider, client = make_provider(dimension=None)
    provider.embed_text("hello")
    assert "dimensions" not in client.embeddings.calls[0]


def test_returned_embedding_is_converted_to_list_of_floats():
    provider, _ = make_provider(vector=(1, 2, 3))
    result = provider.embed_text("hello")
    assert isinstance(result, list)
    assert result == [1.0, 2.0, 3.0]
    assert all(type(value) is float for value in result)


def test_correct_vector_length_is_accepted():
    provider, _ = make_provider(vector=[0.5] * 8, dimension=8)
    assert len(provider.embed_text("hello")) == 8


def test_length_is_not_checked_when_dimension_is_not_configured():
    provider, _ = make_provider(vector=[0.5] * 5, dimension=None)
    assert len(provider.embed_text("hello")) == 5


@pytest.mark.parametrize("vector", [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4]])
def test_wrong_vector_length_raises_provider_error(vector):
    provider, _ = make_provider(vector=vector, dimension=3)
    with pytest.raises(EmbeddingProviderError, match="expected 3"):
        provider.embed_text("hello")


@pytest.mark.parametrize("bad_text", ["", "   ", "\n", "\t", " \t\n "])
def test_empty_or_whitespace_text_raises_value_error(bad_text):
    provider, client = make_provider()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        provider.embed_text(bad_text)
    assert client.embeddings.calls == []


@pytest.mark.parametrize("bad_text", [None, 123, 4.5, ["text"], b"text", {"a": 1}])
def test_non_string_text_raises_type_error(bad_text):
    provider, client = make_provider()
    with pytest.raises(TypeError, match="text must be a str"):
        provider.embed_text(bad_text)
    assert client.embeddings.calls == []


def test_non_embedding_config_raises_type_error():
    with pytest.raises(TypeError, match="config must be an EmbeddingConfig"):
        OpenAIEmbeddingProvider({"model": "x"}, client=FakeClient())


def test_missing_api_key_raises_error_when_real_client_is_needed():
    config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
    with mock.patch(CREATE_CLIENT) as create_client:
        with pytest.raises(EmbeddingProviderError, match=ENV_API_KEY):
            OpenAIEmbeddingProvider(config)
    create_client.assert_not_called()


def test_injected_client_prevents_creation_of_real_client():
    config = EmbeddingConfig(provider="openai", model="m", api_key=FAKE_SECRET)
    client = FakeClient()
    with mock.patch(CREATE_CLIENT) as create_client:
        provider = OpenAIEmbeddingProvider(config, client=client)
        provider.embed_text("hello")
    create_client.assert_not_called()
    assert len(client.embeddings.calls) == 1


def test_injected_client_works_without_api_key():
    provider, _ = make_provider(api_key=None)
    assert provider.embed_text("hello") == [0.1, 0.2, 0.3]


def test_real_client_is_created_from_config_when_not_injected():
    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        api_key=FAKE_SECRET,
        base_url="https://gateway.example.com/v1",
        dimension=3,
    )
    fake_client = FakeClient()
    with mock.patch(CREATE_CLIENT, return_value=fake_client) as create_client:
        provider = OpenAIEmbeddingProvider(config)
    create_client.assert_called_once_with(FAKE_SECRET, "https://gateway.example.com/v1")
    assert provider.embed_text("hello") == [0.1, 0.2, 0.3]


def test_client_factory_passes_api_key_and_base_url_to_sdk():
    fake_sdk = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"openai": fake_sdk}):
        client = _create_openai_client(FAKE_SECRET, "https://gateway.example.com/v1")
    fake_sdk.OpenAI.assert_called_once_with(
        api_key=FAKE_SECRET, base_url="https://gateway.example.com/v1"
    )
    assert client is fake_sdk.OpenAI.return_value


def test_client_factory_omits_base_url_when_not_configured():
    fake_sdk = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"openai": fake_sdk}):
        _create_openai_client(FAKE_SECRET, None)
    fake_sdk.OpenAI.assert_called_once_with(api_key=FAKE_SECRET)


def test_missing_sdk_raises_provider_error_without_secret():
    with mock.patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(EmbeddingProviderError, match="pip install openai") as info:
            _create_openai_client(FAKE_SECRET, None)
    assert FAKE_SECRET not in str(info.value)
    assert isinstance(info.value.__cause__, ImportError)


def test_client_failure_is_wrapped_in_provider_error():
    provider, _ = make_provider(error=ConnectionError("boom"))
    with pytest.raises(EmbeddingProviderError, match="request failed"):
        provider.embed_text("hello")


def test_original_exception_is_preserved_as_cause():
    original = TimeoutError("timed out")
    provider, _ = make_provider(error=original)
    with pytest.raises(EmbeddingProviderError) as info:
        provider.embed_text("hello")
    assert info.value.__cause__ is original


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(bad_value):
    provider, _ = make_provider(vector=[0.1, bad_value, 0.3])
    with pytest.raises(EmbeddingProviderError, match="non-finite"):
        provider.embed_text("hello")


@pytest.mark.parametrize("bad_value", ["0.5", None, [0.5], True, b"1"])
def test_non_numeric_values_are_rejected(bad_value):
    provider, _ = make_provider(vector=[0.1, bad_value, 0.3])
    with pytest.raises(EmbeddingProviderError, match="non-numeric"):
        provider.embed_text("hello")


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(),
        SimpleNamespace(data=[]),
        SimpleNamespace(data=None),
        SimpleNamespace(data=[SimpleNamespace()]),
    ],
)
def test_malformed_response_raises_provider_error(response):
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError, match="did not contain an embedding"):
        provider.embed_text("hello")


@pytest.mark.parametrize("embedding", ["abc", b"abc", 5, None])
def test_non_sequence_embedding_is_rejected(embedding):
    response = SimpleNamespace(data=[SimpleNamespace(embedding=embedding)])
    provider, _ = make_provider(response=response)
    with pytest.raises(EmbeddingProviderError):
        provider.embed_text("hello")


def test_empty_embedding_is_rejected():
    provider, _ = make_provider(vector=[], dimension=None)
    with pytest.raises(EmbeddingProviderError, match="empty embedding"):
        provider.embed_text("hello")


def test_api_key_never_appears_in_error_messages():
    leaky_error = RuntimeError(f"401 Unauthorized: bad key {FAKE_SECRET}")
    failures = [
        make_provider(api_key=FAKE_SECRET, error=leaky_error)[0],
        make_provider(api_key=FAKE_SECRET, vector=[0.1], dimension=3)[0],
        make_provider(api_key=FAKE_SECRET, vector=[math.nan, 0.1, 0.2])[0],
        make_provider(api_key=FAKE_SECRET, vector=["x", 0.1, 0.2])[0],
        make_provider(api_key=FAKE_SECRET, response=SimpleNamespace())[0],
    ]
    for provider in failures:
        with pytest.raises(EmbeddingProviderError) as info:
            provider.embed_text("hello")
        assert FAKE_SECRET not in str(info.value)
        assert FAKE_SECRET not in repr(info.value)


def test_api_key_is_not_exposed_by_the_provider_repr():
    provider, _ = make_provider(api_key=FAKE_SECRET)
    assert FAKE_SECRET not in repr(provider)
    assert FAKE_SECRET not in str(provider)


def test_missing_key_error_does_not_contain_secrets():
    config = EmbeddingConfig(provider="openai", model="m")
    with pytest.raises(EmbeddingProviderError) as info:
        OpenAIEmbeddingProvider(config)
    assert FAKE_SECRET not in str(info.value)


def test_no_network_access_is_required():
    with no_network():
        provider, _ = make_provider()
        assert provider.embed_text("hello") == [0.1, 0.2, 0.3]
        with pytest.raises(ValueError):
            provider.embed_text("  ")
        bad_provider, _ = make_provider(vector=[0.1])
        with pytest.raises(EmbeddingProviderError):
            bad_provider.embed_text("hello")


def test_separate_instances_behave_consistently():
    first, _ = make_provider(vector=[0.25, 0.5, 0.75])
    second, _ = make_provider(vector=[0.25, 0.5, 0.75])
    assert first.embed_text("same text") == second.embed_text("same text")


def test_same_instance_is_repeatable():
    provider, client = make_provider()
    assert provider.embed_text("hello") == provider.embed_text("hello")
    assert len(client.embeddings.calls) == 2
