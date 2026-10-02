"""Tests for the embedding provider factory (Step 10F).

No test makes a real network request or a real OpenAI call. The OpenAI
provider is replaced with a fake wherever it would be constructed.
"""

from __future__ import annotations

import dataclasses
import socket

import pytest

from nexora.embedding import factory
from nexora.embedding.config import EmbeddingConfig
from nexora.embedding.factory import create_embedding_provider
from nexora.embedding.hash_provider import HashEmbeddingProvider

SECRET = "sk-test-super-secret-key-123"


class FakeOpenAIProvider:
    """Records constructor arguments; never touches the network."""

    instances: list["FakeOpenAIProvider"] = []

    def __init__(self, config, client=None):
        self.config = config
        self.client = client
        FakeOpenAIProvider.instances.append(self)

    def embed_text(self, text):
        return [0.0]

    def embed_texts(self, texts):
        return [[0.0] for _ in texts]


class FakeHashProvider:
    """Records the keyword/positional arguments the factory passes."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def embed_text(self, text):
        return [0.0]

    def embed_texts(self, texts):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if anything in these tests tries to open a connection."""

    def _blocked(*args, **kwargs):
        raise AssertionError("Network access attempted during factory test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    FakeOpenAIProvider.instances.clear()


@pytest.fixture
def fake_openai(monkeypatch):
    monkeypatch.setattr(factory, "OpenAIEmbeddingProvider", FakeOpenAIProvider)
    return FakeOpenAIProvider


def _assert_provider_interface(obj):
    assert callable(getattr(obj, "embed_text", None))
    assert callable(getattr(obj, "embed_texts", None))


# 1. Hash provider creation
def test_hash_provider_created_with_defaults():
    provider = create_embedding_provider(EmbeddingConfig(provider="hash"))
    assert isinstance(provider, HashEmbeddingProvider)


def test_hash_provider_default_config_works():
    provider = create_embedding_provider(EmbeddingConfig())
    assert isinstance(provider, HashEmbeddingProvider)


def test_hash_provider_honors_configured_dimension():
    provider = create_embedding_provider(
        EmbeddingConfig(provider="hash", dimension=16)
    )
    assert len(provider.embed_text("hello")) == 16


def test_hash_provider_uses_own_default_dimension_when_none(monkeypatch):
    monkeypatch.setattr(factory, "HashEmbeddingProvider", FakeHashProvider)
    provider = create_embedding_provider(
        EmbeddingConfig(provider="hash", dimension=None)
    )
    assert provider.args == ()
    assert provider.kwargs == {}


def test_hash_provider_receives_configured_dimension(monkeypatch):
    monkeypatch.setattr(factory, "HashEmbeddingProvider", FakeHashProvider)
    provider = create_embedding_provider(
        EmbeddingConfig(provider="hash", dimension=32)
    )
    assert provider.kwargs == {"dimension": 32}


# 2. OpenAI provider creation
def test_openai_provider_created(fake_openai):
    config = EmbeddingConfig(provider="openai", model="m", api_key=SECRET)
    provider = create_embedding_provider(config)
    assert isinstance(provider, FakeOpenAIProvider)
    assert len(fake_openai.instances) == 1


# 3. Unsupported provider
@pytest.mark.parametrize("name", ["nope", "cohere", "hash2"])
def test_unsupported_provider_raises_value_error(name):
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        create_embedding_provider(EmbeddingConfig(provider=name))


def test_unsupported_provider_does_not_fall_back(fake_openai):
    with pytest.raises(ValueError):
        create_embedding_provider(EmbeddingConfig(provider="unknown"))
    assert fake_openai.instances == []


def test_unsupported_provider_error_lists_supported_names():
    with pytest.raises(ValueError) as exc_info:
        create_embedding_provider(EmbeddingConfig(provider="unknown"))
    message = str(exc_info.value)
    assert "hash" in message
    assert "openai" in message


# 4. Invalid config argument
@pytest.mark.parametrize(
    "bad", [None, "hash", {"provider": "hash"}, 123, object()]
)
def test_invalid_config_argument_raises_type_error(bad):
    with pytest.raises(TypeError, match="EmbeddingConfig"):
        create_embedding_provider(bad)


# 5. Returned object satisfies the EmbeddingProvider interface
def test_hash_result_satisfies_provider_interface():
    _assert_provider_interface(
        create_embedding_provider(EmbeddingConfig(provider="hash"))
    )


def test_openai_result_satisfies_provider_interface(fake_openai):
    _assert_provider_interface(
        create_embedding_provider(EmbeddingConfig(provider="openai"))
    )


def test_hash_provider_produces_vectors_through_factory():
    provider = create_embedding_provider(EmbeddingConfig(provider="hash"))
    vectors = provider.embed_texts(["a", "b"])
    assert len(vectors) == 2
    assert provider.embed_text("a") == vectors[0]


# 6. Supplied configuration is passed correctly
def test_openai_receives_the_same_config_object(fake_openai):
    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        api_key=SECRET,
        base_url="https://example.invalid/v1",
        dimension=256,
        batch_size=8,
    )
    provider = create_embedding_provider(config)
    assert provider.config is config
    assert provider.client is None


# 7. No real network request occurs
def test_factory_makes_no_network_calls_for_hash():
    # The autouse fixture blocks socket.connect; this must still succeed.
    create_embedding_provider(EmbeddingConfig(provider="hash"))


def test_factory_makes_no_network_calls_for_openai(fake_openai):
    create_embedding_provider(
        EmbeddingConfig(provider="openai", api_key=SECRET)
    )


# 8. API keys are not exposed
def test_api_key_not_in_unsupported_provider_error():
    config = EmbeddingConfig(provider="unknown", api_key=SECRET)
    with pytest.raises(ValueError) as exc_info:
        create_embedding_provider(config)
    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)


def test_api_key_not_in_invalid_config_error():
    with pytest.raises(TypeError) as exc_info:
        create_embedding_provider({"api_key": SECRET})
    assert SECRET not in str(exc_info.value)


def test_api_key_not_in_config_repr():
    config = EmbeddingConfig(provider="openai", api_key=SECRET)
    assert SECRET not in repr(config)
    assert SECRET not in str(config)


def test_api_key_not_logged(fake_openai, caplog):
    with caplog.at_level("DEBUG"):
        create_embedding_provider(
            EmbeddingConfig(provider="openai", api_key=SECRET)
        )
    assert SECRET not in caplog.text


# 9. Factory does not mutate the supplied configuration
@pytest.mark.parametrize("provider", ["hash", "openai"])
def test_factory_does_not_mutate_config(provider, fake_openai):
    config = EmbeddingConfig(
        provider=provider, model="m", api_key=SECRET, dimension=24
    )
    before = dataclasses.asdict(config)
    create_embedding_provider(config)
    assert dataclasses.asdict(config) == before


# 10. Provider construction errors propagate
def test_openai_construction_error_propagates(monkeypatch):
    class Boom:
        def __init__(self, config, client=None):
            raise RuntimeError("openai construction failed")

    monkeypatch.setattr(factory, "OpenAIEmbeddingProvider", Boom)
    with pytest.raises(RuntimeError, match="openai construction failed"):
        create_embedding_provider(EmbeddingConfig(provider="openai"))


def test_hash_construction_error_propagates(monkeypatch):
    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("hash construction failed")

    monkeypatch.setattr(factory, "HashEmbeddingProvider", Boom)
    with pytest.raises(RuntimeError, match="hash construction failed"):
        create_embedding_provider(EmbeddingConfig(provider="hash"))


def test_invalid_hash_dimension_error_propagates():
    with pytest.raises((ValueError, TypeError)):
        create_embedding_provider(
            EmbeddingConfig(provider="hash", dimension=-5)
        )
