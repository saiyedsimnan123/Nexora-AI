import dataclasses
import os
from unittest import mock

import pytest

from nexora.embedding import EmbeddingConfig
from nexora.embedding.config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_BATCH_SIZE,
    ENV_DIMENSION,
    ENV_MODEL,
    ENV_PROVIDER,
)

FAKE_SECRET = "test-secret-not-a-real-key-123"


def test_default_configuration():
    config = EmbeddingConfig()
    assert config.provider == DEFAULT_PROVIDER
    assert config.model == DEFAULT_MODEL
    assert config.api_key is None
    assert config.base_url is None
    assert config.dimension is None
    assert config.batch_size is None
    assert config.has_api_key is False


def test_custom_provider():
    assert EmbeddingConfig(provider="local").provider == "local"


def test_custom_model():
    assert EmbeddingConfig(model="my-model-v1").model == "my-model-v1"


def test_custom_base_url():
    config = EmbeddingConfig(base_url="http://localhost:8080/v1")
    assert config.base_url == "http://localhost:8080/v1"


def test_optional_api_key():
    assert EmbeddingConfig().api_key is None
    config = EmbeddingConfig(api_key=FAKE_SECRET)
    assert config.api_key == FAKE_SECRET
    assert config.has_api_key is True


def test_api_key_is_not_exposed_in_repr_or_str():
    config = EmbeddingConfig(
        provider="api", model="m", api_key=FAKE_SECRET, dimension=8
    )
    assert FAKE_SECRET not in repr(config)
    assert FAKE_SECRET not in str(config)
    assert "api_key" not in repr(config)
    assert "provider='api'" in repr(config)


def test_api_key_loaded_from_env_is_not_exposed_in_repr():
    config = EmbeddingConfig.from_env({ENV_API_KEY: FAKE_SECRET})
    assert config.api_key == FAKE_SECRET
    assert FAKE_SECRET not in repr(config)


@pytest.mark.parametrize("bad_api_key", ["", "   "])
def test_blank_api_key_raises_value_error(bad_api_key):
    with pytest.raises(ValueError, match="api_key"):
        EmbeddingConfig(api_key=bad_api_key)


def test_non_string_api_key_raises_type_error():
    with pytest.raises(TypeError, match="api_key"):
        EmbeddingConfig(api_key=12345)


@pytest.mark.parametrize("bad_dimension", [0, -1, -768])
def test_non_positive_dimension_raises_value_error(bad_dimension):
    with pytest.raises(ValueError, match="dimension must be positive"):
        EmbeddingConfig(dimension=bad_dimension)


@pytest.mark.parametrize("bad_dimension", [1.5, "384", True])
def test_non_int_dimension_raises_type_error(bad_dimension):
    with pytest.raises(TypeError, match="dimension must be an int"):
        EmbeddingConfig(dimension=bad_dimension)


@pytest.mark.parametrize("bad_batch", [0, -1, -32])
def test_non_positive_batch_size_raises_value_error(bad_batch):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        EmbeddingConfig(batch_size=bad_batch)


@pytest.mark.parametrize("bad_batch", [2.5, "32", False])
def test_non_int_batch_size_raises_type_error(bad_batch):
    with pytest.raises(TypeError, match="batch_size must be an int"):
        EmbeddingConfig(batch_size=bad_batch)


@pytest.mark.parametrize("bad_provider", ["", "   ", "\t\n"])
def test_empty_provider_raises_value_error(bad_provider):
    with pytest.raises(ValueError, match="provider"):
        EmbeddingConfig(provider=bad_provider)


@pytest.mark.parametrize("bad_model", ["", "   ", "\t\n"])
def test_empty_model_raises_value_error(bad_model):
    with pytest.raises(ValueError, match="model"):
        EmbeddingConfig(model=bad_model)


@pytest.mark.parametrize("bad_value", [None, 123])
def test_non_string_provider_and_model_raise_type_error(bad_value):
    with pytest.raises(TypeError, match="provider"):
        EmbeddingConfig(provider=bad_value)
    with pytest.raises(TypeError, match="model"):
        EmbeddingConfig(model=bad_value)


@pytest.mark.parametrize(
    "bad_url",
    ["not a url", "localhost:8080", "ftp://example.com", "http://", "https:///path", "http://exa mple.com"],
)
def test_invalid_base_url_raises_value_error(bad_url):
    with pytest.raises(ValueError, match="base_url"):
        EmbeddingConfig(base_url=bad_url)


def test_non_string_base_url_raises_type_error():
    with pytest.raises(TypeError, match="base_url"):
        EmbeddingConfig(base_url=8080)


@pytest.mark.parametrize(
    "url",
    ["https://api.example.com", "http://localhost:8000", "https://host.example/v1/embeddings"],
)
def test_valid_base_urls_are_accepted(url):
    assert EmbeddingConfig(base_url=url).base_url == url


def test_valid_full_configuration():
    config = EmbeddingConfig(
        provider="self-hosted",
        model="embed-large",
        api_key=FAKE_SECRET,
        base_url="https://embeddings.internal.example/v1",
        dimension=1024,
        batch_size=64,
    )
    assert config.provider == "self-hosted"
    assert config.model == "embed-large"
    assert config.dimension == 1024
    assert config.batch_size == 64
    assert config.has_api_key is True


def test_configuration_is_immutable():
    config = EmbeddingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.provider = "other"


def test_equal_configs_are_equal():
    assert EmbeddingConfig(provider="a", model="b") == EmbeddingConfig(provider="a", model="b")


def test_environment_variable_loading():
    environ = {
        ENV_PROVIDER: "api",
        ENV_MODEL: "embed-small",
        ENV_API_KEY: FAKE_SECRET,
        ENV_BASE_URL: "https://api.example.com/v1",
        ENV_DIMENSION: "256",
        ENV_BATCH_SIZE: "16",
    }
    config = EmbeddingConfig.from_env(environ)
    assert config == EmbeddingConfig(
        provider="api",
        model="embed-small",
        api_key=FAKE_SECRET,
        base_url="https://api.example.com/v1",
        dimension=256,
        batch_size=16,
    )


def test_missing_optional_environment_variables():
    config = EmbeddingConfig.from_env({})
    assert config == EmbeddingConfig()

    partial = EmbeddingConfig.from_env({ENV_PROVIDER: "local"})
    assert partial.provider == "local"
    assert partial.model == DEFAULT_MODEL
    assert partial.api_key is None
    assert partial.base_url is None
    assert partial.dimension is None
    assert partial.batch_size is None


def test_blank_environment_values_are_treated_as_missing():
    environ = {
        ENV_PROVIDER: "  ",
        ENV_MODEL: "",
        ENV_API_KEY: " ",
        ENV_BASE_URL: "",
        ENV_DIMENSION: "  ",
        ENV_BATCH_SIZE: "",
    }
    assert EmbeddingConfig.from_env(environ) == EmbeddingConfig()


def test_environment_values_are_stripped():
    config = EmbeddingConfig.from_env({ENV_PROVIDER: "  local  ", ENV_DIMENSION: " 64 "})
    assert config.provider == "local"
    assert config.dimension == 64


@pytest.mark.parametrize("name", [ENV_DIMENSION, ENV_BATCH_SIZE])
def test_non_integer_environment_number_raises_value_error(name):
    with pytest.raises(ValueError, match=name):
        EmbeddingConfig.from_env({name: "abc"})


def test_invalid_environment_value_fails_validation():
    with pytest.raises(ValueError, match="dimension must be positive"):
        EmbeddingConfig.from_env({ENV_DIMENSION: "0"})
    with pytest.raises(ValueError, match="base_url"):
        EmbeddingConfig.from_env({ENV_BASE_URL: "not a url"})


def test_from_env_reads_os_environ_by_default():
    with mock.patch.dict(os.environ, {ENV_PROVIDER: "local", ENV_MODEL: "m1"}, clear=True):
        config = EmbeddingConfig.from_env()
    assert config.provider == "local"
    assert config.model == "m1"
