import dataclasses

import pytest

from nexora.llm.config import LLMConfig

KEY = "placeholder-key-123"


def test_defaults():
    c = LLMConfig(model="m")
    assert (c.provider, c.api_key, c.base_url, c.timeout) == ("openai", None, None, 30.0)
    assert c.temperature is None and c.max_output_tokens is None


def test_explicit_values_and_frozen():
    c = LLMConfig(model="m", provider="openai", api_key=KEY, base_url="https://x.example/v1",
                  timeout=5, temperature=0.2, max_output_tokens=100)
    assert c.timeout == 5 and c.temperature == 0.2 and c.max_output_tokens == 100
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.model = "other"


@pytest.mark.parametrize("kw,exc", [
    ({"model": ""}, ValueError), ({"model": "  "}, ValueError), ({"model": None}, TypeError),
    ({"provider": ""}, ValueError), ({"provider": 1}, TypeError),
    ({"timeout": 0}, ValueError), ({"timeout": -1}, ValueError),
    ({"timeout": float("inf")}, ValueError), ({"timeout": float("nan")}, ValueError),
    ({"timeout": True}, TypeError), ({"timeout": "5"}, TypeError),
    ({"temperature": 2.5}, ValueError), ({"temperature": -0.1}, ValueError),
    ({"temperature": float("nan")}, ValueError), ({"temperature": True}, TypeError),
    ({"max_output_tokens": 0}, ValueError), ({"max_output_tokens": -5}, ValueError),
    ({"max_output_tokens": True}, TypeError), ({"max_output_tokens": 1.5}, TypeError),
    ({"base_url": "ftp://x.example"}, ValueError), ({"base_url": "not a url"}, ValueError),
    ({"base_url": 5}, TypeError), ({"api_key": 5}, TypeError),
])
def test_invalid_values(kw, exc):
    base = {"model": "m"}
    base.update(kw)
    with pytest.raises(exc):
        LLMConfig(**base)


def test_boundary_values_accepted():
    assert LLMConfig(model="m", temperature=0).temperature == 0
    assert LLMConfig(model="m", temperature=2).temperature == 2
    assert LLMConfig(model="m", max_output_tokens=1).max_output_tokens == 1


def test_api_key_hidden_from_repr_and_errors():
    c = LLMConfig(model="m", api_key=KEY)
    assert KEY not in repr(c) and KEY not in str(c)
    with pytest.raises(ValueError) as info:
        LLMConfig(model="m", api_key=KEY, timeout=-1)
    assert KEY not in str(info.value)


def test_from_env_reads_all_variables():
    env = {
        "NEXORA_LLM_PROVIDER": "openai", "NEXORA_LLM_MODEL": "m1",
        "NEXORA_LLM_API_KEY": KEY, "NEXORA_LLM_BASE_URL": "http://localhost:8000",
        "NEXORA_LLM_TIMEOUT": "12.5", "NEXORA_LLM_TEMPERATURE": "0.7",
        "NEXORA_LLM_MAX_OUTPUT_TOKENS": "256",
    }
    c = LLMConfig.from_env(env)
    assert (c.model, c.api_key, c.base_url) == ("m1", KEY, "http://localhost:8000")
    assert (c.timeout, c.temperature, c.max_output_tokens) == (12.5, 0.7, 256)


def test_from_env_defaults_and_missing_model():
    assert LLMConfig.from_env({"NEXORA_LLM_MODEL": "m"}).provider == "openai"
    with pytest.raises(ValueError):
        LLMConfig.from_env({})
    with pytest.raises(ValueError):
        LLMConfig.from_env({"NEXORA_LLM_MODEL": "   "})


def test_from_env_invalid_numbers_do_not_leak_key():
    env = {"NEXORA_LLM_MODEL": "m", "NEXORA_LLM_API_KEY": KEY, "NEXORA_LLM_TIMEOUT": "abc"}
    with pytest.raises(ValueError) as info:
        LLMConfig.from_env(env)
    assert KEY not in str(info.value)
    with pytest.raises(ValueError):
        LLMConfig.from_env({"NEXORA_LLM_MODEL": "m", "NEXORA_LLM_MAX_OUTPUT_TOKENS": "1.5"})
