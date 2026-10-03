from types import SimpleNamespace

import pytest

from nexora.llm.base import LLMProviderError
from nexora.llm.config import LLMConfig
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.openai_provider import OpenAIProvider

KEY = "placeholder-key-123"
MSGS = [LLMMessage(role="system", content="be brief"), LLMMessage(role="user", content="hi")]


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def fake_client(response=None, error=None):
    return SimpleNamespace(responses=FakeResponses(response, error))


def good_response(**over):
    base = dict(
        output_text="hello there", model="served-model", id="resp_1",
        usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
    )
    base.update(over)
    return SimpleNamespace(**base)


def make(config=None, **client_kw):
    client = fake_client(**client_kw)
    return OpenAIProvider(config or LLMConfig(model="cfg-model", api_key=KEY), client=client), client


def test_construction_validates_config_and_makes_no_client():
    with pytest.raises(TypeError):
        OpenAIProvider({"model": "m"})
    assert OpenAIProvider(LLMConfig(model="m"))._client is None


def test_injected_client_is_used_with_responses_api():
    provider, client = make(response=good_response())
    provider.generate(MSGS)
    (call,) = client.responses.calls
    assert call["model"] == "cfg-model"
    assert call["input"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert "temperature" not in call and "max_output_tokens" not in call


def test_config_defaults_and_overrides():
    cfg = LLMConfig(model="cfg-model", api_key=KEY, temperature=0.3, max_output_tokens=50)
    provider, client = make(cfg, response=good_response())
    provider.generate(MSGS)
    provider.generate(MSGS, model="other", temperature=0.9, max_output_tokens=7)
    first, second = client.responses.calls
    assert (first["temperature"], first["max_output_tokens"]) == (0.3, 50)
    assert (second["model"], second["temperature"], second["max_output_tokens"]) == ("other", 0.9, 7)


def test_zero_temperature_override_is_forwarded():
    provider, client = make(LLMConfig(model="m", temperature=0.8), response=good_response())
    provider.generate(MSGS, temperature=0)
    assert client.responses.calls[0]["temperature"] == 0


def test_response_and_usage_conversion():
    provider, _ = make(response=good_response())
    out = provider.generate(MSGS)
    assert isinstance(out, LLMResponse)
    assert out.text == "hello there" and out.model == "served-model"
    assert out.usage == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    assert out.raw == {"response_id": "resp_1"}


def test_missing_optional_fields_handled():
    provider, _ = make(response=SimpleNamespace(output_text="x"))
    out = provider.generate(MSGS)
    assert out.model == "cfg-model" and out.usage is None and out.raw is None
    partial = good_response(usage=SimpleNamespace(input_tokens=2))
    out = make(response=partial)[0].generate(MSGS)
    assert out.usage == {"input_tokens": 2}


def test_missing_output_text_is_provider_error():
    provider, _ = make(response=SimpleNamespace(output_text=None))
    with pytest.raises(LLMProviderError):
        provider.generate(MSGS)


def test_api_errors_wrapped_without_leaking_key():
    err = ConnectionError(f"failed with key {KEY}")
    provider, _ = make(error=err)
    with pytest.raises(LLMProviderError) as info:
        provider.generate(MSGS)
    assert KEY not in str(info.value)
    assert info.value.__cause__ is err
    assert "ConnectionError" in str(info.value)


@pytest.mark.parametrize("messages,exc", [
    ("hi", TypeError), (None, TypeError), (tuple(MSGS), TypeError),
    ([], ValueError), (["hi"], TypeError),
    ([{"role": "user", "content": "x"}], TypeError),
    ([MSGS[0], object()], TypeError),
])
def test_invalid_messages_rejected_before_api_call(messages, exc):
    provider, client = make(response=good_response())
    with pytest.raises(exc):
        provider.generate(messages)
    assert client.responses.calls == []


@pytest.mark.parametrize("kw,exc", [
    ({"model": ""}, ValueError), ({"model": "   "}, ValueError), ({"model": 5}, TypeError),
    ({"temperature": 2.5}, ValueError), ({"temperature": -1}, ValueError),
    ({"temperature": float("nan")}, ValueError), ({"temperature": True}, TypeError),
    ({"temperature": "0.5"}, TypeError),
    ({"max_output_tokens": 0}, ValueError), ({"max_output_tokens": -3}, ValueError),
    ({"max_output_tokens": True}, TypeError), ({"max_output_tokens": 1.5}, TypeError),
])
def test_invalid_options_rejected_before_api_call(kw, exc):
    provider, client = make(response=good_response())
    with pytest.raises(exc):
        provider.generate(MSGS, **kw)
    assert client.responses.calls == []


def test_empty_model_does_not_fall_back_to_config_model():
    provider, client = make(response=good_response())
    with pytest.raises(ValueError):
        provider.generate(MSGS, model="")
    provider.generate(MSGS, model=None)
    assert client.responses.calls[0]["model"] == "cfg-model"
