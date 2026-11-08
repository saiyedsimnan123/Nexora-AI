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


class FakeChatCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_chat_completions_api_supported_with_proper_conversion():
    chat_resp = SimpleNamespace(
        id="chatcmpl-123",
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="Chat completion answer"))
        ],
        usage=SimpleNamespace(
            prompt_tokens=15,
            completion_tokens=8,
            total_tokens=23,
        ),
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions(response=chat_resp))
    )
    provider = OpenAIProvider(
        LLMConfig(model="gpt-4o-mini", api_key=KEY, temperature=0.7, max_output_tokens=100),
        client=fake_client,
    )
    out = provider.generate(MSGS)

    assert isinstance(out, LLMResponse)
    assert out.text == "Chat completion answer"
    assert out.model == "gpt-4o-mini"
    assert out.usage == {"input_tokens": 15, "output_tokens": 8, "total_tokens": 23}
    assert out.raw == {"response_id": "chatcmpl-123"}

    (call,) = fake_client.chat.completions.calls
    assert call["model"] == "gpt-4o-mini"
    assert call["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 100


def test_reasoning_models_use_max_completion_tokens():
    chat_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Reasoning answer"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions(response=chat_resp))
    )
    provider = OpenAIProvider(
        LLMConfig(model="o1-mini", api_key=KEY, max_output_tokens=250),
        client=fake_client,
    )
    provider.generate(MSGS)
    call = fake_client.chat.completions.calls[0]
    assert "max_completion_tokens" in call
    assert call["max_completion_tokens"] == 250
    assert "max_tokens" not in call


def test_authentication_and_ratelimit_error_wrapping_without_leaking_key():
    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    # 1. Auth error
    auth_err = AuthenticationError(f"Incorrect API key {KEY} provided")
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions(error=auth_err))
    )
    provider = OpenAIProvider(LLMConfig(model="m", api_key=KEY), client=fake_client)
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate(MSGS)
    assert KEY not in str(exc_info.value)
    assert "OpenAI authentication failed" in str(exc_info.value)

    # 2. Rate limit error
    rate_err = RateLimitError(f"Quota exceeded with key {KEY}")
    fake_client2 = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions(error=rate_err))
    )
    provider2 = OpenAIProvider(LLMConfig(model="m", api_key=KEY), client=fake_client2)
    with pytest.raises(LLMProviderError) as exc_info2:
        provider2.generate(MSGS)
    assert KEY not in str(exc_info2.value)
    assert "OpenAI rate limit or quota exceeded" in str(exc_info2.value)


def test_missing_credentials_raises_wrapped_llm_provider_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_LLM_API_KEY", raising=False)
    provider = OpenAIProvider(LLMConfig(model="gpt-4o", api_key=None))
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate(MSGS)
    assert "OpenAI request failed (OpenAIError)" in str(exc_info.value)
    assert "Missing credentials" in str(exc_info.value)
