import pytest

from nexora.llm.base import LLMProvider
from nexora.llm.config import LLMConfig
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.openai_provider import OpenAIProvider
from nexora.llm.service import LLMService, create_llm_provider


class FakeProvider:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.response = LLMResponse(text="answer", model="fake", usage={"total_tokens": 5})

    def generate(self, messages, *, model=None, temperature=None, max_output_tokens=None):
        self.calls.append((messages, model, temperature, max_output_tokens))
        if self.error:
            raise self.error
        return self.response


MSGS = [LLMMessage(role="system", content="s"), LLMMessage(role="user", content="u")]


def test_provider_protocol_runtime_check():
    assert isinstance(FakeProvider(), LLMProvider)
    assert not isinstance(object(), LLMProvider)


def test_service_rejects_bad_provider():
    with pytest.raises(TypeError):
        LLMService(object())


def test_generate_forwards_messages_and_returns_response_unchanged():
    provider = FakeProvider()
    result = LLMService(provider).generate(MSGS)
    assert result is provider.response
    assert provider.calls == [(MSGS, None, None, None)]


def test_optional_parameters_forwarded():
    provider = FakeProvider()
    LLMService(provider).generate(MSGS, model="m2", temperature=0.5, max_output_tokens=64)
    assert provider.calls[0][1:] == ("m2", 0.5, 64)


@pytest.mark.parametrize("messages,exc", [
    ("hi", TypeError), (None, TypeError), (tuple(MSGS), TypeError), ([], ValueError),
    (["hi"], TypeError), ([{"role": "user", "content": "x"}], TypeError),
])
def test_invalid_messages(messages, exc):
    provider = FakeProvider()
    with pytest.raises(exc):
        LLMService(provider).generate(messages)
    assert provider.calls == []


@pytest.mark.parametrize("kw,exc", [
    ({"model": ""}, ValueError), ({"model": 3}, TypeError),
    ({"temperature": 3.0}, ValueError), ({"temperature": True}, TypeError),
    ({"max_output_tokens": 0}, ValueError), ({"max_output_tokens": True}, TypeError),
    ({"max_output_tokens": 2.5}, TypeError),
])
def test_invalid_options(kw, exc):
    provider = FakeProvider()
    with pytest.raises(exc):
        LLMService(provider).generate(MSGS, **kw)
    assert provider.calls == []


def test_provider_errors_propagate_unchanged():
    err = RuntimeError("down")
    with pytest.raises(RuntimeError) as info:
        LLMService(FakeProvider(error=err)).generate(MSGS)
    assert info.value is err


def test_factory_creates_openai_provider_without_network():
    provider = create_llm_provider(LLMConfig(model="m", api_key="placeholder-key-123"))
    assert isinstance(provider, OpenAIProvider)
    assert isinstance(provider, LLMProvider)


def test_factory_rejects_unsupported_provider_and_bad_config():
    with pytest.raises(ValueError):
        create_llm_provider(LLMConfig(model="m", provider="mystery"))
    with pytest.raises(TypeError):
        create_llm_provider({"model": "m"})
