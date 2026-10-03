import dataclasses

import pytest

from nexora.llm.models import LLMMessage, LLMResponse


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_valid_messages(role):
    msg = LLMMessage(role=role, content="hello")
    assert (msg.role, msg.content) == (role, "hello")
    assert LLMMessage(role=role, content="").content == ""


@pytest.mark.parametrize("role", ["tool", "function", "", "USER"])
def test_unsupported_roles(role):
    with pytest.raises(ValueError):
        LLMMessage(role=role, content="x")


def test_invalid_message_types():
    with pytest.raises(TypeError):
        LLMMessage(role=1, content="x")
    with pytest.raises(TypeError):
        LLMMessage(role="user", content=None)


def test_message_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        LLMMessage(role="user", content="x").content = "y"


def test_valid_response():
    r = LLMResponse(text="hi", model="m", usage={"input_tokens": 3}, raw={"response_id": "r1"})
    assert r.text == "hi" and r.usage == {"input_tokens": 3} and r.raw == {"response_id": "r1"}
    assert LLMResponse(text="", model="m").usage is None


def test_response_validation():
    with pytest.raises(TypeError):
        LLMResponse(text=1, model="m")
    with pytest.raises(TypeError):
        LLMResponse(text="t", model=None)
    with pytest.raises(ValueError):
        LLMResponse(text="t", model="  ")
    for bad in ({"a": True}, {"a": 1.5}, {1: 1}):
        with pytest.raises(TypeError):
            LLMResponse(text="t", model="m", usage=bad)
    with pytest.raises(ValueError):
        LLMResponse(text="t", model="m", usage={"a": -1})
    with pytest.raises(TypeError):
        LLMResponse(text="t", model="m", usage=[1])
    with pytest.raises(TypeError):
        LLMResponse(text="t", model="m", raw="x")


@pytest.mark.parametrize("raw", [{"api_key": "x"}, {"headers": {"Authorization": "x"}}, {"a": [{"client_secret": "x"}]}])
def test_raw_rejects_credential_keys(raw):
    with pytest.raises(ValueError):
        LLMResponse(text="t", model="m", raw=raw)


def test_response_frozen_and_defensive_copies():
    usage, raw = {"input_tokens": 1}, {"meta": {"n": 1}}
    r = LLMResponse(text="t", model="m", usage=usage, raw=raw)
    usage["input_tokens"] = 99
    raw["meta"]["n"] = 99
    assert r.usage == {"input_tokens": 1} and r.raw == {"meta": {"n": 1}}
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.text = "x"
