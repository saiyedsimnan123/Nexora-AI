import dataclasses

import pytest

from nexora.research.models import ResearchAnswer


def make(**kw):
    base = dict(text="answer", query="q", model="m")
    base.update(kw)
    return ResearchAnswer(**base)


def test_valid_answer_and_defaults():
    a = make(usage={"input_tokens": 3}, evidence_count=2, raw={"response_id": "r"})
    assert (a.text, a.query, a.model, a.evidence_count) == ("answer", "q", "m", 2)
    assert a.usage == {"input_tokens": 3} and a.raw == {"response_id": "r"}
    d = make(text="")
    assert d.usage is None and d.raw is None and d.evidence_count == 0


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make().text = "x"


@pytest.mark.parametrize("kw,exc", [
    ({"text": None}, TypeError), ({"text": 5}, TypeError),
    ({"query": None}, TypeError), ({"query": 1}, TypeError),
    ({"model": ""}, ValueError), ({"model": "  "}, ValueError),
    ({"model": None}, TypeError), ({"model": 3}, TypeError),
    ({"evidence_count": 1.5}, TypeError), ({"evidence_count": "2"}, TypeError),
    ({"evidence_count": True}, TypeError), ({"evidence_count": False}, TypeError),
    ({"evidence_count": -1}, ValueError),
    ({"usage": [1]}, TypeError), ({"usage": {1: 1}}, TypeError),
    ({"usage": {"a": True}}, TypeError), ({"usage": {"a": 1.5}}, TypeError),
    ({"usage": {"a": -1}}, ValueError),
    ({"raw": "x"}, TypeError), ({"raw": [1]}, TypeError),
])
def test_invalid_fields(kw, exc):
    with pytest.raises(exc):
        make(**kw)


def test_defensive_copies():
    usage, raw = {"input_tokens": 1}, {"meta": {"n": 1}}
    a = make(usage=usage, raw=raw)
    usage["input_tokens"] = 99
    raw["meta"]["n"] = 99
    assert a.usage == {"input_tokens": 1} and a.raw == {"meta": {"n": 1}}


@pytest.mark.parametrize("raw", [{"api_key": "x"}, {"h": {"Authorization": "x"}}, {"a": [{"client_secret": "x"}]}])
def test_secret_like_raw_keys_rejected(raw):
    with pytest.raises(ValueError):
        make(raw=raw)
