import dataclasses

import pytest

from nexora.api.errors import InvalidRequestError
from nexora.api.models import ResearchRequest, ResearchResponse
from nexora.research.models import ResearchAnswer


def test_minimal_request():
    r = ResearchRequest(query="What is attention?")
    assert r.query == "What is attention?"
    assert (r.model, r.temperature, r.max_output_tokens) == (None, None, None)


def test_full_request_preserves_values():
    r = ResearchRequest(query="q", model="m1", temperature=0.5, max_output_tokens=200)
    assert (r.model, r.temperature, r.max_output_tokens) == ("m1", 0.5, 200)
    assert ResearchRequest(query="q", temperature=0).temperature == 0
    assert ResearchRequest(query="q", temperature=2).temperature == 2


def test_query_content_preserved_exactly():
    q = "  Multi\nline  query with é and 'quotes' \t"
    assert ResearchRequest(query=q).query == q


def test_request_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ResearchRequest(query="q").query = "x"


@pytest.mark.parametrize("kw", [
    {"query": ""}, {"query": "   \n"}, {"query": None}, {"query": 5}, {"query": ["q"]},
    {"query": "q", "model": ""}, {"query": "q", "model": "  "}, {"query": "q", "model": 3},
    {"query": "q", "temperature": 2.5}, {"query": "q", "temperature": -0.1},
    {"query": "q", "temperature": float("nan")}, {"query": "q", "temperature": "0.5"},
    {"query": "q", "temperature": True},
    {"query": "q", "max_output_tokens": 0}, {"query": "q", "max_output_tokens": -1},
    {"query": "q", "max_output_tokens": 1.5}, {"query": "q", "max_output_tokens": "10"},
    {"query": "q", "max_output_tokens": True}, {"query": "q", "max_output_tokens": False},
])
def test_invalid_requests_raise_invalid_request_error(kw):
    with pytest.raises(InvalidRequestError):
        ResearchRequest(**kw)


def make_answer(**kw):
    base = dict(text="Answer [E1].", query="q?", model="served", usage={"total_tokens": 7},
                evidence_count=2, raw={"response_id": "resp_123"})
    base.update(kw)
    return ResearchAnswer(**base)


def test_response_from_answer_preserves_public_fields():
    resp = ResearchResponse.from_answer(make_answer(text="  Exact\ntext  ", query="  Why?  "))
    assert resp.text == "  Exact\ntext  " and resp.query == "  Why?  "
    assert resp.model == "served" and resp.evidence_count == 2
    assert resp.usage == {"total_tokens": 7}


def test_response_usage_none_and_defensive_copy():
    assert ResearchResponse.from_answer(make_answer(usage=None)).usage is None
    answer = make_answer()
    resp = ResearchResponse.from_answer(answer)
    resp.usage["total_tokens"] = 999
    assert answer.usage == {"total_tokens": 7}


def test_response_frozen_and_rejects_non_answers():
    resp = ResearchResponse.from_answer(make_answer())
    with pytest.raises(dataclasses.FrozenInstanceError):
        resp.text = "x"
    for bad in (None, "text", {"text": "x"}):
        with pytest.raises(TypeError):
            ResearchResponse.from_answer(bad)


def test_raw_provider_data_is_not_exposed():
    resp = ResearchResponse.from_answer(make_answer(raw={"response_id": "resp_123", "meta": {"x": 1}}))
    assert not hasattr(resp, "raw")
    assert "raw" not in {f.name for f in dataclasses.fields(resp)}
    assert "raw" not in resp.to_dict()
    assert "resp_123" not in repr(resp) and "resp_123" not in str(resp.to_dict())


def test_to_dict_shape_and_independence():
    resp = ResearchResponse.from_answer(make_answer())
    d = resp.to_dict()
    assert set(d) == {"text", "query", "model", "usage", "evidence_count"}
    d["usage"]["total_tokens"] = 0
    assert resp.usage == {"total_tokens": 7}
