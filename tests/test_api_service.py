import inspect

import pytest

import nexora.api.models as models_module
import nexora.api.service as service_module
from nexora.api.errors import InvalidConfigurationError, InvalidRequestError, ResearchServiceError
from nexora.api.models import ResearchRequest, ResearchResponse
from nexora.api.service import ResearchApplicationService
from nexora.research.models import ResearchAnswer


class FakeQueryService:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.answer = ResearchAnswer(text="done", query="q", model="m", usage={"total_tokens": 4},
                                     evidence_count=3, raw={"response_id": "secret-ish-id"})

    def ask(self, query, *, model=None, temperature=None, max_output_tokens=None):
        self.calls.append((query, model, temperature, max_output_tokens))
        if self.error:
            raise self.error
        return self.answer


def test_dependency_injection_and_construction_makes_no_calls():
    fake = FakeQueryService()
    ResearchApplicationService(fake)
    assert fake.calls == []


@pytest.mark.parametrize("bad", [None, object(), "service"])
def test_invalid_query_service_is_configuration_error(bad):
    with pytest.raises(InvalidConfigurationError):
        ResearchApplicationService(bad)


def test_ask_forwards_request_fields():
    fake = FakeQueryService()
    ResearchApplicationService(fake).ask(
        ResearchRequest(query="  my q ", model="m2", temperature=0.3, max_output_tokens=64))
    assert fake.calls == [("  my q ", "m2", 0.3, 64)]
    ResearchApplicationService(fake).ask(ResearchRequest(query="plain"))
    assert fake.calls[1] == ("plain", None, None, None)


def test_answer_converted_to_response_without_raw():
    resp = ResearchApplicationService(FakeQueryService()).ask(ResearchRequest(query="q"))
    assert isinstance(resp, ResearchResponse)
    assert (resp.text, resp.query, resp.model, resp.evidence_count) == ("done", "q", "m", 3)
    assert resp.usage == {"total_tokens": 4}
    assert "secret-ish-id" not in str(resp.to_dict())


@pytest.mark.parametrize("bad", [None, "q", {"query": "q"}])
def test_non_request_rejected_without_calls(bad):
    fake = FakeQueryService()
    with pytest.raises(InvalidRequestError):
        ResearchApplicationService(fake).ask(bad)
    assert fake.calls == []


def test_pipeline_errors_wrapped_with_cause_and_no_detail_leak():
    err = RuntimeError("provider said key=sk-placeholder")
    with pytest.raises(ResearchServiceError) as info:
        ResearchApplicationService(FakeQueryService(error=err)).ask(ResearchRequest(query="q"))
    assert info.value.__cause__ is err
    assert "RuntimeError" in str(info.value) and "sk-placeholder" not in str(info.value)


def test_errors_are_never_returned_as_responses():
    for err in (ValueError("v"), TypeError("t"), ConnectionError("c")):
        with pytest.raises(ResearchServiceError):
            ResearchApplicationService(FakeQueryService(error=err)).ask(ResearchRequest(query="q"))


def test_invalid_request_error_is_a_value_error():
    assert issubclass(InvalidRequestError, ValueError)


def test_no_infrastructure_or_http_imports():
    for module in (models_module, service_module):
        source = inspect.getsource(module).lower()
        for forbidden in ("import openai", "qdrant", "embedding", "vectorstore", "fastapi", "os.environ"):
            assert forbidden not in source
