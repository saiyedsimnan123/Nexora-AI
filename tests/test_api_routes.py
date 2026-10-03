import pytest
from fastapi.testclient import TestClient

from nexora.api.app import create_app
from nexora.api.errors import InvalidConfigurationError, InvalidRequestError, ResearchServiceError
from nexora.api.models import ResearchResponse
from nexora.api.service import ResearchApplicationService
from nexora.research.models import ResearchAnswer

URL = "/api/v1/research"
SECRET = "sk-placeholder-secret"


class FakeQueryService:
    """Stands in for ResearchQueryService; the real application service sits above it."""

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def ask(self, query, *, model=None, temperature=None, max_output_tokens=None):
        self.calls.append((query, model, temperature, max_output_tokens))
        if self.error:
            raise self.error
        return ResearchAnswer(
            text="Based on the evidence [E1].", query=query, model="served-model",
            usage={"total_tokens": 421}, evidence_count=5,
            raw={"response_id": "resp_internal_123"},
        )


class RaisingAppService:
    def __init__(self, error):
        self.error = error

    def ask(self, request):
        raise self.error


def make_client(query_service=None):
    query_service = query_service or FakeQueryService()
    app = create_app(ResearchApplicationService(query_service))
    return TestClient(app, raise_server_exceptions=False), query_service


def test_valid_request_returns_serialized_response():
    client, _ = make_client()
    response = client.post(URL, json={"query": "What datasets?", "model": "m1",
                                      "temperature": 0.2, "max_output_tokens": 500})
    assert response.status_code == 200
    assert response.json() == {
        "text": "Based on the evidence [E1].", "query": "What datasets?", "model": "served-model",
        "usage": {"total_tokens": 421}, "evidence_count": 5,
    }


def test_fields_forwarded_to_pipeline_and_query_preserved_exactly():
    client, qs = make_client()
    query = "  Spaced\nquery with é  "
    client.post(URL, json={"query": query, "model": "m1", "temperature": 0.7, "max_output_tokens": 64})
    assert qs.calls == [(query, "m1", 0.7, 64)]


def test_optional_fields_default_to_none():
    client, qs = make_client()
    assert client.post(URL, json={"query": "q"}).status_code == 200
    assert qs.calls == [("q", None, None, None)]


def test_raw_provider_data_not_exposed():
    client, _ = make_client()
    response = client.post(URL, json={"query": "q"})
    assert "raw" not in response.json()
    assert "resp_internal_123" not in response.text


def test_response_matches_research_response_to_dict():
    client, qs = make_client()
    expected = ResearchResponse.from_answer(qs.ask("q")).to_dict()
    assert client.post(URL, json={"query": "q"}).json() == expected


@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "   \n\t"},
    {"query": "q", "temperature": 2.5}, {"query": "q", "temperature": -0.5},
    {"query": "q", "max_output_tokens": 0}, {"query": "q", "max_output_tokens": -4},
    {"query": "q", "model": ""}, {"query": "q", "model": "   "},
])
def test_semantically_invalid_requests_get_400_without_calling_pipeline(payload):
    client, qs = make_client()
    response = client.post(URL, json=payload)
    assert response.status_code == 400
    assert response.json() == {"error": {"code": "invalid_request", "message": "Invalid research request."}}
    assert qs.calls == []


@pytest.mark.parametrize("payload", [
    {}, {"query": 5}, {"query": None}, {"query": ["q"]},
    {"query": "q", "model": 3}, {"query": "q", "temperature": "0.5"},
    {"query": "q", "max_output_tokens": 1.5}, {"query": "q", "max_output_tokens": "10"},
    {"query": "q", "max_output_tokens": True}, {"query": "q", "unknown_field": 1},
])
def test_wrong_shape_requests_get_422_without_calling_pipeline(payload):
    client, qs = make_client()
    response = client.post(URL, json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert qs.calls == []


def test_bool_temperature_is_rejected():
    client, qs = make_client()
    assert client.post(URL, json={"query": "q", "temperature": True}).status_code in (400, 422)
    assert qs.calls == []


def test_malformed_json_gets_422():
    client, _ = make_client()
    response = client.post(URL, content="{not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["error"]["code"] == "validation_error"


def test_validation_errors_do_not_echo_submitted_input():
    client, _ = make_client()
    response = client.post(URL, json={"query": SECRET, "temperature": "hot"})
    assert response.status_code == 422 and SECRET not in response.text


@pytest.mark.parametrize("error,status,code", [
    (InvalidRequestError("bad input detail"), 400, "invalid_request"),
    (ResearchServiceError(f"failed {SECRET}"), 500, "research_failed"),
    (InvalidConfigurationError(f"misconfigured {SECRET}"), 500, "server_misconfigured"),
])
def test_application_errors_map_to_http_without_leaking_messages(error, status, code):
    client = TestClient(create_app(RaisingAppService(error)), raise_server_exceptions=False)
    response = client.post(URL, json={"query": "q"})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert SECRET not in response.text and "bad input detail" not in response.text
    assert "Traceback" not in response.text


def test_provider_exception_details_not_leaked_through_full_stack():
    cause = ConnectionError(f"https://api.example/v1?key={SECRET}")
    client, _ = make_client(FakeQueryService(error=cause))
    response = client.post(URL, json={"query": "q"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "research_failed"
    for leaked in (SECRET, "ConnectionError", "api.example"):
        assert leaked not in response.text


def test_dependency_injection_uses_supplied_service():
    class Recording:
        def __init__(self):
            self.requests = []

        def ask(self, request):
            self.requests.append(request)
            return ResearchResponse(text="t", query=request.query, model="m", usage=None, evidence_count=0)

    service = Recording()
    client = TestClient(create_app(service))
    response = client.post(URL, json={"query": "hello", "model": "x"})
    assert response.json() == {"text": "t", "query": "hello", "model": "m",
                               "usage": None, "evidence_count": 0}
    assert service.requests[0].query == "hello" and service.requests[0].model == "x"


def test_repeated_requests_are_deterministic():
    client, _ = make_client()
    first = client.post(URL, json={"query": "q"})
    second = client.post(URL, json={"query": "q"})
    assert first.status_code == second.status_code == 200 and first.json() == second.json()


def test_exact_response_schema_and_types():
    client, _ = make_client()
    body = client.post(URL, json={"query": "q"}).json()
    assert set(body) == {"text", "query", "model", "usage", "evidence_count"}
    assert isinstance(body["text"], str) and isinstance(body["query"], str)
    assert isinstance(body["model"], str) and isinstance(body["usage"], dict)
    assert isinstance(body["evidence_count"], int) and not isinstance(body["evidence_count"], bool)


def test_response_with_null_usage_is_serialized():
    class NoUsage:
        def ask(self, request):
            return ResearchResponse(text="t", query=request.query, model="m", usage=None, evidence_count=0)

    response = TestClient(create_app(NoUsage())).post(URL, json={"query": "q"})
    assert response.status_code == 200 and response.json()["usage"] is None


def test_missing_query_reports_location_not_value():
    client, qs = make_client()
    response = client.post(URL, json={"model": "m1"})
    assert response.status_code == 422
    details = response.json()["error"]["details"]
    assert any(d["loc"][-1] == "query" for d in details)
    assert qs.calls == []


# ---- final hardening: dependency resolution and request-boundary cases ----
from types import SimpleNamespace  # noqa: E402

from nexora.api.routes import get_research_service  # noqa: E402


def test_get_research_service_returns_exact_injected_service():
    service = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(research_service=service)))
    assert get_research_service(request) is service


def test_create_app_stores_exact_service_on_app_state():
    service = ResearchApplicationService(FakeQueryService())
    assert create_app(service).state.research_service is service


@pytest.mark.parametrize("content,content_type", [
    ("hello world", "text/plain"),
    ("", "application/json"),
    ("", None),
    ("{broken json", "application/json"),
    ("5", "application/json"),
    ('"just a string"', "application/json"),
    ("null", "application/json"),
    ("[]", "application/json"),
    ('[{"query": "q"}]', "application/json"),
])
def test_non_object_or_unparseable_bodies_get_422_and_skip_pipeline(content, content_type):
    client, qs = make_client()
    headers = {"Content-Type": content_type} if content_type else {}
    response = client.post(URL, content=content, headers=headers)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert qs.calls == []


def test_missing_content_type_with_valid_json_is_controlled():
    # Behaviour differs across FastAPI versions: either parsed as JSON or rejected.
    client, qs = make_client()
    response = client.post(URL, content='{"query": "q"}')
    assert response.status_code in (200, 422)
    if response.status_code == 422:
        assert qs.calls == []
    else:
        assert qs.calls == [("q", None, None, None)]


def test_unexpected_exception_response_is_fully_generic():
    class Broken:
        def ask(self, request):
            raise KeyError(f"/srv/app/secret_module.py {SECRET}")

    client = TestClient(create_app(Broken()), raise_server_exceptions=False)
    response = client.post(URL, json={"query": "q"})
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": "An internal error occurred."}}
    for leaked in (SECRET, "KeyError", "Traceback", "secret_module", ".py"):
        assert leaked not in response.text
