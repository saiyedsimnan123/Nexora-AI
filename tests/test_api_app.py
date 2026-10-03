import inspect

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import nexora.api.app as app_module
import nexora.api.routes as routes_module
from nexora.api.app import create_app
from nexora.api.errors import InvalidConfigurationError


class ExplodingService:
    """Any use of the service fails the test: health must not touch it."""

    def __init__(self):
        self.calls = 0

    def ask(self, request):
        self.calls += 1
        raise AssertionError("service must not be called")


def test_app_can_be_created():
    assert isinstance(create_app(ExplodingService()), FastAPI)


@pytest.mark.parametrize("bad", [None, object(), "service"])
def test_create_app_rejects_invalid_service(bad):
    with pytest.raises(InvalidConfigurationError):
        create_app(bad)


def test_health_ok_and_deterministic_without_touching_service():
    service = ExplodingService()
    client = TestClient(create_app(service))
    first, second = client.get("/health"), client.get("/health")
    assert first.status_code == 200 and first.json() == {"status": "ok"}
    assert second.json() == first.json()
    assert service.calls == 0


def test_unknown_route_uses_consistent_error_shape():
    client = TestClient(create_app(ExplodingService()))
    response = client.get("/nope")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "http_error" and "detail" not in body


def test_wrong_method_uses_consistent_error_shape():
    client = TestClient(create_app(ExplodingService()))
    response = client.get("/api/v1/research")
    assert response.status_code == 405 and "error" in response.json()


def test_unexpected_exception_is_generic_500():
    class Broken:
        def ask(self, request):
            raise KeyError("sk-placeholder-secret")

    client = TestClient(create_app(Broken()), raise_server_exceptions=False)
    response = client.post("/api/v1/research", json={"query": "q"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "sk-placeholder-secret" not in response.text


def test_modules_have_no_infrastructure_imports():
    for module in (app_module, routes_module):
        source = inspect.getsource(module).lower()
        for forbidden in ("import openai", "from openai", "qdrant", "embedding", "vectorstore", "os.environ"):
            assert forbidden not in source


def test_openapi_metadata_and_routes():
    from nexora import __version__

    app = create_app(ExplodingService())
    schema = app.openapi()
    assert schema["info"]["title"] == "Nexora AI"
    assert schema["info"]["version"] == __version__
    assert schema["info"]["description"]
    assert "/health" in schema["paths"] and "/api/v1/research" in schema["paths"]
    assert "post" in schema["paths"]["/api/v1/research"]


def test_health_rejects_other_methods():
    client = TestClient(create_app(ExplodingService()))
    response = client.post("/health")
    assert response.status_code == 405 and response.json()["error"]["code"] == "http_error"


def test_openapi_describes_research_and_health_operations():
    paths = create_app(ExplodingService()).openapi()["paths"]
    assert "get" in paths["/health"]
    operation = paths["/api/v1/research"]["post"]
    assert "requestBody" in operation
    ok = operation["responses"]["200"]
    assert "application/json" in ok["content"]
    assert "schema" in ok["content"]["application/json"]
