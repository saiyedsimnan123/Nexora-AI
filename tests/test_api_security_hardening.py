"""Tests for API security hardening, HTTP security headers, and documentation route guards."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nexora.api.app import create_app
from nexora.api.service import ResearchApplicationService


class _FakeResearchService:
    def ask(self, request: Any) -> Any:
        return MagicMock()


@pytest.fixture
def fake_research_service() -> _FakeResearchService:
    return _FakeResearchService()


def test_security_headers_present_on_api_responses(fake_research_service: _FakeResearchService) -> None:
    app = create_app(fake_research_service, enforce_auth=False)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "accelerometer=()" in response.headers.get("permissions-policy", "")
    assert "default-src 'self'" in response.headers.get("content-security-policy", "")
    assert "frame-ancestors 'none'" in response.headers.get("content-security-policy", "")


def test_security_headers_present_on_error_responses(fake_research_service: _FakeResearchService) -> None:
    app = create_app(fake_research_service, enforce_auth=False)
    client = TestClient(app)

    response = client.get("/nonexistent-endpoint-404")
    assert response.status_code == 404
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "default-src 'self'" in response.headers.get("content-security-policy", "")


def test_hsts_omitted_on_plain_http(fake_research_service: _FakeResearchService) -> None:
    app = create_app(fake_research_service, enforce_auth=False)
    client = TestClient(app, base_url="http://testserver")

    response = client.get("/health")
    assert response.status_code == 200
    # Over plain HTTP, HSTS must not be sent
    assert "strict-transport-security" not in response.headers


def test_hsts_emitted_over_https(fake_research_service: _FakeResearchService) -> None:
    app = create_app(fake_research_service, enforce_auth=False)
    client = TestClient(app, base_url="https://testserver")

    response = client.get("/health")
    assert response.status_code == 200
    hsts = response.headers.get("strict-transport-security")
    assert hsts is not None
    assert "max-age=" in hsts
    assert "includeSubDomains" in hsts


def test_hsts_emitted_with_forwarded_proto_https(fake_research_service: _FakeResearchService) -> None:
    app = create_app(fake_research_service, enforce_auth=False)
    client = TestClient(app, base_url="http://testserver")

    response = client.get("/health", headers={"x-forwarded-proto": "https"})
    assert response.status_code == 200
    hsts = response.headers.get("strict-transport-security")
    assert hsts is not None
    assert "max-age=" in hsts


@pytest.mark.parametrize(
    ("env_var", "env_val"),
    [
        ("APP_ENV", "production"),
        ("APP_ENV", "prod"),
        ("ENVIRONMENT", "production"),
        ("ENVIRONMENT", "prod"),
        ("NEXORA_APP_MODE", "production"),
        ("NEXORA_APP_MODE", "prod"),
        ("RAILWAY_ENVIRONMENT", "production"),
    ],
)
def test_docs_disabled_in_production_by_default(
    fake_research_service: _FakeResearchService,
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    env_val: str,
) -> None:
    # Clear all production indicators
    for var in ("APP_ENV", "ENVIRONMENT", "NEXORA_APP_MODE", "RAILWAY_ENVIRONMENT", "NEXORA_ENABLE_DOCS"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv(env_var, env_val)

    app = create_app(fake_research_service, enforce_auth=False)
    client = TestClient(app)

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_docs_enabled_when_explicitly_configured(
    fake_research_service: _FakeResearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("NEXORA_ENABLE_DOCS", "true")

    app = create_app(fake_research_service, enforce_auth=False)
    client = TestClient(app)

    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_docs_enabled_in_development_by_default(
    fake_research_service: _FakeResearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("NEXORA_APP_MODE", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("NEXORA_ENABLE_DOCS", raising=False)

    app = create_app(fake_research_service, enforce_auth=False)
    client = TestClient(app)

    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_ready_endpoint_does_not_leak_enforced_flag(fake_research_service: _FakeResearchService) -> None:
    app = create_app(fake_research_service, enforce_auth=False)
    client = TestClient(app)

    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert "checks" in body
    auth_check = body["checks"].get("auth")
    assert auth_check is not None
    assert auth_check["status"] == "ready"
    assert "enforced" not in auth_check
