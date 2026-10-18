"""Tests for CORS security configuration and preflight enforcement."""

import pytest
from fastapi.testclient import TestClient

from nexora.api.app import create_app
from nexora.config.cors import CORSConfig, CORSConfigurationError


class FakeResearchService:
    def ask(self, request):
        from types import SimpleNamespace
        return SimpleNamespace(
            to_dict=lambda: {
                "text": "answer",
                "query": getattr(request, "query", "q"),
                "model": "fake-model",
                "evidence_count": 1,
            }
        )


def test_cors_disabled_by_default():
    """Verify that by default without CORS configuration, no CORS headers are returned."""
    app = create_app(FakeResearchService(), enforce_auth=False)
    client = TestClient(app)

    # Simple GET request with Origin header
    resp = client.get("/health", headers={"Origin": "http://evil.com"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_cors_explicit_allowed_origins():
    """Verify allowed origins receive CORS headers and disallowed origins do not."""
    cors = CORSConfig(
        enabled=True,
        allow_origins=("http://localhost:3000", "https://app.nexora.ai"),
        allow_credentials=False,
    )
    app = create_app(FakeResearchService(), enforce_auth=False, cors_config=cors)
    client = TestClient(app)

    # 1. Allowed origin gets Access-Control-Allow-Origin
    resp_allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp_allowed.status_code == 200
    assert resp_allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"

    resp_allowed2 = client.get("/health", headers={"Origin": "https://app.nexora.ai"})
    assert resp_allowed2.status_code == 200
    assert resp_allowed2.headers.get("access-control-allow-origin") == "https://app.nexora.ai"

    # 2. Disallowed origin does NOT get Access-Control-Allow-Origin
    resp_disallowed = client.get("/health", headers={"Origin": "http://evil.com"})
    assert resp_disallowed.status_code == 200
    assert "access-control-allow-origin" not in resp_disallowed.headers


def test_cors_wildcard_rejected_with_credentials():
    """Reject wildcard '*' origin when allow_credentials is True."""
    with pytest.raises(CORSConfigurationError) as exc:
        CORSConfig(
            enabled=True,
            allow_origins=("*",),
            allow_credentials=True,
        )
    assert "cannot contain wildcard '*' when allow_credentials is True" in str(exc.value)


def test_cors_preflight_options():
    """Verify preflight OPTIONS request returns appropriate headers and max-age."""
    cors = CORSConfig(
        enabled=True,
        allow_origins=("http://localhost:3000",),
        allow_credentials=True,
        allow_methods=("GET", "POST", "OPTIONS"),
        allow_headers=("Authorization", "Content-Type", "X-API-Key"),
        max_age=3600,
    )
    app = create_app(FakeResearchService(), enforce_auth=False, cors_config=cors)
    client = TestClient(app)

    resp = client.options(
        "/api/v1/research",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert resp.headers.get("access-control-allow-credentials") == "true"
    assert "POST" in resp.headers.get("access-control-allow-methods", "")
    assert "authorization" in resp.headers.get("access-control-allow-headers", "").lower()
    assert resp.headers.get("access-control-max-age") == "3600"


def test_cors_invalid_origins_rejected():
    """Reject origins that do not have valid http:// or https:// schemes."""
    with pytest.raises(CORSConfigurationError) as exc:
        CORSConfig(allow_origins=("ftp://bad.com",))
    assert "must start with http:// or https://" in str(exc.value)

    with pytest.raises(CORSConfigurationError) as exc:
        CORSConfig(allow_origins=("not-a-url",))
    assert "must start with http:// or https://" in str(exc.value)


def test_cors_from_env():
    """Verify loading CORS settings from environment variables."""
    env = {
        "NEXORA_CORS_ALLOWED_ORIGINS": "http://localhost:3000, https://research.nexora.ai",
        "NEXORA_CORS_ALLOW_CREDENTIALS": "true",
        "NEXORA_CORS_MAX_AGE": "1200",
    }
    cfg = CORSConfig.from_env(env)
    assert cfg.enabled is True
    assert cfg.allow_origins == ("http://localhost:3000", "https://research.nexora.ai")
    assert cfg.allow_credentials is True
    assert cfg.max_age == 1200


def test_cors_from_env_wildcard_with_credentials_fails():
    """Verify that setting wildcard origins and credentials in environment raises error."""
    env = {
        "NEXORA_CORS_ALLOWED_ORIGINS": "*",
        "NEXORA_CORS_ALLOW_CREDENTIALS": "true",
    }
    with pytest.raises(CORSConfigurationError):
        CORSConfig.from_env(env)


def test_cors_headers_present_on_rate_limited_429_response():
    """Verify that rate-limited 429 responses include CORS headers for allowed origins."""
    from nexora.config.rate_limit import RateLimitConfig

    cors = CORSConfig(
        enabled=True,
        allow_origins=("http://localhost:3000",),
    )
    rate_limit = RateLimitConfig(
        enabled=True,
        research_query_limit=1,
        window_seconds=60,
    )
    app = create_app(
        FakeResearchService(),
        enforce_auth=False,
        cors_config=cors,
        rate_limit_config=rate_limit,
    )
    client = TestClient(app)

    # First request: 200 with CORS
    r1 = client.post(
        "/api/v1/research",
        headers={"Origin": "http://localhost:3000"},
        json={"query": "q1"},
    )
    assert r1.status_code == 200
    assert r1.headers.get("access-control-allow-origin") == "http://localhost:3000"

    # Second request: 429 with CORS
    r2 = client.post(
        "/api/v1/research",
        headers={"Origin": "http://localhost:3000"},
        json={"query": "q2"},
    )
    assert r2.status_code == 429
    assert r2.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert r2.headers.get("x-ratelimit-remaining") == "0"

