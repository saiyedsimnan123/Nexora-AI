"""Tests for configurable inbound rate limiting, endpoint tiers, and abuse protection."""

import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient

from nexora.api.app import create_app
from nexora.api.rate_limit import InMemoryRateLimitStorage, get_caller_identity, categorize_route
from nexora.config.rate_limit import RateLimitConfig, RateLimitConfigurationError


class FakeResearchService:
    def ask(self, request):
        return SimpleNamespace(
            to_dict=lambda: {
                "text": "answer",
                "query": getattr(request, "query", "q"),
                "model": "fake-model",
                "evidence_count": 1,
            }
        )


class FakeAnalysisService:
    def analyze_paper(self, paper_id, **kwargs):
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_id": str(paper_id),
                "title": "Fake Paper",
                "problem_and_objectives": {"summary": "Objectives"},
                "methodology_and_algorithms": {"summary": "Methodology"},
                "models_and_datasets": {"summary": "Dataset"},
                "experiments_and_metrics": {"summary": "Experiments"},
                "results_and_findings": {"summary": "Results"},
                "limitations": {"summary": "Limitations"},
                "future_work": {"summary": "Future work"},
                "evidence_count": 1,
                "model": "fake-model",
                "usage": None,
            }
        )


class FakeDiscoveryService:
    def search(self, query):
        return SimpleNamespace(
            to_dict=lambda: {
                "provider": "arxiv",
                "query": query.query if hasattr(query, "query") else str(query),
                "total_results": 0,
                "offset": 0,
                "limit": 10,
                "papers": [],
            }
        )


class ControllableClock:
    def __init__(self, initial_time: float = 1000.0) -> None:
        self.current_time = initial_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


def test_rate_limiting_disabled_by_default():
    """Verify that when rate limiting is disabled, requests proceed without limit headers."""
    app = create_app(FakeResearchService(), enforce_auth=False)
    client = TestClient(app)

    for _ in range(25):
        resp = client.post("/api/v1/research", json={"query": "test"})
        assert resp.status_code == 200
        assert "x-ratelimit-limit" not in resp.headers


def test_rate_limiting_allowed_and_headers():
    """Verify allowed requests receive standard rate limit headers."""
    cfg = RateLimitConfig(
        enabled=True,
        research_query_limit=3,
        window_seconds=60,
    )
    storage = InMemoryRateLimitStorage()
    app = create_app(
        FakeResearchService(),
        enforce_auth=False,
        rate_limit_config=cfg,
        rate_limit_storage=storage,
    )
    client = TestClient(app)

    # 1. First request
    resp1 = client.post("/api/v1/research", json={"query": "test 1"})
    assert resp1.status_code == 200
    assert resp1.headers.get("x-ratelimit-limit") == "3"
    assert resp1.headers.get("x-ratelimit-remaining") == "2"

    # 2. Second request
    resp2 = client.post("/api/v1/research", json={"query": "test 2"})
    assert resp2.status_code == 200
    assert resp2.headers.get("x-ratelimit-remaining") == "1"

    # 3. Third request
    resp3 = client.post("/api/v1/research", json={"query": "test 3"})
    assert resp3.status_code == 200
    assert resp3.headers.get("x-ratelimit-remaining") == "0"


def test_rate_limiting_exceeded_returns_429():
    """Verify exceeding rate limit returns HTTP 429, Retry-After, and error JSON."""
    clock = ControllableClock(1000.0)
    storage = InMemoryRateLimitStorage(time_func=clock)
    cfg = RateLimitConfig(
        enabled=True,
        discovery_search_limit=2,
        window_seconds=60,
    )
    app = create_app(
        FakeResearchService(),
        discovery_service=FakeDiscoveryService(),
        enforce_auth=False,
        rate_limit_config=cfg,
        rate_limit_storage=storage,
    )
    client = TestClient(app)

    # Hits 1 & 2 succeed
    r1 = client.get("/api/v1/discovery/search?query=test")
    assert r1.status_code == 200

    r2 = client.get("/api/v1/discovery/search?query=test")
    assert r2.status_code == 200

    # Hit 3 exceeds limit -> 429
    r3 = client.get("/api/v1/discovery/search?query=test")
    assert r3.status_code == 429
    assert r3.headers.get("x-ratelimit-limit") == "2"
    assert r3.headers.get("x-ratelimit-remaining") == "0"
    assert int(r3.headers.get("retry-after", "0")) > 0

    body = r3.json()
    assert body["error"]["code"] == "rate_limited"
    assert "discovery_search" in body["error"]["message"]


def test_rate_limiting_sliding_window_reset():
    """Verify advancing time past window resets rate limit quota."""
    clock = ControllableClock(1000.0)
    storage = InMemoryRateLimitStorage(time_func=clock)
    cfg = RateLimitConfig(
        enabled=True,
        research_query_limit=2,
        window_seconds=60,
    )
    app = create_app(
        FakeResearchService(),
        enforce_auth=False,
        rate_limit_config=cfg,
        rate_limit_storage=storage,
    )
    client = TestClient(app)

    client.post("/api/v1/research", json={"query": "q1"})
    client.post("/api/v1/research", json={"query": "q2"})
    # Exceeded
    r_blocked = client.post("/api/v1/research", json={"query": "q3"})
    assert r_blocked.status_code == 429

    # Advance clock by 61 seconds (past window)
    clock.advance(61.0)

    # Next request succeeds
    r_after = client.post("/api/v1/research", json={"query": "q4"})
    assert r_after.status_code == 200
    assert r_after.headers.get("x-ratelimit-remaining") == "1"


def test_rate_limiting_cost_tiering():
    """Verify that different endpoint categories have separate rate limit quotas."""
    cfg = RateLimitConfig(
        enabled=True,
        heavy_analysis_limit=1,
        discovery_search_limit=5,
        window_seconds=60,
    )
    storage = InMemoryRateLimitStorage()
    app = create_app(
        FakeResearchService(),
        analysis_service=FakeAnalysisService(),
        discovery_service=FakeDiscoveryService(),
        enforce_auth=False,
        rate_limit_config=cfg,
        rate_limit_storage=storage,
    )
    client = TestClient(app)

    # 1. Exhaust heavy analysis limit
    r_ana = client.post("/api/v1/papers/analyze", json={"paper_id": "00000000-0000-0000-0000-000000000001"})
    assert r_ana.status_code == 200
    r_ana2 = client.post("/api/v1/papers/analyze", json={"paper_id": "00000000-0000-0000-0000-000000000001"})
    assert r_ana2.status_code == 429

    # 2. Discovery search quota is independent and still available!
    r_disc = client.get("/api/v1/discovery/search?query=test")
    assert r_disc.status_code == 200
    assert r_disc.headers.get("x-ratelimit-remaining") == "4"


def test_rate_limiting_caller_isolation():
    """Verify different callers (API keys) do not deplete each other's quota."""
    cfg = RateLimitConfig(
        enabled=True,
        research_query_limit=1,
        window_seconds=60,
    )
    storage = InMemoryRateLimitStorage()
    app = create_app(
        FakeResearchService(),
        enforce_auth=False,
        rate_limit_config=cfg,
        rate_limit_storage=storage,
    )
    client = TestClient(app)

    # Caller A hits limit
    r_a1 = client.post(
        "/api/v1/research",
        headers={"Authorization": "Bearer nexora_token_aaaaaa"},
        json={"query": "q"},
    )
    assert r_a1.status_code == 200

    r_a2 = client.post(
        "/api/v1/research",
        headers={"Authorization": "Bearer nexora_token_aaaaaa"},
        json={"query": "q"},
    )
    assert r_a2.status_code == 429

    # Caller B with different token is not blocked!
    r_b = client.post(
        "/api/v1/research",
        headers={"Authorization": "Bearer nexora_token_bbbbbb"},
        json={"query": "q"},
    )
    assert r_b.status_code == 200


def test_caller_identity_hashes_secret_keys():
    """Verify raw tokens are never present in caller identities; only SHA-256 prefixes."""
    from starlette.requests import Request

    secret = "nexora_super_secret_key_12345"
    scope = {
        "type": "http",
        "headers": [(b"authorization", f"Bearer {secret}".encode("utf-8"))],
        "client": ("127.0.0.1", 1234),
    }
    req = Request(scope)
    identity = get_caller_identity(req)

    assert identity.startswith("key:")
    assert secret not in identity
    assert len(identity) == 20  # "key:" + 16 hex chars


def test_untrusted_spoofed_headers_are_ignored():
    """Verify client-supplied headers like X-User-Id or fake X-Forwarded-For are ignored."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [
            (b"x-user-id", b"admin"),
            (b"x-forwarded-for", b"8.8.8.8"),
            (b"x-real-ip", b"1.1.1.1"),
        ],
        "client": ("192.168.1.50", 1234),
    }
    req = Request(scope)
    # Without trusted_proxies, caller identity resolves directly to client host
    identity = get_caller_identity(req, trusted_proxies=())
    assert identity == "ip:192.168.1.50"
    assert "8.8.8.8" not in identity
    assert "admin" not in identity


def test_rate_limiting_health_exempt():
    """Verify /health is unmetered."""
    cfg = RateLimitConfig(enabled=True, default_limit=1, window_seconds=60)
    app = create_app(FakeResearchService(), enforce_auth=False, rate_limit_config=cfg)
    client = TestClient(app)

    for _ in range(10):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "x-ratelimit-limit" not in resp.headers


def test_rate_limit_config_validation():
    """Verify invalid limit parameters fail fast."""
    with pytest.raises(RateLimitConfigurationError):
        RateLimitConfig(heavy_analysis_limit=0)

    with pytest.raises(RateLimitConfigurationError):
        RateLimitConfig(window_seconds=-5)

    with pytest.raises(RateLimitConfigurationError):
        RateLimitConfig(backend="unsupported_backend")


def test_rate_limit_config_from_env():
    """Verify loading rate limit settings from environment variables."""
    env = {
        "NEXORA_RATE_LIMIT_ENABLED": "true",
        "NEXORA_RATE_LIMIT_HEAVY_ANALYSIS": "5",
        "NEXORA_RATE_LIMIT_DISCOVERY_SEARCH": "15",
        "NEXORA_RATE_LIMIT_BACKEND": "memory",
        "NEXORA_RATE_LIMIT_TRUSTED_PROXIES": "10.0.0.1, 10.0.0.2",
    }
    cfg = RateLimitConfig.from_env(env)
    assert cfg.enabled is True
    assert cfg.heavy_analysis_limit == 5
    assert cfg.discovery_search_limit == 15
    assert cfg.trusted_proxies == ("10.0.0.1", "10.0.0.2")
