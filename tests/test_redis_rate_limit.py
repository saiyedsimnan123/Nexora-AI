"""Tests for atomic Redis-backed distributed rate limiting (Milestone 36)."""

from __future__ import annotations

import collections
import math
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from nexora.api.app import create_app
from nexora.api.rate_limit import (
    RedisRateLimitStorage,
    RateLimitMiddleware,
    _LUA_SLIDING_WINDOW,
)
from nexora.config.rate_limit import RateLimitConfig, RateLimitConfigurationError


class FakeRedisClient:
    """Accurate in-memory simulation of Redis sorted sets & Lua sliding-window execution."""

    def __init__(self) -> None:
        # zsets: key -> dict[member, score]
        self.zsets: dict[str, dict[str, float]] = collections.defaultdict(dict)
        self.expires: dict[str, float] = {}
        self.closed: bool = False

    def eval(self, script: str, numkeys: int, key: str, *args: str) -> list[int]:
        if self.closed:
            raise ConnectionError("Redis client is closed")

        now = float(args[0])
        limit = int(args[1])
        window = float(args[2])
        member_id = str(args[3])
        cutoff = now - window

        zset = self.zsets[key]

        # 1. ZREMRANGEBYSCORE key -inf cutoff
        to_remove = [m for m, score in zset.items() if score <= cutoff]
        for m in to_remove:
            del zset[m]

        # 2. ZCARD key
        current_count = len(zset)

        # 3. Check limit
        if current_count >= limit:
            # Find oldest score
            oldest_time = min(zset.values()) if zset else now
            retry_after = max(1, math.ceil(oldest_time + window - now))
            return [0, 0, retry_after]
        else:
            # ZADD key now member_id
            zset[member_id] = now
            remaining = max(0, limit - current_count - 1)
            return [1, remaining, 0]

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# 1. RateLimitConfig Validation
# ---------------------------------------------------------------------------


def test_create_app_rejects_redis_backend_without_storage_or_url() -> None:
    from nexora.api.errors import InvalidConfigurationError

    cfg = RateLimitConfig(enabled=True, backend="redis", redis_url=None)
    with pytest.raises(InvalidConfigurationError, match="no distributed RateLimitStorage backend"):
        create_app(
            MagicMock(ask=MagicMock()),
            enforce_auth=False,
            rate_limit_config=cfg,
            rate_limit_storage=None,
        )


def test_config_from_env_parses_redis_settings() -> None:
    env = {
        "NEXORA_RATE_LIMIT_ENABLED": "true",
        "NEXORA_RATE_LIMIT_BACKEND": "redis",
        "NEXORA_RATE_LIMIT_REDIS_URL": "redis://redis-cluster:6379/1",
        "NEXORA_RATE_LIMIT_REDIS_FAIL_CLOSED": "false",
    }
    cfg = RateLimitConfig.from_env(env)
    assert cfg.enabled is True
    assert cfg.backend == "redis"
    assert cfg.redis_url == "redis://redis-cluster:6379/1"
    assert cfg.redis_fail_closed is False


def test_config_from_env_fallback_to_redis_url_var() -> None:
    env = {
        "NEXORA_RATE_LIMIT_ENABLED": "true",
        "NEXORA_RATE_LIMIT_BACKEND": "redis",
        "REDIS_URL": "redis://fallback:6379/0",
    }
    cfg = RateLimitConfig.from_env(env)
    assert cfg.redis_url == "redis://fallback:6379/0"
    assert cfg.redis_fail_closed is True  # Default is fail closed


# ---------------------------------------------------------------------------
# 2. RedisRateLimitStorage Mechanics & Window Expiry
# ---------------------------------------------------------------------------


def test_storage_sliding_window_quota_and_expiry() -> None:
    fake_redis = FakeRedisClient()
    current_time = 1000.0

    storage = RedisRateLimitStorage(
        redis_client=fake_redis,
        time_func=lambda: current_time,
    )

    limit = 3
    window = 10

    # 1. First 3 requests allowed
    ok1, rem1, retry1 = storage.check_and_increment("test:key", limit, window)
    assert ok1 is True
    assert rem1 == 2
    assert retry1 == 0

    ok2, rem2, retry2 = storage.check_and_increment("test:key", limit, window)
    assert ok2 is True
    assert rem2 == 1

    ok3, rem3, retry3 = storage.check_and_increment("test:key", limit, window)
    assert ok3 is True
    assert rem3 == 0

    # 2. Fourth request rejected (quota exceeded)
    ok4, rem4, retry4 = storage.check_and_increment("test:key", limit, window)
    assert ok4 is False
    assert rem4 == 0
    assert retry4 > 0

    # 3. Advance time halfway through window (5 seconds) -> still rejected
    current_time = 1005.0
    ok5, rem5, retry5 = storage.check_and_increment("test:key", limit, window)
    assert ok5 is False

    # 4. Advance time past window (11 seconds) -> oldest hits expired, allowed again
    current_time = 1011.0
    ok6, rem6, retry6 = storage.check_and_increment("test:key", limit, window)
    assert ok6 is True
    assert rem6 == 2


def test_storage_ownership_and_close() -> None:
    fake_redis = FakeRedisClient()
    # Externally injected client: storage does NOT own it
    storage = RedisRateLimitStorage(redis_client=fake_redis)
    storage.close()
    assert fake_redis.closed is False  # Not closed because not owned

    # Force close
    storage.close(force=True)
    assert fake_redis.closed is True


# ---------------------------------------------------------------------------
# 3. Fail-Closed Security Policy on Redis Failure
# ---------------------------------------------------------------------------


def test_storage_fails_closed_by_default_on_redis_error() -> None:
    failing_redis = MagicMock()
    failing_redis.eval.side_effect = ConnectionError("Connection refused to Redis cluster")

    storage = RedisRateLimitStorage(
        redis_client=failing_redis,
        fail_closed=True,
    )

    allowed, remaining, retry_after = storage.check_and_increment("test:key", limit=10, window_seconds=60)
    # Fail-closed policy: deny access to protect resources during degradation
    assert allowed is False
    assert remaining == 0
    assert retry_after == 60


def test_storage_fails_open_when_explicitly_configured() -> None:
    failing_redis = MagicMock()
    failing_redis.eval.side_effect = TimeoutError("Redis socket timeout")

    storage = RedisRateLimitStorage(
        redis_client=failing_redis,
        fail_closed=False,
    )

    allowed, remaining, retry_after = storage.check_and_increment("test:key", limit=10, window_seconds=60)
    # Opt-in fail-open policy
    assert allowed is True
    assert remaining == 1
    assert retry_after == 0


# ---------------------------------------------------------------------------
# 4. Concurrency Simulation
# ---------------------------------------------------------------------------


def test_storage_atomic_concurrency_simulation() -> None:
    import concurrent.futures

    fake_redis = FakeRedisClient()
    storage = RedisRateLimitStorage(
        redis_client=fake_redis,
        time_func=time.time,
    )

    limit = 20
    window = 30
    total_workers = 50

    def make_hit() -> bool:
        allowed, _, _ = storage.check_and_increment("concurrent:client:1", limit, window)
        return allowed

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: make_hit(), range(total_workers)))

    allowed_count = sum(1 for r in results if r is True)
    denied_count = sum(1 for r in results if r is False)

    # Exactly 20 requests must be allowed, and 30 denied
    assert allowed_count == limit
    assert denied_count == total_workers - limit


# ---------------------------------------------------------------------------
# 5. Full End-to-End FastAPI App Integration with Request Correlation
# ---------------------------------------------------------------------------


def test_e2e_redis_rate_limiting_with_request_id_correlation() -> None:
    fake_redis = FakeRedisClient()
    storage = RedisRateLimitStorage(redis_client=fake_redis)

    config = RateLimitConfig(
        enabled=True,
        backend="redis",
        redis_url="redis://dummy:6379/0",
        window_seconds=60,
        research_query_limit=2,
        default_limit=10,
    )

    mock_research = MagicMock()
    mock_res = MagicMock()
    mock_res.to_dict.return_value = {
        "text": "grounded response",
        "query": "What is attention?",
        "model": "test-model",
        "evidence_count": 1,
        "usage": None,
    }
    mock_research.ask.return_value = mock_res

    app = create_app(
        mock_research,
        rate_limit_config=config,
        rate_limit_storage=storage,
        enforce_auth=False,
    )

    client = TestClient(app)

    # Request 1: Allowed
    r1 = client.post(
        "/api/v1/research",
        json={"query": "What is attention?"},
        headers={"X-Request-ID": "req-trace-1"},
    )
    assert r1.status_code == 200
    assert r1.headers.get("X-Request-ID") == "req-trace-1"
    assert r1.headers.get("X-RateLimit-Limit") == "2"
    assert r1.headers.get("X-RateLimit-Remaining") == "1"

    # Request 2: Allowed
    r2 = client.post(
        "/api/v1/research",
        json={"query": "What is transformer?"},
        headers={"X-Request-ID": "req-trace-2"},
    )
    assert r2.status_code == 200
    assert r2.headers.get("X-Request-ID") == "req-trace-2"
    assert r2.headers.get("X-RateLimit-Limit") == "2"
    assert r2.headers.get("X-RateLimit-Remaining") == "0"

    # Request 3: Rate Limited (429)
    r3 = client.post(
        "/api/v1/research",
        json={"query": "What is BERT?"},
        headers={"X-Request-ID": "req-trace-3"},
    )
    assert r3.status_code == 429
    # Crucial: Request ID must be echoed even on 429 rate limit responses!
    assert r3.headers.get("X-Request-ID") == "req-trace-3"
    assert r3.headers.get("Retry-After") is not None
    assert r3.headers.get("X-RateLimit-Limit") == "2"
    assert r3.headers.get("X-RateLimit-Remaining") == "0"
    data = r3.json()
    assert data["error"]["code"] == "rate_limited"
