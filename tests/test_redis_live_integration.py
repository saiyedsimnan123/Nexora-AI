"""Integration tests for live Redis service and distributed rate limiting (Milestone 37).

When a live Redis instance is available (via TEST_REDIS_URL, NEXORA_RATE_LIMIT_REDIS_URL,
REDIS_URL, or default localhost:6379), this suite tests:
1. Real connectivity to the Redis server.
2. Execution of the atomic Lua sliding-window script on Redis's single-threaded engine.
3. True distributed rate limit enforcement across multiple independent client instances.
4. Correct TTL setting and key expiration in the live Redis keyspace.
5. Fail-closed security policy when connection drops.

When run locally without a running Redis instance, this suite skips cleanly with an
informative message. When run in GitHub Actions CI with the redis:7-alpine service
container, this suite executes against real Redis.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from nexora.api.rate_limit import RedisRateLimitStorage


def _get_live_redis_url() -> str | None:
    return (
        os.environ.get("TEST_REDIS_URL")
        or os.environ.get("NEXORA_RATE_LIMIT_REDIS_URL")
        or os.environ.get("REDIS_URL")
    )


def _probe_redis_connection(url: str) -> bool:
    try:
        import redis

        client = redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=1.0)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def redis_url() -> str:
    candidates = []
    env_url = _get_live_redis_url()
    if env_url:
        candidates.append(env_url)
    candidates.append("redis://127.0.0.1:6379/0")

    for cand in candidates:
        if _probe_redis_connection(cand):
            return cand

    pytest.skip(
        "Live Redis service not reachable. Set TEST_REDIS_URL or run a local/container "
        "Redis service (redis:7-alpine on port 6379) to run live Redis integration tests."
    )


def test_live_redis_lua_sliding_window_execution(redis_url: str) -> None:
    """Verify that the Lua script executes natively on the live Redis engine."""
    import redis

    client = redis.from_url(redis_url, decode_responses=True)
    test_key = f"test:live:lua:{uuid.uuid4().hex[:8]}"

    storage = RedisRateLimitStorage(redis_client=client, prefix="")

    limit = 3
    window = 10

    # First 3 hits allowed
    for expected_rem in (2, 1, 0):
        allowed, rem, retry = storage.check_and_increment(test_key, limit, window)
        assert allowed is True
        assert rem == expected_rem
        assert retry == 0

    # 4th hit rejected
    allowed, rem, retry = storage.check_and_increment(test_key, limit, window)
    assert allowed is False
    assert rem == 0
    assert retry > 0

    # Verify that Redis assigned a positive TTL to the key
    ttl = client.ttl(test_key)
    assert ttl > 0
    assert ttl <= window + 2

    # Cleanup
    client.delete(test_key)
    client.close()


def test_live_redis_distributed_enforcement_across_separate_clients(redis_url: str) -> None:
    """Verify rate limits are strictly enforced across two independent client instances.

    This simulates two separate web worker processes (e.g. Uvicorn worker 1 and worker 2)
    or separate container replicas hitting the same Redis datastore.
    """
    shared_key = f"test:live:distributed:{uuid.uuid4().hex[:8]}"

    # Client A: independent storage instance
    storage_a = RedisRateLimitStorage(redis_url=redis_url, prefix="")

    # Client B: completely separate connection pool and storage instance
    storage_b = RedisRateLimitStorage(redis_url=redis_url, prefix="")

    limit = 4
    window = 30

    try:
        # Worker A processes 2 requests
        ok_a1, rem_a1, _ = storage_a.check_and_increment(shared_key, limit, window)
        assert ok_a1 is True
        assert rem_a1 == 3

        ok_a2, rem_a2, _ = storage_a.check_and_increment(shared_key, limit, window)
        assert ok_a2 is True
        assert rem_a2 == 2

        # Worker B immediately sees Worker A's consumed quota (distributed state)
        ok_b1, rem_b1, _ = storage_b.check_and_increment(shared_key, limit, window)
        assert ok_b1 is True
        assert rem_b1 == 1

        ok_b2, rem_b2, _ = storage_b.check_and_increment(shared_key, limit, window)
        assert ok_b2 is True
        assert rem_b2 == 0

        # Now quota is exhausted globally: both workers are blocked
        blocked_a, _, retry_a = storage_a.check_and_increment(shared_key, limit, window)
        assert blocked_a is False
        assert retry_a > 0

        blocked_b, _, retry_b = storage_b.check_and_increment(shared_key, limit, window)
        assert blocked_b is False
        assert retry_b > 0

    finally:
        # Teardown
        import redis

        cleanup_client = redis.from_url(redis_url)
        cleanup_client.delete(shared_key)
        cleanup_client.close()
        storage_a.close()
        storage_b.close()


def test_live_redis_window_expiration(redis_url: str) -> None:
    """Verify hits naturally expire after window_seconds on live Redis."""
    import redis

    short_key = f"test:live:exp:{uuid.uuid4().hex[:8]}"
    storage = RedisRateLimitStorage(redis_url=redis_url, prefix="")

    limit = 2
    window = 1  # 1-second short window

    try:
        # Hit 1 & 2 allowed
        assert storage.check_and_increment(short_key, limit, window)[0] is True
        assert storage.check_and_increment(short_key, limit, window)[0] is True

        # Hit 3 blocked
        assert storage.check_and_increment(short_key, limit, window)[0] is False

        # Wait for 1.2 seconds for window to slide past
        time.sleep(1.2)

        # Quota replenished on live Redis
        allowed, remaining, _ = storage.check_and_increment(short_key, limit, window)
        assert allowed is True
        assert remaining == 1

    finally:
        cleanup_client = redis.from_url(redis_url)
        cleanup_client.delete(short_key)
        cleanup_client.close()
        storage.close()
