"""Inbound rate limiting middleware, storage protocols, and caller identity resolution."""

from __future__ import annotations

import collections
import hashlib
import time
from typing import Callable, Protocol, Sequence
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from nexora.config.rate_limit import RateLimitConfig, RateLimitConfigurationError


class RateLimitStorage(Protocol):
    """Protocol for rate limit counters.

    Implementations can be in-memory (single process) or distributed (e.g. Redis).
    """

    def check_and_increment(
        self, key: str, limit: int, window_seconds: int
    ) -> tuple[bool, int, int]:
        """Check if request for `key` is allowed and record the hit.

        Args:
            key: Rate limit key (e.g., "heavy_analysis:token:a1b2c3d4").
            limit: Maximum allowed requests in the time window.
            window_seconds: Time window duration in seconds.

        Returns:
            tuple of (allowed: bool, remaining: int, retry_after: int)
        """
        ...


class InMemoryRateLimitStorage:
    """Thread-safe sliding-window rate limit storage in local process memory.

    Deployment Notice:
        This storage tracks request counts in local process memory. In deployments
        running multiple worker processes (such as Uvicorn with `--workers N`) or
        distributed containers across Kubernetes pods, each worker maintains its
        own independent memory. For global rate enforcement across multiple
        processes, use a shared storage backend (e.g. Redis) implementing
        the RateLimitStorage protocol.
    """

    def __init__(self, time_func: Callable[[], float] = time.time) -> None:
        import threading

        self._lock = threading.Lock()
        self._history: dict[str, collections.deque[float]] = {}
        self._time_func = time_func

    def check_and_increment(
        self, key: str, limit: int, window_seconds: int
    ) -> tuple[bool, int, int]:
        now = self._time_func()
        cutoff = now - window_seconds

        with self._lock:
            deq = self._history.setdefault(key, collections.deque())

            # Evict timestamps outside sliding window
            while deq and deq[0] <= cutoff:
                deq.popleft()

            if len(deq) >= limit:
                # Limit exceeded; calculate retry_after from oldest hit in window
                oldest = deq[0]
                retry_after = max(1, int(oldest + window_seconds - now))
                return False, 0, retry_after

            # Hit allowed; record current timestamp
            deq.append(now)
            remaining = max(0, limit - len(deq))
            return True, remaining, 0

    def clear(self) -> None:
        """Reset all in-memory counters (for test teardown)."""
        with self._lock:
            self._history.clear()

    def check_readiness(self) -> tuple[bool, str]:
        """Check if in-memory rate limit storage is operational."""
        return True, "in_memory_ready"


_LUA_SLIDING_WINDOW = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local window = tonumber(ARGV[3])
local member_id = ARGV[4]
local cutoff = now - window

-- 1. Evict entries outside the sliding window
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)

-- 2. Count current hits in the window
local current_count = redis.call('ZCARD', key)

if current_count >= limit then
    -- Limit exceeded; find oldest timestamp to compute retry_after
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = 1
    if oldest and #oldest >= 2 then
        local oldest_time = tonumber(oldest[2])
        retry_after = math.max(1, math.ceil(oldest_time + window - now))
    end
    return {0, 0, retry_after}
else
    -- Hit allowed; record current member and score
    redis.call('ZADD', key, now, member_id)
    redis.call('EXPIRE', key, math.ceil(window) + 2)
    local remaining = math.max(0, limit - current_count - 1)
    return {1, remaining, 0}
end
"""


class RedisRateLimitStorage:
    """Atomic, distributed sliding-window rate limit storage backed by Redis.

    Uses an atomic Lua script executing sliding-window log pruning and counting,
    preventing race conditions under high concurrency across multiple workers/nodes.
    """

    def __init__(
        self,
        redis_client: Any = None,
        redis_url: str | None = None,
        *,
        prefix: str = "nexora:ratelimit:",
        connect_timeout: float = 2.0,
        socket_timeout: float = 2.0,
        fail_closed: bool = True,
        time_func: Callable[[], float] = time.time,
    ) -> None:
        self._prefix = prefix
        self._fail_closed = fail_closed
        self._time_func = time_func
        self._owns_client = False

        if redis_client is not None:
            self._client = redis_client
        elif redis_url:
            import redis

            self._client = redis.from_url(
                redis_url,
                socket_connect_timeout=connect_timeout,
                socket_timeout=socket_timeout,
                decode_responses=True,
            )
            self._owns_client = True
        else:
            raise RateLimitConfigurationError(
                "Either redis_client or redis_url must be provided for RedisRateLimitStorage"
            )

        self._lua_script = _LUA_SLIDING_WINDOW

    def check_and_increment(
        self, key: str, limit: int, window_seconds: int
    ) -> tuple[bool, int, int]:
        """Atomically check and record a rate-limit hit using a Redis Lua script."""
        import logging
        import uuid

        now = self._time_func()
        member_id = f"{now}:{uuid.uuid4().hex[:8]}"
        full_key = f"{self._prefix}{key}"

        try:
            # Execute atomic sliding-window Lua script
            eval_fn = getattr(self._client, "eval", None)
            if not callable(eval_fn):
                raise RuntimeError("Redis client does not support eval()")

            res = eval_fn(
                self._lua_script,
                1,
                full_key,
                str(now),
                str(limit),
                str(window_seconds),
                member_id,
            )
            # res is expected to be [allowed, remaining, retry_after]
            allowed = bool(res[0])
            remaining = int(res[1])
            retry_after = int(res[2])
            return allowed, remaining, retry_after

        except Exception as exc:
            logging.getLogger(__name__).error(
                "Redis rate limit evaluation failed for key %s (%s): %s",
                full_key,
                type(exc).__name__,
                exc,
            )
            if self._fail_closed:
                # Security policy: fail closed to prevent DDoS / abuse on infrastructure degradation
                return False, 0, max(1, window_seconds)
            # Explicit opt-in fail-open policy
            return True, 1, 0

    def close(self, force: bool = False) -> None:
        """Close underlying Redis connection pool if owned (or if forced)."""
        if (getattr(self, "_owns_client", False) or force) and self._client is not None and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:
                pass

    def check_readiness(self) -> tuple[bool, str]:
        """Check if Redis connection is responsive via PING."""
        try:
            ping_fn = getattr(self._client, "ping", None)
            if callable(ping_fn):
                res = ping_fn()
                if res:
                    return True, "redis_ready"
            return False, "redis_ping_failed"
        except Exception as exc:
            return False, f"redis_unavailable: {type(exc).__name__}"



def get_caller_identity(request: Request, trusted_proxies: Sequence[str] = ()) -> str:
    """Derive a tamper-resistant caller identifier for rate limiting.

    Security Rules:
      1. If authenticated via Bearer token or X-API-Key:
         Hash the raw key using SHA-256 and use a 16-character digest prefix.
         Never log, leak, or store the raw secret key.
      2. If unauthenticated:
         Use the remote client IP.
         Do not trust arbitrary client headers (e.g. X-User-Id, Client-IP).
         Only inspect X-Forwarded-For if request.client.host is explicitly
         in trusted_proxies.
    """
    # 1. Bearer Token in Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.strip().startswith("Bearer "):
        token = auth_header.strip()[7:].strip()
        if token:
            key_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
            return f"key:{key_hash}"

    # 2. X-API-Key header
    x_api_key = request.headers.get("X-API-Key")
    if x_api_key and x_api_key.strip():
        token = x_api_key.strip()
        key_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
        return f"key:{key_hash}"

    # 3. Client IP resolution
    client_host = request.client.host if request.client else "127.0.0.1"
    if trusted_proxies and client_host in trusted_proxies:
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            hops = [ip.strip() for ip in xff.split(",") if ip.strip()]
            for hop in reversed(hops):
                if hop not in trusted_proxies:
                    return f"ip:{hop}"

    return f"ip:{client_host}"


def categorize_route(path: str, method: str) -> str | None:
    """Categorize request into an endpoint cost tier.

    Returns None for unmetered endpoints (e.g. /health, /docs).
    """
    clean_path = path.rstrip("/")

    # Exclude system and documentation routes
    if clean_path in ("/health", "/ready", "/health/ready", "/docs", "/openapi.json", "/redoc") or clean_path.startswith("/docs/"):
        return None

    # Heavy analysis & synthesis operations
    if (
        clean_path == "/api/v1/papers/analyze"
        or (clean_path.startswith("/api/v1/papers/") and clean_path.endswith("/analyze"))
        or clean_path == "/api/v1/papers/compare"
        or clean_path == "/api/v1/research/compare"
        or clean_path == "/api/v1/research/gaps"
        or (clean_path.startswith("/api/v1/papers/") and clean_path.endswith("/gaps"))
        or clean_path == "/api/v1/research/verify"
    ):
        return "heavy_analysis"

    # Discovery search
    if clean_path == "/api/v1/discovery/search":
        return "discovery_search"

    # Discovery import
    if clean_path == "/api/v1/discovery/import":
        return "discovery_import"

    # Document upload/ingest
    if clean_path == "/api/v1/documents/ingest":
        return "document_ingest"

    # Research / RAG query
    if (
        clean_path == "/api/v1/research"
        or clean_path == "/api/v1/research/query"
        or clean_path == "/api/v1/rag/query"
    ):
        return "research_query"

    # Default API reads and resource metadata
    if clean_path.startswith("/api/v1/"):
        return "default"

    return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Inbound rate limiting middleware enforcing tiered limits by endpoint cost."""

    def __init__(
        self,
        app: Any,
        config: RateLimitConfig,
        storage: RateLimitStorage | None = None,
    ) -> None:
        super().__init__(app)
        self.config = config
        if config.enabled and config.backend == "redis" and storage is None:
            if config.redis_url:
                storage = RedisRateLimitStorage(
                    redis_url=config.redis_url,
                    fail_closed=config.redis_fail_closed,
                )
            else:
                raise RateLimitConfigurationError(
                    "Redis rate limit backend ('redis') is configured, but no distributed RateLimitStorage backend was provided. "
                    "Set NEXORA_RATE_LIMIT_REDIS_URL or pass a custom RateLimitStorage instance."
                )
        self.storage = storage or InMemoryRateLimitStorage()

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        if not self.config.enabled:
            return await call_next(request)

        category = categorize_route(request.url.path, request.method)
        if category is None:
            return await call_next(request)

        limit = self._resolve_limit(category)
        if limit <= 0:
            return await call_next(request)

        caller_id = get_caller_identity(request, self.config.trusted_proxies)
        key = f"{category}:{caller_id}"

        allowed, remaining, retry_after = self.storage.check_and_increment(
            key, limit, self.config.window_seconds
        )

        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after),
                },
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": (
                            f"Rate limit exceeded for category '{category}'. "
                            f"Please retry after {retry_after} seconds."
                        ),
                    }
                },
            )

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(self.config.window_seconds)
        return response

    def _resolve_limit(self, category: str) -> int:
        if category == "heavy_analysis":
            return self.config.heavy_analysis_limit
        if category == "research_query":
            return self.config.research_query_limit
        if category == "discovery_search":
            return self.config.discovery_search_limit
        if category == "discovery_import":
            return self.config.discovery_import_limit
        if category == "document_ingest":
            return self.config.document_ingest_limit
        return self.config.default_limit
