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
    if clean_path in ("/health", "/docs", "/openapi.json", "/redoc") or clean_path.startswith("/docs/"):
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
            raise RateLimitConfigurationError(
                "Redis rate limit backend ('redis') is configured, but no distributed RateLimitStorage backend was provided. "
                "Pass a custom RateLimitStorage instance or set backend to 'memory'."
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
