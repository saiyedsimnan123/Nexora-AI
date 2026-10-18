"""Inbound rate limiting configuration for Nexora API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


class RateLimitConfigurationError(ValueError):
    """Invalid rate limit configuration."""


_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})
SUPPORTED_BACKENDS = ("memory", "redis")


def _parse_bool(name: str, raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise RateLimitConfigurationError(f"{name} must be one of true/false, 1/0, yes/no, on/off")


def _parse_int(name: str, raw: str) -> int:
    try:
        val = int(raw.strip())
        if val <= 0:
            raise ValueError()
        return val
    except ValueError:
        raise RateLimitConfigurationError(f"{name} must be a positive integer") from None


@dataclass(frozen=True)
class RateLimitConfig:
    """Configurable inbound rate limits by endpoint cost and caller.

    Cost Tiers (requests per window):
      - heavy_analysis_limit: multi-paper comparison, structured analysis, research gaps, claim verification.
      - research_query_limit: vector + BM25 hybrid grounded RAG queries.
      - discovery_search_limit: external arXiv search requests.
      - discovery_import_limit: safe PDF downloading, parsing, and collection import.
      - document_ingest_limit: direct document file upload, parsing, and vector indexing.
      - default_limit: standard read/metadata queries.
    """

    enabled: bool = False
    window_seconds: int = 60
    heavy_analysis_limit: int = 10
    research_query_limit: int = 30
    discovery_search_limit: int = 20
    discovery_import_limit: int = 10
    document_ingest_limit: int = 10
    default_limit: int = 120
    backend: str = "memory"
    trusted_proxies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise RateLimitConfigurationError("enabled must be a boolean")
        for name, val in (
            ("window_seconds", self.window_seconds),
            ("heavy_analysis_limit", self.heavy_analysis_limit),
            ("research_query_limit", self.research_query_limit),
            ("discovery_search_limit", self.discovery_search_limit),
            ("discovery_import_limit", self.discovery_import_limit),
            ("document_ingest_limit", self.document_ingest_limit),
            ("default_limit", self.default_limit),
        ):
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                raise RateLimitConfigurationError(f"{name} must be a positive integer")

        if self.backend not in SUPPORTED_BACKENDS:
            raise RateLimitConfigurationError(
                f"backend must be one of {SUPPORTED_BACKENDS}, got {self.backend!r}"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RateLimitConfig:
        """Load rate limit configuration from environment variables.

        Variables:
          NEXORA_RATE_LIMIT_ENABLED: bool (defaults to False unless configured)
          NEXORA_RATE_LIMIT_WINDOW_SECONDS: int (default 60)
          NEXORA_RATE_LIMIT_HEAVY_ANALYSIS: int (default 10)
          NEXORA_RATE_LIMIT_RESEARCH_QUERY: int (default 30)
          NEXORA_RATE_LIMIT_DISCOVERY_SEARCH: int (default 20)
          NEXORA_RATE_LIMIT_DISCOVERY_IMPORT: int (default 10)
          NEXORA_RATE_LIMIT_DOCUMENT_INGEST: int (default 10)
          NEXORA_RATE_LIMIT_DEFAULT: int (default 120)
          NEXORA_RATE_LIMIT_BACKEND: str (default "memory")
          NEXORA_RATE_LIMIT_TRUSTED_PROXIES: comma-separated list of IP addresses
        """
        env = os.environ if environ is None else environ

        raw_enabled = env.get("NEXORA_RATE_LIMIT_ENABLED")
        enabled = False
        if raw_enabled is not None and raw_enabled.strip():
            enabled = _parse_bool("NEXORA_RATE_LIMIT_ENABLED", raw_enabled)

        kwargs: dict[str, Any] = {"enabled": enabled}

        for var, field_name in (
            ("NEXORA_RATE_LIMIT_WINDOW_SECONDS", "window_seconds"),
            ("NEXORA_RATE_LIMIT_HEAVY_ANALYSIS", "heavy_analysis_limit"),
            ("NEXORA_RATE_LIMIT_RESEARCH_QUERY", "research_query_limit"),
            ("NEXORA_RATE_LIMIT_DISCOVERY_SEARCH", "discovery_search_limit"),
            ("NEXORA_RATE_LIMIT_DISCOVERY_IMPORT", "discovery_import_limit"),
            ("NEXORA_RATE_LIMIT_DOCUMENT_INGEST", "document_ingest_limit"),
            ("NEXORA_RATE_LIMIT_DEFAULT", "default_limit"),
        ):
            raw = env.get(var)
            if raw is not None and raw.strip():
                kwargs[field_name] = _parse_int(var, raw)

        raw_backend = env.get("NEXORA_RATE_LIMIT_BACKEND")
        if raw_backend is not None and raw_backend.strip():
            kwargs["backend"] = raw_backend.strip().lower()

        raw_proxies = env.get("NEXORA_RATE_LIMIT_TRUSTED_PROXIES")
        if raw_proxies is not None and raw_proxies.strip():
            kwargs["trusted_proxies"] = tuple(
                p.strip() for p in raw_proxies.split(",") if p.strip()
            )

        return cls(**kwargs)
