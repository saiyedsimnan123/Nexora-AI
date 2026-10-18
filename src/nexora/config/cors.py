"""CORS configuration for Nexora API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


class CORSConfigurationError(ValueError):
    """Invalid CORS configuration."""


_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})


def _parse_bool(name: str, raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise CORSConfigurationError(f"{name} must be one of true/false, 1/0, yes/no, on/off")


@dataclass(frozen=True)
class CORSConfig:
    """CORS policy configuration.

    Security Rules:
      - Origins are empty (disabled) by default.
      - Wildcard '*' is strictly prohibited when allow_credentials is True.
      - Origins must be explicit scheme + host + optional port (or '*').
    """

    enabled: bool = False
    allow_origins: tuple[str, ...] = ()
    allow_credentials: bool = False
    allow_methods: tuple[str, ...] = ("GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD")
    allow_headers: tuple[str, ...] = ("Authorization", "Content-Type", "Accept", "X-API-Key")
    max_age: int = 600

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise CORSConfigurationError("enabled must be a boolean")
        if not isinstance(self.allow_credentials, bool):
            raise CORSConfigurationError("allow_credentials must be a boolean")
        if not isinstance(self.max_age, int) or isinstance(self.max_age, bool) or self.max_age < 0:
            raise CORSConfigurationError("max_age must be a non-negative integer")

        if self.allow_credentials and "*" in self.allow_origins:
            raise CORSConfigurationError(
                "CORS allow_origins cannot contain wildcard '*' when allow_credentials is True"
            )

        for origin in self.allow_origins:
            if not isinstance(origin, str) or not origin.strip():
                raise CORSConfigurationError("Each origin in allow_origins must be a non-empty string")
            cleaned = origin.strip()
            if cleaned != "*" and not (cleaned.startswith("http://") or cleaned.startswith("https://")):
                raise CORSConfigurationError(
                    f"Invalid origin {origin!r}: must start with http:// or https:// (or '*')"
                )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> CORSConfig:
        """Load CORS configuration from environment variables.

        Variables:
          NEXORA_CORS_ENABLED: bool
          NEXORA_CORS_ALLOWED_ORIGINS: comma-separated list of origins
          NEXORA_CORS_ALLOW_CREDENTIALS: bool
          NEXORA_CORS_ALLOW_METHODS: comma-separated list of HTTP methods
          NEXORA_CORS_ALLOW_HEADERS: comma-separated list of headers
          NEXORA_CORS_MAX_AGE: integer seconds
        """
        env = os.environ if environ is None else environ

        raw_origins = env.get("NEXORA_CORS_ALLOWED_ORIGINS")
        origins: list[str] = []
        if raw_origins is not None and raw_origins.strip():
            origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

        raw_enabled = env.get("NEXORA_CORS_ENABLED")
        if raw_enabled is not None and raw_enabled.strip():
            enabled = _parse_bool("NEXORA_CORS_ENABLED", raw_enabled)
        else:
            # Enable automatically if allowed origins are explicitly supplied
            enabled = len(origins) > 0

        raw_creds = env.get("NEXORA_CORS_ALLOW_CREDENTIALS")
        allow_credentials = (
            _parse_bool("NEXORA_CORS_ALLOW_CREDENTIALS", raw_creds)
            if raw_creds is not None and raw_creds.strip()
            else False
        )

        raw_methods = env.get("NEXORA_CORS_ALLOW_METHODS")
        methods: tuple[str, ...] = ("GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD")
        if raw_methods is not None and raw_methods.strip():
            methods = tuple(m.strip().upper() for m in raw_methods.split(",") if m.strip())

        raw_headers = env.get("NEXORA_CORS_ALLOW_HEADERS")
        headers: tuple[str, ...] = ("Authorization", "Content-Type", "Accept", "X-API-Key")
        if raw_headers is not None and raw_headers.strip():
            headers = tuple(h.strip() for h in raw_headers.split(",") if h.strip())

        raw_max_age = env.get("NEXORA_CORS_MAX_AGE")
        max_age = 600
        if raw_max_age is not None and raw_max_age.strip():
            try:
                max_age = int(raw_max_age.strip())
            except ValueError:
                raise CORSConfigurationError("NEXORA_CORS_MAX_AGE must be an integer") from None

        return cls(
            enabled=enabled,
            allow_origins=tuple(origins),
            allow_credentials=allow_credentials,
            allow_methods=methods,
            allow_headers=headers,
            max_age=max_age,
        )
