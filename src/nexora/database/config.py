"""PostgreSQL connection configuration.

Pure configuration: no connection is opened, no driver is imported, and the
environment is read only inside ``DatabaseConfig.from_env``.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

_SCHEME_PATTERN = re.compile(r"^postgresql(\+[a-z][a-z0-9_]*)?$")
_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})


class DatabaseConfigurationError(ValueError):
    """Invalid database configuration. Messages never contain the URL or password."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, repr=False)
class DatabaseConfig:
    """Immutable PostgreSQL pool configuration.

    Defaults (production-oriented, SQLAlchemy-compatible):
      pool_size=5          persistent connections kept in the pool
      max_overflow=10      extra connections allowed under load
      pool_timeout=30.0    seconds to wait for a free connection
      pool_recycle=1800    recycle connections after 30 minutes (-1 disables)
      echo=False           SQL logging off
    """

    url: str = field(repr=False)
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 1800
    echo: bool = False

    def __post_init__(self) -> None:
        self._validate_url(self.url)
        if not _is_int(self.pool_size) or self.pool_size < 1:
            raise DatabaseConfigurationError("pool_size must be an integer >= 1")
        if not _is_int(self.max_overflow) or self.max_overflow < 0:
            raise DatabaseConfigurationError("max_overflow must be an integer >= 0")
        if (
            isinstance(self.pool_timeout, bool)
            or not isinstance(self.pool_timeout, (int, float))
            or not math.isfinite(self.pool_timeout)
            or self.pool_timeout <= 0
        ):
            raise DatabaseConfigurationError("pool_timeout must be a finite number > 0")
        if not _is_int(self.pool_recycle) or self.pool_recycle < -1:
            raise DatabaseConfigurationError("pool_recycle must be an integer >= -1")
        if not isinstance(self.echo, bool):
            raise DatabaseConfigurationError("echo must be a boolean")

    @staticmethod
    def _validate_url(url: Any) -> None:
        if not isinstance(url, str):
            raise DatabaseConfigurationError("url must be a string")
        if not url.strip():
            raise DatabaseConfigurationError("url must not be empty")
        try:
            parts = urlsplit(url.strip())
            parts.port  # raises ValueError for a malformed port
        except ValueError:
            raise DatabaseConfigurationError("url is not a valid URL") from None
        if "://" not in url or not _SCHEME_PATTERN.match(parts.scheme):
            raise DatabaseConfigurationError(
                "url must be a PostgreSQL URL (postgresql://, postgresql+<driver>://)"
            )

    def redacted_url(self) -> str:
        """The URL with any password replaced by ``***`` (safe for logs)."""
        parts = urlsplit(self.url.strip())
        netloc = parts.netloc
        if "@" in netloc:
            userinfo, host = netloc.rsplit("@", 1)
            if ":" in userinfo:
                netloc = f"{userinfo.split(':', 1)[0]}:***@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

    def __repr__(self) -> str:
        return (
            f"DatabaseConfig(url={self.redacted_url()!r}, pool_size={self.pool_size!r}, "
            f"max_overflow={self.max_overflow!r}, pool_timeout={self.pool_timeout!r}, "
            f"pool_recycle={self.pool_recycle!r}, echo={self.echo!r})"
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> DatabaseConfig:
        """Load configuration from NEXORA_DATABASE_* variables.

        NEXORA_DATABASE_URL is required. The optional variables fall back to the
        defaults when unset or blank; any other invalid value raises
        DatabaseConfigurationError (it is never silently ignored).
        """
        env = os.environ if environ is None else environ
        url = env.get("NEXORA_DATABASE_URL")
        if url is None or not url.strip():
            raise DatabaseConfigurationError("NEXORA_DATABASE_URL is required")

        kwargs: dict[str, Any] = {}
        for name, key, parse in (
            ("NEXORA_DATABASE_POOL_SIZE", "pool_size", _parse_int),
            ("NEXORA_DATABASE_MAX_OVERFLOW", "max_overflow", _parse_int),
            ("NEXORA_DATABASE_POOL_TIMEOUT", "pool_timeout", _parse_float),
            ("NEXORA_DATABASE_POOL_RECYCLE", "pool_recycle", _parse_int),
            ("NEXORA_DATABASE_ECHO", "echo", _parse_bool),
        ):
            raw = env.get(name)
            if raw is not None and raw.strip():
                kwargs[key] = parse(name, raw.strip())
        return cls(url=url.strip(), **kwargs)


def _parse_int(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise DatabaseConfigurationError(f"{name} must be an integer") from None


def _parse_float(name: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise DatabaseConfigurationError(f"{name} must be a number") from None


def _parse_bool(name: str, raw: str) -> bool:
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise DatabaseConfigurationError(f"{name} must be one of true/false, 1/0, yes/no, on/off")
