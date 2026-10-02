"""Validated, offline configuration for the future Qdrant vector store.

``QdrantConfig`` holds only what a Qdrant client will need in order to be
constructed later: server URL, optional API key, collection name and request
timeout. It is deliberately inert. Creating one, or calling
:meth:`QdrantConfig.from_env`, never contacts Qdrant, never creates a client
and never touches the network; the Qdrant SDK is not imported here.

Environment variables read by :meth:`QdrantConfig.from_env` (and only there,
never at import time):

* ``NEXORA_QDRANT_URL`` - required, ``http://`` or ``https://`` URL.
* ``NEXORA_QDRANT_API_KEY`` - optional.
* ``NEXORA_QDRANT_COLLECTION`` - optional, defaults to ``"nexora"``.
* ``NEXORA_QDRANT_TIMEOUT`` - optional, seconds, defaults to ``30.0``.

Surrounding whitespace is stripped from environment values, and an optional
variable that is unset or blank is treated as absent.

The API key is a secret: it is excluded from ``repr()`` and is never included
in error messages raised by this module.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Self
from urllib.parse import urlsplit

__all__ = ["QdrantConfig"]

ENV_QDRANT_URL = "NEXORA_QDRANT_URL"
ENV_QDRANT_API_KEY = "NEXORA_QDRANT_API_KEY"
ENV_QDRANT_COLLECTION = "NEXORA_QDRANT_COLLECTION"
ENV_QDRANT_TIMEOUT = "NEXORA_QDRANT_TIMEOUT"

DEFAULT_COLLECTION_NAME = "nexora"
DEFAULT_TIMEOUT = 30.0

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _validate_url(url: object) -> None:
    if not isinstance(url, str):
        raise TypeError(f"url must be a string, got {type(url).__name__}")
    if not url.strip():
        raise ValueError("url must not be empty")
    try:
        parts = urlsplit(url)
        # Accessing .port validates the port component (non-numeric / range).
        parts.port
    except ValueError:
        raise ValueError("url is not a valid URL") from None
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise ValueError("url must start with http:// or https://")
    if not parts.hostname:
        raise ValueError("url must include a host")


def _validate_api_key(api_key: object) -> None:
    if api_key is None:
        return
    if not isinstance(api_key, str):
        raise TypeError(f"api_key must be a string or None, got {type(api_key).__name__}")
    if not api_key.strip():
        raise ValueError("api_key must not be empty; use None for no API key")


def _validate_collection_name(name: object) -> None:
    if not isinstance(name, str):
        raise TypeError(f"collection_name must be a string, got {type(name).__name__}")
    if not name.strip():
        raise ValueError("collection_name must not be empty")
    # The name is used as a URL path segment by Qdrant's REST API.
    if "/" in name or any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise ValueError("collection_name must not contain '/' or control characters")


def _validate_timeout(timeout: object) -> None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError(f"timeout must be an int or float, got {type(timeout).__name__}")
    if isinstance(timeout, float) and not math.isfinite(timeout):
        raise ValueError("timeout must be finite")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")


@dataclass(frozen=True)
class QdrantConfig:
    """Immutable, validated Qdrant connection settings (no I/O)."""

    url: str
    api_key: str | None = field(default=None, repr=False)
    collection_name: str = DEFAULT_COLLECTION_NAME
    timeout: float = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        _validate_url(self.url)
        _validate_api_key(self.api_key)
        _validate_collection_name(self.collection_name)
        _validate_timeout(self.timeout)

    @classmethod
    def from_env(cls) -> Self:
        """Build a config from ``NEXORA_QDRANT_*`` environment variables."""
        url = _read_env(ENV_QDRANT_URL)
        if url is None:
            raise ValueError(
                f"{ENV_QDRANT_URL} environment variable is required and must not be empty"
            )

        api_key = _read_env(ENV_QDRANT_API_KEY)
        collection_name = _read_env(ENV_QDRANT_COLLECTION) or DEFAULT_COLLECTION_NAME

        raw_timeout = _read_env(ENV_QDRANT_TIMEOUT)
        if raw_timeout is None:
            timeout = DEFAULT_TIMEOUT
        else:
            try:
                timeout = float(raw_timeout)
            except ValueError:
                raise ValueError(f"{ENV_QDRANT_TIMEOUT} must be a number of seconds") from None

        return cls(
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            timeout=timeout,
        )


def _read_env(name: str) -> str | None:
    """Return the stripped value of ``name``, or None if unset or blank."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None
