"""Configuration model for selecting and configuring an embedding provider.

``EmbeddingConfig`` only *describes* a provider. It performs no network calls
and has no provider-specific logic, so the rest of Nexora can stay independent
of which provider is in use.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

ENV_PROVIDER = "NEXORA_EMBEDDING_PROVIDER"
ENV_MODEL = "NEXORA_EMBEDDING_MODEL"
ENV_API_KEY = "NEXORA_EMBEDDING_API_KEY"
ENV_BASE_URL = "NEXORA_EMBEDDING_BASE_URL"
ENV_DIMENSION = "NEXORA_EMBEDDING_DIMENSION"
ENV_BATCH_SIZE = "NEXORA_EMBEDDING_BATCH_SIZE"

DEFAULT_PROVIDER = "hash"
DEFAULT_MODEL = "hash-sha256"


def _require_name(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a str, got {type(value).__name__}"
        )
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty or blank")


def _check_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an int, got {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


def _env_text(environ: Mapping[str, str], name: str) -> str | None:
    """Return the stripped value of ``name``, or None if unset or blank."""
    value = environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _env_int(environ: Mapping[str, str], name: str) -> int | None:
    value = _env_text(environ, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


@dataclass(frozen=True)
class EmbeddingConfig:
    """Immutable, validated description of an embedding provider.

    Attributes:
        provider: Provider name used later to select an implementation.
        model: Model name for that provider.
        api_key: Optional secret. Local providers do not need one. It is
            excluded from ``repr()``/``str()``; use ``has_api_key`` to check
            presence without exposing it.
        base_url: Optional http(s) base URL (e.g. a self-hosted endpoint).
        dimension: Embedding dimension, when known.
        batch_size: Optional batch size; ``None`` lets the provider decide.
    """

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    dimension: int | None = None
    batch_size: int | None = None

    def __post_init__(self) -> None:
        _require_name("provider", self.provider)
        _require_name("model", self.model)

        if self.api_key is not None:
            if not isinstance(self.api_key, str):
                raise TypeError(
                    f"api_key must be a str or None, got "
                    f"{type(self.api_key).__name__}"
                )
            if not self.api_key.strip():
                raise ValueError("api_key must not be blank; use None instead")

        if self.base_url is not None:
            self._check_base_url(self.base_url)

        if self.dimension is not None:
            _check_positive_int("dimension", self.dimension)
        if self.batch_size is not None:
            _check_positive_int("batch_size", self.batch_size)

    @staticmethod
    def _check_base_url(value: object) -> None:
        if not isinstance(value, str):
            raise TypeError(
                f"base_url must be a str or None, got {type(value).__name__}"
            )
        parsed = urlparse(value)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or any(ch.isspace() for ch in value)
        ):
            raise ValueError(
                f"base_url must be a valid http(s) URL, got {value!r}"
            )

    @property
    def has_api_key(self) -> bool:
        """Whether an API key is set (without exposing it)."""
        return self.api_key is not None

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "EmbeddingConfig":
        """Build a config from ``NEXORA_EMBEDDING_*`` environment variables.

        Variables that are unset or blank fall back to the defaults (or
        ``None`` for optional fields). No network calls are made.

        Args:
            environ: Mapping to read from. Defaults to ``os.environ``, read
                at call time.

        Raises:
            ValueError: If a numeric variable is not an integer, or if any
                resulting value fails validation.
        """
        env = os.environ if environ is None else environ
        return cls(
            provider=_env_text(env, ENV_PROVIDER) or DEFAULT_PROVIDER,
            model=_env_text(env, ENV_MODEL) or DEFAULT_MODEL,
            api_key=_env_text(env, ENV_API_KEY),
            base_url=_env_text(env, ENV_BASE_URL),
            dimension=_env_int(env, ENV_DIMENSION),
            batch_size=_env_int(env, ENV_BATCH_SIZE),
        )
