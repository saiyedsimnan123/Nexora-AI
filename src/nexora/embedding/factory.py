"""Embedding provider factory.

Centralizes construction of embedding providers so the rest of Nexora
depends only on ``EmbeddingProvider`` and never on how a concrete provider
is built.

Design notes:
- The factory performs no network calls. Constructing a provider must stay
  cheap and offline.
- The factory never logs or formats secrets. Error messages mention only
  the provider name, never the configuration object or its API key.
- The supplied (frozen) configuration is only read, never modified.
- There is no silent fallback: an unsupported provider raises ``ValueError``.
"""

from __future__ import annotations

from nexora.embedding.base import EmbeddingProvider
from nexora.embedding.config import EmbeddingConfig
from nexora.embedding.hash_provider import HashEmbeddingProvider
from nexora.embedding.openai_provider import OpenAIEmbeddingProvider

PROVIDER_HASH = "hash"
PROVIDER_OPENAI = "openai"
SUPPORTED_PROVIDERS = (PROVIDER_HASH, PROVIDER_OPENAI)


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Create an embedding provider from ``config``.

    Args:
        config: The embedding configuration. Only ``config.provider`` is
            used to choose the provider; provider-specific settings are
            read from the same object.

    Returns:
        A provider implementing ``EmbeddingProvider``.

    Raises:
        TypeError: If ``config`` is not an ``EmbeddingConfig`` instance.
        ValueError: If ``config.provider`` is not a supported provider name.
            Errors raised while constructing the provider itself propagate
            unchanged.
    """
    if not isinstance(config, EmbeddingConfig):
        raise TypeError(
            "config must be an EmbeddingConfig instance, "
            f"got {type(config).__name__}"
        )

    provider_name = config.provider
    normalized = (
        provider_name.strip().lower() if isinstance(provider_name, str) else None
    )

    if normalized == PROVIDER_HASH:
        return _create_hash_provider(config)
    if normalized == PROVIDER_OPENAI:
        return OpenAIEmbeddingProvider(config)

    raise ValueError(
        f"Unsupported embedding provider: {provider_name!r}. "
        f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def _create_hash_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Build the hash provider, honoring ``config.dimension`` when set."""
    if config.dimension is None:
        return HashEmbeddingProvider()
    return HashEmbeddingProvider(dimension=config.dimension)
