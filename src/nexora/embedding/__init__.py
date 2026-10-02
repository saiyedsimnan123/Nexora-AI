"""Nexora embedding subsystem: provider interface, config and test provider."""

from nexora.embedding.base import EmbeddingProvider
from nexora.embedding.config import EmbeddingConfig
from nexora.embedding.hash_provider import DEFAULT_DIMENSION, HashEmbeddingProvider

__all__ = [
    "DEFAULT_DIMENSION",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
]

