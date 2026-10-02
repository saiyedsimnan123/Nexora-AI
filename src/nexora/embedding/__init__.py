"""Public API of the ``nexora.embedding`` package.

Only the stable, intended embedding components are exported here. Importing
this package performs no network activity and reads no environment values.
"""

from nexora.embedding.base import EmbeddingProvider
from nexora.embedding.cache import EmbeddingCache, InMemoryEmbeddingCache
from nexora.embedding.config import EmbeddingConfig
from nexora.embedding.factory import (
    PROVIDER_HASH,
    PROVIDER_OPENAI,
    SUPPORTED_PROVIDERS,
    create_embedding_provider,
)
from nexora.embedding.hash_provider import DEFAULT_DIMENSION, HashEmbeddingProvider
from nexora.embedding.pipeline import EmbeddingPipeline
from nexora.embedding.service import EmbeddingService

__all__ = [
    "DEFAULT_DIMENSION",
    "EmbeddingCache",
    "EmbeddingConfig",
    "EmbeddingPipeline",
    "EmbeddingProvider",
    "EmbeddingService",
    "HashEmbeddingProvider",
    "InMemoryEmbeddingCache",
    "PROVIDER_HASH",
    "PROVIDER_OPENAI",
    "SUPPORTED_PROVIDERS",
    "create_embedding_provider",
]
