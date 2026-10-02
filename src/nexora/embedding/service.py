"""Embedding service: orchestrates an ``EmbeddingProvider`` and an ``EmbeddingCache``.

``EmbeddingService`` is the layer the rest of Nexora should call to get
embeddings. It checks the cache first and calls the provider only for texts
that are not cached yet, so repeated work is avoided. It knows nothing about
how vectors are produced or how the cache stores them: it depends only on the
``EmbeddingProvider`` and ``EmbeddingCache`` interfaces.
"""

import hashlib

from nexora.embedding.base import EmbeddingProvider, validate_texts
from nexora.embedding.cache import EmbeddingCache

_KEY_PREFIX = "nexora:embedding:v1"


def _build_cache_key(namespace: str, text: str) -> str:
    """Build the versioned cache key for ``text`` within ``namespace``."""
    namespace_bytes = namespace.encode("utf-8")
    payload = (
        len(namespace_bytes).to_bytes(8, "big")
        + namespace_bytes
        + text.encode("utf-8", errors="surrogatepass")
    )
    return f"{_KEY_PREFIX}:{namespace}:{hashlib.sha256(payload).hexdigest()}"


class EmbeddingService:
    """Embeds text through a provider, using an optional cache."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: EmbeddingCache | None = None,
        cache_namespace: str = "default",
    ) -> None:
        if not isinstance(provider, EmbeddingProvider):
            raise TypeError(
                f"provider must implement EmbeddingProvider, "
                f"got {type(provider).__name__}"
            )
        if cache is not None and not isinstance(cache, EmbeddingCache):
            raise TypeError(
                f"cache must implement EmbeddingCache or be None, "
                f"got {type(cache).__name__}"
            )
        if not isinstance(cache_namespace, str):
            raise TypeError(
                f"cache_namespace must be a str, got {type(cache_namespace).__name__}"
            )
        if not cache_namespace.strip():
            raise ValueError("cache_namespace must not be empty or whitespace-only")

        self._provider = provider
        self._cache = cache
        self._cache_namespace = cache_namespace

    def cache_key(self, text: str) -> str:
        """Return a deterministic cache key for ``text``."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        return _build_cache_key(self._cache_namespace, text)

    def embed_text(self, text: str) -> list[float]:
        """Return one embedding, using the cache when enabled."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        cache = self._cache
        if cache is None:
            return self._provider.embed_text(text)

        key = self.cache_key(text)
        cached = cache.get(key)
        if cached is not None:
            return cached

        vector = self._provider.embed_text(text)
        cache.set(key, vector)
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings in input order, using the cache when enabled."""
        validate_texts(texts)
        if not texts:
            return []

        cache = self._cache
        if cache is None:
            return self._provider.embed_texts(texts)
        return self._embed_texts_with_cache(cache, texts)

    def _embed_texts_with_cache(
        self, cache: EmbeddingCache, texts: list[str]
    ) -> list[list[float]]:
        keys = [self.cache_key(text) for text in texts]

        resolved: dict[str, list[float]] = {}
        missing: dict[str, str] = {}

        for key, text in zip(keys, texts):
            if key in resolved or key in missing:
                continue
            cached = cache.get(key)
            if cached is None:
                missing[key] = text
            else:
                resolved[key] = cached

        if missing:
            self._embed_missing(cache, missing, resolved)

        return [list(resolved[key]) for key in keys]

    def _embed_missing(
        self,
        cache: EmbeddingCache,
        missing: dict[str, str],
        resolved: dict[str, list[float]],
    ) -> None:
        """Embed unique missing texts in one call and cache the results."""
        vectors = self._provider.embed_texts(list(missing.values()))

        if not isinstance(vectors, list):
            raise RuntimeError(
                f"Provider returned {type(vectors).__name__} instead of a list "
                f"of {len(missing)} vectors"
            )
        if len(vectors) != len(missing):
            raise RuntimeError(
                f"Provider returned {len(vectors)} vectors "
                f"for {len(missing)} unique texts"
            )

        for key, vector in zip(missing, vectors):
            cache.set(key, vector)
            resolved[key] = vector
