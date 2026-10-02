"""Provider-independent embedding interface for Nexora AI.

Everything in Nexora that needs embeddings should depend on
``EmbeddingProvider`` only, never on a concrete provider. Concrete providers
(local, API-based, self-hosted, test) live in their own modules and are
selected through ``EmbeddingConfig``.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for anything that can turn text into an embedding vector."""

    def embed_text(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        ...
