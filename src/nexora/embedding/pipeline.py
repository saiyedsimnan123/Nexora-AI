"""Embedding pipeline.

A thin orchestration layer over ``EmbeddingService``.

The pipeline is the entry point other Nexora components should use to turn
text into vectors. It intentionally contains no provider selection, no
caching and no validation of its own: all of that belongs to the service
(and the components behind it). The pipeline only delegates, which keeps a
single place for future orchestration steps to be added deliberately.
"""

from __future__ import annotations

from nexora.embedding.service import EmbeddingService


class EmbeddingPipeline:
    """Delegates embedding requests to an ``EmbeddingService``.

    Results are returned exactly as the service produced them. Ordering,
    duplicate handling and input validation are the service's
    responsibility, and any error it raises propagates unchanged.
    """

    def __init__(self, service: EmbeddingService) -> None:
        """Create a pipeline around ``service``.

        Construction only stores the service; it performs no work and no
        network activity.
        """
        self._service = service

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text via the service and return its vector."""
        return self._service.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts via the service.

        Returns one vector per input text, in the service's order.
        """
        return self._service.embed_texts(texts)
