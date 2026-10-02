"""Embedding integration layer.

Converts text chunks into structured ``EmbeddedChunk`` objects through the
existing ``EmbeddingPipeline``.

Dependency direction::

    EmbeddingIntegration -> EmbeddingPipeline -> EmbeddingService -> EmbeddingProvider

This module contains no provider selection, no caching, no environment
access and no network calls. Detailed input validation (for example,
rejecting empty or non-string chunks) belongs to ``EmbeddingService`` and
the providers; errors they raise propagate unchanged. The only checks made
here are the ones this layer needs in order to pair chunks with vectors
correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexora.embedding.pipeline import EmbeddingPipeline


@dataclass(frozen=True)
class EmbeddedChunk:
    """A text chunk paired with its embedding vector.

    Attributes:
        text: The original chunk text, unchanged.
        embedding: The vector produced for ``text``.
        index: Zero-based position of the chunk in the input batch.
    """

    text: str
    embedding: list[float]
    index: int


class EmbeddingIntegration:
    """Embeds text chunks via an ``EmbeddingPipeline``."""

    def __init__(self, pipeline: EmbeddingPipeline) -> None:
        """Store ``pipeline``. Performs no embedding work or network activity."""
        self._pipeline = pipeline

    def embed_chunks(self, chunks: list[str]) -> list[EmbeddedChunk]:
        """Embed ``chunks`` in a single batch call and pair each with its vector.

        Args:
            chunks: Text chunks, in the order they should be returned.

        Returns:
            One ``EmbeddedChunk`` per input chunk, in input order, with
            indexes starting at 0. An empty input returns ``[]`` without
            calling the pipeline.

        Raises:
            TypeError: If ``chunks`` is not a list (a bare string would
                otherwise be treated as a sequence of characters).
            ValueError: If the pipeline returns a different number of
                vectors than chunks, since pairing them would be wrong.
            Exception: Any error raised by the pipeline propagates unchanged.
        """
        if not isinstance(chunks, list):
            raise TypeError(
                f"chunks must be a list of strings, got {type(chunks).__name__}"
            )
        if not chunks:
            return []

        vectors = self._pipeline.embed_texts(chunks)

        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedding pipeline returned "
                f"{len(vectors)} vectors for {len(chunks)} chunks"
            )

        return [
            EmbeddedChunk(text=text, embedding=vector, index=index)
            for index, (text, vector) in enumerate(zip(chunks, vectors))
        ]
