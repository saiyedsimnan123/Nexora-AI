"""Deterministic hash-based embedding provider for development and testing."""

import hashlib
import math
import struct

DEFAULT_DIMENSION = 128

_FLOATS_PER_BLOCK = 8  # one SHA-256 digest (32 bytes) -> 8 x 4-byte values


class HashEmbeddingProvider:
    """Deterministic hash-based embedding provider for development/testing.

    WARNING: this is NOT a semantic embedding model. Vectors are derived from
    a SHA-256 hash of the exact text, so texts with similar meaning (or even
    similar wording) get unrelated vectors. It must not be used for
    production semantic retrieval. Its only purpose is to let Nexora's
    pipeline and tests run without an external API or a large model download.

    Properties:
        * Deterministic: the same text always gives the same vector, across
          instances and across Python processes (``hash()`` is not used).
        * Different texts normally give different vectors.
        * Vectors have exactly ``dimension`` floats and unit length (L2).

    It structurally satisfies ``nexora.embedding.base.EmbeddingProvider``.
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        """Create a provider.

        Args:
            dimension: Length of the output vectors. Must be a positive int.

        Raises:
            TypeError: If ``dimension`` is not an int.
            ValueError: If ``dimension`` is not positive.
        """
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError(
                f"dimension must be an int, got {type(dimension).__name__}"
            )
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Length of the vectors this provider produces."""
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Return a deterministic pseudo-embedding for ``text``.

        Args:
            text: Non-empty text to embed.

        Returns:
            A list of ``dimension`` floats forming a unit-length vector.

        Raises:
            TypeError: If ``text`` is not a str.
            ValueError: If ``text`` is empty or whitespace-only.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        data = text.encode("utf-8")
        values: list[float] = []
        block = 0
        while len(values) < self._dimension:
            digest = hashlib.sha256(f"{block}:".encode("ascii") + data).digest()
            for (number,) in struct.iter_unpack(">I", digest):
                # Map an unsigned 32-bit integer to a float in [-1.0, 1.0).
                values.append(number / 2**31 - 1.0)
            block += 1
        values = values[: self._dimension]

        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:  # practically impossible; keeps the output valid
            return values
        return [v / norm for v in values]
