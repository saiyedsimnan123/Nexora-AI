"""Provider-independent embedding interface for Nexora AI.

Everything in Nexora that needs embeddings should depend on
``EmbeddingProvider`` only, never on a concrete provider. Concrete providers
(local, API-based, self-hosted, test) live in their own modules and are
selected through ``EmbeddingConfig``.
"""

from typing import Protocol, runtime_checkable


def validate_texts(texts: object) -> None:
    """Validate the input of a batch embedding call.

    Args:
        texts: The value passed to ``embed_texts``.

    Raises:
        TypeError: If ``texts`` is not a ``list``, or an element is not a str.
        ValueError: If an element is empty or whitespace-only.
    """
    if not isinstance(texts, list):
        raise TypeError(f"texts must be a list of str, got {type(texts).__name__}")
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] must be a str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError(f"texts[{index}] must not be empty or whitespace-only")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for anything that can turn text into embedding vectors."""

    def embed_text(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text, in the same order as ``texts``.

        An empty list returns ``[]``. Texts are never modified.
        """
        ...
