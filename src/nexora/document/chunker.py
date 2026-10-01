"""Deterministic character-based text chunking for Nexora AI.

This is the first, intentionally simple chunker: it slices text into
fixed-size windows with a fixed overlap. Sentence-aware or token-aware
chunking can be added later once retrieval requirements are known.
"""


def _validate_int(name: str, value: object) -> None:
    """Raise TypeError unless ``value`` is a real int (bools are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split text into overlapping character-based chunks.

    The text is sliced into windows of at most ``chunk_size`` characters.
    Each window starts ``chunk_size - chunk_overlap`` characters after the
    previous one, so consecutive chunks share exactly ``chunk_overlap``
    characters. Chunking stops as soon as a chunk reaches the end of the
    text, so no chunk is a pure duplicate of the tail of the previous one.

    The text is never modified: chunks are exact slices of the input, so
    punctuation, numbers, symbols, citations and technical terms are kept
    as-is, and the original order is preserved. The function is
    deterministic and uses only the standard library.

    Args:
        text: Cleaned text to split.
        chunk_size: Maximum characters per chunk. Must be a positive int.
        chunk_overlap: Characters shared by consecutive chunks. Must be a
            non-negative int strictly smaller than ``chunk_size``.

    Returns:
        A list of chunks. Empty or whitespace-only input returns ``[]``.

    Raises:
        TypeError: If ``text`` is not a str, or either size is not an int.
        ValueError: If ``chunk_size`` is not positive, ``chunk_overlap`` is
            negative, or ``chunk_overlap`` is not smaller than ``chunk_size``.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    _validate_int("chunk_size", chunk_size)
    _validate_int("chunk_overlap", chunk_overlap)

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(
            f"chunk_overlap must be non-negative, got {chunk_overlap}"
        )
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})"
        )

    if not text.strip():
        return []

    step = chunk_size - chunk_overlap
    length = len(text)
    chunks: list[str] = []
    start = 0

    while True:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start += step

    return chunks
