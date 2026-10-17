"""Deterministic character-based text chunking for Nexora AI.

Provides deterministic chunking with exact provenance tracking:
- Fixed-size windowing with configurable overlap.
- Page-aware chunking preserving 1-based page numbers and sequential order.
- Punctuation, symbols, citations, and terminology preserved without alteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


def _validate_int(name: str, value: object) -> None:
    """Raise TypeError unless ``value`` is a real int (bools are rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


@dataclass(frozen=True)
class ProvenanceChunk:
    """A deterministic text chunk preserving page number and document order.

    Attributes:
        text: The text slice of this chunk.
        chunk_index: Zero-based sequential position within the document.
        page_number: 1-based page number where this chunk originated, or None.
    """

    text: str
    chunk_index: int
    page_number: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a str, got {type(self.text).__name__}")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise TypeError(f"chunk_index must be an int, got {type(self.chunk_index).__name__}")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {self.chunk_index}")
        if self.page_number is not None:
            if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
                raise TypeError(f"page_number must be an int or None, got {type(self.page_number).__name__}")
            if self.page_number <= 0:
                raise ValueError(f"page_number must be positive, got {self.page_number}")


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


def chunk_pages(
    pages: Sequence[Any],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    preprocessor: Callable[[str], str] | None = None,
) -> list[ProvenanceChunk]:
    """Split ordered document pages into deterministic chunks with page numbers.

    Iterates through pages in document order. Each page's text is cleaned using
    the optional ``preprocessor`` and then partitioned into windows using ``chunk_text``.
    Every produced chunk maintains its exact 1-based page number and sequential
    document chunk index.

    Args:
        pages: Sequence of DocumentPage instances, (page_number, text) tuples,
            or dicts containing 'page_number' and 'text'.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Shared characters between consecutive chunks on the same page.
        preprocessor: Optional callable to clean page text before chunking.

    Returns:
        A list of ProvenanceChunk objects in document order.
    """
    _validate_int("chunk_size", chunk_size)
    _validate_int("chunk_overlap", chunk_overlap)

    if not isinstance(pages, (list, tuple)):
        raise TypeError(f"pages must be a sequence, got {type(pages).__name__}")

    result: list[ProvenanceChunk] = []

    for page_entry in pages:
        if hasattr(page_entry, "page_number") and hasattr(page_entry, "text"):
            page_num = page_entry.page_number
            raw_text = page_entry.text
        elif isinstance(page_entry, (tuple, list)) and len(page_entry) >= 2:
            page_num = page_entry[0]
            raw_text = page_entry[1]
        elif isinstance(page_entry, dict) and "page_number" in page_entry and "text" in page_entry:
            page_num = page_entry["page_number"]
            raw_text = page_entry["text"]
        else:
            raise TypeError(f"Unsupported page entry format: {type(page_entry).__name__}")

        if isinstance(page_num, bool) or not isinstance(page_num, int) or page_num <= 0:
            raise ValueError(f"page_number must be a positive integer, got {page_num!r}")
        if not isinstance(raw_text, str):
            raise TypeError(f"page text must be a str, got {type(raw_text).__name__}")

        text = preprocessor(raw_text) if preprocessor is not None else raw_text
        if not text.strip():
            continue

        page_text_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for text_chunk in page_text_chunks:
            result.append(
                ProvenanceChunk(
                    text=text_chunk,
                    chunk_index=len(result),
                    page_number=page_num,
                )
            )

    return result
