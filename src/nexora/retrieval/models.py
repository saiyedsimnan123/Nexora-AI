"""Retrieval data models.

Framework-independent, immutable value objects used by the retrieval layer.
They depend only on the Python standard library so that any vector store
(Qdrant, in-memory, ...) can be adapted to them later.

Vector dimension is deliberately not validated here; that belongs to the
vector-store layer. IDs are opaque strings supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_non_empty_str(value: object, name: str) -> None:
    """Raise unless ``value`` is a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _require_finite_number(value: object, name: str) -> None:
    """Raise unless ``value`` is a finite int or float (bool is rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    try:
        finite = isfinite(value)
    except OverflowError:  # integers too large to convert to float
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")


def _require_str_key_dict(value: object, name: str) -> None:
    """Raise unless ``value`` is a dict whose keys are all strings."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")


@dataclass(frozen=True)
class VectorRecord:
    """A vector plus payload that can later be stored in a vector database."""

    id: str
    vector: list[float]
    payload: dict[str, object]

    def __post_init__(self) -> None:
        _require_non_empty_str(self.id, "id")
        if not isinstance(self.vector, list):
            raise TypeError("vector must be a list")
        if not self.vector:
            raise ValueError("vector must not be empty")
        for position, value in enumerate(self.vector):
            _require_finite_number(value, f"vector[{position}]")
        _require_str_key_dict(self.payload, "payload")

        object.__setattr__(self, "vector", list(self.vector))
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk returned by retrieval, with its relevance score.

    Score semantics depend on the distance/similarity metric in use, so no
    range is enforced.
    """

    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        _require_non_empty_str(self.document_id, "document_id")
        _require_non_empty_str(self.chunk_id, "chunk_id")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _require_finite_number(self.score, "score")
        _require_str_key_dict(self.metadata, "metadata")

        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SearchResult:
    """The ordered result of one retrieval operation."""

    query: str
    results: list[RetrievedChunk]

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.results, list):
            raise TypeError("results must be a list")
        for position, item in enumerate(self.results):
            if not isinstance(item, RetrievedChunk):
                raise TypeError(f"results[{position}] must be a RetrievedChunk")

        object.__setattr__(self, "results", list(self.results))
      
