"""Research evidence context builder (step 11H).

Turns quality-filtered retrieval results into a traceable, immutable
EvidenceContext. No rewriting, truncation of chunks, or answer generation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any


def _non_empty_str(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class EvidenceItem:
    """One retrieved chunk, with its position in the final context."""

    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, object]
    position: int

    def __post_init__(self) -> None:
        _non_empty_str("document_id", self.document_id)
        _non_empty_str("chunk_id", self.chunk_id)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be a number")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        if not all(isinstance(k, str) for k in self.metadata):
            raise TypeError("metadata keys must be strings")
        _non_negative_int("position", self.position)
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))


@dataclass(frozen=True)
class EvidenceContext:
    """Ordered evidence for a query, ready for a future RAG layer."""

    query: str
    items: list[EvidenceItem]
    total_candidates: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.items, (list, tuple)):
            raise TypeError("items must be a list")
        if not all(isinstance(i, EvidenceItem) for i in self.items):
            raise TypeError("items must contain only EvidenceItem objects")
        _non_negative_int("total_candidates", self.total_candidates)
        object.__setattr__(self, "items", list(self.items))


class EvidenceContextBuilder:
    """Builds an EvidenceContext from a RetrievalQualityService.

    ``max_total_characters`` caps the combined length of evidence text. Whole
    items are kept in order until the next one would exceed the cap; chunks are
    never cut. A cap of 0 therefore yields no items.
    """

    def __init__(
        self, retrieval_service: Any, *, max_total_characters: int | None = None
    ) -> None:
        if not callable(getattr(retrieval_service, "retrieve", None)):
            raise TypeError("retrieval_service must provide retrieve(query)")
        if max_total_characters is not None:
            _non_negative_int("max_total_characters", max_total_characters)
        self._retrieval = retrieval_service
        self._max_chars = max_total_characters

    def build(self, query: str) -> EvidenceContext:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        chunks = self._retrieval.retrieve(query).results
        items: list[EvidenceItem] = []
        used = 0
        for chunk in chunks:
            if self._max_chars is not None and used + len(chunk.text) > self._max_chars:
                break
            used += len(chunk.text)
            items.append(
                EvidenceItem(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=chunk.score,
                    metadata=chunk.metadata,
                    position=len(items),
                )
            )
        return EvidenceContext(query=query, items=items, total_candidates=len(chunks))
