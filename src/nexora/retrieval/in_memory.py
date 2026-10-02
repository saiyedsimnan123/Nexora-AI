"""In-memory implementation of the Nexora VectorStore contract.

Intended for development, unit testing, and local experimentation.
It is NOT a production vector database.

Uses only the Python standard library and Nexora retrieval modules.
"""

from __future__ import annotations

import copy
import math

from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.store import VectorStore

__all__ = ["InMemoryVectorStore"]


def _validate_vector(vector: object, *, name: str, allow_tuple: bool) -> list[float]:
    """Validate a vector and return a fresh list of floats."""
    allowed = (list, tuple) if allow_tuple else (list,)

    if not isinstance(vector, allowed):
        raise TypeError(f"{name} must be a list of numbers")

    if len(vector) == 0:
        raise ValueError(f"{name} must not be empty")

    cleaned: list[float] = []

    for value in vector:
        if isinstance(value, bool):
            raise TypeError(f"{name} must not contain bool values")

        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain only numbers")

        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"{name} must contain finite numbers"
            ) from exc

        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite numbers")

        cleaned.append(number)

    return cleaned


def _validate_positive_int(value: object, *, name: str) -> int:
    """Validate a positive, non-bool integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two equal-length vectors.

    Returns 0.0 if either vector is a zero vector.

    The vectors are rescaled by their largest absolute component first
    to reduce overflow risk when handling very large finite values.
    """
    scale_a = max(abs(x) for x in a)
    scale_b = max(abs(x) for x in b)

    if scale_a == 0.0 or scale_b == 0.0:
        return 0.0

    normalized_a = [x / scale_a for x in a]
    normalized_b = [x / scale_b for x in b]

    norm_a = math.sqrt(sum(x * x for x in normalized_a))
    norm_b = math.sqrt(sum(x * x for x in normalized_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = sum(
        x * y
        for x, y in zip(normalized_a, normalized_b)
    )

    score = dot / (norm_a * norm_b)

    if not math.isfinite(score):
        return 0.0

    return max(-1.0, min(1.0, score))


def _text_field(
    payload: dict[str, object],
    key: str,
    fallback: str,
) -> str:
    """Return a payload string field or the fallback."""
    value = payload.get(key)

    if isinstance(value, str):
        if value != "" or key == "text":
            return value

    return fallback


class InMemoryVectorStore:
    """Dictionary-backed VectorStore using cosine similarity."""

    def __init__(self, dimension: int | None = None) -> None:
        if dimension is not None:
            dimension = _validate_positive_int(
                dimension,
                name="dimension",
            )

        self._dimension: int | None = dimension
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimension(self) -> int | None:
        """Return the established vector dimension."""
        return self._dimension

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records atomically.

        If duplicate IDs occur within the same batch, the last record
        wins.
        """
        if not isinstance(records, list):
            raise TypeError("records must be a list")

        # Phase 1: validate and prepare everything.
        # The actual store is not modified during this phase.
        staged: dict[str, VectorRecord] = {}
        dimension = self._dimension

        for record in records:
            if not isinstance(record, VectorRecord):
                raise TypeError(
                    "every record must be a VectorRecord"
                )

            vector = _validate_vector(
                record.vector,
                name="record vector",
                allow_tuple=True,
            )

            if dimension is None:
                dimension = len(vector)

            elif len(vector) != dimension:
                raise ValueError(
                    f"vector dimension {len(vector)} does not match "
                    f"store dimension {dimension}"
                )

            payload = copy.deepcopy(dict(record.payload))

            staged[record.id] = VectorRecord(
                id=record.id,
                vector=vector,
                payload=payload,
            )

        # Phase 2: commit only after the entire batch is valid.
        self._records.update(staged)
        self._dimension = dimension

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, ids: list[str]) -> None:
        """Delete records by ID.

        Missing IDs are ignored.
        """
        if not isinstance(ids, list):
            raise TypeError("ids must be a list")

        for item in ids:
            if not isinstance(item, str):
                raise TypeError("every id must be a string")

            if item == "":
                raise ValueError("ids must not be empty")

        for item in ids:
            self._records.pop(item, None)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
    ) -> SearchResult:
        """Return top records by cosine similarity.

        Results are sorted by:
        1. score descending
        2. record ID ascending for deterministic tie-breaking
        """
        query = _validate_vector(
            vector,
            name="query vector",
            allow_tuple=False,
        )

        limit = _validate_positive_int(
            limit,
            name="limit",
        )

        if (
            self._dimension is not None
            and len(query) != self._dimension
        ):
            raise ValueError(
                f"query dimension {len(query)} does not match "
                f"store dimension {self._dimension}"
            )

        scored: list[tuple[float, str]] = []

        for record_id, record in self._records.items():
            score = _cosine_similarity(
                query,
                list(record.vector),
            )
            scored.append((score, record_id))

        scored.sort(
            key=lambda item: (-item[0], item[1])
        )

        chunks = [
            self._to_chunk(
                self._records[record_id],
                score,
            )
            for score, record_id in scored[:limit]
        ]

        return SearchResult(
            query="",
            results=chunks,
        )

    # ------------------------------------------------------------------
    # Result conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_chunk(
        record: VectorRecord,
        score: float,
    ) -> RetrievedChunk:
        """Convert a VectorRecord into a RetrievedChunk."""
        payload = copy.deepcopy(dict(record.payload))

        return RetrievedChunk(
            document_id=_text_field(
                payload,
                "document_id",
                record.id,
            ),
            chunk_id=_text_field(
                payload,
                "chunk_id",
                record.id,
            ),
            text=_text_field(
                payload,
                "text",
                "",
            ),
            score=score,
            metadata=copy.deepcopy(payload),
        )

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)


# Static protocol assurance.
_PROTOCOL_CHECK: type[VectorStore] = InMemoryVectorStore 
