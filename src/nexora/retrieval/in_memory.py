"""In-memory implementation of the ``VectorStore`` contract.

``InMemoryVectorStore`` keeps vector records in a per-instance dictionary and
ranks them by cosine similarity. It is intended for tests and development: it
is not persistent and has no external infrastructure dependencies. Stored data
is defensively copied on the way in and on the way out, and ``upsert`` is
atomic (the whole batch is validated before the store is touched).
"""

import copy
import heapq
import math

from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.store import VectorStore

__all__ = ["InMemoryVectorStore"]


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_vector(vector: object, name: str) -> None:
    """Require a non-empty list of finite, non-bool numbers."""
    if not isinstance(vector, list):
        raise TypeError(f"{name} must be a list of numbers, got {type(vector).__name__}")
    if not vector:
        raise ValueError(f"{name} must not be empty")
    for index, value in enumerate(vector):
        if not _is_real_number(value):
            raise TypeError(f"{name}[{index}] must be a number, got {type(value).__name__}")
        try:
            finite = math.isfinite(value)
        except OverflowError:  # int too large to represent as a float
            finite = False
        if not finite:
            raise ValueError(f"{name}[{index}] must be finite")


def _validate_dimension(dimension: object) -> None:
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError(f"dimension must be an int, got {type(dimension).__name__}")
    if dimension <= 0:
        raise ValueError("dimension must be greater than zero")


def _validate_limit(limit: object) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise TypeError(f"limit must be an int, got {type(limit).__name__}")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")


def _unit_vector(vector: list[float]) -> list[float] | None:
    """Return ``vector`` scaled to length 1, or None if its magnitude is zero.

    ``math.hypot`` avoids overflow for very large components, and dividing
    first keeps the later dot product within [-1, 1].
    """
    norm = math.hypot(*vector)
    if norm == 0.0:
        return None
    return [component / norm for component in vector]


def _cosine_similarity(a_unit: list[float] | None, b: list[float]) -> float:
    """Cosine similarity of a pre-normalised vector and ``b``; 0.0 for zero vectors."""
    if a_unit is None:
        return 0.0
    b_unit = _unit_vector(b)
    if b_unit is None:
        return 0.0
    dot = math.fsum(x * y for x, y in zip(a_unit, b_unit))
    return max(-1.0, min(1.0, dot))  # guard against rounding just outside [-1, 1]


def _text_or_default(value: object, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _to_chunk(record: VectorRecord, score: float) -> RetrievedChunk:
    payload = record.payload
    text = payload.get("text")
    return RetrievedChunk(
        document_id=_text_or_default(payload.get("document_id"), record.id),
        chunk_id=_text_or_default(payload.get("chunk_id"), record.id),
        text=text if isinstance(text, str) else "",
        score=score,
        metadata=copy.deepcopy(payload),
    )


class InMemoryVectorStore:
    """A deterministic, non-persistent ``VectorStore`` using cosine similarity."""

    def __init__(self, dimension: int | None = None) -> None:
        """Create an empty store.

        ``dimension=None`` infers the dimension from the first upserted record.
        """
        if dimension is not None:
            _validate_dimension(dimension)
        self._dimension: int | None = dimension
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimension(self) -> int | None:
        """The established vector dimension, or None if not yet known."""
        return self._dimension

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records; the batch is applied atomically.

        If an id appears more than once in the batch, the last occurrence wins.
        """
        if not isinstance(records, list):
            raise TypeError(f"records must be a list, got {type(records).__name__}")

        dimension = self._dimension
        prepared: list[VectorRecord] = []
        for index, record in enumerate(records):
            if not isinstance(record, VectorRecord):
                raise TypeError(
                    f"records[{index}] must be a VectorRecord, got {type(record).__name__}"
                )
            vector = list(record.vector)
            _validate_vector(vector, f"records[{index}].vector")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError(
                    f"records[{index}].vector has dimension {len(vector)}, "
                    f"expected {dimension}"
                )
            prepared.append(
                VectorRecord(
                    id=record.id,
                    vector=vector,
                    payload=copy.deepcopy(record.payload),
                )
            )

        # Everything is valid: only now is the store mutated.
        for record in prepared:
            self._records[record.id] = record
        self._dimension = dimension

    def delete(self, ids: list[str]) -> None:
        """Delete records by id; ids that are not stored are ignored."""
        if not isinstance(ids, list):
            raise TypeError(f"ids must be a list, got {type(ids).__name__}")
        for index, record_id in enumerate(ids):
            if not isinstance(record_id, str):
                raise TypeError(f"ids[{index}] must be a string, got {type(record_id).__name__}")
            if not record_id:
                raise ValueError(f"ids[{index}] must not be empty")
        for record_id in ids:
            self._records.pop(record_id, None)

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
    ) -> SearchResult:
        """Return up to ``limit`` records ranked by cosine similarity.

        Ties are broken by record id ascending. The protocol supplies only a
        vector, so the result's ``query`` is always ``""``.
        """
        _validate_limit(limit)
        _validate_vector(vector, "vector")
        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError(
                f"vector has dimension {len(vector)}, expected {self._dimension}"
            )

        query_unit = _unit_vector(vector)
        scored = [
            (_cosine_similarity(query_unit, record.vector), record)
            for record in self._records.values()
        ]
        best = heapq.nsmallest(limit, scored, key=lambda item: (-item[0], item[1].id))
        return SearchResult(
            query="",
            results=[_to_chunk(record, score) for score, record in best],
        )

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)


# Static check that the class satisfies the contract (no runtime cost).
_: type[VectorStore] = InMemoryVectorStore
