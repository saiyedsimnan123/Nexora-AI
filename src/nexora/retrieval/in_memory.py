"""In-memory ``VectorStore`` for development and testing.

``InMemoryVectorStore`` keeps records in a dictionary and ranks them by cosine
similarity with a plain scan. It is meant for tests, local experiments and
small data sets; it is not thread-safe, not persistent and not optimized for
large collections. It structurally satisfies ``nexora.retrieval.store.VectorStore``
and uses only the standard library plus Nexora's retrieval models.
"""

import copy
import dataclasses
import heapq
import math

from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord


def _check_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_id(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must not be empty or whitespace-only")


def _validated_vector(vector: object, name: str) -> list[float]:
    """Validate ``vector`` and return a fresh ``list[float]`` copy."""
    if not isinstance(vector, list):
        raise TypeError(f"{name} must be a list, got {type(vector).__name__}")
    if not vector:
        raise ValueError(f"{name} must not be empty")
    result: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{index}] must be a number, got {type(value).__name__}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return result


def _scaled(vector: list[float]) -> tuple[list[float], float] | None:
    """Scale by the largest magnitude (avoids overflow); None for a zero vector."""
    peak = max(abs(value) for value in vector)
    if peak == 0.0:
        return None
    scaled = [value / peak for value in vector]
    return scaled, math.sqrt(math.fsum(value * value for value in scaled))


def _cosine(
    query: tuple[list[float], float] | None, other: tuple[list[float], float] | None
) -> float:
    """Cosine similarity in [-1.0, 1.0]; 0.0 if either vector is zero."""
    if query is None or other is None:
        return 0.0
    dot = math.fsum(a * b for a, b in zip(query[0], other[0]))
    score = dot / (query[1] * other[1])
    return max(-1.0, min(1.0, score)) + 0.0  # "+ 0.0" turns -0.0 into 0.0


def _non_blank_str(value: object, default: str) -> str:
    return value if isinstance(value, str) and value.strip() else default


def _to_chunk(record: VectorRecord, score: float) -> RetrievedChunk:
    payload = record.payload
    text = payload.get("text")
    return RetrievedChunk(
        document_id=_non_blank_str(payload.get("document_id"), record.id),
        chunk_id=_non_blank_str(payload.get("chunk_id"), record.id),
        text=text if isinstance(text, str) else "",
        score=score,
        metadata=copy.deepcopy(payload),
    )


class InMemoryVectorStore:
    """Dictionary-backed vector store using cosine similarity.

    Args:
        dimension: Required vector length. If ``None``, the first successful
            upsert establishes it. Once established it never resets, even if
            every record is deleted.

    Raises:
        TypeError: If ``dimension`` is not an int (bools are rejected).
        ValueError: If ``dimension`` is not positive.
    """

    def __init__(self, dimension: int | None = None) -> None:
        if dimension is not None:
            _check_positive_int("dimension", dimension)
        self._dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimension(self) -> int | None:
        """The established vector dimension, or ``None`` if not yet known."""
        return self._dimension

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records atomically.

        The whole batch is validated and copied before the store changes, so
        one invalid record leaves the store untouched. For duplicate ids in a
        batch the last occurrence wins.

        Raises:
            TypeError: If ``records`` is not a list, an element is not a
                ``VectorRecord``, or an id, vector or payload has a wrong type.
            ValueError: If an id is blank, or a vector is empty, non-finite or
                does not match the store dimension.
        """
        if not isinstance(records, list):
            raise TypeError(f"records must be a list, got {type(records).__name__}")

        dimension = self._dimension
        staged: dict[str, VectorRecord] = {}
        for index, record in enumerate(records):
            if not isinstance(record, VectorRecord):
                raise TypeError(
                    f"records[{index}] must be a VectorRecord, got {type(record).__name__}"
                )
            _check_id(f"records[{index}].id", record.id)
            vector = _validated_vector(record.vector, f"records[{index}].vector")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError(
                    f"records[{index}].vector has {len(vector)} dimensions, expected {dimension}"
                )
            if not isinstance(record.payload, dict):
                raise TypeError(
                    f"records[{index}].payload must be a dict, "
                    f"got {type(record.payload).__name__}"
                )
            staged[record.id] = dataclasses.replace(
                record, vector=vector, payload=copy.deepcopy(record.payload)
            )

        self._dimension = dimension
        self._records.update(staged)

    def delete(self, ids: list[str]) -> None:
        """Remove records by id; missing ids are ignored.

        Raises:
            TypeError: If ``ids`` is not a list or an element is not a str.
            ValueError: If an id is empty or whitespace-only.
        """
        if not isinstance(ids, list):
            raise TypeError(f"ids must be a list, got {type(ids).__name__}")
        for index, record_id in enumerate(ids):
            _check_id(f"ids[{index}]", record_id)
        for record_id in ids:
            self._records.pop(record_id, None)

    def search(self, vector: list[float], *, limit: int = 10) -> SearchResult:
        """Return up to ``limit`` records ranked by cosine similarity.

        Highest score first; equal scores are ordered by record id. An empty
        store returns an empty result. The result's ``query`` is always ``""``.

        Raises:
            TypeError: If ``vector`` is not a list of numbers or ``limit`` is
                not an int (bools are rejected).
            ValueError: If ``limit`` is not positive, the vector is empty or
                non-finite, or its length differs from the store dimension.
        """
        _check_positive_int("limit", limit)
        query = _validated_vector(vector, "vector")
        if self._dimension is not None and len(query) != self._dimension:
            raise ValueError(
                f"vector has {len(query)} dimensions, expected {self._dimension}"
            )

        prepared = _scaled(query)
        scored = (
            (_cosine(prepared, _scaled(record.vector)), record)
            for record in self._records.values()
        )
        top = heapq.nsmallest(limit, scored, key=lambda item: (-item[0], item[1].id))
        return SearchResult(query="", results=[_to_chunk(r, s) for s, r in top])

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)
