"""Qdrant adapter implementing the VectorStore protocol (step 11E)."""

from __future__ import annotations

import math
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.qdrant_config import QdrantConfig


class QdrantStoreError(RuntimeError):
    """Raised when the Qdrant service/client fails (not for bad input)."""


_NAMESPACE = uuid.UUID("6f1c2d3e-4a5b-5c6d-8e7f-0a1b2c3d4e5f")
_ID_KEY = "_nexora_id"


def _validate_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("id must be a non-empty string")
    return value


def _to_point_id(value: str) -> str:
    """Map an opaque Nexora ID to a valid Qdrant point ID (deterministic)."""
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return str(uuid.uuid5(_NAMESPACE, value))


class QdrantVectorStore:
    def __init__(self, config: QdrantConfig, *, dimension: int, client: Any = None) -> None:
        if not isinstance(config, QdrantConfig):
            raise TypeError("config must be a QdrantConfig")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._config = config
        self._dimension = dimension
        self._collection = config.collection_name
        if client is None:
            client = QdrantClient(
                url=config.url, api_key=config.api_key, timeout=config.timeout
            )
        self._client = client

    @property
    def dimension(self) -> int:
        return self._dimension

    def _call(self, operation: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # re-raised as a distinct infrastructure error
            raise QdrantStoreError(
                f"Qdrant {operation} failed ({type(exc).__name__})"
            ) from exc

    def _check_vector(self, vector: Any) -> list[float]:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("vector must be a non-empty list of numbers")
        for x in vector:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError("vector values must be numbers")
            if not math.isfinite(x):
                raise ValueError("vector values must be finite")
        if len(vector) != self._dimension:
            raise ValueError(
                f"vector dimension {len(vector)} != expected {self._dimension}"
            )
        return [float(x) for x in vector]

    def ensure_collection(self) -> None:
        exists = self._call(
            "collection_exists", self._client.collection_exists, self._collection
        )
        if exists:
            return
        self._call(
            "create_collection",
            self._client.create_collection,
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._dimension, distance=models.Distance.COSINE
            ),
        )

    def upsert(self, records: list[VectorRecord]) -> None:
        if not isinstance(records, list):
            raise TypeError("records must be a list")
        points = []
        for record in records:
            if not isinstance(record, VectorRecord):
                raise TypeError("every item must be a VectorRecord")
            original_id = _validate_id(record.id)
            vector = self._check_vector(record.vector)
            payload = dict(record.payload or {})
            payload[_ID_KEY] = original_id
            points.append(
                models.PointStruct(
                    id=_to_point_id(original_id), vector=vector, payload=payload
                )
            )
        if not points:
            return
        self._call("upsert", self._client.upsert, self._collection, points=points)

    def delete(self, ids: list[str]) -> None:
        if not isinstance(ids, list):
            raise TypeError("ids must be a list")
        valid = [_to_point_id(_validate_id(i)) for i in ids]
        if not valid:
            return
        self._call(
            "delete",
            self._client.delete,
            self._collection,
            points_selector=models.PointIdsList(points=valid),
        )

    def search(self, vector: list[float], *, limit: int = 10) -> SearchResult:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        query = self._check_vector(vector)
        response = self._call(
            "search",
            self._client.query_points,
            self._collection,
            query=query,
            limit=limit,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            original_id = payload.get(_ID_KEY, str(point.id))
            results.append(
                RetrievedChunk(
                    document_id=payload.get("document_id", original_id),
                    chunk_id=payload.get("chunk_id", original_id),
                    text=payload.get("text", ""),
                    score=point.score,
                    metadata=dict(payload),
                )
            )
        return SearchResult(query="", results=results)

    def count(self) -> int:
        result = self._call(
            "count", self._client.count, self._collection, exact=True
        )
        return int(result.count)
