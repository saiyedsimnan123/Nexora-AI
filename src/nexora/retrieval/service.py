"""Retrieval service: query text -> embedding -> VectorStore search (step 11F)."""

from __future__ import annotations

import math
from typing import Any

from nexora.retrieval.models import RetrievedChunk, SearchResult
from nexora.retrieval.store import VectorStore


class RetrievalService:
    """Application-level retrieval boundary.

    Depends only on an embedding service (``embed_text``) and the VectorStore
    protocol, so it works with any store implementation.
    """

    def __init__(
        self,
        embedding_service: Any,
        vector_store: VectorStore,
        *,
        dimension: int | None = None,
        cache_queries: bool = True,
        max_cache_entries: int = 256,
        bm25_index: Any = None,
    ) -> None:
        if not callable(getattr(embedding_service, "embed_text", None)):
            raise TypeError("embedding_service must provide embed_text(text)")
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must implement the VectorStore protocol")
        if not isinstance(cache_queries, bool):
            raise TypeError("cache_queries must be a boolean")
        if isinstance(max_cache_entries, bool) or not isinstance(max_cache_entries, int):
            raise TypeError("max_cache_entries must be an integer")
        if max_cache_entries <= 0:
            raise ValueError("max_cache_entries must be a positive integer")

        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._bm25_index = bm25_index

        self._dimension = dimension
        if self._dimension is None:
            store_dim = getattr(vector_store, "dimension", None)
            if isinstance(store_dim, int) and not isinstance(store_dim, bool) and store_dim > 0:
                self._dimension = store_dim
            else:
                emb_dim = getattr(embedding_service, "dimension", None)
                if isinstance(emb_dim, int) and not isinstance(emb_dim, bool) and emb_dim > 0:
                    self._dimension = emb_dim

        self._cache_queries = cache_queries
        self._max_cache_entries = max_cache_entries
        self._query_cache: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def max_cache_entries(self) -> int:
        return self._max_cache_entries

    @property
    def cache_queries(self) -> bool:
        return self._cache_queries

    @property
    def cache_size(self) -> int:
        """Return the number of entries currently stored in the query cache."""
        return len(self._query_cache)

    @property
    def bm25_index(self) -> Any:
        """Optional BM25 lexical index."""
        return self._bm25_index

    def clear_cache(self) -> None:
        """Clear all entries currently stored in the query cache."""
        self._query_cache.clear()

    def _evict_oldest(self) -> None:
        """Safely evict the oldest entry from the query cache without failing on an empty cache."""
        if not self._query_cache:
            return
        try:
            first_key = next(iter(self._query_cache))
            del self._query_cache[first_key]
        except (StopIteration, KeyError):
            pass

    def _validate_vector(self, vector: Any) -> list[float]:
        if not isinstance(vector, list):
            raise TypeError(f"Embedding service must return a list, got {type(vector).__name__}")
        if not vector:
            raise ValueError("Embedding service returned an empty vector")
        for i, val in enumerate(vector):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"vector[{i}] must be a number, got {type(val).__name__}")
            if not math.isfinite(val):
                raise ValueError(f"vector[{i}] must be finite")
        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError(
                f"Query vector dimension {len(vector)} does not match expected dimension {self._dimension}"
            )
        return vector

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        mode: str = "vector",
        rrf_k: int = 60,
        paper_id: str | None = None,
        document_id: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> SearchResult:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        if mode not in ("vector", "bm25", "hybrid"):
            raise ValueError(f"mode must be 'vector', 'bm25', or 'hybrid', got {mode!r}")

        effective_filter: dict[str, Any] = dict(filter) if filter else {}
        if paper_id is not None:
            effective_filter["paper_id"] = str(paper_id)
        if document_id is not None:
            effective_filter["document_id"] = str(document_id)

        target_paper_id = effective_filter.get("paper_id")
        target_doc_id = effective_filter.get("document_id")

        if mode == "bm25":
            if self._bm25_index is None:
                raise ValueError("BM25 retrieval requested but no bm25_index was configured")
            return self._bm25_index.search(
                query,
                limit=limit,
                paper_id=target_paper_id,
                document_id=target_doc_id,
            )

        if self._cache_queries and query in self._query_cache:
            vector = list(self._query_cache[query])
        else:
            vector = self._embedding_service.embed_text(query)
            self._validate_vector(vector)
            if self._cache_queries:
                if len(self._query_cache) >= self._max_cache_entries:
                    self._evict_oldest()
                self._query_cache[query] = list(vector)

        try:
            raw_result = self._vector_store.search(
                vector, limit=limit, filter=effective_filter or None
            )
        except TypeError:
            raw_result = self._vector_store.search(vector, limit=limit)
            if effective_filter:
                filtered_chunks = [
                    c for c in raw_result.results
                    if all(str(c.metadata.get(k)) == str(v) for k, v in effective_filter.items())
                ]
                raw_result = SearchResult(query=raw_result.query, results=filtered_chunks)

        if not isinstance(raw_result, SearchResult):
            raise TypeError(
                f"Vector store search must return a SearchResult, got {type(raw_result).__name__}"
            )

        if mode == "hybrid":
            if self._bm25_index is None:
                raise ValueError("Hybrid retrieval requested but no bm25_index was configured")
            from nexora.retrieval.bm25 import reciprocal_rank_fusion

            bm25_res = self._bm25_index.search(
                query,
                limit=limit,
                paper_id=target_paper_id,
                document_id=target_doc_id,
            )
            return reciprocal_rank_fusion(raw_result, bm25_res, rrf_k=rrf_k, limit=limit)

        return raw_result
