"""Retrieval service: query text -> embedding -> VectorStore search (step 11F)."""

from __future__ import annotations

from typing import Any

from nexora.retrieval.models import SearchResult
from nexora.retrieval.store import VectorStore


class RetrievalService:
    """Application-level retrieval boundary.

    Depends only on an embedding service (``embed_text``) and the VectorStore
    protocol, so it works with any store implementation.
    """

    def __init__(self, embedding_service: Any, vector_store: VectorStore) -> None:
        if not callable(getattr(embedding_service, "embed_text", None)):
            raise TypeError("embedding_service must provide embed_text(text)")
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must implement the VectorStore protocol")
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(self, query: str, *, limit: int = 10) -> SearchResult:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        vector = self._embedding_service.embed_text(query)
        return self._vector_store.search(vector, limit=limit)
