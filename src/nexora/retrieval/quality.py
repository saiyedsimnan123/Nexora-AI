"""Retrieval quality layer: filters and diversifies raw vector-search candidates (11G)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from nexora.retrieval.models import RetrievedChunk, SearchResult


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class RetrievalQualityPolicy:
    """Immutable settings for post-processing retrieval candidates.

    ``candidate_limit`` defaults to ``3 * max_results`` when not given.
    """

    min_score: float | None = None
    max_results: int = 10
    max_results_per_document: int | None = None
    candidate_limit: int | None = None

    def __post_init__(self) -> None:
        if self.min_score is not None:
            if isinstance(self.min_score, bool) or not isinstance(
                self.min_score, (int, float)
            ):
                raise TypeError("min_score must be a number or None")
            if not math.isfinite(self.min_score):
                raise ValueError("min_score must be finite")
        _positive_int("max_results", self.max_results)
        if self.max_results_per_document is not None:
            _positive_int("max_results_per_document", self.max_results_per_document)
        if self.candidate_limit is None:
            object.__setattr__(self, "candidate_limit", self.max_results * 3)
        else:
            _positive_int("candidate_limit", self.candidate_limit)
            if self.candidate_limit < self.max_results:
                raise ValueError("candidate_limit must be >= max_results")


class RetrievalQualityService:
    """Wraps RetrievalService and cleans up its ranked candidates."""

    def __init__(self, retrieval_service: Any, policy: RetrievalQualityPolicy) -> None:
        if not callable(getattr(retrieval_service, "retrieve", None)):
            raise TypeError("retrieval_service must provide retrieve(query, *, limit)")
        if not isinstance(policy, RetrievalQualityPolicy):
            raise TypeError("policy must be a RetrievalQualityPolicy")
        self._retrieval = retrieval_service
        self._policy = policy

    def retrieve(self, query: str) -> SearchResult:
        candidates = self._retrieval.retrieve(
            query, limit=self._policy.candidate_limit
        )
        selected = self._select(candidates.results)
        return SearchResult(query=query, results=selected)

    def _select(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        policy = self._policy
        seen_chunks: set[str] = set()
        per_document: dict[str, int] = {}
        selected: list[RetrievedChunk] = []
        for chunk in candidates:
            if len(selected) >= policy.max_results:
                break
            if policy.min_score is not None and chunk.score < policy.min_score:
                continue
            if chunk.chunk_id in seen_chunks:
                continue
            if policy.max_results_per_document is not None:
                if per_document.get(chunk.document_id, 0) >= policy.max_results_per_document:
                    continue
            seen_chunks.add(chunk.chunk_id)
            per_document[chunk.document_id] = per_document.get(chunk.document_id, 0) + 1
            selected.append(chunk)
        return selected
