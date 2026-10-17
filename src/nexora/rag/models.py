"""Data models for the end-to-end RAG pipeline.

Defines contracts for RAG requests, attributed sources, and grounded responses:
- AttributedSource: Traceable paper metadata, chunk ID, relevance score, and excerpt.
- RAGRequest: Parameterized research question with retrieval and generation options.
- RAGResponse: Complete answer payload with supporting sources, citation verification,
  and explicit mock vs. live model demarcation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from nexora.llm.config import validate_max_output_tokens, validate_temperature


@dataclass(frozen=True)
class AttributedSource:
    """One retrieved piece of research evidence with source attribution."""

    label: str
    document_id: str
    chunk_id: str
    paper_id: str | None
    paper_title: str
    score: float
    text: str
    text_excerpt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string, e.g. '[E1]'")
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.paper_title, str) or not self.paper_title.strip():
            raise ValueError("paper_title must be a non-empty string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be a numeric value")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.text_excerpt, str):
            raise TypeError("text_excerpt must be a string")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    @property
    def page_number(self) -> int | None:
        """Page number from metadata if present and valid."""
        val = self.metadata.get("page_number")
        if val is None:
            val = self.metadata.get("page")
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            return None
        return val

    def to_dict(self) -> dict[str, Any]:
        """Convert attributed source to a JSON-serializable dictionary."""
        d = {
            "label": self.label,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "score": round(float(self.score), 6),
            "text": self.text,
            "text_excerpt": self.text_excerpt,
            "metadata": dict(self.metadata),
        }
        if self.page_number is not None:
            d["page_number"] = self.page_number
        return d


@dataclass(frozen=True)
class RAGRequest:
    """A user research question with retrieval and generation configurations."""

    query: str
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    min_score: float | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    paper_id: str | None = None
    document_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not self.query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if self.retrieval_mode not in ("bm25", "vector", "hybrid"):
            raise ValueError(f"retrieval_mode must be 'bm25', 'vector', or 'hybrid', got {self.retrieval_mode!r}")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {self.top_k}")
        if self.min_score is not None:
            if isinstance(self.min_score, bool) or not isinstance(self.min_score, (int, float)):
                raise TypeError("min_score must be a number or None")
        validate_temperature(self.temperature)
        validate_max_output_tokens(self.max_output_tokens)


@dataclass(frozen=True)
class RAGResponse:
    """Comprehensive grounded answer from the RAG pipeline."""

    query: str
    answer: str
    sources: list[AttributedSource]
    cited_sources: list[AttributedSource]
    is_grounded: bool
    is_mock: bool
    model: str
    status: str = "success"
    error: str | None = None
    usage: dict[str, int] | None = None
    retrieval_mode: str = "hybrid"
    evidence_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.answer, str):
            raise TypeError("answer must be a string")
        if not isinstance(self.sources, (list, tuple)):
            raise TypeError("sources must be a list of AttributedSource")
        if not all(isinstance(s, AttributedSource) for s in self.sources):
            raise TypeError("sources must contain only AttributedSource instances")
        if not isinstance(self.cited_sources, (list, tuple)):
            raise TypeError("cited_sources must be a list of AttributedSource")
        if not all(isinstance(s, AttributedSource) for s in self.cited_sources):
            raise TypeError("cited_sources must contain only AttributedSource instances")
        if not isinstance(self.is_grounded, bool):
            raise TypeError("is_grounded must be a boolean")
        if not isinstance(self.is_mock, bool):
            raise TypeError("is_mock must be a boolean")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.status not in ("success", "insufficient_evidence", "error"):
            raise ValueError(f"status must be 'success', 'insufficient_evidence', or 'error', got {self.status!r}")

        object.__setattr__(self, "sources", list(self.sources))
        object.__setattr__(self, "cited_sources", list(self.cited_sources))
        if self.usage is not None:
            object.__setattr__(self, "usage", dict(self.usage))

    @property
    def has_citations(self) -> bool:
        """True if the generated answer contains at least one citation."""
        return len(self.cited_sources) > 0

    @property
    def uncited_sources(self) -> list[AttributedSource]:
        """Sources retrieved in the context but not explicitly cited in the answer."""
        cited_chunk_ids = {s.chunk_id for s in self.cited_sources}
        return [s for s in self.sources if s.chunk_id not in cited_chunk_ids]

    def to_dict(self) -> dict[str, Any]:
        """Convert complete response to a structured JSON-serializable dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "cited_sources": [s.to_dict() for s in self.cited_sources],
            "is_grounded": self.is_grounded,
            "is_mock": self.is_mock,
            "model": self.model,
            "status": self.status,
            "error": self.error,
            "usage": dict(self.usage) if self.usage else None,
            "retrieval_mode": self.retrieval_mode,
            "evidence_count": self.evidence_count,
        }
