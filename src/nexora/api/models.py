"""External request/response contracts (plain dataclasses, no HTTP dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nexora.api.errors import InvalidRequestError
from nexora.llm.config import validate_max_output_tokens, validate_temperature
from nexora.research.models import ResearchAnswer


@dataclass(frozen=True)
class ResearchRequest:
    """A research question plus optional generation controls.

    Validation failures raise InvalidRequestError. The query is kept exactly as
    supplied (no stripping or normalization).
    """

    query: str
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise InvalidRequestError("query must be a string")
        if not self.query.strip():
            raise InvalidRequestError("query must not be empty or whitespace-only")
        if self.model is not None:
            if not isinstance(self.model, str):
                raise InvalidRequestError("model must be a string")
            if not self.model.strip():
                raise InvalidRequestError("model must not be empty")
        try:
            validate_temperature(self.temperature)
            validate_max_output_tokens(self.max_output_tokens)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(str(exc)) from None


@dataclass(frozen=True)
class ResearchResponse:
    """Public view of a ResearchAnswer. Provider ``raw`` data is deliberately omitted."""

    text: str
    query: str
    model: str
    usage: dict[str, int] | None
    evidence_count: int

    def __post_init__(self) -> None:
        if self.usage is not None:
            object.__setattr__(self, "usage", dict(self.usage))

    @classmethod
    def from_answer(cls, answer: ResearchAnswer) -> ResearchResponse:
        if not isinstance(answer, ResearchAnswer):
            raise TypeError("answer must be a ResearchAnswer")
        return cls(
            text=answer.text,
            query=answer.query,
            model=answer.model,
            usage=answer.usage,
            evidence_count=answer.evidence_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "text": self.text,
            "query": self.query,
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
            "evidence_count": self.evidence_count,
        }


@dataclass(frozen=True)
class DocumentIngestResponse:
    """Public view of a document ingestion result."""

    document_id: str
    paper_id: str
    chunks_count: int
    indexed_chunks_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise InvalidRequestError("document_id must be a non-empty string")
        if not isinstance(self.paper_id, str) or not self.paper_id.strip():
            raise InvalidRequestError("paper_id must be a non-empty string")
        if isinstance(self.chunks_count, bool) or not isinstance(self.chunks_count, int) or self.chunks_count < 0:
            raise InvalidRequestError("chunks_count must be a non-negative integer")
        if isinstance(self.indexed_chunks_count, bool) or not isinstance(self.indexed_chunks_count, int) or self.indexed_chunks_count < 0:
            raise InvalidRequestError("indexed_chunks_count must be a non-negative integer")

    @classmethod
    def from_result(cls, result: Any) -> DocumentIngestResponse:
        return cls(
            document_id=str(result.document_id),
            paper_id=str(result.paper_id),
            chunks_count=int(result.chunks_count),
            indexed_chunks_count=int(result.indexed_chunks_count),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "document_id": self.document_id,
            "paper_id": self.paper_id,
            "chunks_count": self.chunks_count,
            "indexed_chunks_count": self.indexed_chunks_count,
        }
