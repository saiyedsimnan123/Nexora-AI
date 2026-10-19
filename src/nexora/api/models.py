"""External request/response contracts (plain dataclasses, no HTTP dependencies)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

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
    collection_id: str | None = None

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


@dataclass(frozen=True)
class PaperAnalysisRequest:
    """Request to perform structured analysis on an individual research paper."""

    paper_id: str
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_id, str):
            raise InvalidRequestError("paper_id must be a string")
        if not self.paper_id.strip():
            raise InvalidRequestError("paper_id must not be empty or whitespace-only")
        from uuid import UUID
        try:
            UUID(self.paper_id.strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError("Invalid paper_id: must be a valid UUID.")
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
class PaperAnalysisResponse:
    """Public view of StructuredPaperAnalysis (provider raw data deliberately omitted)."""

    paper_id: str
    title: str | None
    problem_and_objectives: dict[str, Any]
    methodology_and_algorithms: dict[str, Any]
    models_and_datasets: dict[str, Any]
    experiments_and_metrics: dict[str, Any]
    results_and_findings: dict[str, Any]
    limitations: dict[str, Any]
    future_work: dict[str, Any]
    evidence_count: int
    model: str
    usage: dict[str, int] | None

    @classmethod
    def from_analysis(cls, analysis: Any) -> PaperAnalysisResponse:
        d = analysis.to_dict()
        return cls(
            paper_id=d["paper_id"],
            title=d["title"],
            problem_and_objectives=d["problem_and_objectives"],
            methodology_and_algorithms=d["methodology_and_algorithms"],
            models_and_datasets=d["models_and_datasets"],
            experiments_and_metrics=d["experiments_and_metrics"],
            results_and_findings=d["results_and_findings"],
            limitations=d["limitations"],
            future_work=d["future_work"],
            evidence_count=d["evidence_count"],
            model=d["model"],
            usage=d["usage"],
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "problem_and_objectives": dict(self.problem_and_objectives),
            "methodology_and_algorithms": dict(self.methodology_and_algorithms),
            "models_and_datasets": dict(self.models_and_datasets),
            "experiments_and_metrics": dict(self.experiments_and_metrics),
            "results_and_findings": dict(self.results_and_findings),
            "limitations": dict(self.limitations),
            "future_work": dict(self.future_work),
            "evidence_count": self.evidence_count,
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
        }


@dataclass(frozen=True)
class PaperComparisonRequest:
    """Request to compare two or more research papers across aligned dimensions."""

    paper_ids: list[str]
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_ids, (list, tuple)):
            raise InvalidRequestError("paper_ids must be a list of strings")
        if len(self.paper_ids) < 2:
            raise InvalidRequestError("At least 2 paper IDs are required for comparison.")
        from uuid import UUID
        cleaned: list[str] = []
        for idx, pid in enumerate(self.paper_ids):
            if not isinstance(pid, str) or not pid.strip():
                raise InvalidRequestError(f"paper_ids[{idx}] must be a non-empty string")
            try:
                UUID(pid.strip())
            except (ValueError, AttributeError):
                raise InvalidRequestError(f"Invalid paper_id {pid!r}: must be a valid UUID.")
            cleaned.append(str(UUID(pid.strip())))
        if len(set(cleaned)) != len(cleaned):
            raise InvalidRequestError("paper_ids must be unique with no duplicates.")
        object.__setattr__(self, "paper_ids", cleaned)
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
class PaperComparisonResponse:
    """Public view of PaperComparison (provider raw data deliberately omitted)."""

    paper_ids: list[str]
    dimensions: dict[str, Any]
    overall_summary: str
    model: str
    usage: dict[str, int] | None

    @classmethod
    def from_comparison(cls, comparison: Any) -> PaperComparisonResponse:
        d = comparison.to_dict()
        return cls(
            paper_ids=d["paper_ids"],
            dimensions=d["dimensions"],
            overall_summary=d["overall_summary"],
            model=d["model"],
            usage=d["usage"],
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "paper_ids": list(self.paper_ids),
            "dimensions": dict(self.dimensions),
            "overall_summary": self.overall_summary,
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
        }


@dataclass(frozen=True)
class ResearchCompareRequest:
    """Request to compare research papers across 7 dimensions with evidence synthesis."""

    paper_ids: list[str]
    query: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_ids, (list, tuple)):
            raise InvalidRequestError("paper_ids must be a list of strings")
        if len(self.paper_ids) < 2:
            raise InvalidRequestError("At least 2 paper IDs are required for comparison.")
        if len(self.paper_ids) > 5:
            raise InvalidRequestError("Comparison exceeds maximum of 5 papers.")

        from uuid import UUID

        cleaned: list[str] = []
        for idx, pid in enumerate(self.paper_ids):
            if not isinstance(pid, str) or not pid.strip():
                raise InvalidRequestError(f"paper_ids[{idx}] must be a non-empty string")
            try:
                UUID(pid.strip())
            except (ValueError, AttributeError):
                raise InvalidRequestError(f"Invalid paper_id {pid!r}: must be a valid UUID.")
            cleaned.append(str(UUID(pid.strip())))
        if len(set(cleaned)) != len(cleaned):
            raise InvalidRequestError("paper_ids must be unique with no duplicates.")
        object.__setattr__(self, "paper_ids", cleaned)

        if self.query is not None:
            if not isinstance(self.query, str):
                raise InvalidRequestError("query must be a string or None")

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
class ResearchCompareResponse:
    """Public view of CrossPaperComparisonResult (provider raw data deliberately omitted)."""

    paper_ids: list[str]
    paper_titles: dict[str, str | None]
    matrix: dict[str, Any]
    conclusion: str
    conclusion_reasoning: str
    comparability_status: str
    overall_summary: str
    model: str
    usage: dict[str, int] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_ids", list(self.paper_ids))
        object.__setattr__(self, "paper_titles", dict(self.paper_titles))
        object.__setattr__(self, "matrix", copy.deepcopy(dict(self.matrix)))
        if self.usage is not None:
            object.__setattr__(self, "usage", dict(self.usage))

    @classmethod
    def from_result(cls, result: Any) -> ResearchCompareResponse:
        d = result.to_dict()
        return cls(
            paper_ids=d["paper_ids"],
            paper_titles=d["paper_titles"],
            matrix=d["matrix"],
            conclusion=d["conclusion"],
            conclusion_reasoning=d["conclusion_reasoning"],
            comparability_status=d["comparability_status"],
            overall_summary=d["overall_summary"],
            model=d["model"],
            usage=d.get("usage"),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "paper_ids": list(self.paper_ids),
            "paper_titles": dict(self.paper_titles),
            "matrix": copy.deepcopy(self.matrix),
            "conclusion": self.conclusion,
            "conclusion_reasoning": self.conclusion_reasoning,
            "comparability_status": self.comparability_status,
            "overall_summary": self.overall_summary,
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
        }


@dataclass(frozen=True)
class ResearchVerifyRequest:
    """A research question to answer and verify, or an answer to verify against retrieved evidence."""

    query: str
    text: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    expected_paper_id: str | None = None
    collection_ids: Sequence[str] | None = None
    paper_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise InvalidRequestError("query must be a string")
        if not self.query.strip():
            raise InvalidRequestError("query must not be empty or whitespace-only")
        if self.text is not None:
            if not isinstance(self.text, str):
                raise InvalidRequestError("text must be a string or None")
        if self.model is not None:
            if not isinstance(self.model, str):
                raise InvalidRequestError("model must be a string")
            if not self.model.strip():
                raise InvalidRequestError("model must not be empty")
        if self.expected_paper_id is not None:
            if not isinstance(self.expected_paper_id, str):
                raise InvalidRequestError("expected_paper_id must be a string")
        if self.paper_id is not None:
            if not isinstance(self.paper_id, str):
                raise InvalidRequestError("paper_id must be a string")
        if self.collection_ids is not None:
            if not isinstance(self.collection_ids, (list, tuple)):
                raise InvalidRequestError("collection_ids must be a sequence of strings")
            for c in self.collection_ids:
                if not isinstance(c, str) or not c.strip():
                    raise InvalidRequestError("collection_ids must contain non-empty strings")
        try:
            validate_temperature(self.temperature)
            validate_max_output_tokens(self.max_output_tokens)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(str(exc)) from None


@dataclass(frozen=True)
class ResearchVerifyResponse:
    """Public view of a research answer with its claim-to-evidence verification result."""

    text: str
    query: str
    model: str
    usage: dict[str, int] | None
    evidence_count: int
    verification: dict[str, Any]

    def __post_init__(self) -> None:
        if self.usage is not None:
            object.__setattr__(self, "usage", dict(self.usage))
        if not isinstance(self.verification, dict):
            raise TypeError("verification must be a dict")
        object.__setattr__(self, "verification", copy.deepcopy(self.verification))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "text": self.text,
            "query": self.query,
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
            "evidence_count": self.evidence_count,
            "verification": copy.deepcopy(self.verification),
        }


@dataclass(frozen=True)
class ResearchGapsRequest:
    """Request for research gap discovery across specific papers or a collection."""

    paper_ids: list[str] | None = None
    collection_id: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.paper_ids is None and self.collection_id is None:
            raise InvalidRequestError("Either 'paper_ids' or 'collection_id' must be provided.")
        if self.paper_ids is not None and self.collection_id is not None:
            raise InvalidRequestError("Provide either 'paper_ids' or 'collection_id', not both.")

        if self.paper_ids is not None:
            if not isinstance(self.paper_ids, (list, tuple)):
                raise InvalidRequestError("paper_ids must be a list of strings")
            if len(self.paper_ids) == 0:
                raise InvalidRequestError("paper_ids must not be empty")
            if len(self.paper_ids) > 5:
                raise InvalidRequestError("paper_ids must contain at most 5 papers")

            from uuid import UUID

            cleaned: list[str] = []
            for idx, pid in enumerate(self.paper_ids):
                if not isinstance(pid, str) or not pid.strip():
                    raise InvalidRequestError(f"paper_ids[{idx}] must be a non-empty string")
                try:
                    UUID(pid.strip())
                except (ValueError, AttributeError):
                    raise InvalidRequestError(f"Invalid paper_id {pid!r}: must be a valid UUID.")
                cleaned.append(str(UUID(pid.strip())))
            if len(set(cleaned)) != len(cleaned):
                raise InvalidRequestError("paper_ids must be unique with no duplicates.")
            object.__setattr__(self, "paper_ids", cleaned)

        if self.collection_id is not None:
            if not isinstance(self.collection_id, str) or not self.collection_id.strip():
                raise InvalidRequestError("collection_id must be a non-empty string")
            from uuid import UUID

            try:
                col_uuid = UUID(self.collection_id.strip())
            except (ValueError, AttributeError):
                raise InvalidRequestError(
                    f"Invalid collection_id {self.collection_id!r}: must be a valid UUID."
                )
            object.__setattr__(self, "collection_id", str(col_uuid))

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
class ResearchGapsResponse:
    """Public view of ResearchGapReport (provider raw data deliberately omitted)."""

    paper_ids: list[str]
    collection_id: str | None
    documented_gaps: list[dict[str, Any]]
    inferred_gaps: list[dict[str, Any]]
    total_gaps_count: int
    summary: str
    limitations: list[str]
    model: str
    usage: dict[str, int] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_ids", list(self.paper_ids))
        object.__setattr__(self, "documented_gaps", copy.deepcopy(list(self.documented_gaps)))
        object.__setattr__(self, "inferred_gaps", copy.deepcopy(list(self.inferred_gaps)))
        object.__setattr__(self, "limitations", list(self.limitations))
        if self.usage is not None:
            object.__setattr__(self, "usage", dict(self.usage))

    @classmethod
    def from_report(cls, report: Any) -> ResearchGapsResponse:
        d = report.to_dict()
        return cls(
            paper_ids=d["paper_ids"],
            collection_id=d.get("collection_id"),
            documented_gaps=d["documented_gaps"],
            inferred_gaps=d["inferred_gaps"],
            total_gaps_count=d["total_gaps_count"],
            summary=d["summary"],
            limitations=d["limitations"],
            model=d["model"],
            usage=d.get("usage"),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return {
            "paper_ids": list(self.paper_ids),
            "collection_id": self.collection_id,
            "documented_gaps": copy.deepcopy(self.documented_gaps),
            "inferred_gaps": copy.deepcopy(self.inferred_gaps),
            "total_gaps_count": self.total_gaps_count,
            "summary": self.summary,
            "limitations": list(self.limitations),
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
        }


@dataclass(frozen=True)
class DiscoverySearchRequest:
    """Request query parameters for searching scholarly research papers."""

    query: str
    categories: tuple[str, ...] = ()
    offset: int = 0
    limit: int = 10
    sort_by: str = "relevance"
    sort_order: str = "descending"

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise InvalidRequestError("query must not be empty or whitespace-only")
        if not isinstance(self.categories, (list, tuple)):
            raise InvalidRequestError("categories must be a list or tuple of strings")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise InvalidRequestError("offset must be a non-negative integer")
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit < 1
            or self.limit > 100
        ):
            raise InvalidRequestError("limit must be an integer between 1 and 100")
        if self.sort_by not in ("relevance", "submitted_date", "last_updated"):
            raise InvalidRequestError(
                f"sort_by must be one of ('relevance', 'submitted_date', 'last_updated'), got {self.sort_by!r}"
            )
        if self.sort_order not in ("descending", "ascending"):
            raise InvalidRequestError(
                f"sort_order must be one of ('descending', 'ascending'), got {self.sort_order!r}"
            )
        object.__setattr__(self, "categories", tuple(str(c).strip() for c in self.categories if str(c).strip()))


@dataclass(frozen=True)
class DiscoverySearchResponse:
    """Public view of scholarly search results."""

    provider: str
    query: str
    total_results: int
    offset: int
    limit: int
    papers: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "query": self.query,
            "total_results": self.total_results,
            "offset": self.offset,
            "limit": self.limit,
            "papers": copy.deepcopy(self.papers),
        }


@dataclass(frozen=True)
class DiscoveryImportRequest:
    """Request for explicitly importing and ingesting a paper."""

    collection_id: str
    external_id: str
    durable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.collection_id, str) or not self.collection_id.strip():
            raise InvalidRequestError("collection_id is required")
        if not isinstance(self.external_id, str) or not self.external_id.strip():
            raise InvalidRequestError("external_id is required")


@dataclass(frozen=True)
class DiscoveryImportResponse:
    """Public view of explicit paper import result."""

    paper_id: str
    collection_id: str
    external_id: str
    status: str
    message: str
    title: str
    document_id: str | None = None
    chunks_count: int = 0
    indexed_chunks_count: int = 0
    doi: str | None = None
    metadata: dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "collection_id": self.collection_id,
            "external_id": self.external_id,
            "status": self.status,
            "message": self.message,
            "title": self.title,
            "document_id": self.document_id,
            "chunks_count": self.chunks_count,
            "indexed_chunks_count": self.indexed_chunks_count,
            "doi": self.doi,
            "metadata": dict(self.metadata),
        }


# Re-export RAG contracts for external consumers of the API package
from nexora.rag.models import AttributedSource, RAGRequest, RAGResponse

__all__ = [
    "AttributedSource",
    "DiscoveryImportRequest",
    "DiscoveryImportResponse",
    "DiscoverySearchRequest",
    "DiscoverySearchResponse",
    "DocumentIngestResponse",
    "PaperAnalysisRequest",
    "PaperAnalysisResponse",
    "PaperComparisonRequest",
    "PaperComparisonResponse",
    "RAGRequest",
    "RAGResponse",
    "ResearchCompareRequest",
    "ResearchCompareResponse",
    "ResearchGapsRequest",
    "ResearchGapsResponse",
    "ResearchRequest",
    "ResearchResponse",
    "ResearchVerifyRequest",
    "ResearchVerifyResponse",
]



