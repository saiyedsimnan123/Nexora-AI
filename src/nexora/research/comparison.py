"""Multi-paper comparison models and schema (Milestone 12C-24).

Defines validated, modular dataclasses for comparing two or more research papers
across aligned research dimensions while preserving per-paper evidence claims.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nexora.llm.models import _check_no_secret_keys
from nexora.research.analysis import ANALYSIS_DIMENSIONS, AnalysisField


class ConclusionClassification(str, Enum):
    """Standardized classification of cross-paper evaluation outcomes."""

    DEMONSTRATED_IMPROVEMENT = "Demonstrated improvement"
    LIMITED_EVIDENCE = "Limited evidence"
    CONFLICTING_EVIDENCE = "Conflicting evidence"
    INSUFFICIENT_EVIDENCE = "Insufficient evidence"


COMPARISON_MATRIX_DIMENSIONS = (
    "methods",
    "datasets",
    "baselines",
    "metrics",
    "results",
    "limitations",
    "comparability",
)

VALID_CONCLUSIONS = (
    ConclusionClassification.DEMONSTRATED_IMPROVEMENT.value,
    ConclusionClassification.LIMITED_EVIDENCE.value,
    ConclusionClassification.CONFLICTING_EVIDENCE.value,
    ConclusionClassification.INSUFFICIENT_EVIDENCE.value,
)


@dataclass(frozen=True)
class MatrixEvidenceSource:
    """A verified evidence chunk supporting a cross-paper matrix cell."""

    document_id: str
    chunk_id: str
    paper_id: str
    paper_title: str | None = None
    page_number: int | None = None
    text_excerpt: str = ""
    score: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise TypeError("document_id must be a non-empty string")
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise TypeError("chunk_id must be a non-empty string")
        if not isinstance(self.paper_id, str) or not self.paper_id.strip():
            raise TypeError("paper_id must be a non-empty string")
        if self.page_number is not None:
            if isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or self.page_number <= 0:
                raise TypeError("page_number must be a positive integer or None")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be a float or int")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "page_number": self.page_number,
            "text_excerpt": self.text_excerpt,
            "score": round(float(self.score), 4),
        }
        return d


@dataclass(frozen=True)
class MatrixPaperFinding:
    """The reported findings for a single paper on a matrix dimension."""

    paper_id: str
    paper_title: str | None = None
    summary: str = "Not reported in the provided evidence."
    status: str = "reported"  # "reported", "not_reported", "uncertain"
    sources: list[MatrixEvidenceSource] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.paper_id, str) or not self.paper_id.strip():
            raise TypeError("paper_id must be a non-empty string")
        if not isinstance(self.summary, str):
            raise TypeError("summary must be a string")
        if self.status not in ("reported", "not_reported", "uncertain"):
            raise ValueError(f"status must be 'reported', 'not_reported', or 'uncertain', got {self.status!r}")
        if not isinstance(self.sources, (list, tuple)):
            raise TypeError("sources must be a list of MatrixEvidenceSource")
        for idx, src in enumerate(self.sources):
            if not isinstance(src, MatrixEvidenceSource):
                raise TypeError(f"sources[{idx}] must be a MatrixEvidenceSource")
        if not isinstance(self.details, dict):
            raise TypeError("details must be a dictionary")
        object.__setattr__(self, "sources", list(self.sources))
        object.__setattr__(self, "details", dict(self.details))

    @classmethod
    def not_reported(cls, paper_id: str, paper_title: str | None = None) -> MatrixPaperFinding:
        return cls(
            paper_id=paper_id,
            paper_title=paper_title,
            summary="Not reported in the provided evidence.",
            status="not_reported",
            sources=[],
            details={},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "summary": self.summary,
            "status": self.status,
            "sources": [s.to_dict() for s in self.sources],
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class MatrixDimensionComparison:
    """Comparative synthesis for one dimension across all compared papers."""

    dimension: str
    per_paper: dict[str, MatrixPaperFinding]
    comparison_summary: str
    key_differences: list[str] = field(default_factory=list)
    key_similarities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, str) or not self.dimension.strip():
            raise TypeError("dimension must be a non-empty string")
        if not isinstance(self.per_paper, dict):
            raise TypeError("per_paper must be a dictionary mapping paper_id to MatrixPaperFinding")
        for pid, fld in self.per_paper.items():
            if not isinstance(pid, str) or not pid.strip():
                raise TypeError("per_paper keys must be non-empty paper_id strings")
            if not isinstance(fld, MatrixPaperFinding):
                raise TypeError(f"per_paper[{pid!r}] must be a MatrixPaperFinding")
        if not isinstance(self.comparison_summary, str):
            raise TypeError("comparison_summary must be a string")
        if not self.comparison_summary.strip():
            raise ValueError("comparison_summary must not be empty or whitespace-only")
        if not isinstance(self.key_differences, (list, tuple)):
            raise TypeError("key_differences must be a list of strings")
        for idx, diff in enumerate(self.key_differences):
            if not isinstance(diff, str) or not diff.strip():
                raise TypeError(f"key_differences[{idx}] must be a non-empty string")
        if not isinstance(self.key_similarities, (list, tuple)):
            raise TypeError("key_similarities must be a list of strings")
        for idx, sim in enumerate(self.key_similarities):
            if not isinstance(sim, str) or not sim.strip():
                raise TypeError(f"key_similarities[{idx}] must be a non-empty string")

        object.__setattr__(self, "per_paper", dict(self.per_paper))
        object.__setattr__(self, "key_differences", list(self.key_differences))
        object.__setattr__(self, "key_similarities", list(self.key_similarities))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "per_paper": {pid: fld.to_dict() for pid, fld in self.per_paper.items()},
            "comparison_summary": self.comparison_summary,
            "key_differences": list(self.key_differences),
            "key_similarities": list(self.key_similarities),
        }


@dataclass(frozen=True)
class CrossPaperComparisonResult:
    """Complete validated result of cross-paper comparison and evidence synthesis."""

    paper_ids: list[str]
    paper_titles: dict[str, str | None]
    matrix: dict[str, MatrixDimensionComparison]
    conclusion: str
    conclusion_reasoning: str
    comparability_status: str
    overall_summary: str
    model: str = "unknown"
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_ids, (list, tuple)):
            raise TypeError("paper_ids must be a list of strings")
        if len(self.paper_ids) < 2:
            raise ValueError("At least 2 paper IDs are required for cross-paper comparison")
        for idx, pid in enumerate(self.paper_ids):
            if not isinstance(pid, str) or not pid.strip():
                raise TypeError(f"paper_ids[{idx}] must be a non-empty string")
        if len(set(self.paper_ids)) != len(self.paper_ids):
            raise ValueError("paper_ids must be unique with no duplicates")

        if not isinstance(self.paper_titles, dict):
            raise TypeError("paper_titles must be a dictionary")

        if not isinstance(self.matrix, dict):
            raise TypeError("matrix must be a dictionary")
        for dim, mdc in self.matrix.items():
            if not isinstance(dim, str):
                raise TypeError("matrix keys must be strings")
            if not isinstance(mdc, MatrixDimensionComparison):
                raise TypeError(f"matrix[{dim!r}] must be a MatrixDimensionComparison")

        if not isinstance(self.conclusion, str) or self.conclusion not in VALID_CONCLUSIONS:
            raise ValueError(f"conclusion must be one of {VALID_CONCLUSIONS}, got {self.conclusion!r}")

        if not isinstance(self.conclusion_reasoning, str) or not self.conclusion_reasoning.strip():
            raise ValueError("conclusion_reasoning must be a non-empty string")

        if not isinstance(self.comparability_status, str) or not self.comparability_status.strip():
            raise ValueError("comparability_status must be a non-empty string")

        if not isinstance(self.overall_summary, str) or not self.overall_summary.strip():
            raise ValueError("overall_summary must be a non-empty string")

        if not isinstance(self.model, str) or not self.model.strip():
            raise TypeError("model must be a non-empty string")

        if self.usage is not None:
            if not isinstance(self.usage, dict):
                raise TypeError("usage must be a dict or None")
            object.__setattr__(self, "usage", dict(self.usage))

        if self.raw is not None:
            if not isinstance(self.raw, dict):
                raise TypeError("raw must be a dict or None")
            _check_no_secret_keys(self.raw)
            object.__setattr__(self, "raw", copy.deepcopy(self.raw))

        object.__setattr__(self, "paper_ids", list(self.paper_ids))
        object.__setattr__(self, "paper_titles", dict(self.paper_titles))
        object.__setattr__(self, "matrix", dict(self.matrix))

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_ids": list(self.paper_ids),
            "paper_titles": dict(self.paper_titles),
            "matrix": {dim: mdc.to_dict() for dim, mdc in self.matrix.items()},
            "conclusion": self.conclusion,
            "conclusion_reasoning": self.conclusion_reasoning,
            "comparability_status": self.comparability_status,
            "overall_summary": self.overall_summary,
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
        }


@dataclass(frozen=True)
class FieldComparison:
    """Comparison of a single dimension across multiple papers."""

    dimension: str
    per_paper: dict[str, AnalysisField]
    comparison_summary: str
    key_differences: list[str] = field(default_factory=list)
    key_similarities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, str) or not self.dimension.strip():
            raise TypeError("dimension must be a non-empty string")
        if not isinstance(self.per_paper, dict):
            raise TypeError("per_paper must be a dictionary mapping paper_id to AnalysisField")
        for pid, fld in self.per_paper.items():
            if not isinstance(pid, str) or not pid.strip():
                raise TypeError("per_paper keys must be non-empty paper_id strings")
            if not isinstance(fld, AnalysisField):
                raise TypeError(f"per_paper[{pid!r}] must be an AnalysisField")
        if not isinstance(self.comparison_summary, str):
            raise TypeError("comparison_summary must be a string")
        if not self.comparison_summary.strip():
            raise ValueError("comparison_summary must not be empty or whitespace-only")
        if not isinstance(self.key_differences, (list, tuple)):
            raise TypeError("key_differences must be a list of strings")
        for idx, diff in enumerate(self.key_differences):
            if not isinstance(diff, str) or not diff.strip():
                raise TypeError(f"key_differences[{idx}] must be a non-empty string")
        if not isinstance(self.key_similarities, (list, tuple)):
            raise TypeError("key_similarities must be a list of strings")
        for idx, sim in enumerate(self.key_similarities):
            if not isinstance(sim, str) or not sim.strip():
                raise TypeError(f"key_similarities[{idx}] must be a non-empty string")

        object.__setattr__(self, "per_paper", dict(self.per_paper))
        object.__setattr__(self, "key_differences", list(self.key_differences))
        object.__setattr__(self, "key_similarities", list(self.key_similarities))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation of the field comparison."""
        return {
            "dimension": self.dimension,
            "per_paper": {pid: fld.to_dict() for pid, fld in self.per_paper.items()},
            "comparison_summary": self.comparison_summary,
            "key_differences": list(self.key_differences),
            "key_similarities": list(self.key_similarities),
        }


@dataclass(frozen=True)
class PaperComparison:
    """Complete validated multi-paper comparison result."""

    paper_ids: list[str]
    dimensions: dict[str, FieldComparison]
    overall_summary: str
    model: str = "unknown"
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_ids, (list, tuple)):
            raise TypeError("paper_ids must be a list of strings")
        if len(self.paper_ids) < 2:
            raise ValueError("At least 2 paper IDs are required for comparison")
        for idx, pid in enumerate(self.paper_ids):
            if not isinstance(pid, str) or not pid.strip():
                raise TypeError(f"paper_ids[{idx}] must be a non-empty string")
        if len(set(self.paper_ids)) != len(self.paper_ids):
            raise ValueError("paper_ids must be unique with no duplicates")

        if not isinstance(self.dimensions, dict):
            raise TypeError("dimensions must be a dictionary mapping dimension name to FieldComparison")
        for dim, fc in self.dimensions.items():
            if not isinstance(dim, str):
                raise TypeError("dimensions keys must be strings")
            if not isinstance(fc, FieldComparison):
                raise TypeError(f"dimensions[{dim!r}] must be a FieldComparison")

        if not isinstance(self.overall_summary, str):
            raise TypeError("overall_summary must be a string")
        if not self.overall_summary.strip():
            raise ValueError("overall_summary must not be empty or whitespace-only")
        if not isinstance(self.model, str) or not self.model.strip():
            raise TypeError("model must be a non-empty string")

        if self.usage is not None:
            if not isinstance(self.usage, dict):
                raise TypeError("usage must be a dict or None")
            for key, count in self.usage.items():
                if not isinstance(key, str):
                    raise TypeError("usage keys must be strings")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise TypeError("usage values must be non-negative integers")
            object.__setattr__(self, "usage", dict(self.usage))

        if self.raw is not None:
            if not isinstance(self.raw, dict):
                raise TypeError("raw must be a dict or None")
            _check_no_secret_keys(self.raw)
            object.__setattr__(self, "raw", copy.deepcopy(self.raw))

        object.__setattr__(self, "paper_ids", list(self.paper_ids))
        object.__setattr__(self, "dimensions", dict(self.dimensions))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation of the multi-paper comparison."""
        return {
            "paper_ids": list(self.paper_ids),
            "dimensions": {dim: fc.to_dict() for dim, fc in self.dimensions.items()},
            "overall_summary": self.overall_summary,
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
        }
