"""Structured paper analysis models and schema (Milestone 12C-24).

Defines validated, modular dataclasses for structured research paper analysis
across 7 standardized research dimensions with strict evidence traceability.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from nexora.llm.models import _check_no_secret_keys

ANALYSIS_DIMENSIONS: tuple[str, ...] = (
    "problem_and_objectives",
    "methodology_and_algorithms",
    "models_and_datasets",
    "experiments_and_metrics",
    "results_and_findings",
    "limitations",
    "future_work",
)

VALID_STATUSES: frozenset[str] = frozenset({"reported", "not_reported", "uncertain"})
NOT_REPORTED_TEXT: str = "Not reported in the provided evidence."


@dataclass(frozen=True)
class AnalysisField:
    """A structured analysis section or claim with evidence traceability."""

    summary: str
    status: str = "reported"
    evidence_labels: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str):
            raise TypeError("summary must be a string")
        if not self.summary.strip():
            raise ValueError("summary must not be empty or whitespace-only")
        if not isinstance(self.status, str) or self.status not in VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(VALID_STATUSES)}, got {self.status!r}"
            )
        if not isinstance(self.evidence_labels, (list, tuple)):
            raise TypeError("evidence_labels must be a list of strings")
        for idx, lbl in enumerate(self.evidence_labels):
            if not isinstance(lbl, str) or not lbl.strip():
                raise TypeError(f"evidence_labels[{idx}] must be a non-empty string")
        if not isinstance(self.sources, (list, tuple)):
            raise TypeError("sources must be a list of dictionaries")
        for idx, src in enumerate(self.sources):
            if not isinstance(src, dict):
                raise TypeError(f"sources[{idx}] must be a dictionary")
        if not isinstance(self.details, dict):
            raise TypeError("details must be a dictionary")

        object.__setattr__(self, "evidence_labels", list(self.evidence_labels))
        object.__setattr__(self, "sources", copy.deepcopy(list(self.sources)))
        object.__setattr__(self, "details", copy.deepcopy(self.details))

    @property
    def is_reported(self) -> bool:
        """True if the information was explicitly found in the paper evidence."""
        return self.status == "reported"

    @property
    def is_missing(self) -> bool:
        """True if the information was not reported in the paper evidence."""
        return self.status == "not_reported"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation of this analysis field."""
        return {
            "summary": self.summary,
            "status": self.status,
            "evidence_labels": list(self.evidence_labels),
            "sources": copy.deepcopy(self.sources),
            "details": copy.deepcopy(self.details),
        }

    @classmethod
    def not_reported(cls, message: str = NOT_REPORTED_TEXT) -> AnalysisField:
        """Factory for an explicitly missing/unreported field."""
        return cls(summary=message, status="not_reported", evidence_labels=[], sources=[], details={})


@dataclass(frozen=True)
class StructuredPaperAnalysis:
    """Complete validated structured analysis of an individual research paper."""

    paper_id: str
    problem_and_objectives: AnalysisField
    methodology_and_algorithms: AnalysisField
    models_and_datasets: AnalysisField
    experiments_and_metrics: AnalysisField
    results_and_findings: AnalysisField
    limitations: AnalysisField
    future_work: AnalysisField
    title: str | None = None
    evidence_count: int = 0
    model: str = "unknown"
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_id, str) or not self.paper_id.strip():
            raise TypeError("paper_id must be a non-empty string")
        if self.title is not None and not isinstance(self.title, str):
            raise TypeError("title must be a string or None")
        if not isinstance(self.model, str) or not self.model.strip():
            raise TypeError("model must be a non-empty string")
        if isinstance(self.evidence_count, bool) or not isinstance(self.evidence_count, int):
            raise TypeError("evidence_count must be an integer")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")

        fields_to_check = {
            "problem_and_objectives": self.problem_and_objectives,
            "methodology_and_algorithms": self.methodology_and_algorithms,
            "models_and_datasets": self.models_and_datasets,
            "experiments_and_metrics": self.experiments_and_metrics,
            "results_and_findings": self.results_and_findings,
            "limitations": self.limitations,
            "future_work": self.future_work,
        }
        for name, val in fields_to_check.items():
            if not isinstance(val, AnalysisField):
                raise TypeError(f"{name} must be an AnalysisField, got {type(val).__name__}")

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

    def get_field(self, dimension: str) -> AnalysisField:
        """Retrieve an analysis field by dimension name."""
        if dimension not in ANALYSIS_DIMENSIONS:
            raise KeyError(
                f"Unknown dimension {dimension!r}. Expected one of {ANALYSIS_DIMENSIONS}"
            )
        return getattr(self, dimension)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation of the structured paper analysis."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "problem_and_objectives": self.problem_and_objectives.to_dict(),
            "methodology_and_algorithms": self.methodology_and_algorithms.to_dict(),
            "models_and_datasets": self.models_and_datasets.to_dict(),
            "experiments_and_metrics": self.experiments_and_metrics.to_dict(),
            "results_and_findings": self.results_and_findings.to_dict(),
            "limitations": self.limitations.to_dict(),
            "future_work": self.future_work.to_dict(),
            "evidence_count": self.evidence_count,
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
        }

    @classmethod
    def empty(
        cls,
        paper_id: str,
        *,
        title: str | None = None,
        model: str = "none",
        message: str = NOT_REPORTED_TEXT,
    ) -> StructuredPaperAnalysis:
        """Create a structured analysis where all dimensions are explicitly not reported."""
        return cls(
            paper_id=paper_id,
            title=title,
            problem_and_objectives=AnalysisField.not_reported(message),
            methodology_and_algorithms=AnalysisField.not_reported(message),
            models_and_datasets=AnalysisField.not_reported(message),
            experiments_and_metrics=AnalysisField.not_reported(message),
            results_and_findings=AnalysisField.not_reported(message),
            limitations=AnalysisField.not_reported(message),
            future_work=AnalysisField.not_reported(message),
            evidence_count=0,
            model=model,
            usage=None,
        )
