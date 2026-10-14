"""Multi-paper comparison models and schema (Milestone 12C-24).

Defines validated, modular dataclasses for comparing two or more research papers
across aligned research dimensions while preserving per-paper evidence claims.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from nexora.llm.models import _check_no_secret_keys
from nexora.research.analysis import ANALYSIS_DIMENSIONS, AnalysisField


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
