"""Research gap data models (Milestone 12C-26).

Implements evidence-backed research gap models:
- Distinguishes author-documented gaps from AI-inferred candidate gaps.
- Preserves traceable chunk, document, and paper metadata.
- Records qualifying or contradicting evidence when present.
- Enforces strict uncertainty/confidence bounds and zero-hallucination guarantees.
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from nexora.llm.models import _check_no_secret_keys


class GapCategory(str, Enum):
    """Category of a research gap."""

    DOCUMENTED = "documented"
    INFERRED = "inferred"


VALID_GAP_CATEGORIES = {c.value for c in GapCategory}


@dataclass(frozen=True)
class ResearchGap:
    """An individual research gap with traceable evidence and reasoning."""

    gap_id: str
    category: str
    title: str
    description: str
    reasoning: str
    source_paper_ids: list[str]
    evidence_labels: list[str]
    supporting_sources: list[dict[str, Any]]
    contradicting_sources: list[dict[str, Any]]
    confidence: float
    limitations: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.gap_id, str) or not self.gap_id.strip():
            raise TypeError("gap_id must be a non-empty string")
        if not isinstance(self.category, str):
            raise TypeError("category must be a string")
        if self.category not in VALID_GAP_CATEGORIES:
            raise ValueError(
                f"category must be one of {VALID_GAP_CATEGORIES}, got {self.category!r}"
            )
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.reasoning, str) or not self.reasoning.strip():
            raise ValueError("reasoning must be a non-empty string")
        if not isinstance(self.source_paper_ids, (list, tuple)):
            raise TypeError("source_paper_ids must be a list of strings")
        if not isinstance(self.evidence_labels, (list, tuple)):
            raise TypeError("evidence_labels must be a list of strings")
        if not isinstance(self.supporting_sources, (list, tuple)):
            raise TypeError("supporting_sources must be a list of dicts")
        if not isinstance(self.contradicting_sources, (list, tuple)):
            raise TypeError("contradicting_sources must be a list of dicts")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a float between 0.0 and 1.0")
        if not (0.0 <= self.confidence <= 1.0) or not math.isfinite(self.confidence):
            raise ValueError("confidence must be a finite number between 0.0 and 1.0")
        if not isinstance(self.limitations, (list, tuple)):
            raise TypeError("limitations must be a list of strings")

        # Defensive copies
        object.__setattr__(self, "source_paper_ids", list(self.source_paper_ids))
        object.__setattr__(self, "evidence_labels", list(self.evidence_labels))
        object.__setattr__(self, "supporting_sources", copy.deepcopy(list(self.supporting_sources)))
        object.__setattr__(self, "contradicting_sources", copy.deepcopy(list(self.contradicting_sources)))
        object.__setattr__(self, "limitations", list(self.limitations))

    @property
    def is_documented(self) -> bool:
        """True if the gap was explicitly stated by paper authors (limitation / future work)."""
        return self.category == GapCategory.DOCUMENTED.value

    @property
    def is_inferred(self) -> bool:
        """True if the gap was inferred across papers as a candidate hypothesis."""
        return self.category == GapCategory.INFERRED.value

    @property
    def has_contradictions(self) -> bool:
        """True if any paper in the context disputes, qualifies, or resolves this gap."""
        return len(self.contradicting_sources) > 0

    def to_dict(self) -> dict[str, Any]:
        """Defensive dictionary serialization."""
        return {
            "gap_id": self.gap_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "reasoning": self.reasoning,
            "source_paper_ids": list(self.source_paper_ids),
            "evidence_labels": list(self.evidence_labels),
            "supporting_sources": copy.deepcopy(self.supporting_sources),
            "contradicting_sources": copy.deepcopy(self.contradicting_sources),
            "confidence": self.confidence,
            "limitations": list(self.limitations),
            "is_documented": self.is_documented,
            "is_inferred": self.is_inferred,
            "has_contradictions": self.has_contradictions,
        }


@dataclass(frozen=True)
class ResearchGapReport:
    """Comprehensive report of documented and inferred research gaps."""

    paper_ids: list[str]
    collection_id: str | None
    documented_gaps: list[ResearchGap]
    inferred_gaps: list[ResearchGap]
    total_gaps_count: int
    summary: str
    limitations: list[str]
    model: str
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_ids, (list, tuple)):
            raise TypeError("paper_ids must be a list of strings")
        if self.collection_id is not None and not isinstance(self.collection_id, str):
            raise TypeError("collection_id must be a string or None")
        if not isinstance(self.documented_gaps, (list, tuple)):
            raise TypeError("documented_gaps must be a list of ResearchGap objects")
        if not isinstance(self.inferred_gaps, (list, tuple)):
            raise TypeError("inferred_gaps must be a list of ResearchGap objects")
        if isinstance(self.total_gaps_count, bool) or not isinstance(self.total_gaps_count, int):
            raise TypeError("total_gaps_count must be an integer")
        if self.total_gaps_count < 0:
            raise ValueError("total_gaps_count must be non-negative")
        if not isinstance(self.summary, str):
            raise TypeError("summary must be a string")
        if not isinstance(self.limitations, (list, tuple)):
            raise TypeError("limitations must be a list of strings")
        if not isinstance(self.model, str):
            raise TypeError("model must be a string")

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
        object.__setattr__(self, "documented_gaps", list(self.documented_gaps))
        object.__setattr__(self, "inferred_gaps", list(self.inferred_gaps))
        object.__setattr__(self, "limitations", list(self.limitations))

    @property
    def all_gaps(self) -> list[ResearchGap]:
        return self.documented_gaps + self.inferred_gaps

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary representation."""
        return {
            "paper_ids": list(self.paper_ids),
            "collection_id": self.collection_id,
            "documented_gaps": [g.to_dict() for g in self.documented_gaps],
            "inferred_gaps": [g.to_dict() for g in self.inferred_gaps],
            "total_gaps_count": self.total_gaps_count,
            "summary": self.summary,
            "limitations": list(self.limitations),
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
        }

    @classmethod
    def empty(
        cls,
        paper_ids: list[str],
        reason: str = "Insufficient evidence to identify research gaps.",
        collection_id: str | None = None,
    ) -> ResearchGapReport:
        """Factory for empty/insufficient evidence scenarios to prevent hallucinations."""
        return cls(
            paper_ids=list(paper_ids),
            collection_id=collection_id,
            documented_gaps=[],
            inferred_gaps=[],
            total_gaps_count=0,
            summary=reason,
            limitations=[
                "No research gaps could be verified due to empty or missing evidence chunks.",
                "Candidate gaps were not synthesized without grounded source material.",
            ],
            model="none",
            usage=None,
            raw=None,
        )
