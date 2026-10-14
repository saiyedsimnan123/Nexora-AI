"""Research gap data models (Milestone 12C-26 & 12C-27).

Implements hardened evidence-backed research gap models:
- Distinguishes author-documented gaps from AI-inferred candidate gaps.
- Enforces strict immutability against shared reference mutation.
- Validates required source metadata (label, document_id, chunk_id, text).
- Enforces bounded confidence (documented confidence 1.0 requires verified sources; inferred strictly < 1.0).
- Preserves traceable chunk, document, and paper metadata.
- Records qualifying or contradicting evidence when present.
"""

from __future__ import annotations

import copy
import math
from enum import Enum
from typing import Any, Sequence

from nexora.llm.models import _check_no_secret_keys


class GapCategory(str, Enum):
    """Category of a research gap."""

    DOCUMENTED = "documented"
    INFERRED = "inferred"


VALID_GAP_CATEGORIES = frozenset({c.value for c in GapCategory})
_REQUIRED_SOURCE_KEYS = frozenset({"label", "document_id", "chunk_id", "text"})


def _validate_source_record(source: Any, field_name: str, index: int) -> dict[str, Any]:
    """Validate required metadata and ensure no secrets inside an evidence source."""
    if not isinstance(source, dict):
        raise TypeError(f"{field_name}[{index}] must be a dict, got {type(source).__name__}")
    _check_no_secret_keys(source)
    for key in _REQUIRED_SOURCE_KEYS:
        if key not in source:
            raise ValueError(f"{field_name}[{index}] is missing required key {key!r}")
        val = source[key]
        if key == "text":
            if not isinstance(val, str):
                raise TypeError(f"{field_name}[{index}]['text'] must be a string")
        else:
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{field_name}[{index}][{key!r}] must be a non-empty string")

    paper_id = source.get("paper_id")
    if paper_id is not None and not isinstance(paper_id, str):
        raise TypeError(f"{field_name}[{index}]['paper_id'] must be a string or None")

    return dict(source)


class ResearchGap:
    """An individual research gap with traceable evidence, reasoning, and strict immutability."""

    def __init__(
        self,
        gap_id: str,
        category: str,
        title: str,
        description: str,
        reasoning: str,
        source_paper_ids: Sequence[str],
        evidence_labels: Sequence[str],
        supporting_sources: Sequence[dict[str, Any]],
        contradicting_sources: Sequence[dict[str, Any]],
        confidence: float,
        limitations: Sequence[str],
    ) -> None:
        if not isinstance(gap_id, str) or not gap_id.strip():
            raise TypeError("gap_id must be a non-empty string")
        if not isinstance(category, str):
            raise TypeError("category must be a string")
        if category not in VALID_GAP_CATEGORIES:
            raise ValueError(
                f"category must be one of {VALID_GAP_CATEGORIES}, got {category!r}"
            )
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("reasoning must be a non-empty string")
        if not isinstance(source_paper_ids, (list, tuple)):
            raise TypeError("source_paper_ids must be a sequence of strings")
        for idx, pid in enumerate(source_paper_ids):
            if not isinstance(pid, str) or not pid.strip():
                raise TypeError(f"source_paper_ids[{idx}] must be a non-empty string")

        if not isinstance(evidence_labels, (list, tuple)):
            raise TypeError("evidence_labels must be a sequence of strings")
        for idx, lbl in enumerate(evidence_labels):
            if not isinstance(lbl, str) or not lbl.strip():
                raise TypeError(f"evidence_labels[{idx}] must be a non-empty string")

        if not isinstance(supporting_sources, (list, tuple)):
            raise TypeError("supporting_sources must be a sequence of dicts")
        validated_supporting = [
            _validate_source_record(s, "supporting_sources", idx)
            for idx, s in enumerate(supporting_sources)
        ]

        if not isinstance(contradicting_sources, (list, tuple)):
            raise TypeError("contradicting_sources must be a sequence of dicts")
        validated_contradicting = [
            _validate_source_record(s, "contradicting_sources", idx)
            for idx, s in enumerate(contradicting_sources)
        ]

        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("confidence must be a float between 0.0 and 1.0")
        if not (0.0 <= confidence <= 1.0) or not math.isfinite(confidence):
            raise ValueError("confidence must be a finite number between 0.0 and 1.0")

        # Provenance safety rule: Documented gaps with confidence 1.0 require verified supporting sources
        if category == GapCategory.DOCUMENTED.value:
            if confidence == 1.0 and len(validated_supporting) == 0:
                raise ValueError(
                    "Documented gap with confidence 1.0 must have at least one supporting source chunk."
                )

        # Provenance safety rule: Inferred candidate gaps are hypotheses and cannot have confidence 1.0
        if category == GapCategory.INFERRED.value:
            if confidence >= 1.0:
                raise ValueError(
                    "Inferred candidate gaps cannot have confidence 1.0; candidate hypotheses must be strictly < 1.0."
                )

        if not isinstance(limitations, (list, tuple)):
            raise TypeError("limitations must be a sequence of strings")
        for idx, lim in enumerate(limitations):
            if not isinstance(lim, str):
                raise TypeError(f"limitations[{idx}] must be a string")

        # Freeze internal copies
        self._gap_id = gap_id
        self._category = category
        self._title = title
        self._description = description
        self._reasoning = reasoning
        self._source_paper_ids = tuple(source_paper_ids)
        self._evidence_labels = tuple(evidence_labels)
        self._supporting_sources = tuple(copy.deepcopy(s) for s in validated_supporting)
        self._contradicting_sources = tuple(copy.deepcopy(s) for s in validated_contradicting)
        self._confidence = float(confidence)
        self._limitations = tuple(limitations)
        self._initialized = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError(f"Cannot modify immutable {self.__class__.__name__} instance attribute {name!r}")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Cannot delete attribute {name!r} from immutable {self.__class__.__name__}")

    @property
    def gap_id(self) -> str:
        return self._gap_id

    @property
    def category(self) -> str:
        return self._category

    @property
    def title(self) -> str:
        return self._title

    @property
    def description(self) -> str:
        return self._description

    @property
    def reasoning(self) -> str:
        return self._reasoning

    @property
    def source_paper_ids(self) -> list[str]:
        return list(self._source_paper_ids)

    @property
    def evidence_labels(self) -> list[str]:
        return list(self._evidence_labels)

    @property
    def supporting_sources(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self._supporting_sources))

    @property
    def contradicting_sources(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self._contradicting_sources))

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def limitations(self) -> list[str]:
        return list(self._limitations)

    @property
    def is_documented(self) -> bool:
        """True if the gap was explicitly stated by paper authors (limitation / future work)."""
        return self._category == GapCategory.DOCUMENTED.value

    @property
    def is_inferred(self) -> bool:
        """True if the gap was inferred across papers as a candidate hypothesis."""
        return self._category == GapCategory.INFERRED.value

    @property
    def has_contradictions(self) -> bool:
        """True if any paper in the context disputes, qualifies, or resolves this gap."""
        return len(self._contradicting_sources) > 0

    def to_dict(self) -> dict[str, Any]:
        """Defensive dictionary serialization."""
        return {
            "gap_id": self._gap_id,
            "category": self._category,
            "title": self._title,
            "description": self._description,
            "reasoning": self._reasoning,
            "source_paper_ids": list(self._source_paper_ids),
            "evidence_labels": list(self._evidence_labels),
            "supporting_sources": copy.deepcopy(list(self._supporting_sources)),
            "contradicting_sources": copy.deepcopy(list(self._contradicting_sources)),
            "confidence": self._confidence,
            "limitations": list(self._limitations),
            "is_documented": self.is_documented,
            "is_inferred": self.is_inferred,
            "has_contradictions": self.has_contradictions,
        }

    def __repr__(self) -> str:
        return (
            f"ResearchGap(gap_id={self._gap_id!r}, category={self._category!r}, "
            f"title={self._title!r}, confidence={self._confidence:.2f})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ResearchGap):
            return False
        return (
            self._gap_id == other._gap_id
            and self._category == other._category
            and self._title == other._title
            and self._description == other._description
            and self._reasoning == other._reasoning
            and self._source_paper_ids == other._source_paper_ids
            and self._evidence_labels == other._evidence_labels
            and self._supporting_sources == other._supporting_sources
            and self._contradicting_sources == other._contradicting_sources
            and math.isclose(self._confidence, other._confidence)
            and self._limitations == other._limitations
        )


class ResearchGapReport:
    """Comprehensive report of documented and inferred research gaps with strict immutability."""

    def __init__(
        self,
        paper_ids: Sequence[str],
        collection_id: str | None,
        documented_gaps: Sequence[ResearchGap],
        inferred_gaps: Sequence[ResearchGap],
        total_gaps_count: int,
        summary: str,
        limitations: Sequence[str],
        model: str,
        usage: dict[str, int] | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(paper_ids, (list, tuple)):
            raise TypeError("paper_ids must be a sequence of strings")
        for idx, pid in enumerate(paper_ids):
            if not isinstance(pid, str) or not pid.strip():
                raise TypeError(f"paper_ids[{idx}] must be a non-empty string")

        if collection_id is not None and not isinstance(collection_id, str):
            raise TypeError("collection_id must be a string or None")

        if not isinstance(documented_gaps, (list, tuple)):
            raise TypeError("documented_gaps must be a sequence of ResearchGap objects")
        for idx, g in enumerate(documented_gaps):
            if not isinstance(g, ResearchGap):
                raise TypeError(f"documented_gaps[{idx}] must be a ResearchGap")

        if not isinstance(inferred_gaps, (list, tuple)):
            raise TypeError("inferred_gaps must be a sequence of ResearchGap objects")
        for idx, g in enumerate(inferred_gaps):
            if not isinstance(g, ResearchGap):
                raise TypeError(f"inferred_gaps[{idx}] must be a ResearchGap")

        if isinstance(total_gaps_count, bool) or not isinstance(total_gaps_count, int):
            raise TypeError("total_gaps_count must be an integer")
        if total_gaps_count < 0:
            raise ValueError("total_gaps_count must be non-negative")

        if not isinstance(summary, str):
            raise TypeError("summary must be a string")
        if not isinstance(limitations, (list, tuple)):
            raise TypeError("limitations must be a sequence of strings")
        for idx, lim in enumerate(limitations):
            if not isinstance(lim, str):
                raise TypeError(f"limitations[{idx}] must be a string")

        if not isinstance(model, str):
            raise TypeError("model must be a string")

        if usage is not None:
            if not isinstance(usage, dict):
                raise TypeError("usage must be a dict or None")
            _check_no_secret_keys(usage)
            self._usage: dict[str, int] | None = dict(usage)
        else:
            self._usage = None

        if raw is not None:
            if not isinstance(raw, dict):
                raise TypeError("raw must be a dict or None")
            _check_no_secret_keys(raw)
            self._raw: dict[str, Any] | None = copy.deepcopy(raw)
        else:
            self._raw = None

        self._paper_ids = tuple(paper_ids)
        self._collection_id = collection_id
        self._documented_gaps = tuple(documented_gaps)
        self._inferred_gaps = tuple(inferred_gaps)
        self._total_gaps_count = total_gaps_count
        self._summary = summary
        self._limitations = tuple(limitations)
        self._model = model
        self._initialized = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError(f"Cannot modify immutable {self.__class__.__name__} instance attribute {name!r}")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Cannot delete attribute {name!r} from immutable {self.__class__.__name__}")

    @property
    def paper_ids(self) -> list[str]:
        return list(self._paper_ids)

    @property
    def collection_id(self) -> str | None:
        return self._collection_id

    @property
    def documented_gaps(self) -> list[ResearchGap]:
        return list(self._documented_gaps)

    @property
    def inferred_gaps(self) -> list[ResearchGap]:
        return list(self._inferred_gaps)

    @property
    def total_gaps_count(self) -> int:
        return self._total_gaps_count

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def limitations(self) -> list[str]:
        return list(self._limitations)

    @property
    def model(self) -> str:
        return self._model

    @property
    def usage(self) -> dict[str, int] | None:
        return None if self._usage is None else dict(self._usage)

    @property
    def raw(self) -> dict[str, Any] | None:
        return None if self._raw is None else copy.deepcopy(self._raw)

    @property
    def all_gaps(self) -> list[ResearchGap]:
        return list(self._documented_gaps) + list(self._inferred_gaps)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary representation."""
        return {
            "paper_ids": list(self._paper_ids),
            "collection_id": self._collection_id,
            "documented_gaps": [g.to_dict() for g in self._documented_gaps],
            "inferred_gaps": [g.to_dict() for g in self._inferred_gaps],
            "total_gaps_count": self._total_gaps_count,
            "summary": self._summary,
            "limitations": list(self._limitations),
            "model": self._model,
            "usage": None if self._usage is None else dict(self._usage),
        }

    @classmethod
    def empty(
        cls,
        paper_ids: Sequence[str],
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

    def __repr__(self) -> str:
        return (
            f"ResearchGapReport(papers={len(self._paper_ids)}, "
            f"documented={len(self._documented_gaps)}, inferred={len(self._inferred_gaps)}, "
            f"total={self._total_gaps_count})"
        )
