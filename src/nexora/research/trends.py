"""Data models for temporal research trend analysis across paper collections."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrendTrajectory(str, Enum):
    """Classification of how a method, architecture, dataset, or metric evolves over time."""

    EMERGING = "emerging"
    ESTABLISHED = "established"
    DECLINING = "declining"
    INSUFFICIENT_DATA = "insufficient_data"


class TrendCategory(str, Enum):
    """Standardized categories for research trend items."""

    METHOD = "method"
    ARCHITECTURE = "architecture"
    DATASET = "dataset"
    METRIC = "metric"


VALID_TRAJECTORIES = {t.value for t in TrendTrajectory}
VALID_CATEGORIES = {c.value for c in TrendCategory}


@dataclass(frozen=True)
class TrendCitation:
    """Citation linking a trend finding to an isolated paper chunk."""

    paper_id: str
    chunk_id: str | None
    evidence_label: str
    text_excerpt: str | None = None
    period: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "chunk_id": self.chunk_id,
            "evidence_label": self.evidence_label,
            "text_excerpt": self.text_excerpt,
            "period": self.period,
        }


@dataclass
class TrendItem:
    """A research concept with historical occurrence tracking and trajectory classification."""

    name: str
    category: str
    trajectory: str
    trajectory_reasoning: str
    occurrences_by_period: dict[str, int] = field(default_factory=dict)
    first_seen_period: str | None = None
    last_seen_period: str | None = None
    total_occurrences: int = 0
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence_labels: list[str] = field(default_factory=list)
    summary: str = ""

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            self.category = TrendCategory.METHOD.value
        if self.trajectory not in VALID_TRAJECTORIES:
            self.trajectory = TrendTrajectory.INSUFFICIENT_DATA.value
        if not self.total_occurrences and self.occurrences_by_period:
            self.total_occurrences = sum(self.occurrences_by_period.values())
        if self.occurrences_by_period:
            sorted_periods = sorted(self.occurrences_by_period.keys())
            if not self.first_seen_period:
                self.first_seen_period = sorted_periods[0]
            if not self.last_seen_period:
                self.last_seen_period = sorted_periods[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "trajectory": self.trajectory,
            "trajectory_reasoning": self.trajectory_reasoning,
            "occurrences_by_period": copy.deepcopy(self.occurrences_by_period),
            "first_seen_period": self.first_seen_period,
            "last_seen_period": self.last_seen_period,
            "total_occurrences": self.total_occurrences,
            "citations": copy.deepcopy(self.citations),
            "evidence_labels": list(self.evidence_labels),
            "summary": self.summary,
        }


@dataclass
class TemporalBucket:
    """Aggregation of papers and research findings within a temporal period."""

    period: str
    paper_count: int
    paper_ids: list[str] = field(default_factory=list)
    paper_titles: dict[str, str] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "paper_count": self.paper_count,
            "paper_ids": list(self.paper_ids),
            "paper_titles": copy.deepcopy(self.paper_titles),
            "highlights": list(self.highlights),
        }


@dataclass
class ResearchTrendsReport:
    """Comprehensive temporal research trend report across a collection."""

    collection_id: str
    paper_ids: list[str]
    time_unit: str
    temporal_buckets: list[TemporalBucket] = field(default_factory=list)
    trends: list[TrendItem] = field(default_factory=list)
    missing_date_paper_ids: list[str] = field(default_factory=list)
    summary: str = ""
    limitations: list[str] = field(default_factory=list)
    model: str = "none"
    usage: dict[str, int] | None = None
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "paper_ids": list(self.paper_ids),
            "time_unit": self.time_unit,
            "temporal_buckets": [b.to_dict() for b in self.temporal_buckets],
            "trends": [t.to_dict() for t in self.trends],
            "missing_date_paper_ids": list(self.missing_date_paper_ids),
            "summary": self.summary,
            "limitations": list(self.limitations),
            "model": self.model,
            "usage": copy.deepcopy(self.usage) if self.usage else None,
        }
