"""Data models for evidence-grounded literature review synthesis across paper collections."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SynthesisFinding:
    """A synthesized finding supported by evidence across one or more papers."""

    claim: str
    status: str
    evidence_labels: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    supporting_paper_ids: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status,
            "evidence_labels": list(self.evidence_labels),
            "sources": copy.deepcopy(self.sources),
            "supporting_paper_ids": list(self.supporting_paper_ids),
            "summary": self.summary,
        }


@dataclass
class SynthesisDivergence:
    """A methodological, empirical, or conceptual disagreement between research papers."""

    topic: str
    description: str
    perspectives: dict[str, str] = field(default_factory=dict)
    evidence_labels: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "description": self.description,
            "perspectives": copy.deepcopy(self.perspectives),
            "evidence_labels": list(self.evidence_labels),
            "sources": copy.deepcopy(self.sources),
        }


@dataclass
class SynthesisLimitation:
    """A recurring limitation documented across multiple papers."""

    description: str
    affected_paper_ids: list[str] = field(default_factory=list)
    evidence_labels: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "affected_paper_ids": list(self.affected_paper_ids),
            "evidence_labels": list(self.evidence_labels),
            "sources": copy.deepcopy(self.sources),
        }


@dataclass
class LiteratureSynthesisReport:
    """Structured literature review synthesis across a corpus of research papers."""

    collection_id: str
    paper_ids: list[str]
    topic: str | None = None
    executive_summary: str = ""
    common_findings: list[SynthesisFinding] = field(default_factory=list)
    disagreements: list[SynthesisDivergence] = field(default_factory=list)
    primary_methods: list[str] = field(default_factory=list)
    recurring_limitations: list[SynthesisLimitation] = field(default_factory=list)
    open_research_gaps: list[str] = field(default_factory=list)
    evidence_count: int = 0
    is_grounded: bool = True
    limitations: list[str] = field(default_factory=list)
    model: str = "none"
    usage: dict[str, int] | None = None
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "paper_ids": list(self.paper_ids),
            "topic": self.topic,
            "executive_summary": self.executive_summary,
            "common_findings": [f.to_dict() for f in self.common_findings],
            "disagreements": [d.to_dict() for d in self.disagreements],
            "primary_methods": list(self.primary_methods),
            "recurring_limitations": [l.to_dict() for l in self.recurring_limitations],
            "open_research_gaps": list(self.open_research_gaps),
            "evidence_count": self.evidence_count,
            "is_grounded": self.is_grounded,
            "limitations": list(self.limitations),
            "model": self.model,
            "usage": copy.deepcopy(self.usage) if self.usage else None,
        }
