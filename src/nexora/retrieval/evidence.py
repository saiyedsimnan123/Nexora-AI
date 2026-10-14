"""Research evidence context builder (step 11H).

Turns quality-filtered retrieval results into a traceable, immutable
EvidenceContext. No rewriting, truncation of chunks, or answer generation.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any


def _non_empty_str(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class EvidenceItem:
    """One retrieved chunk, with its position in the final context."""

    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, object]
    position: int

    def __post_init__(self) -> None:
        _non_empty_str("document_id", self.document_id)
        _non_empty_str("chunk_id", self.chunk_id)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be a number")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        if not all(isinstance(k, str) for k in self.metadata):
            raise TypeError("metadata keys must be strings")
        _non_negative_int("position", self.position)
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    @property
    def paper_id(self) -> str | None:
        """Paper ID from metadata, if present."""
        val = self.metadata.get("paper_id")
        return str(val) if val is not None else None

    @property
    def label(self) -> str:
        """Standard citation label, e.g. '[E1]'."""
        return f"[E{self.position + 1}]"


@dataclass(frozen=True)
class EvidenceContext:
    """Ordered evidence for a query, ready for a future RAG layer."""

    query: str
    items: list[EvidenceItem]
    total_candidates: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.items, (list, tuple)):
            raise TypeError("items must be a list")
        if not all(isinstance(i, EvidenceItem) for i in self.items):
            raise TypeError("items must contain only EvidenceItem objects")
        _non_negative_int("total_candidates", self.total_candidates)
        object.__setattr__(self, "items", list(self.items))

    def get_by_label(self, label: str) -> EvidenceItem | None:
        """Retrieve an evidence item by its citation label like '[E1]' or 'E1'."""
        if not isinstance(label, str):
            return None
        clean = label.strip().strip("[]").upper()
        if clean.startswith("E") and clean[1:].isdigit():
            idx = int(clean[1:]) - 1
            if 0 <= idx < len(self.items):
                return self.items[idx]
        return None


class EvidenceContextBuilder:
    """Builds an EvidenceContext from a RetrievalQualityService.

    ``max_total_characters`` caps the combined length of evidence text. Whole
    items are kept in order until the next one would exceed the cap; chunks are
    never cut. A cap of 0 therefore yields no items.
    """

    def __init__(
        self, retrieval_service: Any, *, max_total_characters: int | None = None
    ) -> None:
        if not callable(getattr(retrieval_service, "retrieve", None)):
            raise TypeError("retrieval_service must provide retrieve(query)")
        if max_total_characters is not None:
            _non_negative_int("max_total_characters", max_total_characters)
        self._retrieval = retrieval_service
        self._max_chars = max_total_characters

    def build(self, query: str) -> EvidenceContext:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        chunks = self._retrieval.retrieve(query).results
        items: list[EvidenceItem] = []
        used = 0
        for chunk in chunks:
            if self._max_chars is not None and used + len(chunk.text) > self._max_chars:
                break
            used += len(chunk.text)
            items.append(
                EvidenceItem(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=chunk.score,
                    metadata=chunk.metadata,
                    position=len(items),
                )
            )
        return EvidenceContext(query=query, items=items, total_candidates=len(chunks))


_CITATION_PATTERN = re.compile(r"\[E(\d+)\]")


@dataclass(frozen=True)
class CitationVerificationResult:
    """Quantitative result of verifying citations against supplied evidence.

    NOTE ON SEMANTICS:
    Citation validity and reference integrity verify that citation labels (e.g. `[E1]`)
    syntactically resolve to chunks present in the supplied EvidenceContext.
    This confirms reference integrity and traceability back to source documents,
    but does NOT prove semantic claim support, natural language entailment, or the
    underlying factual truth of generated claims.
    """

    cited_labels: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    unreferenced_labels: list[str]
    is_fully_grounded: bool
    citation_precision: float
    sources: list[dict[str, Any]]

    @property
    def has_valid_references(self) -> bool:
        """Alias for is_fully_grounded: confirms all cited labels resolve to supplied evidence."""
        return self.is_fully_grounded


def extract_citations(text: str) -> list[str]:
    """Return all unique [E#] citation labels found in text, in order of appearance."""
    if not isinstance(text, str):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for match in _CITATION_PATTERN.finditer(text):
        label = match.group(0)
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def verify_citations(text: str, context: EvidenceContext) -> CitationVerificationResult:
    """Verify citations in text against the supplied EvidenceContext.

    Checks syntactic reference resolution against supplied evidence items.
    Returns structured metrics and traceable chunk source metadata.

    NOTE: Confirms reference-level validity only; does not perform semantic claim verification.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(context, EvidenceContext):
        raise TypeError("context must be an EvidenceContext")

    cited = extract_citations(text)
    num_items = len(context.items)

    valid: list[str] = []
    invalid: list[str] = []
    sources: list[dict[str, Any]] = []

    for label in cited:
        match = _CITATION_PATTERN.match(label)
        if match:
            idx = int(match.group(1))
            if 1 <= idx <= num_items:
                valid.append(label)
                item = context.items[idx - 1]
                sources.append({
                    "label": label,
                    "document_id": item.document_id,
                    "chunk_id": item.chunk_id,
                    "paper_id": item.paper_id,
                    "filename": item.metadata.get("filename"),
                    "score": item.score,
                    "text": item.text,
                    "position": item.position,
                })
            else:
                invalid.append(label)
        else:
            invalid.append(label)

    valid_set = set(valid)
    all_supplied = [f"[E{i + 1}]" for i in range(num_items)]
    unreferenced = [l for l in all_supplied if l not in valid_set]

    total_cited = len(cited)
    precision = (len(valid) / total_cited) if total_cited > 0 else (1.0 if num_items == 0 else 0.0)
    is_grounded = len(invalid) == 0 and (total_cited > 0 or num_items == 0)

    return CitationVerificationResult(
        cited_labels=cited,
        valid_citations=valid,
        invalid_citations=invalid,
        unreferenced_labels=unreferenced,
        is_fully_grounded=is_grounded,
        citation_precision=precision,
        sources=sources,
    )


def filter_unsupplied_citations(text: str, max_evidence_count: int) -> str:
    """Strip any [E#] citation where # > max_evidence_count or # < 1.

    Ensures that generated answers cite ONLY evidence supplied to the step.
    """
    if not isinstance(text, str):
        return text

    def _replace(match: re.Match) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= max_evidence_count:
            return match.group(0)
        return ""

    filtered = _CITATION_PATTERN.sub(_replace, text)
    filtered = re.sub(r" {2,}", " ", filtered)
    filtered = re.sub(r" ([,.;:!?])", r"\1", filtered)
    return filtered
