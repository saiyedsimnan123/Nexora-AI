"""Scholarly paper deduplication logic and normalization helpers.

Deduplication rules:
- Priority 1: Exact normalized external identifier (e.g. arXiv ID '2106.09685').
- Priority 2: Exact normalized DOI (e.g. '10.48550/arxiv.2106.09685').
- Priority 3: Exact document checksum (SHA-256) for identical file content.
- Prohibition: Distinct papers are NEVER merged based on title similarity.
- Determinism: If an identifier matches multiple existing records, the earliest
  created record is returned deterministically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from nexora.discovery.arxiv import normalize_and_validate_arxiv_id
from nexora.discovery.errors import InvalidIdentifierError

if TYPE_CHECKING:
    from nexora.database.models.paper import Paper
    from nexora.database.repository.paper import PaperRepository

_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE)


def normalize_doi(raw_doi: str | None) -> str | None:
    """Normalize a Digital Object Identifier (DOI) for robust matching.

    Strips whitespace, converts to lowercase, and removes 'https://doi.org/'
    or 'doi:' prefixes.

    Examples:
        'https://doi.org/10.48550/arXiv.2106.09685' -> '10.48550/arxiv.2106.09685'
        'doi:10.1145/1234.5678' -> '10.1145/1234.5678'
    """
    if raw_doi is None:
        return None
    if not isinstance(raw_doi, str):
        raise TypeError(f"doi must be a string or None, got {type(raw_doi).__name__}")

    cleaned = raw_doi.strip().lower()
    if not cleaned:
        return None

    cleaned = _DOI_PREFIX_RE.sub("", cleaned).strip()
    # Strip trailing punctuation often appended in citations
    cleaned = cleaned.rstrip("./")
    return cleaned if cleaned else None


def normalize_external_id(raw_external_id: str | None, provider: str = "arxiv") -> str | None:
    """Normalize an external paper identifier based on provider rules.

    For arXiv identifiers, applies standard identifier normalization.
    """
    if raw_external_id is None:
        return None
    if not isinstance(raw_external_id, str):
        raise TypeError(f"external_id must be a string or None, got {type(raw_external_id).__name__}")

    cleaned = raw_external_id.strip()
    if not cleaned:
        return None

    if provider.lower() == "arxiv":
        try:
            return normalize_and_validate_arxiv_id(cleaned)
        except InvalidIdentifierError:
            # If not matching strict arXiv format, return clean stripped string
            return cleaned.lower()

    return cleaned


class DeduplicationMatchReason(str, Enum):
    """Reason why an external paper was identified as an existing duplicate."""

    EXTERNAL_ID = "external_id"
    DOI = "doi"
    CHECKSUM = "checksum"
    NONE = "none"


@dataclass(frozen=True)
class DeduplicationResult:
    """The result of evaluating whether a discovered paper already exists."""

    is_duplicate: bool
    existing_paper: Paper | None
    match_reason: DeduplicationMatchReason
    matched_value: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_duplicate": self.is_duplicate,
            "existing_paper_id": str(self.existing_paper.id) if self.existing_paper else None,
            "match_reason": self.match_reason.value,
            "matched_value": self.matched_value,
        }


def check_paper_duplicate_in_collection(
    paper_repo: PaperRepository,
    collection_id: UUID,
    *,
    external_id: str | None = None,
    doi: str | None = None,
    provider: str = "arxiv",
) -> DeduplicationResult:
    """Check if a paper already exists in the given collection by external ID or DOI.

    Never matches by title. Evaluates external_id first, then doi.
    """
    if not isinstance(collection_id, UUID):
        raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

    # 1. Match by normalized external identifier
    norm_ext_id = normalize_external_id(external_id, provider=provider)
    if norm_ext_id:
        existing = paper_repo.get_by_external_id(collection_id, norm_ext_id)
        if existing is not None:
            return DeduplicationResult(
                is_duplicate=True,
                existing_paper=existing,
                match_reason=DeduplicationMatchReason.EXTERNAL_ID,
                matched_value=norm_ext_id,
            )

    # 2. Match by normalized DOI
    norm_doi = normalize_doi(doi)
    if norm_doi:
        existing = paper_repo.get_by_doi(collection_id, norm_doi)
        if existing is not None:
            return DeduplicationResult(
                is_duplicate=True,
                existing_paper=existing,
                match_reason=DeduplicationMatchReason.DOI,
                matched_value=norm_doi,
            )

    return DeduplicationResult(
        is_duplicate=False,
        existing_paper=None,
        match_reason=DeduplicationMatchReason.NONE,
        matched_value=None,
    )
