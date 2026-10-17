"""Data models for external scholarly paper discovery and search."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class ScholarlyPaperMetadata:
    """Normalized metadata for a paper discovered from an external scholarly source.

    All fields originating from external sources are treated as untrusted input
    and cleaned upon construction. Canonical URLs are constructed from verified
    identifiers rather than accepting arbitrary external links.
    """

    external_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    publication_date: datetime.date | None = None
    doi: str | None = None
    categories: tuple[str, ...] = field(default_factory=tuple)
    primary_category: str | None = None
    abstract_url: str | None = None
    pdf_url: str | None = None
    license: str | None = None
    source: str = "arxiv"
    comment: str | None = None
    journal_ref: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.external_id, str) or not self.external_id.strip():
            raise TypeError("external_id must be a non-empty string")
        if not isinstance(self.title, str) or not self.title.strip():
            raise TypeError("title must be a non-empty string")
        if not isinstance(self.abstract, str):
            raise TypeError("abstract must be a string")
        if not isinstance(self.authors, (list, tuple)):
            raise TypeError("authors must be a sequence of strings")
        for idx, author in enumerate(self.authors):
            if not isinstance(author, str) or not author.strip():
                raise TypeError(f"authors[{idx}] must be a non-empty string")
        if not isinstance(self.categories, (list, tuple)):
            raise TypeError("categories must be a sequence of strings")
        if self.publication_date is not None and not isinstance(
            self.publication_date, datetime.date
        ):
            raise TypeError("publication_date must be a datetime.date or None")
        if self.doi is not None and (not isinstance(self.doi, str) or not self.doi.strip()):
            raise TypeError("doi must be a non-empty string or None")
        if self.primary_category is not None and (
            not isinstance(self.primary_category, str) or not self.primary_category.strip()
        ):
            raise TypeError("primary_category must be a non-empty string or None")
        if self.abstract_url is not None and not isinstance(self.abstract_url, str):
            raise TypeError("abstract_url must be a string or None")
        if self.pdf_url is not None and not isinstance(self.pdf_url, str):
            raise TypeError("pdf_url must be a string or None")
        if not isinstance(self.source, str) or not self.source.strip():
            raise TypeError("source must be a non-empty string")
        if not isinstance(self.raw_metadata, dict):
            raise TypeError("raw_metadata must be a dictionary")

        object.__setattr__(self, "authors", tuple(str(a).strip() for a in self.authors))
        object.__setattr__(self, "categories", tuple(str(c).strip() for c in self.categories))
        object.__setattr__(self, "raw_metadata", dict(self.raw_metadata))

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a JSON-compatible dictionary."""
        return {
            "external_id": self.external_id,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "publication_date": (
                self.publication_date.isoformat() if self.publication_date else None
            ),
            "doi": self.doi,
            "categories": list(self.categories),
            "primary_category": self.primary_category,
            "abstract_url": self.abstract_url,
            "pdf_url": self.pdf_url,
            "license": self.license,
            "source": self.source,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
        }


SORT_BY_OPTIONS = ("relevance", "submitted_date", "last_updated")
SORT_ORDER_OPTIONS = ("descending", "ascending")


@dataclass(frozen=True)
class ScholarlySearchQuery:
    """Request query parameters for searching an external scholarly provider."""

    query: str
    categories: tuple[str, ...] = field(default_factory=tuple)
    authors: tuple[str, ...] = field(default_factory=tuple)
    offset: int = 0
    limit: int = 10
    sort_by: str = "relevance"
    sort_order: str = "descending"

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.categories, (list, tuple)):
            raise TypeError("categories must be a sequence of strings")
        if not isinstance(self.authors, (list, tuple)):
            raise TypeError("authors must be a sequence of strings")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit <= 0
            or self.limit > 100
        ):
            raise ValueError("limit must be an integer between 1 and 100")
        if self.sort_by not in SORT_BY_OPTIONS:
            raise ValueError(f"sort_by must be one of {SORT_BY_OPTIONS}, got {self.sort_by!r}")
        if self.sort_order not in SORT_ORDER_OPTIONS:
            raise ValueError(
                f"sort_order must be one of {SORT_ORDER_OPTIONS}, got {self.sort_order!r}"
            )

        # Allow empty query only if category or author filters are specified
        if not self.query.strip() and not self.categories and not self.authors:
            raise ValueError("query, categories, or authors must be specified")

        object.__setattr__(
            self, "categories", tuple(str(c).strip() for c in self.categories if str(c).strip())
        )
        object.__setattr__(
            self, "authors", tuple(str(a).strip() for a in self.authors if str(a).strip())
        )


@dataclass(frozen=True)
class ScholarlySearchResult:
    """Structured result of a scholarly metadata search query."""

    provider: str
    query: str
    total_results: int
    offset: int
    limit: int
    papers: tuple[ScholarlyPaperMetadata, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise TypeError("provider must be a non-empty string")
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if (
            isinstance(self.total_results, bool)
            or not isinstance(self.total_results, int)
            or self.total_results < 0
        ):
            raise ValueError("total_results must be a non-negative integer")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(self.papers, (list, tuple)):
            raise TypeError("papers must be a sequence of ScholarlyPaperMetadata")
        for idx, paper in enumerate(self.papers):
            if not isinstance(paper, ScholarlyPaperMetadata):
                raise TypeError(
                    f"papers[{idx}] must be a ScholarlyPaperMetadata, got {type(paper).__name__}"
                )

        object.__setattr__(self, "papers", tuple(self.papers))

    def to_dict(self) -> dict[str, Any]:
        """Serialize search result to a JSON-compatible dictionary."""
        return {
            "provider": self.provider,
            "query": self.query,
            "total_results": self.total_results,
            "offset": self.offset,
            "limit": self.limit,
            "papers": [p.to_dict() for p in self.papers],
        }
