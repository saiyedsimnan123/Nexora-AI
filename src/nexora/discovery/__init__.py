"""Scholarly discovery subsystem for Nexora AI.

Provides provider-agnostic abstractions and concrete scholarly repository clients
(e.g. arXiv) for searching external academic research and retrieving paper metadata.
"""

from __future__ import annotations

from nexora.discovery.arxiv import (
    ArxivRateLimitPolicy,
    ArxivSourceProvider,
    build_canonical_abstract_url,
    build_canonical_pdf_url,
    normalize_and_validate_arxiv_id,
)
from nexora.discovery.deduplication import (
    DeduplicationMatchReason,
    DeduplicationResult,
    check_paper_duplicate_in_collection,
    normalize_doi,
    normalize_external_id,
)
from nexora.discovery.downloader import (
    DownloadedPDF,
    SafePDFDownloader,
    validate_url_safety,
)
from nexora.discovery.errors import (
    DiscoveryConnectionError,
    DiscoveryError,
    DiscoveryRateLimitError,
    DiscoveryResponseError,
    DiscoveryTimeoutError,
    InvalidIdentifierError,
    InvalidPDFError,
    OversizedPDFError,
    PDFNotFoundError,
    PaperNotFoundError,
    SSRFProtectionError,
)
from nexora.discovery.models import (
    ScholarlyPaperMetadata,
    ScholarlySearchQuery,
    ScholarlySearchResult,
)
from nexora.discovery.provider import ScholarlySourceProvider

__all__ = [
    "ScholarlyPaperMetadata",
    "ScholarlySearchQuery",
    "ScholarlySearchResult",
    "ScholarlySourceProvider",
    "ArxivSourceProvider",
    "ArxivRateLimitPolicy",
    "DownloadedPDF",
    "SafePDFDownloader",
    "validate_url_safety",
    "DeduplicationMatchReason",
    "DeduplicationResult",
    "check_paper_duplicate_in_collection",
    "normalize_doi",
    "normalize_external_id",
    "normalize_and_validate_arxiv_id",
    "build_canonical_abstract_url",
    "build_canonical_pdf_url",
    "DiscoveryError",
    "InvalidIdentifierError",
    "DiscoveryRateLimitError",
    "DiscoveryTimeoutError",
    "DiscoveryConnectionError",
    "DiscoveryResponseError",
    "PaperNotFoundError",
    "SSRFProtectionError",
    "OversizedPDFError",
    "InvalidPDFError",
    "PDFNotFoundError",
]
