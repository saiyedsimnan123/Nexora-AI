"""Typed errors for external scholarly paper discovery."""

from __future__ import annotations


class DiscoveryError(Exception):
    """Base exception for scholarly discovery and external metadata retrieval."""


class InvalidIdentifierError(DiscoveryError, ValueError):
    """Raised when an external paper identifier (e.g. arXiv ID) is malformed or invalid."""


class DiscoveryRateLimitError(DiscoveryError):
    """Raised when an external discovery provider throttles requests (e.g. HTTP 429)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class DiscoveryTimeoutError(DiscoveryError):
    """Raised when an external discovery provider request times out."""


class DiscoveryConnectionError(DiscoveryError):
    """Raised when an external discovery provider cannot be reached (e.g. DNS or connection dropped)."""


class DiscoveryResponseError(DiscoveryError):
    """Raised when an external discovery provider returns HTTP 5xx or unparseable content."""


class PaperNotFoundError(DiscoveryError, LookupError):
    """Raised when a specific paper identifier is not found on the external provider."""


class SSRFProtectionError(DiscoveryError, PermissionError):
    """Raised when a requested URL or redirect target resolves to a private, loopback, or forbidden address."""


class OversizedPDFError(DiscoveryError, ValueError):
    """Raised when a remote PDF exceeds the configured maximum download size."""


class InvalidPDFError(DiscoveryError, ValueError):
    """Raised when downloaded content is not a valid PDF (e.g. HTML, empty, or missing '%PDF-' signature)."""


class PDFNotFoundError(DiscoveryError, LookupError):
    """Raised when a remote PDF endpoint returns HTTP 404 Not Found."""
