"""Expected application-level errors, ready for HTTP exception handlers."""

from __future__ import annotations


class InvalidRequestError(ValueError):
    """The caller's request is malformed."""


class InvalidConfigurationError(RuntimeError):
    """The application service was wired with unusable dependencies."""


class ResearchServiceError(RuntimeError):
    """The research pipeline failed; the original error is chained as __cause__."""


class FileTooLargeError(InvalidRequestError):
    """The uploaded file exceeds the configured maximum upload size."""


class DocumentNotFoundError(LookupError):
    """The requested paper or document was not found."""


class DocumentIngestionApiError(RuntimeError):
    """Document ingestion failed; the original error is chained as __cause__."""
