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


class AnalysisServiceError(RuntimeError):
    """Structured paper analysis failed; the original error is chained as __cause__."""


class ComparisonServiceError(RuntimeError):
    """Multi-paper comparison failed; the original error is chained as __cause__."""


class VerificationServiceError(RuntimeError):
    """Answer grounding verification failed; the original error is chained as __cause__."""


class GapDiscoveryServiceError(RuntimeError):
    """Research gap discovery failed; the original error is chained as __cause__."""


class TrendServiceError(RuntimeError):
    """Research trend analysis or literature synthesis failed; the original error is chained as __cause__."""


class AuthenticationError(Exception):
    """Authentication credential is missing, malformed, invalid, expired, or revoked."""


class ForbiddenError(Exception):
    """Caller is authenticated but not authorized to access or modify the requested resource."""

