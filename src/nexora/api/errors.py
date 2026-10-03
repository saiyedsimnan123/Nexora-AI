"""Expected application-level errors, ready for a future HTTP exception handler.

Suggested mapping: InvalidRequestError -> 4xx, ResearchServiceError -> 5xx/502,
InvalidConfigurationError -> 5xx (a deployment/wiring problem, not the caller's).
"""

from __future__ import annotations


class InvalidRequestError(ValueError):
    """The caller's request is malformed."""


class InvalidConfigurationError(RuntimeError):
    """The application service was wired with unusable dependencies."""


class ResearchServiceError(RuntimeError):
    """The research pipeline failed; the original error is chained as __cause__."""
