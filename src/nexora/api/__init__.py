"""Nexora API boundary: request/response contracts and application service."""

from nexora.api.errors import (
    AnalysisServiceError,
    ComparisonServiceError,
    DocumentIngestionApiError,
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidConfigurationError,
    InvalidRequestError,
    ResearchServiceError,
    VerificationServiceError,
)
from nexora.api.models import (
    DocumentIngestResponse,
    PaperAnalysisRequest,
    PaperAnalysisResponse,
    PaperComparisonRequest,
    PaperComparisonResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchVerifyRequest,
    ResearchVerifyResponse,
)
from nexora.api.service import ResearchApplicationService

__all__ = [
    "AnalysisServiceError",
    "ComparisonServiceError",
    "DocumentIngestResponse",
    "DocumentIngestionApiError",
    "DocumentNotFoundError",
    "FileTooLargeError",
    "InvalidConfigurationError",
    "InvalidRequestError",
    "PaperAnalysisRequest",
    "PaperAnalysisResponse",
    "PaperComparisonRequest",
    "PaperComparisonResponse",
    "ResearchApplicationService",
    "ResearchRequest",
    "ResearchResponse",
    "ResearchServiceError",
    "ResearchVerifyRequest",
    "ResearchVerifyResponse",
    "VerificationServiceError",
]

