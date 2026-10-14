"""Nexora research answer and analysis layer."""

from nexora.research.analysis import (
    ANALYSIS_DIMENSIONS,
    AnalysisField,
    StructuredPaperAnalysis,
)
from nexora.research.analysis_service import (
    PaperAnalysisService,
    PaperComparisonService,
)
from nexora.research.comparison import FieldComparison, PaperComparison
from nexora.research.gap import (
    GapCategory,
    ResearchGap,
    ResearchGapReport,
)
from nexora.research.gap_prompt import ResearchGapPromptBuilder
from nexora.research.gap_service import ResearchGapService
from nexora.research.models import ResearchAnswer
from nexora.research.prompt import (
    PaperAnalysisPromptBuilder,
    PaperComparisonPromptBuilder,
)
from nexora.research.query import ResearchQueryService
from nexora.research.service import ResearchAnswerService
from nexora.research.verification import (
    AnswerVerificationResult,
    ClaimToEvidenceVerifier,
    ClaimVerification,
    ClaimVerificationPromptBuilder,
    ClaimVerificationStatus,
    RawClaim,
    extract_claims_from_text,
)

__all__ = [
    "ANALYSIS_DIMENSIONS",
    "AnalysisField",
    "AnswerVerificationResult",
    "ClaimToEvidenceVerifier",
    "ClaimVerification",
    "ClaimVerificationPromptBuilder",
    "ClaimVerificationStatus",
    "FieldComparison",
    "GapCategory",
    "PaperAnalysisPromptBuilder",
    "PaperAnalysisService",
    "PaperComparisonPromptBuilder",
    "PaperComparisonService",
    "PaperComparison",
    "RawClaim",
    "ResearchAnswer",
    "ResearchAnswerService",
    "ResearchGap",
    "ResearchGapPromptBuilder",
    "ResearchGapReport",
    "ResearchGapService",
    "ResearchQueryService",
    "StructuredPaperAnalysis",
    "extract_claims_from_text",
]


