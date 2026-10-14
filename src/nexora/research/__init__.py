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
from nexora.research.models import ResearchAnswer
from nexora.research.prompt import (
    PaperAnalysisPromptBuilder,
    PaperComparisonPromptBuilder,
)
from nexora.research.query import ResearchQueryService
from nexora.research.service import ResearchAnswerService

__all__ = [
    "ANALYSIS_DIMENSIONS",
    "AnalysisField",
    "FieldComparison",
    "PaperAnalysisPromptBuilder",
    "PaperAnalysisService",
    "PaperComparisonPromptBuilder",
    "PaperComparisonService",
    "PaperComparison",
    "ResearchAnswer",
    "ResearchAnswerService",
    "ResearchQueryService",
    "StructuredPaperAnalysis",
]
