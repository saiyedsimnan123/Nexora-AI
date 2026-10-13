"""Document intelligence layer for Nexora AI.

This package contains the document ingestion, processing, persistence,
and vector indexing functionality for research documents.
"""

from nexora.document.ingestion import (
    DocumentIngestionService,
    IngestionError,
    IngestionResult,
    VectorIndexingError,
)
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.service import DocumentPersistenceService

__all__ = [
    "DocumentChunk",
    "DocumentIngestionService",
    "DocumentPersistenceService",
    "IngestionError",
    "IngestionResult",
    "ProcessedDocument",
    "VectorIndexingError",
]
