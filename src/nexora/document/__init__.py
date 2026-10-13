"""Document intelligence layer for Nexora AI.

This package contains the document ingestion, processing, and persistence
functionality for research documents.
"""

from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.service import DocumentPersistenceService

__all__ = [
    "DocumentChunk",
    "DocumentPersistenceService",
    "ProcessedDocument",
]
