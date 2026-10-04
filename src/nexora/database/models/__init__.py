"""Database ORM models.

Importing this package registers every model's table on ``Base.metadata``.
"""

from nexora.database.models.base import (
    NAMING_CONVENTION,
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    utc_now,
)
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.paper import Paper
from nexora.database.models.paper_reference import PaperReference
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace

__all__ = [
    "NAMING_CONVENTION",
    "Base",
    "Collection",
    "Document",
    "DocumentChunkRecord",
    "Paper",
    "PaperReference",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "User",
    "Workspace",
    "utc_now",
]
