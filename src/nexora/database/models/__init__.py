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
from nexora.database.models.api_key import ApiKey
from nexora.database.models.author import Author
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document, IndexingStatus
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.paper import Paper
from nexora.database.models.paper_author import PaperAuthor
from nexora.database.models.paper_reference import PaperReference
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace

__all__ = [
    "ApiKey",
    "NAMING_CONVENTION",
    "Author",
    "Base",
    "Collection",
    "Document",
    "DocumentChunkRecord",
    "IndexingStatus",
    "Paper",
    "PaperAuthor",
    "PaperReference",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "User",
    "Workspace",
    "utc_now",
]
