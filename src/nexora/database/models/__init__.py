"""Database ORM models.

Importing this package registers every model's table on ``Base.metadata``.
"""

from nexora.database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace

__all__ = [
    "Base",
    "Collection",
    "Document",
    "DocumentChunkRecord",
    "Paper",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "User",
    "Workspace",
]
