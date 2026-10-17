"""Repository exports."""

from nexora.database.repository.author import AuthorRepository
from nexora.database.repository.base import (
    BaseRepository,
    ConstraintError,
    DuplicateRecordError,
    RecordNotFoundError,
    RepositoryError,
)
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.database.repository.paper_author import PaperAuthorRepository
from nexora.database.repository.paper_reference import PaperReferenceRepository
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository

__all__ = [
    "AuthorRepository",
    "BaseRepository",
    "CollectionRepository",
    "ConstraintError",
    "DocumentChunkRepository",
    "DocumentRepository",
    "DuplicateRecordError",
    "PaperAuthorRepository",
    "PaperReferenceRepository",
    "PaperRepository",
    "RecordNotFoundError",
    "RepositoryError",
    "UserRepository",
    "WorkspaceRepository",
]
