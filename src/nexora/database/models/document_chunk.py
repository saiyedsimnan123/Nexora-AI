"""DocumentChunkRecord ORM model.

The PostgreSQL record of one processed text chunk of a ``Document``. It
stores chunk text and position only. Embedding vectors are deliberately not
stored here: the vector store is their canonical home.

This module contains ORM definitions and model-level validation only: it
creates no engine or session, opens no connection, reads no files and
performs no network access.

The plain ``nexora.document.models.DocumentChunk`` dataclass is a different
class (an in-memory processing result); this record is its persisted
metadata counterpart, hence the distinct name.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from nexora.database.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from nexora.database.models.document import Document

CHUNK_TEXT_MAX_LENGTH = 100_000


class DocumentChunkRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One chunk of a document, identified by its zero-based position.

    ``(document_id, chunk_index)`` is unique, so a document cannot hold two
    chunks at the same position. Chunk text is stored as given.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_index", name="document_id_chunk_index"
        ),
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        CheckConstraint("length(trim(text)) > 0", name="text_not_blank"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(String(CHUNK_TEXT_MAX_LENGTH), nullable=False)

    # Deletion is owned by the database (``ondelete="CASCADE"``).
    document: Mapped[Document] = relationship(back_populates="chunks")

    @validates("chunk_index")
    def _validate_chunk_index(self, key: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"chunk_index must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError("chunk_index must be non-negative")
        return value

    @validates("text")
    def _validate_text(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"text must be a str, got {type(value).__name__}")
        if not value.strip():
            raise ValueError("text must not be blank")
        if len(value) > CHUNK_TEXT_MAX_LENGTH:
            raise ValueError(f"text must be at most {CHUNK_TEXT_MAX_LENGTH} characters")
        return value

    def __repr__(self) -> str:
        return f"DocumentChunkRecord(id={self.id!r})"
