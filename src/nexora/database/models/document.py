"""Document ORM model.

A ``Document`` is a stored file (for example a PDF) that belongs to one
``Paper``. This module contains ORM definitions and model-level validation
only: it creates no engine or session, opens no connection, reads no files
and performs no network access. The file itself lives in external storage
and is referenced by ``storage_key``.
"""

from __future__ import annotations

import datetime
from enum import Enum
import re
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from nexora.database.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from nexora.database.models.document_chunk import DocumentChunkRecord
    from nexora.database.models.paper import Paper

FILENAME_MAX_LENGTH = 255
MIME_TYPE_MAX_LENGTH = 255
CHECKSUM_LENGTH = 64
STORAGE_KEY_MAX_LENGTH = 1024
INDEXING_STATUS_MAX_LENGTH = 32
INDEXING_ERROR_MAX_LENGTH = 2000

_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")

_REQUIRED_TEXT_LIMITS = {
    "filename": FILENAME_MAX_LENGTH,
    "mime_type": MIME_TYPE_MAX_LENGTH,
    "storage_key": STORAGE_KEY_MAX_LENGTH,
}


class IndexingStatus(str, Enum):
    """Lifecycle status of vector indexing for a Document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """File metadata for a paper.

    Database-level checks use only ``length`` and ``trim``, which behave the
    same on SQLite and PostgreSQL. The SHA-256 hex format is enforced by the
    model validator.
    """

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("length(trim(filename)) > 0", name="filename_not_blank"),
        CheckConstraint("length(trim(mime_type)) > 0", name="mime_type_not_blank"),
        CheckConstraint(
            "length(trim(storage_key)) > 0", name="storage_key_not_blank"
        ),
        CheckConstraint("file_size >= 0", name="file_size_non_negative"),
        CheckConstraint(
            f"length(checksum) = {CHECKSUM_LENGTH}", name="checksum_length"
        ),
        CheckConstraint(
            "indexing_status IN ('pending', 'processing', 'completed', 'failed')",
            name="indexing_status_valid",
        ),
        CheckConstraint(
            "indexing_attempts >= 0",
            name="indexing_attempts_non_negative",
        ),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(FILENAME_MAX_LENGTH), nullable=False)
    mime_type: Mapped[str] = mapped_column(
        String(MIME_TYPE_MAX_LENGTH), nullable=False
    )
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str] = mapped_column(String(CHECKSUM_LENGTH), nullable=False)
    storage_key: Mapped[str] = mapped_column(
        String(STORAGE_KEY_MAX_LENGTH), nullable=False
    )
    indexing_status: Mapped[str] = mapped_column(
        String(INDEXING_STATUS_MAX_LENGTH),
        nullable=False,
        default=IndexingStatus.PENDING.value,
        server_default=text("'pending'"),
        index=True,
    )
    indexing_error: Mapped[str | None] = mapped_column(
        String(INDEXING_ERROR_MAX_LENGTH),
        nullable=True,
    )
    indexed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    indexing_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    claimed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    claimed_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    paper: Mapped[Paper] = relationship(back_populates="documents")
    # Deletion is owned by the database (``ondelete="CASCADE"``).
    chunks: Mapped[list[DocumentChunkRecord]] = relationship(
        back_populates="document",
        order_by="DocumentChunkRecord.chunk_index",
        passive_deletes="all",
    )

    @validates("filename", "mime_type", "storage_key")
    def _validate_required_text(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a str, got {type(value).__name__}")
        if not value.strip():
            raise ValueError(f"{key} must not be blank")
        limit = _REQUIRED_TEXT_LIMITS[key]
        if len(value) > limit:
            raise ValueError(f"{key} must be at most {limit} characters")
        return value

    @validates("file_size")
    def _validate_file_size(self, key: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"file_size must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError("file_size must be non-negative")
        return value

    @validates("checksum")
    def _validate_checksum(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"checksum must be a str, got {type(value).__name__}")
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("checksum must be 64 hexadecimal characters (SHA-256)")
        return value

    @validates("indexing_status")
    def _validate_indexing_status(self, key: str, value: Any) -> str:
        if isinstance(value, IndexingStatus):
            return value.value
        if not isinstance(value, str):
            raise TypeError(f"indexing_status must be a str or IndexingStatus, got {type(value).__name__}")
        normalized = value.strip().lower()
        valid = {s.value for s in IndexingStatus}
        if normalized not in valid:
            raise ValueError(f"indexing_status must be one of {valid}, got {value!r}")
        return normalized

    @validates("indexing_attempts")
    def _validate_indexing_attempts(self, key: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"indexing_attempts must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError("indexing_attempts must be non-negative")
        return value

    def __repr__(self) -> str:
        return f"Document(id={self.id!r})"
