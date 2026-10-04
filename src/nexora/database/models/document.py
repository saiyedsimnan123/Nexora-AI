"""Document ORM model.

A ``Document`` is a stored file (for example a PDF) that belongs to one
``Paper``. This module contains ORM definitions and model-level validation
only: it creates no engine or session, opens no connection, reads no files,
and performs no external communication. The file itself lives in external
storage and is referenced by ``storage_key``.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from nexora.database.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from nexora.database.models.paper import Paper


FILENAME_MAX_LENGTH = 255
MIME_TYPE_MAX_LENGTH = 255
CHECKSUM_LENGTH = 64
STORAGE_KEY_MAX_LENGTH = 1024

_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")

_REQUIRED_TEXT_LIMITS = {
    "filename": FILENAME_MAX_LENGTH,
    "mime_type": MIME_TYPE_MAX_LENGTH,
    "storage_key": STORAGE_KEY_MAX_LENGTH,
}


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """File metadata for a paper.

    Database-level checks use portable length and whitespace constraints.
    SHA-256 hexadecimal format is enforced by model validation.
    """

    __tablename__ = "documents"

    __table_args__ = (
        CheckConstraint(
            "length(trim(filename)) > 0",
            name="filename_not_blank",
        ),
        CheckConstraint(
            "length(trim(mime_type)) > 0",
            name="mime_type_not_blank",
        ),
        CheckConstraint(
            "length(trim(storage_key)) > 0",
            name="storage_key_not_blank",
        ),
        CheckConstraint(
            "file_size >= 0",
            name="file_size_non_negative",
        ),
        CheckConstraint(
            f"length(checksum) = {CHECKSUM_LENGTH}",
            name="checksum_length",
        ),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    filename: Mapped[str] = mapped_column(
        String(FILENAME_MAX_LENGTH),
        nullable=False,
    )

    mime_type: Mapped[str] = mapped_column(
        String(MIME_TYPE_MAX_LENGTH),
        nullable=False,
    )

    file_size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    checksum: Mapped[str] = mapped_column(
        String(CHECKSUM_LENGTH),
        nullable=False,
    )

    storage_key: Mapped[str] = mapped_column(
        String(STORAGE_KEY_MAX_LENGTH),
        nullable=False,
    )

    paper: Mapped[Paper] = relationship(
        back_populates="documents",
    )

    @validates("filename", "mime_type", "storage_key")
    def _validate_required_text(
        self,
        key: str,
        value: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{key} must be a str, got {type(value).__name__}"
            )

        if not value.strip():
            raise ValueError(f"{key} must not be blank")

        limit = _REQUIRED_TEXT_LIMITS[key]

        if len(value) > limit:
            raise ValueError(
                f"{key} must be at most {limit} characters"
            )

        return value

    @validates("file_size")
    def _validate_file_size(
        self,
        key: str,
        value: int,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"file_size must be an int, got {type(value).__name__}"
            )

        if value < 0:
            raise ValueError("file_size must be non-negative")

        return value

    @validates("checksum")
    def _validate_checksum(
        self,
        key: str,
        value: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"checksum must be a str, got {type(value).__name__}"
            )

        if not _SHA256_HEX.fullmatch(value):
            raise ValueError(
                "checksum must be 64 hexadecimal characters (SHA-256)"
            )

        return value

    def __repr__(self) -> str:
        return f"Document(id={self.id!r})"
