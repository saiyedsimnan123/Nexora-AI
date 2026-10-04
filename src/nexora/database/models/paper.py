"""Paper ORM model.

A ``Paper`` belongs to one ``Collection`` and owns its ``Document`` records.
This module contains ORM definitions and model-level validation only: it
creates no engine or session, opens no connection, and touches neither the
filesystem nor the network.
"""

from __future__ import annotations

import datetime
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Date, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from nexora.database.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from nexora.database.models.collection import Collection
    from nexora.database.models.document import Document
    from nexora.database.models.paper_reference import PaperReference

TITLE_MAX_LENGTH = 500
ABSTRACT_MAX_LENGTH = 20000
DOI_MAX_LENGTH = 255
EXTERNAL_ID_MAX_LENGTH = 255

_OPTIONAL_TEXT_LIMITS = {
    "abstract": ABSTRACT_MAX_LENGTH,
    "doi": DOI_MAX_LENGTH,
    "external_id": EXTERNAL_ID_MAX_LENGTH,
}


class Paper(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A research paper stored in a collection.

    ``doi`` and ``external_id`` are stored as given: neither is normalized
    or globally unique, because DOI formatting varies and external ids are
    provider-specific.
    """

    __tablename__ = "papers"
    __table_args__ = (
        CheckConstraint("length(trim(title)) > 0", name="title_not_blank"),
    )

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(TITLE_MAX_LENGTH), nullable=False)
    abstract: Mapped[str | None] = mapped_column(
        String(ABSTRACT_MAX_LENGTH), nullable=True
    )
    doi: Mapped[str | None] = mapped_column(String(DOI_MAX_LENGTH), nullable=True)
    external_id: Mapped[str | None] = mapped_column(
        String(EXTERNAL_ID_MAX_LENGTH), nullable=True
    )
    publication_date: Mapped[datetime.date | None] = mapped_column(
        Date, nullable=True
    )

    collection: Mapped[Collection] = relationship(back_populates="papers")
    # Deletion is owned by the database (``ondelete="CASCADE"``). The ORM
    # neither deletes nor nulls children when a paper is deleted.
    documents: Mapped[list[Document]] = relationship(
        back_populates="paper",
        passive_deletes="all",
    )
    outgoing_references: Mapped[list[PaperReference]] = relationship(
        "PaperReference",
        foreign_keys="[PaperReference.source_paper_id]",
        back_populates="source_paper",
        passive_deletes="all",
    )
    incoming_references: Mapped[list[PaperReference]] = relationship(
        "PaperReference",
        foreign_keys="[PaperReference.target_paper_id]",
        back_populates="target_paper",
        passive_deletes="all",
    )

    @validates("title")
    def _validate_title(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"title must be a str, got {type(value).__name__}")
        if not value.strip():
            raise ValueError("title must not be blank")
        if len(value) > TITLE_MAX_LENGTH:
            raise ValueError(f"title must be at most {TITLE_MAX_LENGTH} characters")
        return value

    @validates("abstract", "doi", "external_id")
    def _validate_optional_text(self, key: str, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a str or None, got {type(value).__name__}")
        limit = _OPTIONAL_TEXT_LIMITS[key]
        if len(value) > limit:
            raise ValueError(f"{key} must be at most {limit} characters")
        return value

    def __repr__(self) -> str:
        return f"Paper(id={self.id!r})"
