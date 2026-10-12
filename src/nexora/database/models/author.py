"""Author ORM model.

An ``Author`` represents a person who has authored one or more research papers.
This module contains ORM definitions and model-level validation only: it
creates no engine or session, opens no connection, and touches neither the
filesystem nor the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, String
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from nexora.database.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from nexora.database.models.paper_author import PaperAuthor

AUTHOR_NAME_MAX_LENGTH = 500


class Author(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An author of research papers."""

    __tablename__ = "authors"
    __table_args__ = (
        CheckConstraint("length(trim(name)) > 0", name="name_not_blank"),
    )

    name: Mapped[str] = mapped_column(String(AUTHOR_NAME_MAX_LENGTH), nullable=False)

    paper_authorships: Mapped[list[PaperAuthor]] = relationship(
        "PaperAuthor",
        back_populates="author",
        passive_deletes="all",
    )

    @validates("name")
    def _validate_name(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"name must be a str, got {type(value).__name__}")
        if not value.strip():
            raise ValueError("name must not be blank")
        if len(value) > AUTHOR_NAME_MAX_LENGTH:
            raise ValueError(f"name must be at most {AUTHOR_NAME_MAX_LENGTH} characters")
        return value

    def __repr__(self) -> str:
        return f"Author(id={self.id!r}, name={self.name!r})"
