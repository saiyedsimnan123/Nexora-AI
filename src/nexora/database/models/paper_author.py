"""PaperAuthor ORM model.

A ``PaperAuthor`` association connects research papers with their authors.
This module contains ORM definitions and model-level validation only: it
creates no engine or session, opens no connection, and touches neither the
filesystem nor the network.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from nexora.database.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from nexora.database.models.author import Author
    from nexora.database.models.paper import Paper


class PaperAuthor(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Association model between Paper and Author.

    Maintains the author's order on the paper.
    """

    __tablename__ = "paper_authors"
    __table_args__ = (
        CheckConstraint("author_order >= 1", name="author_order_positive"),
        UniqueConstraint("paper_id", "author_id", name="paper_author_unique"),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    author_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("authors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    author_order: Mapped[int] = mapped_column(Integer, nullable=False)

    paper: Mapped[Paper] = relationship("Paper", back_populates="authorships")
    author: Mapped[Author] = relationship("Author", back_populates="paper_authorships")

    @validates("author_order")
    def _validate_author_order(self, key: str, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"author_order must be an integer, got bool")
        if not isinstance(value, int):
            raise TypeError(f"author_order must be an int, got {type(value).__name__}")
        if value < 1:
            raise ValueError("author_order must be >= 1")
        return value

    def __repr__(self) -> str:
        return (
            f"PaperAuthor(id={self.id!r}, paper_id={self.paper_id!r}, "
            f"author_id={self.author_id!r}, author_order={self.author_order!r})"
        )
