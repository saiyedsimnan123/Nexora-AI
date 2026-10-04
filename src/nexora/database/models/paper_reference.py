"""Paper-to-paper citation references between Papers."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from nexora.database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from nexora.database.models.paper import Paper


class PaperReference(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A directed citation from one paper to another."""

    __tablename__ = "paper_references"
    __table_args__ = (
        UniqueConstraint(
            "source_paper_id",
            "target_paper_id",
            name="source_target_unique",
        ),
        CheckConstraint(
            "source_paper_id <> target_paper_id",
            name="no_self_reference",
        ),
    )

    source_paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_paper: Mapped[Paper] = relationship(
        "Paper",
        foreign_keys="[PaperReference.source_paper_id]",
        back_populates="outgoing_references",
        passive_deletes="all",
    )
    target_paper: Mapped[Paper] = relationship(
        "Paper",
        foreign_keys="[PaperReference.target_paper_id]",
        back_populates="incoming_references",
        passive_deletes="all",
    )

    @validates("source_paper_id", "target_paper_id")
    def _validate_paper_ids(self, key: str, value: uuid.UUID) -> uuid.UUID:
        if not isinstance(value, uuid.UUID):
            raise TypeError(f"{key} must be a UUID, got {type(value).__name__}")

        if key == "source_paper_id":
            other = getattr(self, "target_paper_id", None)
            if other is not None and value == other:
                raise ValueError("source_paper_id and target_paper_id must be different")
        else:
            other = getattr(self, "source_paper_id", None)
            if other is not None and value == other:
                raise ValueError("source_paper_id and target_paper_id must be different")

        return value

    def __repr__(self) -> str:
        return f"PaperReference(id={self.id!r})"
