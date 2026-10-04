"""Collection ORM model: owned by one Workspace and owns many Papers."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from nexora.database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from nexora.database.models.paper import Paper
    from nexora.database.models.workspace import Workspace

NAME_MAX_LENGTH = 255
DESCRIPTION_MAX_LENGTH = 2000


class Collection(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "collections"

    __table_args__ = (
        CheckConstraint(
            "length(trim(name)) > 0",
            name="name_not_blank",
        ),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(NAME_MAX_LENGTH),
    )

    description: Mapped[str | None] = mapped_column(
        String(DESCRIPTION_MAX_LENGTH),
    )

    workspace: Mapped[Workspace] = relationship(
        back_populates="collections",
    )

    papers: Mapped[list[Paper]] = relationship(
        back_populates="collection",
        passive_deletes="all",
    )

    def __repr__(self) -> str:
        return f"Collection(id={self.id!r})"
