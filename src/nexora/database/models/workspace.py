"""Workspace ORM model: owned by one User, owns many Collections."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from nexora.database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from nexora.database.models.collection import Collection
    from nexora.database.models.user import User

NAME_MAX_LENGTH = 255
DESCRIPTION_MAX_LENGTH = 2000


class Workspace(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "workspaces"
    __table_args__ = (CheckConstraint("length(trim(name)) > 0", name="name_not_blank"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(NAME_MAX_LENGTH))
    description: Mapped[str | None] = mapped_column(String(DESCRIPTION_MAX_LENGTH))

    user: Mapped[User] = relationship(back_populates="workspaces")
    collections: Mapped[list[Collection]] = relationship(
        back_populates="workspace", passive_deletes="all"
    )

    def __repr__(self) -> str:
        return f"Workspace(id={self.id!r})"
