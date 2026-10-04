"""User ORM model: identity only (no passwords, no authentication)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from nexora.database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from nexora.database.models.workspace import Workspace

EMAIL_MAX_LENGTH = 320  # 64 (local part) + 1 + 255 (domain), per RFC 5321


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A platform user. ``email`` is stored exactly as given and is unique."""

    __tablename__ = "users"

    # unique=True already creates the unique constraint (and its index); no extra index.
    email: Mapped[str] = mapped_column(String(EMAIL_MAX_LENGTH), unique=True)

    # Ownership deletion is handled by the database (ON DELETE CASCADE).
    # passive_deletes="all": the ORM neither loads nor nulls children when a user is deleted.
    workspaces: Mapped[list[Workspace]] = relationship(
        back_populates="user", passive_deletes="all"
    )

    def __repr__(self) -> str:  # id only: no PII, no relationship traversal
        return f"User(id={self.id!r})"
