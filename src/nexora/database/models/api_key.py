"""ApiKey ORM model: provides programmatic authentication tokens for Users."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING
import uuid

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from nexora.database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from nexora.database.models.user import User

NAME_MAX_LENGTH = 255
KEY_HASH_LENGTH = 64
KEY_PREFIX_MAX_LENGTH = 16


class ApiKey(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An API key credential associated with a User.

    The raw secret is never stored in plaintext. Only the SHA-256 hex digest
    is stored in ``key_hash`` for constant-time lookup.
    """

    __tablename__ = "api_keys"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(NAME_MAX_LENGTH), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(KEY_HASH_LENGTH), unique=True, index=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(KEY_PREFIX_MAX_LENGTH), nullable=False)
    expires_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="api_keys")

    def __repr__(self) -> str:
        # Never expose key_hash or raw secret in repr
        return f"ApiKey(id={self.id!r}, prefix={self.key_prefix!r}, name={self.name!r})"
