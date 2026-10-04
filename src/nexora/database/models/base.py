"""ORM foundation: declarative Base, naming convention, UUID key and timestamp mixins.

Definitions only: no Engine, Session or table is created here, nothing connects
to a database, and no environment variable is read.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, MetaData, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Deterministic constraint/index names so Alembic migrations are stable and
# constraints can be dropped by name. PostgreSQL limits identifiers to 63
# characters; SQLAlchemy truncates over-long convention names with a stable hash.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for every Nexora model.

    ``uuid.UUID`` maps to SQLAlchemy's portable ``Uuid`` (native UUID on
    PostgreSQL, CHAR(32) elsewhere); ``datetime`` maps to a timezone-aware
    ``DateTime``.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {
        uuid.UUID: Uuid(as_uuid=True),
        datetime: DateTime(timezone=True),
    }


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class UUIDPrimaryKeyMixin:
    """Opt-in UUID primary key, generated in Python (no database extension needed)."""

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    """Opt-in timezone-aware ``created_at`` / ``updated_at`` columns.

    Values are set client-side (aware UTC) and also have a database
    ``server_default`` of ``now()`` so rows inserted outside the ORM still get one.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        server_default=func.now(),
        nullable=False,
    )
