"""Durable research RAG query job ORM model.

Represents a durable, bounded background job for executing research RAG queries,
evidence retrieval, and answer synthesis.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
import uuid

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from nexora.database.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from nexora.database.models.collection import Collection
    from nexora.database.models.paper import Paper
    from nexora.database.models.user import User

JOB_STATUS_MAX_LENGTH = 32
LAST_ERROR_MAX_LENGTH = 2000
CLAIMED_BY_MAX_LENGTH = 255


class ResearchJobStatus(str, Enum):
    """Lifecycle status of a research RAG background job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A durable background job record for synthesizing research query answers."""

    __tablename__ = "research_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed')",
            name="valid_research_job_status",
        ),
        CheckConstraint("attempts >= 0", name="research_job_attempts_non_negative"),
        CheckConstraint("max_attempts > 0", name="research_job_max_attempts_positive"),
        Index("ix_research_jobs_claim", "status", "next_retry_at", "lease_expires_at"),
        Index("ix_research_jobs_sweeper", "status", "lease_expires_at", "attempts"),
        Index("ix_research_jobs_user_status", "user_id", "status"),
        Index("ix_research_jobs_guest_status", "guest_session_id", "status"),
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    guest_session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    collection_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    retrieval_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="hybrid",
        server_default=text("'hybrid'"),
    )
    top_k: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=5,
        server_default=text("5"),
    )
    min_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    temperature: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    max_output_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(JOB_STATUS_MAX_LENGTH),
        nullable=False,
        default=ResearchJobStatus.QUEUED.value,
        server_default=text("'queued'"),
        index=True,
    )
    result: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB(astext_type=String()), "postgresql"),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        String(LAST_ERROR_MAX_LENGTH),
        nullable=True,
    )
    attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        server_default=text("3"),
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    claimed_by: Mapped[str | None] = mapped_column(
        String(CLAIMED_BY_MAX_LENGTH),
        nullable=True,
    )
    claimed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    lease_expires_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    next_retry_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped[User | None] = relationship()
    collection: Mapped[Collection | None] = relationship()
    paper: Mapped[Paper | None] = relationship()
