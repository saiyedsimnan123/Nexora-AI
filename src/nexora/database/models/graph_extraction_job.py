"""Knowledge graph extraction job ORM model.

Represents a durable, bounded background job for extracting scientific entities
and relationships from research papers and documents into the knowledge graph.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    false,
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
    from nexora.database.models.document import Document
    from nexora.database.models.paper import Paper

JOB_STATUS_MAX_LENGTH = 32
LAST_ERROR_MAX_LENGTH = 2000
CLAIMED_BY_MAX_LENGTH = 255


class GraphExtractionStatus(str, Enum):
    """Lifecycle status of a knowledge graph extraction job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GraphExtractionJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A durable background job record for extracting a paper's knowledge graph."""

    __tablename__ = "graph_extraction_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed')",
            name="valid_job_status",
        ),
        CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        CheckConstraint("max_attempts > 0", name="max_attempts_positive"),
        Index("ix_graph_jobs_claim", "status", "next_retry_at", "lease_expires_at"),
        Index("ix_graph_jobs_sweeper", "status", "lease_expires_at", "attempts"),
        Index(
            "uq_graph_jobs_active_paper",
            "paper_id",
            unique=True,
            sqlite_where=text("status IN ('pending', 'processing')"),
            postgresql_where=text("status IN ('pending', 'processing')"),
        ),
    )

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(JOB_STATUS_MAX_LENGTH),
        nullable=False,
        default=GraphExtractionStatus.PENDING.value,
        server_default=text("'pending'"),
        index=True,
    )
    metadata_extracted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
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
    last_error: Mapped[str | None] = mapped_column(
        String(LAST_ERROR_MAX_LENGTH),
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
    claimed_by: Mapped[str | None] = mapped_column(
        String(CLAIMED_BY_MAX_LENGTH),
        nullable=True,
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
    job_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB(astext_type=String()), "postgresql"),
        nullable=True,
    )

    # Relationships
    collection: Mapped[Collection] = relationship()
    paper: Mapped[Paper] = relationship()
    document: Mapped[Document | None] = relationship()
