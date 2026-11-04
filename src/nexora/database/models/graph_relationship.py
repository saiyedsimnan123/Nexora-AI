"""Knowledge graph relationship ORM model.

Represents a typed, directed relationship between graph endpoints (Paper or Entity)
within an authorized collection, with grounded evidence provenance.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any
import uuid

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from nexora.database.models.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from nexora.database.models.collection import Collection
    from nexora.database.models.document_chunk import DocumentChunkRecord
    from nexora.database.models.graph_entity import GraphEntity
    from nexora.database.models.paper import Paper

RELATIONSHIP_TYPE_MAX_LENGTH = 100
ENDPOINT_TYPE_MAX_LENGTH = 20
EVIDENCE_TEXT_MAX_LENGTH = 5000
EXTRACTION_METHOD_MAX_LENGTH = 100


class GraphEndpointType(str, Enum):
    """Permitted endpoint types for knowledge graph relationships."""

    PAPER = "paper"
    ENTITY = "entity"


VALID_ENDPOINT_TYPES: frozenset[str] = frozenset(t.value for t in GraphEndpointType)


class GraphRelationship(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A typed relationship connecting papers and entities with evidence provenance."""

    __tablename__ = "graph_relationships"
    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "relationship_type",
            "source_type",
            "source_id",
            "target_type",
            "target_id",
            name="uq_graph_relationships_identity",
        ),
        CheckConstraint("length(trim(relationship_type)) > 0", name="type_not_blank"),
        CheckConstraint("source_type IN ('paper', 'entity')", name="source_type_valid"),
        CheckConstraint("target_type IN ('paper', 'entity')", name="target_type_valid"),
        CheckConstraint(
            "NOT (source_type = target_type AND source_id = target_id)",
            name="no_self_loop",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="confidence_range",
        ),
        CheckConstraint(
            "page_number IS NULL OR page_number >= 1",
            name="page_number_positive",
        ),
        CheckConstraint(
            "evidence_text IS NULL OR length(trim(evidence_text)) > 0",
            name="evidence_not_blank",
        ),
        CheckConstraint(
            "((source_type = 'paper' AND source_paper_id IS NOT NULL AND source_paper_id = source_id AND source_entity_id IS NULL) OR "
            "(source_type = 'entity' AND source_entity_id IS NOT NULL AND source_entity_id = source_id AND source_paper_id IS NULL))",
            name="source_endpoint_consistency",
        ),
        CheckConstraint(
            "((target_type = 'paper' AND target_paper_id IS NOT NULL AND target_paper_id = target_id AND target_entity_id IS NULL) OR "
            "(target_type = 'entity' AND target_entity_id IS NOT NULL AND target_entity_id = target_id AND target_paper_id IS NULL))",
            name="target_endpoint_consistency",
        ),
    )

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(
        String(RELATIONSHIP_TYPE_MAX_LENGTH),
        nullable=False,
        index=True,
    )

    # Generic endpoint identities (NOT NULL for strictly reliable unique constraint across all SQL dialects)
    source_type: Mapped[str] = mapped_column(
        String(ENDPOINT_TYPE_MAX_LENGTH),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        index=True,
    )
    target_type: Mapped[str] = mapped_column(
        String(ENDPOINT_TYPE_MAX_LENGTH),
        nullable=False,
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        index=True,
    )

    # Cascading relational foreign keys
    source_paper_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    source_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    target_paper_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    target_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Evidence provenance
    paper_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    evidence_text: Mapped[str | None] = mapped_column(
        String(EVIDENCE_TEXT_MAX_LENGTH),
        nullable=True,
    )
    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    extraction_method: Mapped[str | None] = mapped_column(
        String(EXTRACTION_METHOD_MAX_LENGTH),
        nullable=True,
    )
    relationship_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    # Relationships
    collection: Mapped[Collection] = relationship("Collection", back_populates="graph_relationships")
    source_paper: Mapped[Paper | None] = relationship(
        "Paper",
        foreign_keys="[GraphRelationship.source_paper_id]",
        passive_deletes="all",
    )
    source_entity: Mapped[GraphEntity | None] = relationship(
        "GraphEntity",
        foreign_keys="[GraphRelationship.source_entity_id]",
        back_populates="outgoing_relationships",
        passive_deletes="all",
    )
    target_paper: Mapped[Paper | None] = relationship(
        "Paper",
        foreign_keys="[GraphRelationship.target_paper_id]",
        passive_deletes="all",
    )
    target_entity: Mapped[GraphEntity | None] = relationship(
        "GraphEntity",
        foreign_keys="[GraphRelationship.target_entity_id]",
        back_populates="incoming_relationships",
        passive_deletes="all",
    )
    asserting_paper: Mapped[Paper | None] = relationship(
        "Paper",
        foreign_keys="[GraphRelationship.paper_id]",
        passive_deletes="all",
    )
    evidence_chunk: Mapped[DocumentChunkRecord | None] = relationship(
        "DocumentChunkRecord",
        foreign_keys="[GraphRelationship.chunk_id]",
        passive_deletes="all",
    )

    def __init__(self, **kwargs: Any) -> None:
        # Automatically sync source/target FK columns from source_type/source_id and target_type/target_id
        src_type = kwargs.get("source_type")
        src_id = kwargs.get("source_id")
        if src_type == GraphEndpointType.PAPER.value or src_type == "paper":
            kwargs.setdefault("source_paper_id", src_id)
            kwargs.setdefault("source_entity_id", None)
        elif src_type == GraphEndpointType.ENTITY.value or src_type == "entity":
            kwargs.setdefault("source_entity_id", src_id)
            kwargs.setdefault("source_paper_id", None)

        tgt_type = kwargs.get("target_type")
        tgt_id = kwargs.get("target_id")
        if tgt_type == GraphEndpointType.PAPER.value or tgt_type == "paper":
            kwargs.setdefault("target_paper_id", tgt_id)
            kwargs.setdefault("target_entity_id", None)
        elif tgt_type == GraphEndpointType.ENTITY.value or tgt_type == "entity":
            kwargs.setdefault("target_entity_id", tgt_id)
            kwargs.setdefault("target_paper_id", None)

        super().__init__(**kwargs)

    @property
    def has_evidence(self) -> bool:
        """Return True if this relationship has supporting document evidence."""
        return bool(self.evidence_text or self.chunk_id is not None or self.paper_id is not None)

    @validates("relationship_type")
    def _validate_relationship_type(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"relationship_type must be a str, got {type(value).__name__}")
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("relationship_type must not be blank")
        if len(cleaned) > RELATIONSHIP_TYPE_MAX_LENGTH:
            raise ValueError(f"relationship_type must be at most {RELATIONSHIP_TYPE_MAX_LENGTH} characters")
        return cleaned

    @validates("source_type", "target_type")
    def _validate_endpoint_type(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a str, got {type(value).__name__}")
        lowered = value.strip().lower()
        if lowered not in VALID_ENDPOINT_TYPES:
            raise ValueError(f"Invalid {key} '{value}'. Must be one of: {sorted(VALID_ENDPOINT_TYPES)}")
        return lowered

    @validates("source_id", "target_id")
    def _validate_endpoint_id(self, key: str, value: uuid.UUID) -> uuid.UUID:
        if isinstance(value, str):
            try:
                value = uuid.UUID(value)
            except ValueError as exc:
                raise TypeError(f"{key} must be a valid UUID, got {value!r}") from exc
        if not isinstance(value, uuid.UUID):
            raise TypeError(f"{key} must be a UUID, got {type(value).__name__}")
        return value

    @validates("confidence")
    def _validate_confidence(self, key: str, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"confidence must be a float, got {type(value).__name__}")
        val = float(value)
        if val < 0.0 or val > 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {val}")
        return val

    @validates("page_number")
    def _validate_page_number(self, key: str, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"page_number must be an int, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"page_number must be >= 1, got {value}")
        return value

    @validates("evidence_text")
    def _validate_evidence_text(self, key: str, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"evidence_text must be a str, got {type(value).__name__}")
        stripped = value.strip()
        if not stripped:
            raise ValueError("evidence_text must not be blank if provided")
        if len(value) > EVIDENCE_TEXT_MAX_LENGTH:
            raise ValueError(f"evidence_text must be at most {EVIDENCE_TEXT_MAX_LENGTH} characters")
        return value

    @validates("extraction_method")
    def _validate_extraction_method(self, key: str, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"extraction_method must be a str, got {type(value).__name__}")
        stripped = value.strip()
        if not stripped:
            return None
        if len(stripped) > EXTRACTION_METHOD_MAX_LENGTH:
            raise ValueError(f"extraction_method must be at most {EXTRACTION_METHOD_MAX_LENGTH} characters")
        return stripped

    @validates("relationship_metadata")
    def _validate_metadata(self, key: str, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"relationship_metadata must be a dict, got {type(value).__name__}")
        return value

    def __repr__(self) -> str:
        return (
            f"GraphRelationship(id={self.id!r}, type={self.relationship_type!r}, "
            f"source={self.source_type}:{self.source_id!r}, "
            f"target={self.target_type}:{self.target_id!r}, "
            f"collection_id={self.collection_id!r})"
        )
