"""Knowledge graph entity ORM model.

Represents a scientific entity (concept, method, dataset, metric, author)
scoped strictly to a collection with canonical deduplication.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import TYPE_CHECKING, Any
import unicodedata
import uuid

from sqlalchemy import CheckConstraint, ForeignKey, String, UniqueConstraint
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
    from nexora.database.models.graph_relationship import GraphRelationship

ENTITY_NAME_MAX_LENGTH = 255
ENTITY_TYPE_MAX_LENGTH = 50
DESCRIPTION_MAX_LENGTH = 5000


class GraphEntityType(str, Enum):
    """Permitted entity types in the scientific knowledge graph."""

    CONCEPT = "concept"
    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    AUTHOR = "author"
    OTHER = "other"


VALID_ENTITY_TYPES: frozenset[str] = frozenset(t.value for t in GraphEntityType)


def normalize_entity_name(name: str) -> str:
    """Return a deterministic normalized canonical form for an entity name.

    Strips whitespace, normalizes Unicode (NFKC), collapses internal whitespace,
    and converts to lowercase.
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a str, got {type(name).__name__}")
    cleaned = unicodedata.normalize("NFKC", name).strip().lower()
    collapsed = re.sub(r"\s+", " ", cleaned)
    if not collapsed:
        raise ValueError("Entity name must not be blank")
    return collapsed


class GraphEntity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A canonical scientific entity belonging to an authorized collection."""

    __tablename__ = "graph_entities"
    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "entity_type",
            "canonical_name",
            name="uq_graph_entities_collection_type_canonical",
        ),
        CheckConstraint("length(trim(name)) > 0", name="name_not_blank"),
        CheckConstraint("length(trim(canonical_name)) > 0", name="canonical_name_not_blank"),
        CheckConstraint(
            "entity_type IN ('concept', 'method', 'dataset', 'metric', 'author', 'other')",
            name="valid_entity_type",
        ),
    )

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(ENTITY_NAME_MAX_LENGTH),
        nullable=False,
    )
    canonical_name: Mapped[str] = mapped_column(
        String(ENTITY_NAME_MAX_LENGTH),
        nullable=False,
        index=True,
    )
    entity_type: Mapped[str] = mapped_column(
        String(ENTITY_TYPE_MAX_LENGTH),
        nullable=False,
        index=True,
    )
    description: Mapped[str | None] = mapped_column(
        String(DESCRIPTION_MAX_LENGTH),
        nullable=True,
    )
    aliases: Mapped[list[str] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )
    entity_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    collection: Mapped[Collection] = relationship(
        "Collection",
        back_populates="graph_entities",
    )
    outgoing_relationships: Mapped[list[GraphRelationship]] = relationship(
        "GraphRelationship",
        foreign_keys="[GraphRelationship.source_entity_id]",
        back_populates="source_entity",
        passive_deletes="all",
    )
    incoming_relationships: Mapped[list[GraphRelationship]] = relationship(
        "GraphRelationship",
        foreign_keys="[GraphRelationship.target_entity_id]",
        back_populates="target_entity",
        passive_deletes="all",
    )

    @validates("name")
    def _validate_name(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"name must be a str, got {type(value).__name__}")
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be blank")
        if len(value) > ENTITY_NAME_MAX_LENGTH:
            raise ValueError(f"name must be at most {ENTITY_NAME_MAX_LENGTH} characters")
        return value

    @validates("canonical_name")
    def _validate_canonical_name(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"canonical_name must be a str, got {type(value).__name__}")
        stripped = value.strip()
        if not stripped:
            raise ValueError("canonical_name must not be blank")
        if len(value) > ENTITY_NAME_MAX_LENGTH:
            raise ValueError(f"canonical_name must be at most {ENTITY_NAME_MAX_LENGTH} characters")
        return value

    @validates("entity_type")
    def _validate_entity_type(self, key: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"entity_type must be a str, got {type(value).__name__}")
        lowered = value.strip().lower()
        if lowered not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type '{value}'. Must be one of: {sorted(VALID_ENTITY_TYPES)}"
            )
        return lowered

    @validates("description")
    def _validate_description(self, key: str, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"description must be a str, got {type(value).__name__}")
        if len(value) > DESCRIPTION_MAX_LENGTH:
            raise ValueError(f"description must be at most {DESCRIPTION_MAX_LENGTH} characters")
        return value

    @validates("aliases")
    def _validate_aliases(self, key: str, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"aliases must be a list or tuple of strings, got {type(value).__name__}")
        cleaned: list[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, str):
                raise TypeError(f"aliases[{idx}] must be a str, got {type(item).__name__}")
            s = item.strip()
            if s and s not in cleaned:
                cleaned.append(s)
        return cleaned

    @validates("entity_metadata")
    def _validate_metadata(self, key: str, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"entity_metadata must be a dict, got {type(value).__name__}")
        return value

    def __repr__(self) -> str:
        return (
            f"GraphEntity(id={self.id!r}, type={self.entity_type!r}, "
            f"canonical={self.canonical_name!r}, collection_id={self.collection_id!r})"
        )
