"""Knowledge graph analytics data models (Stage 5 Step 3.4).

Defines typed Pydantic models for graph statistics, distributions,
author collaborations, and author connectivity metrics.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nexora.database.models.graph_entity import VALID_ENTITY_TYPES


class EntityTypeCount(BaseModel):
    """Count of graph entities grouped by entity type."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    entity_type: str = Field(..., min_length=1, description="Entity type identifier.")
    count: int = Field(..., ge=0, description="Non-negative count of entities of this type.")

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("entity_type cannot be blank")
        return cleaned


class RelationshipTypeCount(BaseModel):
    """Count of graph relationships grouped by relationship type."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    relationship_type: str = Field(..., min_length=1, description="Relationship type identifier.")
    count: int = Field(..., ge=0, description="Non-negative count of relationships of this type.")

    @field_validator("relationship_type")
    @classmethod
    def validate_rel_type(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("relationship_type cannot be blank")
        return cleaned


class CollectionGraphSummary(BaseModel):
    """Summary metrics of knowledge graph entities and relationships in an authorized collection."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    collection_id: UUID = Field(..., description="Authorized collection UUID.")
    total_entities: int = Field(..., ge=0, description="Total count of entities in collection.")
    total_relationships: int = Field(..., ge=0, description="Total count of relationships in collection.")
    entity_counts_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of entities across entity types.",
    )
    relationship_counts_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of relationships across relationship types.",
    )

    @property
    def is_empty(self) -> bool:
        """Return True if the collection contains zero entities and zero relationships."""
        return self.total_entities == 0 and self.total_relationships == 0


class AuthorPaperConnection(BaseModel):
    """Represents a connection between a collection-scoped author entity and a paper."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    author_id: UUID = Field(..., description="Collection-scoped GraphEntity ID of author.")
    author_name: str = Field(..., min_length=1, description="Display name of the author.")
    paper_id: UUID = Field(..., description="UUID of the authored Paper.")
    paper_title: str = Field(..., min_length=1, description="Title of the authored Paper.")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nullable confidence score of the authorship relationship.",
    )
    evidence_text: str | None = Field(
        default=None,
        description="Optional grounded text snippet supporting this authorship connection.",
    )


class AuthorCollaborationRecord(BaseModel):
    """Pairwise author collaboration record aggregated from co-authored papers."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    author_id: UUID = Field(..., description="Primary author GraphEntity UUID.")
    author_name: str = Field(..., min_length=1, description="Display name of primary author.")
    collaborator_id: UUID = Field(..., description="Collaborating author GraphEntity UUID.")
    collaborator_name: str = Field(..., min_length=1, description="Display name of collaborator.")
    shared_paper_ids: list[UUID] = Field(
        default_factory=list,
        description="List of distinct paper UUIDs co-authored by this pair.",
    )
    collaboration_count: int = Field(
        ...,
        ge=1,
        description="Number of distinct co-authored papers (must equal len(shared_paper_ids)).",
    )

    @field_validator("shared_paper_ids")
    @classmethod
    def deduplicate_shared_papers(cls, v: list[UUID]) -> list[UUID]:
        seen: set[UUID] = set()
        deduped: list[UUID] = []
        for p in v:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return deduped

    @model_validator(mode="after")
    def validate_collaboration_consistency(self) -> AuthorCollaborationRecord:
        if self.author_id == self.collaborator_id:
            raise ValueError(
                f"Self-collaboration prohibited: author_id and collaborator_id must be distinct ({self.author_id})"
            )
        if len(self.shared_paper_ids) != self.collaboration_count:
            raise ValueError(
                f"collaboration_count ({self.collaboration_count}) must match count of shared_paper_ids ({len(self.shared_paper_ids)})"
            )
        return self

    @property
    def canonical_pair_keys(self) -> tuple[str, str]:
        """Return sorted string tuple of author UUIDs for canonical undirected pair deduplication."""
        id1, id2 = str(self.author_id), str(self.collaborator_id)
        return (id1, id2) if id1 < id2 else (id2, id1)


class AuthorConnectivityMetrics(BaseModel):
    """Graph connectivity metrics for an author within an authorized collection."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    author_id: UUID = Field(..., description="Collection-scoped GraphEntity ID.")
    author_name: str = Field(..., min_length=1, description="Display name of the author.")
    paper_count: int = Field(..., ge=0, description="Total distinct papers authored in collection.")
    collaborator_count: int = Field(
        ...,
        ge=0,
        description="Total distinct co-authors who shared at least one paper.",
    )
    total_collaborations: int = Field(
        ...,
        ge=0,
        description="Sum of collaborations across all co-authors (sum of shared paper connections).",
    )


class CollectionAnalyticsResult(BaseModel):
    """Combined knowledge graph analytics result for an authorized collection."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    collection_id: UUID = Field(..., description="Authorized collection UUID.")
    summary: CollectionGraphSummary = Field(..., description="Overall summary of entities and relationships.")
    entity_distribution: list[EntityTypeCount] = Field(
        default_factory=list,
        description="Distribution of entity types.",
    )
    relationship_distribution: list[RelationshipTypeCount] = Field(
        default_factory=list,
        description="Distribution of relationship types.",
    )
    top_collaborations: list[AuthorCollaborationRecord] = Field(
        default_factory=list,
        description="Top author collaboration pairs in collection.",
    )
    author_connectivity: list[AuthorConnectivityMetrics] = Field(
        default_factory=list,
        description="Author degree and connectivity metrics in collection.",
    )
