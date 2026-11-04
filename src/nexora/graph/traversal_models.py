"""Data models for knowledge graph traversal, neighborhoods, and paths (Stage 5 Step 3.1)."""

from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nexora.database.models.graph_entity import (
    GraphEntity,
    VALID_ENTITY_TYPES,
)
from nexora.database.models.graph_relationship import (
    GraphEndpointType,
    GraphRelationship,
    VALID_ENDPOINT_TYPES,
)
from nexora.database.models.paper import Paper


class GraphNode(BaseModel):
    """A node in the knowledge graph, uniquely identified by compound key (node_type, id)."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    id: UUID
    node_type: str = Field(..., description="Endpoint type: 'paper' or 'entity'.")
    name: str = Field(..., min_length=1, max_length=500, description="Paper title or entity name.")
    entity_type: str | None = Field(default=None, description="Type if node_type is 'entity'.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("node_type")
    @classmethod
    def validate_node_type(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned not in VALID_ENDPOINT_TYPES:
            raise ValueError(
                f"Invalid node_type '{v}'. Must be one of: {sorted(VALID_ENDPOINT_TYPES)}"
            )
        return cleaned

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip().lower()
        if cleaned not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type '{v}'. Permitted types: {sorted(VALID_ENTITY_TYPES)}"
            )
        return cleaned

    @model_validator(mode="after")
    def validate_node_consistency(self) -> GraphNode:
        if self.node_type == GraphEndpointType.PAPER.value and self.entity_type is not None:
            raise ValueError("entity_type must be None when node_type is 'paper'")
        if self.node_type == GraphEndpointType.ENTITY.value and self.entity_type is None:
            raise ValueError("entity_type is required when node_type is 'entity'")
        return self

    @property
    def node_key(self) -> tuple[str, UUID]:
        """Compound unique identity for graph operations."""
        return (self.node_type, self.id)

    @property
    def key(self) -> str:
        """String representation of the compound key."""
        return f"{self.node_type}:{self.id}"

    def __hash__(self) -> int:
        return hash((self.node_type, self.id))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GraphNode):
            return (self.node_type, self.id) == (other.node_type, other.id)
        return False

    @classmethod
    def from_paper(cls, paper: Paper) -> GraphNode:
        """Construct a GraphNode from a Paper ORM instance."""
        meta: dict[str, Any] = {}
        if paper.doi:
            meta["doi"] = paper.doi
        if paper.external_id:
            meta["external_id"] = paper.external_id
        if paper.publication_date:
            meta["publication_date"] = str(paper.publication_date)

        return cls(
            id=paper.id,
            node_type=GraphEndpointType.PAPER.value,
            name=paper.title,
            entity_type=None,
            metadata=meta,
        )

    @classmethod
    def from_entity(cls, entity: GraphEntity) -> GraphNode:
        """Construct a GraphNode from a GraphEntity ORM instance."""
        meta: dict[str, Any] = {
            "canonical_name": entity.canonical_name,
        }
        if entity.description:
            meta["description"] = entity.description
        if entity.aliases:
            meta["aliases"] = list(entity.aliases)
        if entity.entity_metadata:
            meta["entity_metadata"] = dict(entity.entity_metadata)

        return cls(
            id=entity.id,
            node_type=GraphEndpointType.ENTITY.value,
            name=entity.name,
            entity_type=entity.entity_type,
            metadata=meta,
        )


class GraphEdge(BaseModel):
    """A directed edge in the knowledge graph with preserved evidence provenance."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    id: UUID
    relationship_type: str = Field(..., min_length=1, max_length=100)
    source_type: str = Field(...)
    source_id: UUID
    target_type: str = Field(...)
    target_id: UUID
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_text: str | None = Field(default=None, max_length=5000)
    chunk_id: UUID | None = None
    page_number: int | None = Field(default=None, ge=1)
    extraction_method: str | None = None
    paper_id: UUID | None = Field(default=None, description="Asserting paper provenance.")
    relationship_metadata: dict[str, Any] | None = None

    @field_validator("source_type", "target_type")
    @classmethod
    def validate_endpoint_type(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned not in VALID_ENDPOINT_TYPES:
            raise ValueError(
                f"Invalid endpoint type '{v}'. Must be one of: {sorted(VALID_ENDPOINT_TYPES)}"
            )
        return cleaned

    @field_validator("relationship_type")
    @classmethod
    def validate_rel_type(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("relationship_type cannot be blank")
        return cleaned

    @property
    def source_key(self) -> tuple[str, UUID]:
        """Source compound node identity."""
        return (self.source_type, self.source_id)

    @property
    def target_key(self) -> tuple[str, UUID]:
        """Target compound node identity."""
        return (self.target_type, self.target_id)

    @property
    def has_evidence(self) -> bool:
        """True if the edge has grounded chunk ID or quote excerpt."""
        return self.chunk_id is not None or self.evidence_text is not None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GraphEdge):
            return self.id == other.id
        return False

    @classmethod
    def from_orm(cls, rel: GraphRelationship) -> GraphEdge:
        """Construct a GraphEdge from a GraphRelationship ORM record."""
        return cls(
            id=rel.id,
            relationship_type=rel.relationship_type,
            source_type=rel.source_type,
            source_id=rel.source_id,
            target_type=rel.target_type,
            target_id=rel.target_id,
            confidence=rel.confidence,
            evidence_text=rel.evidence_text,
            chunk_id=rel.chunk_id,
            page_number=rel.page_number,
            extraction_method=rel.extraction_method,
            paper_id=rel.paper_id,
            relationship_metadata=rel.relationship_metadata,
        )


class GraphSubgraph(BaseModel):
    """An explored subgraph bounded by a seed node and expansion depth."""

    model_config = ConfigDict(strict=True, extra="ignore")

    seed_id: UUID
    seed_type: str = Field(...)
    depth: int = Field(default=0, ge=0)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    is_truncated: bool = False
    truncation_reason: str | None = None

    @field_validator("seed_type")
    @classmethod
    def validate_seed_type(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned not in VALID_ENDPOINT_TYPES:
            raise ValueError(f"Invalid seed_type '{v}'")
        return cleaned

    @property
    def seed_key(self) -> tuple[str, UUID]:
        """Seed node compound key."""
        return (self.seed_type, self.seed_id)

    @property
    def node_keys(self) -> set[tuple[str, UUID]]:
        """Set of compound node keys present in this subgraph."""
        return {node.node_key for node in self.nodes}

    def get_node(self, node_type: str, node_id: UUID) -> GraphNode | None:
        """Retrieve node by compound identity."""
        clean_type = node_type.strip().lower()
        for node in self.nodes:
            if node.node_type == clean_type and node.id == node_id:
                return node
        return None


class GraphPath(BaseModel):
    """A directed or undirected path between two nodes in the knowledge graph."""

    model_config = ConfigDict(strict=True, extra="ignore")

    nodes: list[GraphNode] = Field(..., min_length=1)
    edges: list[GraphEdge] = Field(default_factory=list)
    length: int = Field(default=0, ge=0)
    confidence_score: float | None = None
    is_partially_unscored: bool = False

    @model_validator(mode="after")
    def validate_path_structure(self) -> GraphPath:
        if self.edges:
            if len(self.nodes) != len(self.edges) + 1:
                raise ValueError(
                    f"Path structure mismatch: {len(self.nodes)} nodes requires "
                    f"{len(self.nodes) - 1} edges, got {len(self.edges)} edges."
                )
        return self

    @property
    def start_node(self) -> GraphNode:
        return self.nodes[0]

    @property
    def end_node(self) -> GraphNode:
        return self.nodes[-1]

    @property
    def path_keys(self) -> list[tuple[str, UUID]]:
        return [node.node_key for node in self.nodes]
