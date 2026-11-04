"""Data models and schemas for knowledge graph extraction (Stage 5 Step 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nexora.database.models.graph_entity import (
    GraphEntity,
    GraphEntityType,
    VALID_ENTITY_TYPES,
    normalize_entity_name,
)
from nexora.database.models.graph_relationship import (
    GraphEndpointType,
    GraphRelationship,
    VALID_ENDPOINT_TYPES,
)


class GraphExtractionError(Exception):
    """Base exception for knowledge graph extraction failures."""


class InvalidExtractionPayloadError(GraphExtractionError):
    """Raised when an extraction payload fails schema validation."""


class ExtractedEntityCandidate(BaseModel):
    """A scientific entity candidate proposed by extraction."""

    model_config = ConfigDict(strict=True, extra="ignore")

    name: str = Field(..., min_length=1, max_length=255)
    entity_type: str = Field(..., min_length=1, max_length=50)
    description: str | None = Field(default=None, max_length=5000)
    aliases: list[str] | None = None
    entity_metadata: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Entity candidate name cannot be blank")
        return cleaned

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        lowered = v.strip().lower()
        if lowered not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type '{v}'. Permitted types: {sorted(VALID_ENTITY_TYPES)}"
            )
        return lowered

    @property
    def canonical_name(self) -> str:
        return normalize_entity_name(self.name)


class ExtractedRelationshipCandidate(BaseModel):
    """A typed relationship candidate connecting graph endpoints with evidence."""

    model_config = ConfigDict(strict=True, extra="ignore")

    source_name: str = Field(..., min_length=1, max_length=500)
    source_type: str = Field(default=GraphEndpointType.PAPER.value)
    target_name: str = Field(..., min_length=1, max_length=500)
    target_type: str = Field(default=GraphEndpointType.ENTITY.value)
    relationship_type: str = Field(..., min_length=1, max_length=100)
    evidence_label: str | None = None
    evidence_quote: str | None = Field(default=None, max_length=5000)
    page_number: int | None = Field(default=None, ge=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    extraction_method: str | None = Field(default="llm_extraction", max_length=100)
    metadata: dict[str, Any] | None = None

    @field_validator("source_type", "target_type")
    @classmethod
    def validate_endpoint_type(cls, v: str) -> str:
        lowered = v.strip().lower()
        if lowered not in VALID_ENDPOINT_TYPES:
            raise ValueError(
                f"Invalid endpoint type '{v}'. Must be one of: {sorted(VALID_ENDPOINT_TYPES)}"
            )
        return lowered

    @field_validator("relationship_type")
    @classmethod
    def validate_relationship_type(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("relationship_type cannot be blank")
        return cleaned

    @field_validator("evidence_quote")
    @classmethod
    def validate_evidence_quote(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None


class ExtractedGraphPayload(BaseModel):
    """Strict schema for LLM structured output during graph extraction."""

    model_config = ConfigDict(strict=True, extra="ignore")

    entities: list[ExtractedEntityCandidate] = Field(default_factory=list)
    relationships: list[ExtractedRelationshipCandidate] = Field(default_factory=list)


@dataclass
class GraphExtractionResult:
    """Outcome report of a paper's knowledge graph extraction pass."""

    paper_id: UUID
    collection_id: UUID
    status: str = "success"  # "success", "partial", "failed"
    entities_created: int = 0
    entities_reused: int = 0
    relationships_created: int = 0
    relationships_reused: int = 0
    authors_linked: int = 0
    citations_linked: int = 0
    entities: list[GraphEntity] = field(default_factory=list)
    relationships: list[GraphRelationship] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_entities(self) -> int:
        return self.entities_created + self.entities_reused

    @property
    def total_relationships(self) -> int:
        return self.relationships_created + self.relationships_reused
