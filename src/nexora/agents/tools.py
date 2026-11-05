"""Explicit, allowlisted tool registry wrapping existing Nexora Stage 1–3 research services."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from nexora.agents.models import (
    AgentContext,
    AgentSecurityError,
    InvalidToolArgumentError,
    ToolDefinition,
    ToolResult,
    UnknownToolError,
)
from nexora.api.errors import (
    DocumentNotFoundError,
    ForbiddenError,
    InvalidConfigurationError,
    InvalidRequestError,
)
from nexora.research.verification import ClaimToEvidenceVerifier, RawClaim
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem, extract_citations


# ---------------------------------------------------------------------------
# Base Tool Abstraction
# ---------------------------------------------------------------------------

class AgentTool(ABC):
    """Abstract base class for all explicitly allowlisted agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """High-level description of tool purpose and capabilities."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema dictionary describing the expected input parameters."""
        ...

    @abstractmethod
    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize arguments. Raises InvalidToolArgumentError on failure."""
        ...

    @abstractmethod
    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        """Execute the tool within the verified security and tenant context."""
        ...

    def get_definition(self) -> ToolDefinition:
        """Return the ToolDefinition model for this tool."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters_schema=self.parameters_schema,
        )


# ---------------------------------------------------------------------------
# Argument Schemas (Pydantic V2)
# ---------------------------------------------------------------------------

def _clean_uuid_str(v: Any, field_name: str) -> str:
    if not isinstance(v, (str, UUID)):
        raise ValueError(f"{field_name} must be a valid UUID string.")
    try:
        return str(UUID(str(v).strip()))
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid {field_name}: must be a valid UUID.")


class SearchLiteratureInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    query: str = Field(..., min_length=1, max_length=1000, description="Semantic research question.")
    collection_id: str | None = Field(default=None, description="Optional UUID of collection to restrict search.")
    paper_id: str | None = Field(default=None, description="Optional UUID of paper to restrict search.")
    top_k: int | None = Field(default=5, ge=1, le=20, description="Max candidate chunks to retrieve.")

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, v: str | None) -> str | None:
        return _clean_uuid_str(v, "collection_id") if v is not None else None

    @field_validator("paper_id")
    @classmethod
    def validate_paper_id(cls, v: str | None) -> str | None:
        return _clean_uuid_str(v, "paper_id") if v is not None else None


class AnalyzePaperInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    paper_id: str = Field(..., description="UUID of the research paper to analyze.")

    @field_validator("paper_id")
    @classmethod
    def validate_paper_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "paper_id")


class ComparePapersInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    paper_ids: list[str] = Field(..., min_length=2, max_length=5, description="UUIDs of 2 to 5 papers to compare.")
    query: str | None = Field(default=None, max_length=500, description="Optional focus query for comparative synthesis.")

    @field_validator("paper_ids")
    @classmethod
    def validate_paper_ids(cls, v: list[str]) -> list[str]:
        cleaned = [_clean_uuid_str(pid, "paper_id") for pid in v]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("paper_ids must contain unique UUIDs with no duplicates.")
        return cleaned


class DiscoverResearchGapsInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str | None = Field(default=None, description="Optional collection UUID.")
    paper_ids: list[str] | None = Field(default=None, max_length=10, description="Optional list of paper UUIDs (1-10).")

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, v: str | None) -> str | None:
        return _clean_uuid_str(v, "collection_id") if v is not None else None

    @field_validator("paper_ids")
    @classmethod
    def validate_paper_ids(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        if len(v) == 0:
            raise ValueError("paper_ids list cannot be empty when provided.")
        cleaned = [_clean_uuid_str(pid, "paper_id") for pid in v]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("paper_ids must contain unique UUIDs.")
        return cleaned


class AnalyzeTrendsInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str = Field(..., description="UUID of the collection to analyze.")

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "collection_id")


class SynthesizeLiteratureInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str = Field(..., description="UUID of the collection to synthesize.")
    topic: str | None = Field(default=None, max_length=200, description="Optional focus topic.")

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "collection_id")


class VerifyClaimInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    claim: str = Field(..., min_length=1, max_length=1000, description="Scientific assertion to verify.")
    paper_id: str | None = Field(default=None, description="Optional paper UUID scoping the claim.")
    evidence_text: str | None = Field(default=None, max_length=5000, description="Optional raw evidence text.")

    @field_validator("paper_id")
    @classmethod
    def validate_paper_id(cls, v: str | None) -> str | None:
        return _clean_uuid_str(v, "paper_id") if v is not None else None


class SearchGraphEntitiesInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str = Field(..., description="UUID of the authorized collection to search.")
    query: str | None = Field(default=None, max_length=200, description="Optional entity name substring to search.")
    entity_type: str | None = Field(default=None, max_length=50, description="Optional entity classification filter (author, concept, dataset, method, metric, other).")
    limit: int | None = Field(default=10, ge=1, le=20, description="Maximum number of entities to return (maximum 20).")

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "collection_id")

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip().lower()
        if not cleaned:
            return None
        valid_types = ("author", "concept", "dataset", "method", "metric", "other")
        if cleaned not in valid_types:
            raise ValueError(f"Invalid entity_type: '{v}'. Must be one of: {list(valid_types)}")
        return cleaned


class ExploreGraphNeighborhoodInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str = Field(..., description="UUID of the authorized collection.")
    node_id: str = Field(..., description="UUID of the seed entity or paper.")
    node_type: str = Field(..., description="Type of seed node ('entity' or 'paper').")
    max_depth: int | None = Field(default=1, ge=1, le=2, description="Exploration depth (1 or 2; agent hard ceiling: 2).")
    max_nodes: int | None = Field(default=15, ge=1, le=25, description="Maximum total nodes to discover (maximum 25).")
    max_edges: int | None = Field(default=30, ge=1, le=50, description="Maximum total edges to discover (maximum 50).")
    direction: str | None = Field(default="both", description="Edge traversal direction ('outgoing', 'incoming', 'both').")
    relationship_types: list[str] | None = Field(default=None, max_length=20, description="Optional relationship types filter.")
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Minimum edge confidence score.")

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "collection_id")

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "node_id")

    @field_validator("node_type")
    @classmethod
    def validate_node_type(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("node_type must be a string.")
        cleaned = v.strip().lower()
        if cleaned not in ("entity", "paper"):
            raise ValueError("node_type must be either 'entity' or 'paper'.")
        return cleaned

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str | None) -> str:
        if v is None:
            return "both"
        cleaned = v.strip().lower()
        if cleaned not in ("outgoing", "incoming", "both"):
            raise ValueError("direction must be 'outgoing', 'incoming', or 'both'.")
        return cleaned

    @field_validator("relationship_types")
    @classmethod
    def validate_relationship_types(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        if not isinstance(v, (list, tuple)):
            raise ValueError("relationship_types must be a list of strings.")
        if len(v) == 0:
            raise ValueError("relationship_types cannot be an empty list; omit the field to include all types.")
        cleaned = []
        for r in v:
            if not isinstance(r, str) or not r.strip():
                raise ValueError("relationship_types entries must be non-empty strings.")
            cleaned_r = r.strip().lower()
            if len(cleaned_r) > 100:
                raise ValueError("relationship_types entries must not exceed 100 characters.")
            cleaned.append(cleaned_r)
        return cleaned


class FindGraphPathInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str = Field(..., description="UUID of the authorized collection.")
    source_id: str = Field(..., description="Source node UUID.")
    source_type: str = Field(..., description="Source node type ('entity' or 'paper').")
    target_id: str = Field(..., description="Target node UUID.")
    target_type: str = Field(..., description="Target node type ('entity' or 'paper').")
    max_depth: int | None = Field(default=2, ge=1, le=3, description="Maximum traversal depth (maximum 3).")
    max_paths: int | None = Field(default=2, ge=1, le=3, description="Maximum paths to discover (maximum 3).")
    direction: str | None = Field(default="outgoing", description="Traversal direction ('outgoing', 'incoming', 'both').")
    relationship_types: list[str] | None = Field(default=None, max_length=20, description="Optional relationship types filter.")
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Minimum edge confidence score.")

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "collection_id")

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "source_id")

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "target_id")

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("source_type must be a string.")
        cleaned = v.strip().lower()
        if cleaned not in ("entity", "paper"):
            raise ValueError("source_type must be either 'entity' or 'paper'.")
        return cleaned

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("target_type must be a string.")
        cleaned = v.strip().lower()
        if cleaned not in ("entity", "paper"):
            raise ValueError("target_type must be either 'entity' or 'paper'.")
        return cleaned

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str | None) -> str:
        if v is None:
            return "outgoing"
        cleaned = v.strip().lower()
        if cleaned not in ("outgoing", "incoming", "both"):
            raise ValueError("direction must be 'outgoing', 'incoming', or 'both'.")
        return cleaned

    @field_validator("relationship_types")
    @classmethod
    def validate_relationship_types(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        if not isinstance(v, (list, tuple)):
            raise ValueError("relationship_types must be a list of strings.")
        if len(v) == 0:
            raise ValueError("relationship_types cannot be an empty list; omit the field to include all types.")
        cleaned = []
        for r in v:
            if not isinstance(r, str) or not r.strip():
                raise ValueError("relationship_types entries must be non-empty strings.")
            cleaned_r = r.strip().lower()
            if len(cleaned_r) > 100:
                raise ValueError("relationship_types entries must not exceed 100 characters.")
            cleaned.append(cleaned_r)
        return cleaned


class QueryGraphAnalyticsInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str = Field(..., description="UUID of the authorized collection.")
    query_type: str = Field(default="overview", description="Analytics query type: 'overview', 'summary', 'collaborations', or 'connectivity'.")
    top_n: int | None = Field(default=5, ge=1, le=15, description="Maximum top rankings to return (maximum 15).")
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Minimum relationship confidence threshold.")
    include_unscored: bool = Field(default=False, description="Whether to include unscored relationships when threshold is active.")
    author_id: str | None = Field(default=None, description="Optional author UUID to filter collaborations or connectivity.")

    @field_validator("collection_id")
    @classmethod
    def validate_collection_id(cls, v: str) -> str:
        return _clean_uuid_str(v, "collection_id")

    @field_validator("author_id")
    @classmethod
    def validate_author_id(cls, v: str | None) -> str | None:
        return _clean_uuid_str(v, "author_id") if v is not None else None

    @field_validator("query_type")
    @classmethod
    def validate_query_type(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("query_type must be a string.")
        cleaned = v.strip().lower()
        valid_queries = ("overview", "summary", "collaborations", "connectivity")
        if cleaned not in valid_queries:
            raise ValueError(f"Invalid query_type: '{v}'. Must be one of: {list(valid_queries)}")
        return cleaned

    @model_validator(mode="after")
    def validate_author_id_usage(self) -> QueryGraphAnalyticsInput:
        if self.author_id is not None and self.query_type in ("overview", "summary"):
            raise ValueError(
                f"author_id filter is not supported for '{self.query_type}' query type; "
                "author_id is only valid for 'collaborations' and 'connectivity'."
            )
        return self


# ---------------------------------------------------------------------------
# Authorization Helpers
# ---------------------------------------------------------------------------

def _enforce_tenant_context(context: AgentContext) -> None:
    """Validate that context contains required authorization primitives if auth is enabled."""
    if not isinstance(context, AgentContext):
        raise AgentSecurityError("Operation requires a valid AgentContext.")
    if context.auth_enforced:
        if context.user_id is None:
            raise AgentSecurityError("Authentication required: missing authenticated user context.")
        if context.auth_service is None:
            raise AgentSecurityError("Authorization service unavailable.")


def _verify_paper_access(context: AgentContext, paper_id: str) -> None:
    """Verify caller owns the specified paper."""
    if not context.auth_enforced or context.auth_service is None:
        return
    if context.user_id is None:
        raise AgentSecurityError("Authentication required.")
    context.auth_service.verify_paper_ownership(UUID(paper_id), context.user_id)


def _verify_collection_access(context: AgentContext, collection_id: str) -> None:
    """Verify caller owns the specified collection."""
    if not context.auth_enforced or context.auth_service is None:
        return
    if context.user_id is None:
        raise AgentSecurityError("Authentication required.")
    context.auth_service.verify_collection_ownership(UUID(collection_id), context.user_id)


def _verify_graph_collection_access(context: AgentContext, collection_id: str | UUID) -> UUID:
    """Verify that the caller is authenticated and owns the specified collection for graph access.

    Knowledge graph data is strictly collection-scoped and tenant-private.
    Security invariants:
    1. Missing context.user_id must fail closed with AgentSecurityError.
    2. Missing context.auth_service must fail closed with AgentSecurityError.
    3. auth_enforced=False is NEVER an authorization bypass.
    4. Successful verification strictly calls context.auth_service.verify_collection_ownership(col_uuid, context.user_id).
    """
    if not isinstance(context, AgentContext):
        raise AgentSecurityError("Operation requires a valid AgentContext.")

    if context.user_id is None:
        raise AgentSecurityError(
            "Authentication required: collection knowledge graph data is private and cannot be accessed anonymously."
        )

    if context.auth_service is None:
        raise AgentSecurityError(
            "Authorization service unavailable: cannot verify collection ownership."
        )

    if not isinstance(collection_id, UUID):
        try:
            col_uuid = UUID(str(collection_id).strip())
        except (ValueError, AttributeError):
            raise InvalidToolArgumentError(f"collection_id must be a valid UUID, got {collection_id!r}")
    else:
        col_uuid = collection_id

    context.auth_service.verify_collection_ownership(col_uuid, context.user_id)
    return col_uuid


# ---------------------------------------------------------------------------
# Concrete Agent Tools
# ---------------------------------------------------------------------------

class SearchLiteratureTool(AgentTool):
    """Tool wrapping semantic paper retrieval and evidence-grounded answer generation."""

    def __init__(self, query_service: Any) -> None:
        if not callable(getattr(query_service, "ask", None)) and not callable(
            getattr(query_service, "ask_with_context", None)
        ):
            raise TypeError("query_service must provide ask or ask_with_context")
        self._service = query_service

    @property
    def name(self) -> str:
        return "search_literature"

    @property
    def description(self) -> str:
        return "Search indexed literature using semantic query and return grounded evidence chunks and answer."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return SearchLiteratureInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = SearchLiteratureInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            _enforce_tenant_context(context)
            args = SearchLiteratureInput.model_validate(arguments)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except AgentSecurityError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="unauthorized")

        try:
            kwargs: dict[str, Any] = {}
            if args.collection_id is not None:
                _verify_collection_access(context, args.collection_id)
                kwargs["collection_id"] = args.collection_id
            elif args.paper_id is not None:
                _verify_paper_access(context, args.paper_id)
                kwargs["paper_id"] = args.paper_id
            elif context.auth_enforced and context.auth_service is not None and context.user_id is not None:
                allowed_cols = context.auth_service.get_user_collection_ids(context.user_id)
                if not allowed_cols:
                    dur = (time.perf_counter() - start_time) * 1000
                    return ToolResult.ok(
                        self.name,
                        data={"answer": "No indexed research collections found for this user.", "sources": []},
                        summary="No collections found for user.",
                        evidence_labels=[],
                        duration_ms=dur,
                    )
                kwargs["collection_ids"] = [str(cid) for cid in allowed_cols]

            ans = self._service.ask(args.query, **kwargs)
            dur = (time.perf_counter() - start_time) * 1000
            citations = getattr(ans, "citations", [])
            summary = (ans.text[:300] + "...") if len(ans.text) > 300 else ans.text
            return ToolResult.ok(
                self.name,
                data={
                    "text": ans.text,
                    "evidence_count": getattr(ans, "evidence_count", 0),
                    "model": getattr(ans, "model", "default"),
                    "citations": citations,
                },
                summary=summary,
                evidence_labels=citations,
                duration_ms=dur,
            )
        except (DocumentNotFoundError, LookupError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except Exception as exc:
            return ToolResult.failure(self.name, f"Search failed: {type(exc).__name__}", error_type="execution_error")


class AnalyzePaperTool(AgentTool):
    """Tool wrapping structured paper analysis across 7 standardized research dimensions."""

    def __init__(self, analysis_service: Any) -> None:
        if not callable(getattr(analysis_service, "analyze_paper", None)):
            raise TypeError("analysis_service must provide analyze_paper")
        self._service = analysis_service

    @property
    def name(self) -> str:
        return "analyze_paper"

    @property
    def description(self) -> str:
        return "Perform structured research analysis on a specific paper across 7 standardized dimensions."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return AnalyzePaperInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = AnalyzePaperInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            _enforce_tenant_context(context)
            args = AnalyzePaperInput.model_validate(arguments)
            _verify_paper_access(context, args.paper_id)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except (AgentSecurityError, ForbiddenError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            res = self._service.analyze_paper(args.paper_id)
            dur = (time.perf_counter() - start_time) * 1000

            data = res.to_dict() if hasattr(res, "to_dict") else res
            evidence_labels: list[str] = []
            if hasattr(res, "dimensions") and isinstance(res.dimensions, dict):
                for fld in res.dimensions.values():
                    if hasattr(fld, "evidence_labels"):
                        evidence_labels.extend(fld.evidence_labels)

            summary = f"Analysis of paper {args.paper_id}: title='{getattr(res, 'title', None)}', evidence_count={getattr(res, 'evidence_count', 0)}"
            return ToolResult.ok(
                self.name,
                data=data,
                summary=summary,
                evidence_labels=list(dict.fromkeys(evidence_labels)),
                duration_ms=dur,
            )
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except Exception as exc:
            return ToolResult.failure(self.name, f"Analysis failed: {type(exc).__name__}", error_type="execution_error")


class ComparePapersTool(AgentTool):
    """Tool wrapping multi-paper comparative synthesis across 2 to 5 papers."""

    def __init__(self, comparison_service: Any) -> None:
        if not callable(getattr(comparison_service, "compare_cross_paper", None)) and not callable(
            getattr(comparison_service, "compare_papers", None)
        ):
            raise TypeError("comparison_service must provide compare_cross_paper or compare_papers")
        self._service = comparison_service

    @property
    def name(self) -> str:
        return "compare_papers"

    @property
    def description(self) -> str:
        return "Perform multi-paper comparative synthesis across 2 to 5 papers."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return ComparePapersInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = ComparePapersInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            _enforce_tenant_context(context)
            args = ComparePapersInput.model_validate(arguments)
            for pid in args.paper_ids:
                _verify_paper_access(context, pid)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except (AgentSecurityError, ForbiddenError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            if callable(getattr(self._service, "compare_cross_paper", None)):
                res = self._service.compare_cross_paper(args.paper_ids, query=args.query)
            else:
                res = self._service.compare_papers(args.paper_ids)
            dur = (time.perf_counter() - start_time) * 1000

            data = res.to_dict() if hasattr(res, "to_dict") else res
            summary = (
                f"Compared {len(args.paper_ids)} papers. Conclusion: "
                f"{getattr(res, 'conclusion', getattr(res, 'overall_summary', 'completed'))}"
            )
            return ToolResult.ok(
                self.name,
                data=data,
                summary=summary,
                evidence_labels=[],
                duration_ms=dur,
            )
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except Exception as exc:
            return ToolResult.failure(self.name, f"Comparison failed: {type(exc).__name__}", error_type="execution_error")


class DiscoverResearchGapsTool(AgentTool):
    """Tool wrapping evidence-grounded research gap and limitation discovery."""

    def __init__(self, gap_service: Any) -> None:
        if not callable(getattr(gap_service, "discover_gaps_for_collection", None)) and not callable(
            getattr(gap_service, "discover_gaps_for_papers", None)
        ):
            raise TypeError("gap_service must provide discover_gaps_for_collection or discover_gaps_for_papers")
        self._service = gap_service

    @property
    def name(self) -> str:
        return "discover_research_gaps"

    @property
    def description(self) -> str:
        return "Discover documented limitations and future-work research gaps for a collection or paper list."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return DiscoverResearchGapsInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = DiscoverResearchGapsInput.model_validate(arguments)
            if parsed.collection_id is None and parsed.paper_ids is None:
                raise InvalidToolArgumentError("At least one of collection_id or paper_ids must be provided.")
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            _enforce_tenant_context(context)
            args = DiscoverResearchGapsInput.model_validate(arguments)
            if args.collection_id is None and args.paper_ids is None:
                return ToolResult.failure(
                    self.name, "At least one of collection_id or paper_ids must be provided.", error_type="validation_error"
                )
            if args.collection_id is not None:
                _verify_collection_access(context, args.collection_id)
            if args.paper_ids is not None:
                for pid in args.paper_ids:
                    _verify_paper_access(context, pid)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except (AgentSecurityError, ForbiddenError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            if args.collection_id is not None:
                report = self._service.discover_gaps_for_collection(args.collection_id)
            else:
                assert args.paper_ids is not None
                report = self._service.discover_gaps_for_papers(args.paper_ids)
            dur = (time.perf_counter() - start_time) * 1000

            data = report.to_dict() if hasattr(report, "to_dict") else report
            gaps_count = len(getattr(report, "gaps", []))
            summary = f"Discovered {gaps_count} research gaps ({getattr(report, 'documented_limitations_count', 0)} limitations, {getattr(report, 'future_work_count', 0)} future directions)."
            evidence_labels: list[str] = []
            if hasattr(report, "gaps"):
                for g in report.gaps:
                    evidence_labels.extend(getattr(g, "evidence_labels", []))

            return ToolResult.ok(
                self.name,
                data=data,
                summary=summary,
                evidence_labels=list(dict.fromkeys(evidence_labels)),
                duration_ms=dur,
            )
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except Exception as exc:
            return ToolResult.failure(self.name, f"Gap discovery failed: {type(exc).__name__}", error_type="execution_error")


class AnalyzeTrendsTool(AgentTool):
    """Tool wrapping temporal topic, methodology, and metric trend analysis across collections."""

    def __init__(self, trend_service: Any) -> None:
        if not callable(getattr(trend_service, "analyze_collection_trends", None)):
            raise TypeError("trend_service must provide analyze_collection_trends")
        self._service = trend_service

    @property
    def name(self) -> str:
        return "analyze_trends"

    @property
    def description(self) -> str:
        return "Analyze temporal topic and methodology trends across papers in a collection over time."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return AnalyzeTrendsInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = AnalyzeTrendsInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            _enforce_tenant_context(context)
            args = AnalyzeTrendsInput.model_validate(arguments)
            _verify_collection_access(context, args.collection_id)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except (AgentSecurityError, ForbiddenError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            report = self._service.analyze_collection_trends(args.collection_id)
            dur = (time.perf_counter() - start_time) * 1000

            data = report.to_dict() if hasattr(report, "to_dict") else report
            trends_count = len(getattr(report, "trends", []))
            summary = f"Identified {trends_count} trends across {len(getattr(report, 'temporal_buckets', []))} temporal periods."
            evidence_labels: list[str] = []
            if hasattr(report, "trends"):
                for t in report.trends:
                    evidence_labels.extend(getattr(t, "evidence_labels", []))

            return ToolResult.ok(
                self.name,
                data=data,
                summary=summary,
                evidence_labels=list(dict.fromkeys(evidence_labels)),
                duration_ms=dur,
            )
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except Exception as exc:
            return ToolResult.failure(self.name, f"Trend analysis failed: {type(exc).__name__}", error_type="execution_error")


class SynthesizeLiteratureTool(AgentTool):
    """Tool wrapping evidence-grounded literature review synthesis across collections."""

    def __init__(self, trend_service: Any) -> None:
        if not callable(getattr(trend_service, "synthesize_collection_literature", None)):
            raise TypeError("trend_service must provide synthesize_collection_literature")
        self._service = trend_service

    @property
    def name(self) -> str:
        return "synthesize_literature"

    @property
    def description(self) -> str:
        return "Synthesize an evidence-grounded literature review across papers in a collection."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return SynthesizeLiteratureInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = SynthesizeLiteratureInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            _enforce_tenant_context(context)
            args = SynthesizeLiteratureInput.model_validate(arguments)
            _verify_collection_access(context, args.collection_id)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except (AgentSecurityError, ForbiddenError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            synthesis = self._service.synthesize_collection_literature(
                args.collection_id, topic=args.topic
            )
            dur = (time.perf_counter() - start_time) * 1000

            data = synthesis.to_dict() if hasattr(synthesis, "to_dict") else synthesis
            findings_count = len(getattr(synthesis, "findings", []))
            divergences_count = len(getattr(synthesis, "divergences", []))
            summary = f"Synthesized literature on '{getattr(synthesis, 'topic', args.topic or 'Collection')}': {findings_count} findings, {divergences_count} divergences."
            evidence_labels: list[str] = []
            if hasattr(synthesis, "findings"):
                for f in synthesis.findings:
                    evidence_labels.extend(getattr(f, "evidence_labels", []))

            return ToolResult.ok(
                self.name,
                data=data,
                summary=summary,
                evidence_labels=list(dict.fromkeys(evidence_labels)),
                duration_ms=dur,
            )
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except Exception as exc:
            return ToolResult.failure(self.name, f"Literature synthesis failed: {type(exc).__name__}", error_type="execution_error")


class VerifyClaimTool(AgentTool):
    """Tool wrapping evidence grounding verification for scientific assertions."""

    def __init__(self, verifier: Any, session_factory: Any = None) -> None:
        if not callable(getattr(verifier, "verify_claim", None)):
            raise TypeError("verifier must provide verify_claim")
        self._verifier = verifier
        self._session_factory = session_factory

    @property
    def name(self) -> str:
        return "verify_claim"

    @property
    def description(self) -> str:
        return "Verify whether a scientific claim is supported, contradicted, or insufficient relative to cited evidence."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return VerifyClaimInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = VerifyClaimInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            _enforce_tenant_context(context)
            args = VerifyClaimInput.model_validate(arguments)
            if args.paper_id is not None:
                _verify_paper_access(context, args.paper_id)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except (AgentSecurityError, ForbiddenError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            cited_labels = extract_citations(args.claim)
            evidence_items: list[EvidenceItem] = []

            # 1. Build evidence context from explicit text if provided
            if args.evidence_text:
                for idx, lbl in enumerate(cited_labels or ["[E1]"]):
                    evidence_items.append(
                        EvidenceItem(
                            document_id="inline_evidence",
                            chunk_id=f"chunk_{idx}",
                            text=args.evidence_text,
                            score=1.0,
                            metadata={"paper_id": args.paper_id or "inline_paper"},
                            position=idx,
                        )
                    )
            elif args.paper_id and self._session_factory:
                # Load chunks from relational DB
                session = self._session_factory()
                try:
                    from nexora.database.repository.document import DocumentRepository
                    from nexora.database.repository.document_chunk import DocumentChunkRepository
                    from nexora.database.repository.paper import PaperRepository

                    paper_uuid = UUID(args.paper_id)
                    p_repo = PaperRepository(session)
                    paper = p_repo.get_by_id(paper_uuid)
                    if paper is not None:
                        d_repo = DocumentRepository(session)
                        c_repo = DocumentChunkRepository(session)
                        docs = d_repo.get_by_paper_id(paper_uuid)
                        pos = 0
                        for doc in docs:
                            chunks = c_repo.get_by_document_id(doc.id)
                            for chunk in chunks:
                                evidence_items.append(
                                    EvidenceItem(
                                        document_id=str(doc.id),
                                        chunk_id=str(chunk.id),
                                        text=chunk.text,
                                        score=1.0,
                                        metadata={"paper_id": str(paper_uuid)},
                                        position=pos,
                                    )
                                )
                                pos += 1
                finally:
                    if hasattr(session, "close"):
                        session.close()

            evidence_ctx = EvidenceContext(query=args.claim, items=evidence_items, total_candidates=len(evidence_items))
            raw_claim = RawClaim(claim_text=args.claim, cited_labels=cited_labels)
            verification = self._verifier.verify_claim(raw_claim, evidence_ctx)
            dur = (time.perf_counter() - start_time) * 1000

            data = verification.to_dict() if hasattr(verification, "to_dict") else verification
            summary = f"Claim verification: status='{getattr(verification, 'status', 'unknown')}', reasoning='{getattr(verification, 'reasoning', '')[:150]}'"
            return ToolResult.ok(
                self.name,
                data=data,
                summary=summary,
                evidence_labels=cited_labels,
                duration_ms=dur,
            )
        except Exception as exc:
            return ToolResult.failure(self.name, f"Verification failed: {type(exc).__name__}", error_type="execution_error")


# ---------------------------------------------------------------------------
# Knowledge Graph Agent Tools (Stage 5 Step 4.2)
# ---------------------------------------------------------------------------

class SearchGraphEntitiesTool(AgentTool):
    """Tool for searching and listing entities in a collection's knowledge graph."""

    def __init__(self, session_factory: Any = None, *, repository: Any = None) -> None:
        if session_factory is None and repository is None:
            raise TypeError("session_factory or repository must be provided")
        self._session_factory = session_factory
        self._repository = repository

    @property
    def name(self) -> str:
        return "search_graph_entities"

    @property
    def description(self) -> str:
        return "Search or list entities (methods, concepts, datasets, authors) in a collection's knowledge graph."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return SearchGraphEntitiesInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = SearchGraphEntitiesInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            args = SearchGraphEntitiesInput.model_validate(arguments)
            col_uuid = _verify_graph_collection_access(context, args.collection_id)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except AgentSecurityError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="unauthorized")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            from nexora.database.repository.graph_entity import GraphEntityRepository

            eff_limit = min(args.limit or 10, 20)
            if self._repository is not None:
                if args.query and args.query.strip():
                    entities = self._repository.search_entities(
                        collection_id=col_uuid,
                        query=args.query.strip(),
                        entity_type=args.entity_type,
                        limit=eff_limit,
                    )
                else:
                    entities = self._repository.list_by_collection_id(
                        collection_id=col_uuid,
                        entity_type=args.entity_type,
                        limit=eff_limit,
                    )
            elif self._session_factory is not None:
                with self._session_factory() as session:
                    repo = GraphEntityRepository(session)
                    if args.query and args.query.strip():
                        entities = repo.search_entities(
                            collection_id=col_uuid,
                            query=args.query.strip(),
                            entity_type=args.entity_type,
                            limit=eff_limit,
                        )
                    else:
                        entities = repo.list_by_collection_id(
                            collection_id=col_uuid,
                            entity_type=args.entity_type,
                            limit=eff_limit,
                        )
            else:
                return ToolResult.failure(self.name, "Database session factory unavailable.", error_type="execution_error")

            dur = (time.perf_counter() - start_time) * 1000
            data = [
                {
                    "id": str(e.id),
                    "collection_id": str(e.collection_id),
                    "name": e.name,
                    "canonical_name": e.canonical_name,
                    "entity_type": e.entity_type,
                    "description": e.description,
                    "aliases": list(e.aliases or []),
                    "metadata": dict(e.entity_metadata or {}),
                }
                for e in entities
            ]
            summary = (
                f"Found {len(data)} entities in collection graph matching query '{args.query}'."
                if args.query and args.query.strip()
                else f"Listed {len(data)} entities in collection graph."
            )
            return ToolResult.ok(self.name, data=data, summary=summary, duration_ms=dur)
        except Exception as exc:
            return ToolResult.failure(self.name, f"Entity search failed: {type(exc).__name__}", error_type="execution_error")


class ExploreGraphNeighborhoodTool(AgentTool):
    """Tool for exploring bounded 1- or 2-hop local graph topology around an entity or paper."""

    def __init__(self, traversal_service: Any) -> None:
        if not callable(getattr(traversal_service, "explore_multihop", None)):
            raise TypeError("traversal_service must provide explore_multihop")
        self._service = traversal_service

    @property
    def name(self) -> str:
        return "explore_graph_neighborhood"

    @property
    def description(self) -> str:
        return "Explore bounded 1- or 2-hop local graph topology around an entity or paper with full evidence grounding."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return ExploreGraphNeighborhoodInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = ExploreGraphNeighborhoodInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            args = ExploreGraphNeighborhoodInput.model_validate(arguments)
            col_uuid = _verify_graph_collection_access(context, args.collection_id)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except AgentSecurityError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="unauthorized")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            subgraph = self._service.explore_multihop(
                start_id=UUID(args.node_id),
                start_type=args.node_type,
                collection_id=col_uuid,
                max_depth=min(args.max_depth or 1, 2),
                max_nodes=min(args.max_nodes or 15, 25),
                max_edges=min(args.max_edges or 30, 50),
                direction=args.direction or "both",
                relationship_types=args.relationship_types,
                min_confidence=args.min_confidence,
                include_unscored=False,
            )
            dur = (time.perf_counter() - start_time) * 1000

            nodes_data = [
                {
                    "id": str(n.id),
                    "node_type": n.node_type.value if hasattr(n.node_type, "value") else str(n.node_type),
                    "name": n.name,
                    "entity_type": n.entity_type,
                    "description": getattr(n, "description", (n.metadata or {}).get("description") if getattr(n, "metadata", None) else None),
                    "metadata": dict(n.metadata or {}),
                }
                for n in subgraph.nodes
            ]
            edges_data = [
                {
                    "id": str(e.id),
                    "source_type": e.source_type.value if hasattr(e.source_type, "value") else str(e.source_type),
                    "source_id": str(e.source_id),
                    "target_type": e.target_type.value if hasattr(e.target_type, "value") else str(e.target_type),
                    "target_id": str(e.target_id),
                    "relationship_type": e.relationship_type,
                    "confidence": e.confidence,
                    "evidence_text": e.evidence_text,
                    "chunk_id": str(e.chunk_id) if e.chunk_id else None,
                    "paper_id": str(e.paper_id) if e.paper_id else None,
                    "page_number": e.page_number,
                    "extraction_method": e.extraction_method,
                }
                for e in subgraph.edges
            ]

            data = {
                "seed_id": str(subgraph.seed_id),
                "seed_type": subgraph.seed_type.value if hasattr(subgraph.seed_type, "value") else str(subgraph.seed_type),
                "depth": subgraph.depth,
                "nodes": nodes_data,
                "edges": edges_data,
                "is_truncated": subgraph.is_truncated,
                "truncation_reason": subgraph.truncation_reason,
            }
            summary = (
                f"Explored neighborhood around {args.node_type} '{args.node_id}': "
                f"{len(nodes_data)} nodes, {len(edges_data)} edges (depth: {subgraph.depth}, "
                f"truncated: {subgraph.is_truncated})."
            )
            return ToolResult.ok(
                self.name,
                data=data,
                summary=summary,
                evidence_labels=[],
                duration_ms=dur,
            )
        except Exception as exc:
            return ToolResult.failure(self.name, f"Neighborhood exploration failed: {type(exc).__name__}", error_type="execution_error")


class FindGraphPathTool(AgentTool):
    """Tool for finding multi-hop connecting relationship paths between two nodes."""

    def __init__(self, traversal_service: Any) -> None:
        if not callable(getattr(traversal_service, "find_paths", None)):
            raise TypeError("traversal_service must provide find_paths")
        self._service = traversal_service

    @property
    def name(self) -> str:
        return "find_graph_path"

    @property
    def description(self) -> str:
        return "Discover multi-hop connecting paths between two entities or papers within an authorized collection."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return FindGraphPathInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = FindGraphPathInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            args = FindGraphPathInput.model_validate(arguments)
            col_uuid = _verify_graph_collection_access(context, args.collection_id)
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except AgentSecurityError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="unauthorized")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            paths = self._service.find_paths(
                source_id=UUID(args.source_id),
                source_type=args.source_type,
                target_id=UUID(args.target_id),
                target_type=args.target_type,
                collection_id=col_uuid,
                max_depth=min(args.max_depth or 2, 3),
                max_paths=min(args.max_paths or 2, 3),
                direction=args.direction or "outgoing",
                relationship_types=args.relationship_types,
                min_confidence=args.min_confidence,
                include_unscored=False,
            )
            dur = (time.perf_counter() - start_time) * 1000

            paths_data = []
            for p in paths:
                nodes_data = [
                    {
                        "id": str(n.id),
                        "node_type": n.node_type.value if hasattr(n.node_type, "value") else str(n.node_type),
                        "name": n.name,
                        "entity_type": n.entity_type,
                    }
                    for n in p.nodes
                ]
                edges_data = [
                    {
                        "id": str(e.id),
                        "source_type": e.source_type.value if hasattr(e.source_type, "value") else str(e.source_type),
                        "source_id": str(e.source_id),
                        "target_type": e.target_type.value if hasattr(e.target_type, "value") else str(e.target_type),
                        "target_id": str(e.target_id),
                        "relationship_type": e.relationship_type,
                        "confidence": e.confidence,
                        "evidence_text": e.evidence_text,
                        "chunk_id": str(e.chunk_id) if e.chunk_id else None,
                        "paper_id": str(e.paper_id) if e.paper_id else None,
                        "page_number": e.page_number,
                    }
                    for e in p.edges
                ]

                paths_data.append(
                    {
                        "hops": getattr(p, "hops", getattr(p, "length", len(p.edges))),
                        "length": getattr(p, "length", getattr(p, "hops", len(p.edges))),
                        "cumulative_confidence": getattr(p, "cumulative_confidence", getattr(p, "confidence_score", None)),
                        "confidence_score": getattr(p, "confidence_score", getattr(p, "cumulative_confidence", None)),
                        "nodes": nodes_data,
                        "edges": edges_data,
                    }
                )

            summary = (
                f"Discovered {len(paths_data)} path(s) connecting {args.source_type} '{args.source_id}' "
                f"to {args.target_type} '{args.target_id}'."
            )
            return ToolResult.ok(
                self.name,
                data=paths_data,
                summary=summary,
                evidence_labels=[],
                duration_ms=dur,
            )
        except Exception as exc:
            return ToolResult.failure(self.name, f"Path finding failed: {type(exc).__name__}", error_type="execution_error")


class QueryGraphAnalyticsTool(AgentTool):
    """Tool for querying collection-wide knowledge graph statistics, author collaboration networks, or connectivity metrics."""

    def __init__(self, analytics_service: Any) -> None:
        if (
            not callable(getattr(analytics_service, "get_collection_analytics", None))
            or not callable(getattr(analytics_service, "get_author_collaborations", None))
            or not callable(getattr(analytics_service, "get_author_connectivity", None))
            or not callable(getattr(analytics_service, "get_summary", None))
        ):
            raise TypeError("analytics_service must provide analytics methods")
        self._service = analytics_service

    @property
    def name(self) -> str:
        return "query_graph_analytics"

    @property
    def description(self) -> str:
        return "Query collection-wide knowledge graph statistics, author collaboration networks, or connectivity metrics."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return QueryGraphAnalyticsInput.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = QueryGraphAnalyticsInput.model_validate(arguments)
            return parsed.model_dump(exclude_unset=True)
        except ValidationError as exc:
            raise InvalidToolArgumentError(str(exc)) from exc

    def execute(self, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        start_time = time.perf_counter()
        try:
            args = QueryGraphAnalyticsInput.model_validate(arguments)
            col_uuid = _verify_graph_collection_access(context, args.collection_id)
            if args.author_id is not None and args.query_type in ("overview", "summary"):
                raise InvalidToolArgumentError(
                    f"author_id filter is not supported for '{args.query_type}' query type; "
                    "author_id is only valid for 'collaborations' and 'connectivity'."
                )
        except (ValidationError, InvalidToolArgumentError) as exc:
            return ToolResult.failure(self.name, str(exc), error_type="validation_error")
        except AgentSecurityError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="unauthorized")
        except ForbiddenError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="forbidden")
        except DocumentNotFoundError as exc:
            return ToolResult.failure(self.name, str(exc), error_type="not_found")

        try:
            top_n = min(args.top_n or 5, 15)
            author_uuid = UUID(args.author_id) if args.author_id else None

            if args.query_type == "overview":
                res = self._service.get_collection_analytics(
                    collection_id=col_uuid,
                    top_collaborations_limit=top_n,
                    author_connectivity_limit=top_n,
                    min_confidence=args.min_confidence,
                    include_unscored=args.include_unscored,
                )
                data = {
                    "collection_id": str(res.collection_id),
                    "summary": {
                        "total_entities": res.summary.total_entities,
                        "total_relationships": res.summary.total_relationships,
                        "entity_counts_by_type": dict(res.summary.entity_counts_by_type),
                        "relationship_counts_by_type": dict(res.summary.relationship_counts_by_type),
                    },
                    "entity_distribution": [
                        {"entity_type": d.entity_type, "count": d.count}
                        for d in res.entity_distribution
                    ],
                    "relationship_distribution": [
                        {"relationship_type": d.relationship_type, "count": d.count}
                        for d in res.relationship_distribution
                    ],
                    "top_collaborations": [
                        {
                            "author_id": str(c.author_id),
                            "author_name": c.author_name,
                            "collaborator_id": str(c.collaborator_id),
                            "collaborator_name": c.collaborator_name,
                            "collaboration_count": c.collaboration_count,
                            "shared_paper_ids": [str(p) for p in c.shared_paper_ids],
                        }
                        for c in res.top_collaborations
                    ],
                    "author_connectivity": [
                        {
                            "author_id": str(m.author_id),
                            "author_name": m.author_name,
                            "paper_count": m.paper_count,
                            "collaborator_count": getattr(m, "collaborator_count", getattr(m, "coauthor_count", 0)),
                            "total_collaborations": getattr(m, "total_collaborations", getattr(m, "total_degree", 0)),
                            "coauthor_count": getattr(m, "collaborator_count", getattr(m, "coauthor_count", 0)),
                            "total_degree": getattr(m, "total_collaborations", getattr(m, "total_degree", 0)),
                        }
                        for m in res.author_connectivity
                    ],
                }
                summary = (
                    f"Collection graph overview: {res.summary.total_entities} entities, "
                    f"{res.summary.total_relationships} relationships across {len(res.entity_distribution)} entity types."
                )

            elif args.query_type == "summary":
                s = self._service.get_summary(collection_id=col_uuid)
                data = {
                    "collection_id": str(s.collection_id),
                    "total_entities": s.total_entities,
                    "total_relationships": s.total_relationships,
                    "entity_counts_by_type": dict(s.entity_counts_by_type),
                    "relationship_counts_by_type": dict(s.relationship_counts_by_type),
                }
                summary = f"Collection graph summary: {s.total_entities} entities, {s.total_relationships} relationships."

            elif args.query_type == "collaborations":
                collabs = self._service.get_author_collaborations(
                    collection_id=col_uuid,
                    author_id=author_uuid,
                    min_shared_papers=1,
                    min_confidence=args.min_confidence,
                    include_unscored=args.include_unscored,
                    limit=top_n,
                )
                data = [
                    {
                        "author_id": str(c.author_id),
                        "author_name": c.author_name,
                        "collaborator_id": str(c.collaborator_id),
                        "collaborator_name": c.collaborator_name,
                        "collaboration_count": c.collaboration_count,
                        "shared_paper_ids": [str(p) for p in c.shared_paper_ids],
                    }
                    for c in collabs
                ]
                summary = f"Retrieved {len(data)} author collaboration pairs (top_n={top_n})."

            elif args.query_type == "connectivity":
                author_ids = [author_uuid] if author_uuid else None
                conn = self._service.get_author_connectivity(
                    collection_id=col_uuid,
                    author_ids=author_ids,
                    min_confidence=args.min_confidence,
                    include_unscored=args.include_unscored,
                    limit=top_n,
                )
                data = [
                    {
                        "author_id": str(m.author_id),
                        "author_name": m.author_name,
                        "paper_count": m.paper_count,
                        "collaborator_count": getattr(m, "collaborator_count", getattr(m, "coauthor_count", 0)),
                        "total_collaborations": getattr(m, "total_collaborations", getattr(m, "total_degree", 0)),
                        "coauthor_count": getattr(m, "collaborator_count", getattr(m, "coauthor_count", 0)),
                        "total_degree": getattr(m, "total_collaborations", getattr(m, "total_degree", 0)),
                    }
                    for m in conn
                ]
                summary = f"Retrieved {len(data)} author connectivity metrics (top_n={top_n})."
            else:
                return ToolResult.failure(self.name, f"Unsupported query_type '{args.query_type}'.", error_type="validation_error")

            dur = (time.perf_counter() - start_time) * 1000
            return ToolResult.ok(self.name, data=data, summary=summary, duration_ms=dur)
        except Exception as exc:
            return ToolResult.failure(self.name, f"Analytics query failed: {type(exc).__name__}", error_type="execution_error")


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Explicit registry of validated, allowlisted agent tools."""

    def __init__(self, tools: Sequence[AgentTool] | None = None) -> None:
        self._tools: dict[str, AgentTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: AgentTool) -> None:
        """Register an allowlisted tool. Fails on duplicate names."""
        if not isinstance(tool, AgentTool):
            raise TypeError("tool must be an instance of AgentTool")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool:
        """Retrieve a registered tool by name. Fails closed with UnknownToolError."""
        if not isinstance(name, str) or name not in self._tools:
            raise UnknownToolError(f"Tool {name!r} is not registered in the allowlist.")
        return self._tools[name]

    def has(self, name: str) -> bool:
        """Check if tool name is registered."""
        return isinstance(name, str) and name in self._tools

    def list_tools(self) -> list[ToolDefinition]:
        """Return definitions for all allowlisted tools."""
        return [tool.get_definition() for tool in self._tools.values()]

    def list_tool_names(self) -> list[str]:
        """Return list of all registered tool names."""
        return list(self._tools.keys())

    def execute(self, tool_name: str, arguments: dict[str, Any], context: AgentContext) -> ToolResult:
        """Execute a tool by name with safety, parameter validation, and tenant isolation."""
        if not isinstance(tool_name, str) or tool_name not in self._tools:
            return ToolResult.failure(
                tool_name=str(tool_name),
                error=f"Tool {tool_name!r} is not registered in allowlist.",
                error_type="unknown_tool",
            )
        tool = self._tools[tool_name]
        return tool.execute(arguments, context)


def create_default_tool_registry(
    *,
    query_service: Any = None,
    analysis_service: Any = None,
    comparison_service: Any = None,
    gap_service: Any = None,
    trend_service: Any = None,
    verifier: Any = None,
    session_factory: Any = None,
    graph_entity_repository: Any = None,
    graph_traversal_service: Any = None,
    graph_analytics_service: Any = None,
) -> ToolRegistry:
    """Factory creating a populated ToolRegistry with all available research and knowledge graph domain tools."""
    registry = ToolRegistry()

    if query_service is not None:
        registry.register(SearchLiteratureTool(query_service))
    if analysis_service is not None:
        registry.register(AnalyzePaperTool(analysis_service))
    if comparison_service is not None:
        registry.register(ComparePapersTool(comparison_service))
    if gap_service is not None:
        registry.register(DiscoverResearchGapsTool(gap_service))
    if trend_service is not None:
        registry.register(AnalyzeTrendsTool(trend_service))
        registry.register(SynthesizeLiteratureTool(trend_service))
    if verifier is not None:
        registry.register(VerifyClaimTool(verifier, session_factory=session_factory))

    # Knowledge Graph Tools (Stage 5 Step 4.2)
    # 1. SearchGraphEntitiesTool: registered if session_factory or repository is provided
    if session_factory is not None:
        registry.register(SearchGraphEntitiesTool(session_factory=session_factory))
    elif graph_entity_repository is not None:
        registry.register(SearchGraphEntitiesTool(repository=graph_entity_repository))

    # 2. Instantiate graph traversal and analytics services from session_factory if not explicitly injected
    if graph_traversal_service is None and session_factory is not None:
        try:
            from nexora.graph.traversal import KnowledgeGraphTraversalService

            graph_traversal_service = KnowledgeGraphTraversalService(session_factory=session_factory)
        except Exception as exc:
            raise InvalidConfigurationError(
                f"Failed to initialize KnowledgeGraphTraversalService with provided session_factory: {exc}"
            ) from exc

    if graph_analytics_service is None and session_factory is not None:
        try:
            from nexora.graph.analytics_service import KnowledgeGraphAnalyticsService

            graph_analytics_service = KnowledgeGraphAnalyticsService(session_factory=session_factory)
        except Exception as exc:
            raise InvalidConfigurationError(
                f"Failed to initialize KnowledgeGraphAnalyticsService with provided session_factory: {exc}"
            ) from exc

    # 3. ExploreGraphNeighborhoodTool & FindGraphPathTool: registered if traversal service is available
    if graph_traversal_service is not None:
        registry.register(ExploreGraphNeighborhoodTool(graph_traversal_service))
        registry.register(FindGraphPathTool(graph_traversal_service))

    # 4. QueryGraphAnalyticsTool: registered if analytics service is available
    if graph_analytics_service is not None:
        registry.register(QueryGraphAnalyticsTool(graph_analytics_service))

    return registry
