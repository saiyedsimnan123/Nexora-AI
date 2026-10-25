"""Explicit, allowlisted tool registry wrapping existing Nexora Stage 1–3 research services."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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
) -> ToolRegistry:
    """Factory creating a populated ToolRegistry with all available research domain tools."""
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

    return registry
