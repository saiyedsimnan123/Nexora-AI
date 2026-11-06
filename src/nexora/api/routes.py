"""HTTP endpoints. Adapts HTTP requests to application services; no business logic."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from nexora.agents.models import (
    AgentContext,
    AgentExecutionError,
    AgentSecurityError,
    InvalidActionError,
    PlannerServiceError,
)
from nexora.api.errors import (
    AgentServiceError,
    AuthenticationError,
    DocumentNotFoundError,
    FileTooLargeError,
    ForbiddenError,
    GraphServiceError,
    InvalidConfigurationError,
    InvalidRequestError,
    ResearchServiceError,
)
from nexora.api.models import (
    AgentPlanRequest,
    AgentPlanResponse,
    AgentRunRequest,
    AgentRunResponse,
    DocumentIngestResponse,
    PaperAnalysisRequest,
    PaperAnalysisResponse,
    PaperComparisonRequest,
    PaperComparisonResponse,
    ResearchCompareRequest,
    ResearchCompareResponse,
    ResearchGapsRequest,
    ResearchGapsResponse,
    ResearchTrendsRequest,
    ResearchTrendsResponse,
    LiteratureSynthesisRequest,
    LiteratureSynthesisResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchVerifyRequest,
    ResearchVerifyResponse,
)
from nexora.api.service import ResearchApplicationService
from nexora.research.models import ResearchAnswer


class ResearchRequestBody(BaseModel):
    """Wire format. Strict types only; semantic rules live in ResearchRequest."""

    model_config = ConfigDict(strict=True, extra="forbid")

    query: str
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    collection_id: str | None = None


class ResearchResponseBody(BaseModel):
    """Wire format of ResearchResponse (no provider 'raw' data)."""

    text: str
    query: str
    model: str
    usage: dict[str, int] | None = None
    evidence_count: int


class ResearchVerifyRequestBody(BaseModel):
    """Wire format of research verification request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    query: str
    text: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    expected_paper_id: str | None = None


class ResearchVerifyResponseBody(BaseModel):
    """Wire format of research verification response."""

    model_config = ConfigDict(strict=False)

    text: str
    query: str
    model: str
    usage: dict[str, int] | None = None
    evidence_count: int
    verification: dict[str, Any]


class DocumentIngestResponseBody(BaseModel):
    """Wire format of document ingestion result."""

    model_config = ConfigDict(strict=True, extra="forbid")

    document_id: str = Field(
        ...,
        description="UUID of the persisted document record.",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    paper_id: str = Field(
        ...,
        description="UUID of the owning research paper.",
        examples=["9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"],
    )
    chunks_count: int = Field(
        ...,
        ge=0,
        description="Total number of chunks stored in the relational database.",
        examples=[12],
    )
    indexed_chunks_count: int = Field(
        ...,
        ge=0,
        description="Total number of chunks indexed in the vector index.",
        examples=[12],
    )
    graph_job_id: str | None = Field(
        default=None,
        description="UUID of the enqueued durable graph extraction job, if active.",
    )
    graph_status: str | None = Field(
        default=None,
        description="Status of the graph extraction job (e.g. pending, completed).",
    )


class GraphExtractionJobResponseBody(BaseModel):
    """Wire format of durable graph extraction job details."""

    model_config = ConfigDict(strict=False)

    job_id: str
    paper_id: str
    collection_id: str
    status: str
    attempts: int
    max_attempts: int
    metadata_extracted: bool
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    claimed_by: str | None = None
    lease_expires_at: str | None = None


class PaperAnalysisRequestBody(BaseModel):
    """Wire format of structured paper analysis request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    paper_id: str = Field(..., description="UUID of the research paper to analyze.")
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


class PaperAnalysisResponseBody(BaseModel):
    """Wire format of structured paper analysis response."""

    model_config = ConfigDict(strict=False)

    paper_id: str
    title: str | None = None
    problem_and_objectives: dict[str, Any]
    methodology_and_algorithms: dict[str, Any]
    models_and_datasets: dict[str, Any]
    experiments_and_metrics: dict[str, Any]
    results_and_findings: dict[str, Any]
    limitations: dict[str, Any]
    future_work: dict[str, Any]
    evidence_count: int
    model: str
    usage: dict[str, int] | None = None


class PaperComparisonRequestBody(BaseModel):
    """Wire format of multi-paper comparison request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    paper_ids: list[str] = Field(..., min_length=2, description="UUIDs of the papers to compare.")
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


class PaperComparisonResponseBody(BaseModel):
    """Wire format of multi-paper comparison response."""

    model_config = ConfigDict(strict=False)

    paper_ids: list[str]
    dimensions: dict[str, Any]
    overall_summary: str
    model: str
    usage: dict[str, int] | None = None


class ResearchCompareRequestBody(BaseModel):
    """Wire format of cross-paper comparison request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    paper_ids: list[str] = Field(..., min_length=2, max_length=5, description="UUIDs of the papers to compare.")
    query: str | None = Field(default=None, description="Optional comparison query or focus area.")
    model: str | None = Field(default=None, description="Model identifier for synthesis.")
    temperature: float | None = Field(default=None, description="Generation temperature (0.0 to 2.0).")
    max_output_tokens: int | None = Field(default=None, description="Maximum output tokens for synthesis.")


class ResearchCompareResponseBody(BaseModel):
    """Wire format of cross-paper comparison response."""

    model_config = ConfigDict(strict=False)

    paper_ids: list[str]
    paper_titles: dict[str, str | None]
    matrix: dict[str, Any]
    conclusion: str
    conclusion_reasoning: str
    comparability_status: str
    overall_summary: str
    model: str
    usage: dict[str, int] | None = None


class ResearchGapsRequestBody(BaseModel):
    """Wire format of research gap discovery request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    paper_ids: list[str] | None = None
    collection_id: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


class ResearchGapsResponseBody(BaseModel):
    """Wire format of research gap discovery response."""

    model_config = ConfigDict(strict=False)

    paper_ids: list[str]
    collection_id: str | None = None
    documented_gaps: list[dict[str, Any]]
    inferred_gaps: list[dict[str, Any]]
    total_gaps_count: int
    summary: str
    limitations: list[str]
    model: str
    usage: dict[str, int] | None = None


class ResearchTrendsRequestBody(BaseModel):
    """Wire format of temporal research trend analysis request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str
    paper_ids: list[str] | None = None
    time_unit: str = "year"
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


class ResearchTrendsResponseBody(BaseModel):
    """Wire format of temporal research trend analysis response."""

    model_config = ConfigDict(strict=False)

    collection_id: str
    paper_ids: list[str]
    time_unit: str
    temporal_buckets: list[dict[str, Any]]
    trends: list[dict[str, Any]]
    missing_date_paper_ids: list[str]
    summary: str
    limitations: list[str]
    model: str
    usage: dict[str, int] | None = None


class LiteratureSynthesisRequestBody(BaseModel):
    """Wire format of literature synthesis request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str
    paper_ids: list[str] | None = None
    topic: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


class LiteratureSynthesisResponseBody(BaseModel):
    """Wire format of literature synthesis response."""

    model_config = ConfigDict(strict=False)

    collection_id: str
    paper_ids: list[str]
    topic: str | None = None
    executive_summary: str
    common_findings: list[dict[str, Any]]
    disagreements: list[dict[str, Any]]
    primary_methods: list[str]
    recurring_limitations: list[dict[str, Any]]
    open_research_gaps: list[str]
    evidence_count: int
    is_grounded: bool
    limitations: list[str]
    model: str
    usage: dict[str, int] | None = None


class AgentPlanRequestBody(BaseModel):
    """Wire format of research planning request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    goal: str = Field(..., min_length=1, max_length=2000, description="Natural language research problem statement or goal.")
    collection_id: str | None = Field(default=None, description="Optional UUID of a collection to ground the plan in literature.")
    model: str | None = Field(default=None, description="Optional LLM model override.")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Optional generation temperature.")
    max_output_tokens: int | None = Field(default=None, ge=1, le=8192, description="Optional token generation budget.")


class ResearchPlanDimensionBody(BaseModel):
    """Wire format of an individual research plan dimension."""

    model_config = ConfigDict(strict=False)

    name: str
    title: str
    description: str
    details: dict[str, Any] = Field(default_factory=dict)
    evidence_labels: list[str] = Field(default_factory=list)
    supporting_paper_ids: list[str] = Field(default_factory=list)
    is_grounded: bool


class AgentPlanResponseBody(BaseModel):
    """Wire format of research planning response."""

    model_config = ConfigDict(strict=False)

    plan_id: str
    title: str
    research_problem: str
    research_questions: list[str]
    dimensions: dict[str, ResearchPlanDimensionBody]
    collection_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str
    is_mock: bool


class AgentRunRequestBody(BaseModel):
    """Wire format of bounded autonomous agent run request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    goal: str = Field(..., min_length=1, max_length=2000, description="Natural language research task or goal for the agent.")
    max_steps: int | None = Field(default=None, ge=1, le=10, description="Maximum iterative Reason-Act-Observe steps (1-10, default 5).")
    timeout_seconds: float | None = Field(default=None, gt=0.0, le=30.0, description="Maximum execution timeout in seconds (<= 30.0, default 30.0).")
    model: str | None = Field(default=None, description="Optional LLM model override.")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Optional generation temperature.")
    max_output_tokens: int | None = Field(default=None, ge=1, le=8192, description="Optional token generation budget.")


class AgentStepBody(BaseModel):
    """Wire format of an executed agent operational step."""

    model_config = ConfigDict(strict=False)

    step_index: int
    tool_name: str
    tool_input: dict[str, Any]
    output_summary: str
    status: str
    duration_ms: float
    error: str | None = None
    evidence_labels: list[str] = Field(default_factory=list)


class AgentTraceBody(BaseModel):
    """Wire format of the full audit and execution trace of an agent run."""

    model_config = ConfigDict(strict=False)

    trace_id: str
    goal: str
    status: str
    steps: list[AgentStepBody]
    total_duration_ms: float
    total_steps: int
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunResponseBody(BaseModel):
    """Wire format of bounded autonomous agent run result."""

    model_config = ConfigDict(strict=False)

    trace: AgentTraceBody
    final_answer: str | None = None
    status: str
    total_steps: int
    duration_ms: float
    evidence_labels: list[str] = Field(default_factory=list)


class AttributedSourceBody(BaseModel):
    """Wire format of an attributed research evidence source."""

    model_config = ConfigDict(strict=False)

    label: str = Field(..., description="Citation marker, e.g. '[E1]'.")
    document_id: str = Field(..., description="Identifier of the source document.")
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    paper_id: str | None = Field(default=None, description="Research paper UUID if available.")
    paper_title: str = Field(..., description="Title of the paper.")
    score: float = Field(..., description="Relevance score.")
    text: str = Field(..., description="Complete text of the chunk.")
    text_excerpt: str = Field(..., description="Clean excerpt of the evidence.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata key-value pairs.")
    page_number: int | None = Field(default=None, description="Page number of the source document, if known.")


class RAGQueryRequestBody(BaseModel):
    """Wire format for research RAG question query."""

    model_config = ConfigDict(strict=True, extra="forbid")

    query: str = Field(..., description="The research question to be answered.")
    retrieval_mode: str = Field(
        default="hybrid",
        description="Retrieval strategy: 'bm25', 'vector', or 'hybrid'.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        description="Maximum number of evidence chunks to retrieve.",
    )
    min_score: float | None = Field(
        default=None,
        description="Optional minimum relevance score threshold.",
    )
    temperature: float | None = Field(
        default=None,
        description="Generation temperature (0.0 to 2.0).",
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="Maximum output tokens for answer generation.",
    )
    collection_id: str | None = Field(
        default=None,
        description="Optional collection UUID to restrict retrieval to an authorized collection.",
    )
    paper_id: str | None = Field(
        default=None,
        description="Optional paper UUID to restrict retrieval to a specific paper.",
    )


class RAGQueryResponseBody(BaseModel):
    """Wire format for grounded RAG answer response."""

    model_config = ConfigDict(strict=False)

    query: str = Field(..., description="Original research question.")
    answer: str = Field(..., description="Evidence-grounded answer with citations.")
    sources: list[AttributedSourceBody] = Field(
        default_factory=list,
        description="Retrieved evidence chunks with source attribution.",
    )
    cited_sources: list[AttributedSourceBody] = Field(
        default_factory=list,
        description="Evidence chunks explicitly cited in the answer.",
    )
    is_grounded: bool = Field(..., description="Whether the answer is grounded in evidence.")
    is_mock: bool = Field(..., description="Whether a mock generator produced this answer.")
    model: str = Field(..., description="Model identifier used for generation.")
    status: str = Field(
        ...,
        description="Execution status: 'success', 'insufficient_evidence', or 'error'.",
    )
    error: str | None = Field(default=None, description="Error message if generation failed.")
    usage: dict[str, int] | None = Field(default=None, description="Token usage statistics.")
    retrieval_mode: str = Field(..., description="Retrieval mode used ('bm25', 'vector', 'hybrid').")
    evidence_count: int = Field(..., description="Number of evidence chunks retrieved.")


class DiscoveryPaperResponseBody(BaseModel):
    """Wire format of discovered paper metadata."""

    model_config = ConfigDict(strict=False)

    external_id: str = Field(..., description="Canonical arXiv identifier.")
    title: str = Field(..., description="Paper title.")
    authors: list[str] = Field(default_factory=list, description="Paper authors.")
    abstract: str = Field(default="", description="Paper abstract.")
    publication_date: str | None = Field(default=None, description="Publication date.")
    doi: str | None = Field(default=None, description="Digital Object Identifier.")
    categories: list[str] = Field(default_factory=list, description="Subject categories.")
    primary_category: str | None = Field(default=None, description="Primary subject category.")
    abstract_url: str | None = Field(default=None, description="Canonical abstract URL.")
    pdf_url: str | None = Field(default=None, description="Canonical PDF download URL.")
    license: str | None = Field(default=None, description="License URI.")
    source: str = Field(default="arxiv", description="Discovery provider name.")
    comment: str | None = Field(default=None, description="Author comments.")
    journal_ref: str | None = Field(default=None, description="Journal reference.")


class DiscoverySearchResponseBody(BaseModel):
    """Wire format of discovery search result."""

    model_config = ConfigDict(strict=False)

    provider: str = Field(..., description="Provider name, e.g. 'arxiv'.")
    query: str = Field(..., description="Search query string.")
    total_results: int = Field(..., ge=0, description="Total matching results found.")
    offset: int = Field(..., ge=0, description="Pagination offset.")
    limit: int = Field(..., ge=1, description="Pagination limit.")
    papers: list[DiscoveryPaperResponseBody] = Field(
        default_factory=list, description="List of paper metadata results."
    )


class DiscoveryImportRequestBody(BaseModel):
    """Wire format of explicit paper import request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    collection_id: str = Field(..., description="UUID of the destination collection.")
    external_id: str = Field(..., description="arXiv paper identifier to import.")
    durable: bool = Field(default=False, description="Whether to use durable indexing mode.")


class DiscoveryImportResponseBody(BaseModel):
    """Wire format of explicit paper import response."""

    model_config = ConfigDict(strict=False)

    paper_id: str = Field(..., description="UUID of the research paper.")
    collection_id: str = Field(..., description="UUID of the owning collection.")
    external_id: str = Field(..., description="Normalized external paper identifier.")
    status: str = Field(..., description="Import status: 'imported', 'already_imported', or 'failed'.")
    message: str = Field(..., description="Human-readable status message.")
    title: str = Field(..., description="Paper title.")
    document_id: str | None = Field(default=None, description="UUID of the persisted document if available.")
    chunks_count: int = Field(default=0, ge=0, description="Total chunks persisted.")
    indexed_chunks_count: int = Field(default=0, ge=0, description="Total chunks indexed.")
    doi: str | None = Field(default=None, description="Paper DOI if available.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Paper metadata.")


def get_research_service(request: Request) -> ResearchApplicationService:
    """Dependency: the research service injected into the app by create_app()."""
    return request.app.state.research_service


def get_ingestion_service(request: Request) -> Any:
    """Dependency: the document ingestion service injected into the app by create_app()."""
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise InvalidConfigurationError("ingestion_service is not configured")
    return service


def get_analysis_service(request: Request) -> Any:
    """Dependency: the structured paper analysis service injected into the app."""
    service = getattr(request.app.state, "analysis_service", None)
    if service is None:
        raise InvalidConfigurationError("analysis_service is not configured")
    return service


def get_comparison_service(request: Request) -> Any:
    """Dependency: the multi-paper comparison service injected into the app."""
    service = getattr(request.app.state, "comparison_service", None)
    if service is None:
        analysis = getattr(request.app.state, "analysis_service", None)
        if analysis is not None and hasattr(analysis, "compare_papers"):
            return analysis
        raise InvalidConfigurationError("comparison_service is not configured")
    return service


def get_gap_service(request: Request) -> Any:
    """Dependency: the research gap discovery service injected into the app."""
    service = getattr(request.app.state, "gap_service", None)
    if service is None:
        raise InvalidConfigurationError("gap_service is not configured")
    return service


def get_trend_service(request: Request) -> Any:
    """Dependency: the research trend analysis and synthesis service."""
    service = getattr(request.app.state, "trend_service", None)
    if service is None:
        raise InvalidConfigurationError("trend_service is not configured")
    return service


def get_rag_pipeline(request: Request) -> Any:
    """Dependency: the RAG pipeline injected into the app by create_app()."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        from nexora.rag.pipeline import create_configured_rag_pipeline

        pipeline = create_configured_rag_pipeline()
        request.app.state.rag_pipeline = pipeline
    return pipeline


def get_discovery_service(request: Request) -> Any:
    """Dependency: the scholarly discovery service injected into the app."""
    service = getattr(request.app.state, "discovery_service", None)
    if service is None:
        from nexora.discovery.service import DiscoveryService

        service = DiscoveryService(
            session_factory=getattr(request.app.state, "session_factory", None),
            ingestion_service=getattr(request.app.state, "ingestion_service", None),
            document_pipeline=getattr(request.app.state, "document_pipeline", None),
        )
        request.app.state.discovery_service = service
    else:
        if getattr(service, "_session_factory", None) is None:
            service._session_factory = getattr(request.app.state, "session_factory", None)
        if getattr(service, "_ingestion_service", None) is None:
            service._ingestion_service = getattr(request.app.state, "ingestion_service", None)
        if getattr(service, "_document_pipeline", None) is None:
            service._document_pipeline = getattr(request.app.state, "document_pipeline", None)
    return service


def get_tool_registry(request: Request) -> Any:
    """Dependency: the agent tool registry injected into the app."""
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        raise InvalidConfigurationError("tool_registry is not configured")
    return registry


def get_agent_orchestrator(request: Request) -> Any:
    """Dependency: the bounded agent orchestrator injected into the app."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise InvalidConfigurationError("orchestrator is not configured")
    return orchestrator


def get_planner_agent(request: Request) -> Any:
    """Dependency: the research planner agent injected into the app."""
    agent = getattr(request.app.state, "planner_agent", None)
    if agent is None:
        raise InvalidConfigurationError("planner_agent is not configured")
    return agent


class DocumentDetailsResponseBody(BaseModel):
    """Wire format for document details."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., description="UUID of the document.")
    paper_id: str = Field(..., description="UUID of the owning paper.")
    filename: str = Field(..., description="Original filename of the document.")
    file_size: int = Field(..., ge=0, description="Size in bytes.")
    mime_type: str = Field(..., description="MIME type.")
    checksum: str = Field(..., description="SHA-256 hex checksum.")
    indexing_status: str = Field(..., description="Vector indexing lifecycle status.")
    indexing_attempts: int = Field(default=0, ge=0, description="Number of indexing attempts.")
    created_at: str | None = Field(default=None, description="Creation timestamp.")
    updated_at: str | None = Field(default=None, description="Last update timestamp.")


class CollectionDetailsResponseBody(BaseModel):
    """Wire format for collection details."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(..., description="UUID of the collection.")
    workspace_id: str = Field(..., description="UUID of the owning workspace.")
    name: str = Field(..., description="Collection name.")
    description: str | None = Field(default=None, description="Collection description.")
    created_at: str | None = Field(default=None, description="Creation timestamp.")
    updated_at: str | None = Field(default=None, description="Last update timestamp.")


class WorkspaceCollectionsResponseBody(BaseModel):
    """Wire format for listing collections in a workspace."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(..., description="UUID of the workspace.")
    collections: list[CollectionDetailsResponseBody] = Field(
        default_factory=list, description="Collections in the workspace."
    )


def get_auth_service(request: Request) -> Any:
    """Dependency: the authentication and authorization service."""
    service = getattr(request.app.state, "auth_service", None)
    if service is None:
        session_factory = getattr(request.app.state, "session_factory", None)
        if session_factory is not None:
            from nexora.auth.service import AuthenticationService

            service = AuthenticationService(session_factory)
            request.app.state.auth_service = service
    return service


def get_current_user(request: Request) -> Any:
    """Dependency: authenticates the caller if authentication is enforced or credentials are provided."""
    auth_service = get_auth_service(request)
    enforce_auth = getattr(request.app.state, "enforce_auth", False)
    if not enforce_auth:
        if auth_service is not None and (request.headers.get("Authorization") or request.headers.get("X-API-Key")):
            return auth_service.authenticate_request(request)
        return None

    if auth_service is None:
        raise InvalidConfigurationError(
            "Authentication is enforced but no session_factory or auth_service is configured."
        )
    return auth_service.authenticate_request(request)


def get_optional_user(request: Request) -> Any:
    """Dependency: authenticates the caller if credentials are provided, otherwise returns None."""
    has_creds = bool(
        request.headers.get("Authorization") or request.headers.get("X-API-Key")
    )
    if not has_creds:
        return None
    auth_service = get_auth_service(request)
    if auth_service is None:
        enforce_auth = getattr(request.app.state, "enforce_auth", False)
        if enforce_auth:
            raise InvalidConfigurationError(
                "Authentication is enforced but no session_factory or auth_service is configured."
            )
        return None
    return auth_service.authenticate_request(request)


def get_public_rag_pipeline(request: Request) -> Any:
    """Dependency: returns an explicitly isolated RAG pipeline over the public benchmark corpus."""
    pipeline = getattr(request.app.state, "public_rag_pipeline", None)
    if pipeline is not None:
        return pipeline

    enforce_auth = getattr(request.app.state, "enforce_auth", False)
    if not enforce_auth and getattr(request.app.state, "rag_pipeline", None) is not None:
        return request.app.state.rag_pipeline

    from nexora.rag.pipeline import create_offline_rag_pipeline

    pipeline = create_offline_rag_pipeline()
    request.app.state.public_rag_pipeline = pipeline
    return pipeline


def get_public_research_service(request: Request) -> Any:
    """Dependency: returns an isolated research service over the public benchmark corpus."""
    service = getattr(request.app.state, "public_research_service", None)
    if service is not None:
        return service

    enforce_auth = getattr(request.app.state, "enforce_auth", False)
    if not enforce_auth and getattr(request.app.state, "research_service", None) is not None:
        return request.app.state.research_service

    public_pipeline = get_public_rag_pipeline(request)

    class _PublicBenchmarkResearchService:
        def __init__(self, pipeline: Any) -> None:
            self._pipeline = pipeline

        def ask(self, request_obj: Any, **kwargs: Any) -> Any:
            from nexora.api.models import ResearchResponse

            query_text = (
                request_obj.query
                if hasattr(request_obj, "query")
                else str(request_obj)
            )
            resp = self._pipeline.answer(query_text)
            return ResearchResponse(
                text=resp.answer,
                query=query_text,
                model=resp.model,
                usage=resp.usage,
                evidence_count=resp.evidence_count,
            )

    service = _PublicBenchmarkResearchService(public_pipeline)
    request.app.state.public_research_service = service
    return service


health_router = APIRouter()
v1_router = APIRouter(prefix="/api/v1")



@health_router.get("/health")
def health() -> dict[str, str]:
    """Liveness only; touches no dependency."""
    return {"status": "ok"}


@health_router.get("/ready")
@health_router.get("/health/ready")
def readiness(request: Request, response: Response) -> dict[str, Any]:
    """Readiness probe: validates availability of required backend dependencies.

    Checks:
    - database: verifies connectivity via session_factory if configured.
    - auth: checks authentication service readiness when enforce_auth is True.
    - research: checks research service availability.
    - vector_store: checks vector store connectivity via count() if configured.

    Returns HTTP 200 with status='ready' when all required components are operational.
    Returns HTTP 503 with status='unavailable' when any required component is failing.
    """
    session_factory = getattr(request.app.state, "session_factory", None)
    enforce_auth = getattr(request.app.state, "enforce_auth", False)
    research_service = getattr(request.app.state, "research_service", None)
    auth_service = getattr(request.app.state, "auth_service", None)

    checks: dict[str, Any] = {}
    is_ready = True

    # 1. Database check
    if session_factory is not None:
        try:
            from nexora.database.readiness import check_database_readiness

            check_database_readiness(session_factory)
            checks["database"] = {"status": "ready"}
        except Exception as exc:
            checks["database"] = {
                "status": "unavailable",
                "error": type(exc).__name__,
            }
            is_ready = False
    else:
        if enforce_auth:
            checks["database"] = {
                "status": "unavailable",
                "error": "database_not_configured_for_enforced_auth",
            }
            is_ready = False
        else:
            checks["database"] = {"status": "not_configured"}

    # 2. Authentication check
    if enforce_auth:
        if auth_service is None and session_factory is not None and is_ready:
            from nexora.auth.service import AuthenticationService

            try:
                auth_service = AuthenticationService(session_factory)
                request.app.state.auth_service = auth_service
            except Exception:
                pass

        if auth_service is not None and checks.get("database", {}).get("status") == "ready":
            checks["auth"] = {"status": "ready"}
        else:
            checks["auth"] = {"status": "unavailable"}
            is_ready = False
    else:
        checks["auth"] = {"status": "ready"}

    # 3. Research Service check
    if research_service is not None and callable(getattr(research_service, "ask", None)):
        checks["research"] = {"status": "ready"}
    else:
        checks["research"] = {"status": "unavailable"}
        is_ready = False

    # 4. Vector store / Ingestion check
    ingestion_service = getattr(request.app.state, "ingestion_service", None)
    rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
    vector_store = getattr(ingestion_service, "_vector_store", None) or getattr(rag_pipeline, "vector_store", None)
    if vector_store is not None:
        readiness_fn = getattr(vector_store, "check_readiness", None) or getattr(vector_store, "count", None)
        if callable(readiness_fn):
            try:
                readiness_fn()
                checks["vector_store"] = {"status": "ready"}
            except Exception as exc:
                checks["vector_store"] = {
                    "status": "unavailable",
                    "error": type(exc).__name__,
                }
                is_ready = False
        else:
            checks["vector_store"] = {"status": "ready"}

    # 5. Rate limit storage check (when rate limiting is enabled)
    rate_limit_config = getattr(request.app.state, "rate_limit_config", None)
    rate_limit_storage = getattr(request.app.state, "rate_limit_storage", None)
    if rate_limit_config is not None and getattr(rate_limit_config, "enabled", False):
        if rate_limit_storage is not None and hasattr(rate_limit_storage, "check_readiness"):
            ready_ok, ready_msg = rate_limit_storage.check_readiness()
            if ready_ok:
                checks["rate_limit_storage"] = {"status": "ready"}
            else:
                checks["rate_limit_storage"] = {
                    "status": "unavailable",
                    "error": ready_msg,
                }
                is_ready = False
        else:
            checks["rate_limit_storage"] = {
                "status": "unavailable",
                "error": "storage_not_configured_for_enabled_rate_limiting",
            }
            is_ready = False

    if not is_ready:
        response.status_code = 503
        return {"status": "unavailable", "checks": checks}

    return {"status": "ready", "checks": checks}


@v1_router.post("/research", response_model=ResearchResponseBody)
def research(
    body: ResearchRequestBody,
    request: Request,
    current_user: Any = Depends(get_optional_user),
) -> ResearchResponseBody:
    if current_user is None:
        if body.collection_id is not None:
            raise AuthenticationError(
                "Authentication credentials are required to query specific collections."
            )
        service = get_public_research_service(request)
        request_obj = ResearchRequest(
            query=body.query,
            model=body.model,
            temperature=body.temperature,
            max_output_tokens=body.max_output_tokens,
            collection_id=None,
        )
        try:
            ans = service.ask(request_obj)
        except TypeError:
            ans = service.ask(request_obj)
    else:
        service = get_research_service(request)
        auth_service = get_auth_service(request)
        scoped_col_ids: list[str] | None = None
        if auth_service is not None:
            if body.collection_id is not None:
                col, _ = auth_service.verify_collection_ownership(body.collection_id, current_user.id)
                scoped_col_ids = [str(col.id)]
            else:
                user_cols = auth_service.get_user_collection_ids(current_user.id)
                scoped_col_ids = [str(c) for c in user_cols]
        else:
            scoped_col_ids = []

        if scoped_col_ids is not None and len(scoped_col_ids) == 0:
            return ResearchResponseBody(
                text="No evidence found in authorized research collections.",
                query=body.query,
                model=body.model or "unknown",
                evidence_count=0,
            )

        request_obj = ResearchRequest(
            query=body.query,
            model=body.model,
            temperature=body.temperature,
            max_output_tokens=body.max_output_tokens,
            collection_id=body.collection_id,
        )
        try:
            ans = service.ask(request_obj, collection_ids=scoped_col_ids)
        except TypeError:
            ans = service.ask(request_obj)

    if isinstance(ans, ResearchResponse):
        payload = ans.to_dict()
    elif isinstance(ans, ResearchAnswer):
        payload = ResearchResponse.from_answer(ans).to_dict()
    elif hasattr(ans, "to_dict") and callable(ans.to_dict):
        payload = ans.to_dict()
    else:
        payload = {
            "text": getattr(ans, "text", ""),
            "query": getattr(ans, "query", body.query),
            "model": getattr(ans, "model", body.model or "unknown"),
            "usage": getattr(ans, "usage", None),
            "evidence_count": getattr(ans, "evidence_count", 0),
        }
    return ResearchResponseBody(**payload)


@v1_router.post(
    "/research/query",
    response_model=RAGQueryResponseBody,
    summary="Query research papers with grounded RAG",
    description="Retrieve research evidence and generate an attributed answer with verifiable citations.",
)
def research_rag_query(
    body: RAGQueryRequestBody,
    request: Request,
    current_user: Any = Depends(get_optional_user),
) -> RAGQueryResponseBody:
    """Answer a research question using the grounded RAG pipeline."""
    if not body.query.strip():
        raise InvalidRequestError("query must not be empty or whitespace-only")
    if body.retrieval_mode not in ("bm25", "vector", "hybrid"):
        raise InvalidRequestError(
            f"retrieval_mode must be 'bm25', 'vector', or 'hybrid', got {body.retrieval_mode!r}"
        )
    if body.temperature is not None and (body.temperature < 0.0 or body.temperature > 2.0):
        raise InvalidRequestError("temperature must be between 0.0 and 2.0")
    if body.max_output_tokens is not None and body.max_output_tokens <= 0:
        raise InvalidRequestError("max_output_tokens must be greater than 0")

    if current_user is None:
        if body.collection_id is not None or body.paper_id is not None:
            raise AuthenticationError(
                "Authentication credentials are required to query specific collections or papers."
            )
        pipeline = get_public_rag_pipeline(request)
        scoped_collection_ids: list[str] | None = None
        scoped_paper_id: str | None = None
    else:
        auth_service = get_auth_service(request)
        if auth_service is None:
            raise InvalidConfigurationError("auth_service is not configured")

        scoped_collection_ids = None
        scoped_paper_id = None

        if body.collection_id is not None:
            col, _ = auth_service.verify_collection_ownership(body.collection_id, current_user.id)
            scoped_collection_ids = [str(col.id)]
        if body.paper_id is not None:
            paper, col, _ = auth_service.verify_paper_ownership(body.paper_id, current_user.id)
            scoped_paper_id = str(paper.id)
            if scoped_collection_ids is None:
                scoped_collection_ids = [str(col.id)]
            elif str(col.id) not in scoped_collection_ids:
                raise ForbiddenError("Specified paper does not belong to the specified collection.")
        if scoped_collection_ids is None and scoped_paper_id is None:
            user_cols = auth_service.get_user_collection_ids(current_user.id)
            scoped_collection_ids = [str(c) for c in user_cols]

        # Fail closed if caller is scoped but has no authorized collections
        if scoped_collection_ids is not None and len(scoped_collection_ids) == 0:
            from nexora.rag.pipeline import INSUFFICIENT_EVIDENCE_ANSWER
            pipeline = get_rag_pipeline(request)
            model_name = getattr(pipeline.generator, "model_name", "unknown") if hasattr(pipeline, "generator") else "unknown"
            return RAGQueryResponseBody(
                query=body.query,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                sources=[],
                cited_sources=[],
                is_grounded=True,
                is_mock=getattr(pipeline, "is_mock", True),
                model=model_name,
                status="insufficient_evidence",
                retrieval_mode=body.retrieval_mode,
                evidence_count=0,
            )

        pipeline = get_rag_pipeline(request)

    try:
        response = pipeline.answer(
            request=body.query,
            retrieval_mode=body.retrieval_mode,
            top_k=body.top_k,
            min_score=body.min_score,
            temperature=body.temperature,
            max_output_tokens=body.max_output_tokens,
            paper_id=scoped_paper_id,
            collection_ids=scoped_collection_ids,
        )
    except (InvalidRequestError, ValueError) as exc:
        raise InvalidRequestError(str(exc)) from exc
    except Exception as exc:
        raise ResearchServiceError(str(exc)) from exc

    if getattr(response, "status", None) == "error":
        raise ResearchServiceError(getattr(response, "error", None) or "RAG pipeline execution failed.")

    return RAGQueryResponseBody(**response.to_dict())


@v1_router.post(
    "/rag/query",
    response_model=RAGQueryResponseBody,
    summary="Query research papers with grounded RAG (alias)",
    description="Alias for /api/v1/research/query.",
    include_in_schema=False,
)
def rag_query_alias(
    body: RAGQueryRequestBody,
    request: Request,
    current_user: Any = Depends(get_optional_user),
) -> RAGQueryResponseBody:
    return research_rag_query(body, request=request, current_user=current_user)


@v1_router.post(
    "/research/verify",
    response_model=ResearchVerifyResponseBody,
    status_code=200,
    summary="Verify research answer claims against cited evidence",
    description="Verifies claims in a generated or supplied research answer against retrieved evidence context.",
)
def research_verify_endpoint(
    body: ResearchVerifyRequestBody,
    request: Request,
    service: ResearchApplicationService = Depends(get_research_service),
    current_user: Any = Depends(get_current_user),
) -> ResearchVerifyResponseBody:
    scoped_col_ids: list[str] | None = None
    scoped_paper_id: str | None = None

    if current_user is not None:
        auth_service = get_auth_service(request)
        if auth_service is not None:
            if body.expected_paper_id is not None:
                paper, col, _ = auth_service.verify_paper_ownership(body.expected_paper_id, current_user.id)
                scoped_paper_id = str(paper.id)
                scoped_col_ids = [str(col.id)]
            else:
                user_cols = auth_service.get_user_collection_ids(current_user.id)
                scoped_col_ids = [str(c) for c in user_cols]
    elif body.expected_paper_id is not None:
        scoped_paper_id = body.expected_paper_id

    req = ResearchVerifyRequest(
        query=body.query,
        text=body.text,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
        expected_paper_id=body.expected_paper_id,
        collection_ids=scoped_col_ids,
        paper_id=scoped_paper_id,
    )
    try:
        result = service.verify(req, collection_ids=scoped_col_ids, paper_id=scoped_paper_id)
    except TypeError:
        result = service.verify(req)
    return ResearchVerifyResponseBody(**result.to_dict())


@v1_router.post(
    "/papers/analyze",
    response_model=PaperAnalysisResponseBody,
    status_code=200,
    summary="Perform structured analysis on a research paper",
    description="Extracts 7 standardized research dimensions from an ingested paper with evidence citations.",
)
def analyze_paper_endpoint(
    body: PaperAnalysisRequestBody,
    request: Request,
    service: Any = Depends(get_analysis_service),
    current_user: Any = Depends(get_current_user),
) -> PaperAnalysisResponseBody:
    if current_user is not None:
        auth_service = get_auth_service(request)
        if auth_service is not None:
            auth_service.verify_paper_ownership(body.paper_id, current_user.id)
    req = PaperAnalysisRequest(
        paper_id=body.paper_id,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
    )
    result = service.analyze_paper(
        req.paper_id,
        model=req.model,
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
    )
    resp = PaperAnalysisResponse.from_analysis(result)
    return PaperAnalysisResponseBody(**resp.to_dict())


@v1_router.post(
    "/papers/{paper_id}/analyze",
    response_model=PaperAnalysisResponseBody,
    status_code=200,
    summary="Perform structured analysis on a research paper by ID",
    description="Extracts 7 standardized research dimensions from an ingested paper with evidence citations.",
)
def analyze_paper_by_id_endpoint(
    paper_id: str,
    request: Request,
    service: Any = Depends(get_analysis_service),
    current_user: Any = Depends(get_current_user),
) -> PaperAnalysisResponseBody:
    if current_user is not None:
        auth_service = get_auth_service(request)
        if auth_service is not None:
            auth_service.verify_paper_ownership(paper_id, current_user.id)
    req = PaperAnalysisRequest(paper_id=paper_id)
    result = service.analyze_paper(req.paper_id)
    resp = PaperAnalysisResponse.from_analysis(result)
    return PaperAnalysisResponseBody(**resp.to_dict())


@v1_router.post(
    "/papers/compare",
    response_model=PaperComparisonResponseBody,
    status_code=200,
    summary="Compare multiple research papers across aligned dimensions",
    description="Compares two or more papers across standardized research dimensions while preserving per-paper evidence claims.",
)
def compare_papers_endpoint(
    body: PaperComparisonRequestBody,
    request: Request,
    service: Any = Depends(get_comparison_service),
    current_user: Any = Depends(get_current_user),
) -> PaperComparisonResponseBody:
    if current_user is not None:
        auth_service = get_auth_service(request)
        if auth_service is not None:
            for pid in body.paper_ids:
                auth_service.verify_paper_ownership(pid, current_user.id)
    req = PaperComparisonRequest(
        paper_ids=body.paper_ids,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
    )
    result = service.compare_papers(
        req.paper_ids,
        model=req.model,
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
    )
    resp = PaperComparisonResponse.from_comparison(result)
    return PaperComparisonResponseBody(**resp.to_dict())


@v1_router.post(
    "/research/compare",
    response_model=ResearchCompareResponseBody,
    status_code=200,
    summary="Compare research papers across 7-dimension matrix with evidence synthesis",
    description=(
        "Retrieves evidence separately for each paper with strict provenance and isolation. "
        "Constructs a structured comparison matrix for methods, datasets, baselines, metrics, "
        "results, limitations, and comparability. Classifies conclusion as Demonstrated improvement, "
        "Limited evidence, Conflicting evidence, or Insufficient evidence."
    ),
)
def research_compare_endpoint(
    body: ResearchCompareRequestBody,
    request: Request,
    service: Any = Depends(get_comparison_service),
    current_user: Any = Depends(get_current_user),
) -> ResearchCompareResponseBody:
    if current_user is not None:
        auth_service = get_auth_service(request)
        if auth_service is not None:
            for pid in body.paper_ids:
                auth_service.verify_paper_ownership(pid, current_user.id)
    req = ResearchCompareRequest(
        paper_ids=body.paper_ids,
        query=body.query,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
    )
    if hasattr(service, "compare_cross_paper"):
        result = service.compare_cross_paper(
            req.paper_ids,
            query=req.query,
            model=req.model,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
        )
    elif hasattr(service, "compare"):
        result = service.compare(
            req.paper_ids,
            query=req.query,
            model=req.model,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
        )
    else:
        raise InvalidConfigurationError("comparison_service does not support cross-paper comparison")
    resp = ResearchCompareResponse.from_result(result)
    return ResearchCompareResponseBody(**resp.to_dict())


@v1_router.post(
    "/research/gaps",
    response_model=ResearchGapsResponseBody,
    status_code=200,
    summary="Discover evidence-backed research gaps across papers or a collection",
    description="Discovers author-documented and AI-inferred candidate research gaps across specified papers or a collection with traceable evidence.",
)
def discover_research_gaps_endpoint(
    body: ResearchGapsRequestBody,
    request: Request,
    service: Any = Depends(get_gap_service),
    current_user: Any = Depends(get_current_user),
) -> ResearchGapsResponseBody:
    if current_user is not None:
        auth_service = get_auth_service(request)
        if auth_service is not None:
            if body.collection_id is not None:
                auth_service.verify_collection_ownership(body.collection_id, current_user.id)
            if body.paper_ids:
                for pid in body.paper_ids:
                    auth_service.verify_paper_ownership(pid, current_user.id)
    req = ResearchGapsRequest(
        paper_ids=body.paper_ids,
        collection_id=body.collection_id,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
    )
    if req.collection_id is not None:
        result = service.discover_gaps_for_collection(
            req.collection_id,
            model=req.model,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
        )
    else:
        assert req.paper_ids is not None
        result = service.discover_gaps_for_papers(
            req.paper_ids,
            model=req.model,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
        )
    resp = ResearchGapsResponse.from_report(result)
    return ResearchGapsResponseBody(**resp.to_dict())


@v1_router.post(
    "/papers/{paper_id}/gaps",
    response_model=ResearchGapsResponseBody,
    status_code=200,
    summary="Discover evidence-backed research gaps for a single research paper",
    description="Extracts author-documented limitations/future work and candidate gaps for an individual paper.",
)
def discover_paper_gaps_by_id_endpoint(
    paper_id: str,
    request: Request,
    service: Any = Depends(get_gap_service),
    current_user: Any = Depends(get_current_user),
) -> ResearchGapsResponseBody:
    if current_user is not None:
        auth_service = get_auth_service(request)
        if auth_service is not None:
            auth_service.verify_paper_ownership(paper_id, current_user.id)
    req = ResearchGapsRequest(paper_ids=[paper_id])
    assert req.paper_ids is not None
    result = service.discover_gaps_for_papers(req.paper_ids)
    resp = ResearchGapsResponse.from_report(result)
    return ResearchGapsResponseBody(**resp.to_dict())


@v1_router.post(
    "/papers/trends",
    response_model=ResearchTrendsResponseBody,
    status_code=200,
    summary="Analyze temporal research trends across a paper collection",
    description=(
        "Extracts methods, architectures, datasets, and benchmark metrics across papers in an authorized collection, "
        "groups findings by publication year or temporal period, and classifies trajectories as emerging, established, "
        "declining, or insufficient_data strictly based on traceable evidence."
    ),
    responses={
        200: {
            "description": "Temporal research trend analysis completed successfully.",
            "model": ResearchTrendsResponseBody,
        },
        400: {"description": "Malformed request, invalid UUID, or invalid parameters."},
        401: {"description": "Authentication required to access private tenant collections."},
        403: {"description": "Caller does not own or have access to the specified collection or papers."},
        404: {"description": "Referenced collection or paper was not found."},
    },
)
def analyze_research_trends_endpoint(
    body: ResearchTrendsRequestBody,
    request: Request,
    service: Any = Depends(get_trend_service),
    current_user: Any = Depends(get_current_user),
) -> ResearchTrendsResponseBody:
    req = ResearchTrendsRequest(
        collection_id=body.collection_id,
        paper_ids=body.paper_ids,
        time_unit=body.time_unit,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
    )
    if current_user is None:
        from nexora.api.errors import AuthenticationError
        raise AuthenticationError("Authentication required to access research trend analysis.")
    auth_service = get_auth_service(request)
    if auth_service is not None:
        auth_service.verify_collection_ownership(req.collection_id, current_user.id)
        if req.paper_ids:
            for pid in req.paper_ids:
                auth_service.verify_paper_ownership(pid, current_user.id)
    result = service.analyze_collection_trends(
        req.collection_id,
        paper_ids=req.paper_ids,
        time_unit=req.time_unit,
        model=req.model,
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
    )
    resp = ResearchTrendsResponse.from_report(result)
    return ResearchTrendsResponseBody(**resp.to_dict())


@v1_router.post(
    "/collections/{collection_id}/trends",
    response_model=ResearchTrendsResponseBody,
    status_code=200,
    summary="Analyze temporal research trends for a collection by ID",
    description="Convenience endpoint for temporal research trend analysis scoped to a collection ID.",
    responses={
        200: {
            "description": "Temporal research trend analysis completed successfully.",
            "model": ResearchTrendsResponseBody,
        },
        400: {"description": "Malformed request or invalid collection_id."},
        401: {"description": "Authentication required to access private tenant collections."},
        403: {"description": "Caller does not own or have access to the specified collection."},
        404: {"description": "Referenced collection was not found."},
    },
)
def analyze_collection_trends_by_id_endpoint(
    collection_id: str,
    request: Request,
    service: Any = Depends(get_trend_service),
    current_user: Any = Depends(get_current_user),
) -> ResearchTrendsResponseBody:
    req = ResearchTrendsRequest(collection_id=collection_id)
    if current_user is None:
        from nexora.api.errors import AuthenticationError
        raise AuthenticationError("Authentication required to access research trend analysis.")
    auth_service = get_auth_service(request)
    if auth_service is not None:
        auth_service.verify_collection_ownership(req.collection_id, current_user.id)
    result = service.analyze_collection_trends(req.collection_id)
    resp = ResearchTrendsResponse.from_report(result)
    return ResearchTrendsResponseBody(**resp.to_dict())


@v1_router.post(
    "/papers/synthesis",
    response_model=LiteratureSynthesisResponseBody,
    status_code=200,
    summary="Generate evidence-grounded literature review synthesis across a collection",
    description=(
        "Synthesizes common empirical findings, disagreements/divergences, primary methods, recurring limitations, "
        "and open research gaps across an authorized collection, strictly grounded in traceable evidence citations."
    ),
    responses={
        200: {
            "description": "Literature review synthesis completed successfully.",
            "model": LiteratureSynthesisResponseBody,
        },
        400: {"description": "Malformed request, invalid UUID, or invalid parameters."},
        401: {"description": "Authentication required to access private tenant collections."},
        403: {"description": "Caller does not own or have access to the specified collection or papers."},
        404: {"description": "Referenced collection or paper was not found."},
    },
)
def synthesize_literature_endpoint(
    body: LiteratureSynthesisRequestBody,
    request: Request,
    service: Any = Depends(get_trend_service),
    current_user: Any = Depends(get_current_user),
) -> LiteratureSynthesisResponseBody:
    req = LiteratureSynthesisRequest(
        collection_id=body.collection_id,
        paper_ids=body.paper_ids,
        topic=body.topic,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
    )
    if current_user is None:
        from nexora.api.errors import AuthenticationError
        raise AuthenticationError("Authentication required to access literature review synthesis.")
    auth_service = get_auth_service(request)
    if auth_service is not None:
        auth_service.verify_collection_ownership(req.collection_id, current_user.id)
        if req.paper_ids:
            for pid in req.paper_ids:
                auth_service.verify_paper_ownership(pid, current_user.id)
    result = service.synthesize_collection_literature(
        req.collection_id,
        paper_ids=req.paper_ids,
        topic=req.topic,
        model=req.model,
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
    )
    resp = LiteratureSynthesisResponse.from_report(result)
    return LiteratureSynthesisResponseBody(**resp.to_dict())


@v1_router.post(
    "/collections/{collection_id}/synthesis",
    response_model=LiteratureSynthesisResponseBody,
    status_code=200,
    summary="Generate literature review synthesis for a collection by ID",
    description="Convenience endpoint for literature review synthesis scoped to a collection ID.",
    responses={
        200: {
            "description": "Literature review synthesis completed successfully.",
            "model": LiteratureSynthesisResponseBody,
        },
        400: {"description": "Malformed request or invalid collection_id."},
        401: {"description": "Authentication required to access private tenant collections."},
        403: {"description": "Caller does not own or have access to the specified collection."},
        404: {"description": "Referenced collection was not found."},
    },
)
def synthesize_collection_literature_by_id_endpoint(
    collection_id: str,
    request: Request,
    service: Any = Depends(get_trend_service),
    current_user: Any = Depends(get_current_user),
) -> LiteratureSynthesisResponseBody:
    req = LiteratureSynthesisRequest(collection_id=collection_id)
    if current_user is None:
        from nexora.api.errors import AuthenticationError
        raise AuthenticationError("Authentication required to access literature review synthesis.")
    auth_service = get_auth_service(request)
    if auth_service is not None:
        auth_service.verify_collection_ownership(req.collection_id, current_user.id)
    result = service.synthesize_collection_literature(req.collection_id)
    resp = LiteratureSynthesisResponse.from_report(result)
    return LiteratureSynthesisResponseBody(**resp.to_dict())



@v1_router.post(
    "/documents/ingest",
    response_model=DocumentIngestResponseBody,
    status_code=200,
    summary="Upload and ingest a research paper PDF",
    description=(
        "Uploads a PDF file for a given research paper, extracts and chunks text, "
        "persists document and chunk records to the database, and indexes "
        "semantic vectors in the vector index. Enforces transactional rollback "
        "and duplicate detection."
    ),
    responses={
        200: {
            "description": "Document successfully parsed, persisted, and indexed.",
            "model": DocumentIngestResponseBody,
        },
        400: {"description": "Malformed request, missing fields, or invalid/unreadable PDF."},
        404: {"description": "Referenced paper was not found."},
        409: {"description": "Document with identical checksum already exists for this paper."},
        413: {"description": "Uploaded file exceeds maximum allowed size."},
        422: {"description": "Request validation failed."},
        500: {"description": "Indexing failure or unexpected server error."},
    },
)
async def ingest_document(
    request: Request,
    paper_id: str = Form(...),
    file: UploadFile = File(...),
    durable: bool = Form(False),
    service: Any = Depends(get_ingestion_service),
    current_user: Any = Depends(get_current_user),
) -> DocumentIngestResponseBody:
    """Upload and ingest a research paper PDF into relational storage and vector index."""
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise InvalidRequestError("paper_id is required.")

    try:
        paper_uuid = UUID(paper_id.strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid paper_id: must be a valid UUID.")

    if current_user is not None:
        auth_service = get_auth_service(request)
        if auth_service is not None:
            auth_service.verify_paper_ownership(paper_uuid, current_user.id)

    raw_filename = file.filename or ""
    if not raw_filename.lower().endswith(".pdf"):
        raise InvalidRequestError("Only PDF files are supported.")

    content_type = (file.content_type or "").lower()
    if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
        raise InvalidRequestError("Only PDF files are supported.")

    safe_filename = Path(raw_filename).name or "document.pdf"
    max_size = getattr(request.app.state, "max_upload_size_bytes", 50 * 1024 * 1024)
    hasher = hashlib.sha256()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        total_bytes = 0
        try:
            while chunk := await file.read(65536):
                total_bytes += len(chunk)
                if total_bytes > max_size:
                    raise FileTooLargeError(
                        f"Uploaded file exceeds maximum allowed size of {max_size} bytes."
                    )
                tmp_file.write(chunk)
                hasher.update(chunk)
            tmp_file.flush()
        except Exception:
            tmp_file.close()
            if tmp_path.exists():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

    if total_bytes == 0:
        if tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise InvalidRequestError("Uploaded file is empty.")

    file_checksum = hasher.hexdigest()

    try:
        pipeline = getattr(request.app.state, "document_pipeline", None)

        if pipeline is not None and hasattr(pipeline, "_loader") and callable(pipeline._loader):
            try:
                pipeline._loader(str(tmp_path))
            except Exception as exc:
                raise InvalidRequestError("The uploaded file is not a valid or readable PDF.") from exc

        session_factory = getattr(request.app.state, "session_factory", None)
        if session_factory is not None and hasattr(service, "_session_factory"):
            if getattr(service, "_session_factory", None) is None:
                service._session_factory = session_factory

        try:
            if hasattr(service, "process_and_ingest"):
                try:
                    result = service.process_and_ingest(
                        paper_id=paper_uuid,
                        file_path=str(tmp_path),
                        pipeline=pipeline,
                        filename=safe_filename,
                        checksum=file_checksum,
                        durable=durable,
                    )
                except TypeError:
                    try:
                        result = service.process_and_ingest(
                            paper_id=paper_uuid,
                            file_path=str(tmp_path),
                            pipeline=pipeline,
                            filename=safe_filename,
                            checksum=file_checksum,
                        )
                    except TypeError:
                        result = service.process_and_ingest(
                            paper_id=paper_uuid,
                            file_path=str(tmp_path),
                            pipeline=pipeline,
                            filename=safe_filename,
                        )
            elif hasattr(service, "ingest_file"):
                result = service.ingest_file(
                    paper_id=paper_uuid,
                    file_path=str(tmp_path),
                    pipeline=pipeline,
                    filename=safe_filename,
                )
            elif hasattr(service, "ingest"):
                result = service.ingest(
                    paper_id=paper_uuid,
                    file_path=str(tmp_path),
                    filename=safe_filename,
                )
            else:
                raise InvalidConfigurationError(
                    "ingestion_service does not support document ingestion"
                )
        except ValueError as exc:
            raise InvalidRequestError("The uploaded file is not a valid or readable PDF.") from exc

        response_data = DocumentIngestResponse.from_result(result)
        return DocumentIngestResponseBody(**response_data.to_dict())
    finally:
        if tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@v1_router.get(
    "/discovery/search",
    response_model=DiscoverySearchResponseBody,
    status_code=200,
    summary="Search scholarly papers on arXiv",
    description=(
        "Searches external scholarly repositories (arXiv) for papers matching query "
        "and optional category filters. Returns normalized metadata and pagination details. "
        "Does not download PDFs during search."
    ),
    responses={
        200: {"description": "Search results returned successfully.", "model": DiscoverySearchResponseBody},
        400: {"description": "Invalid query or search parameters."},
        422: {"description": "Validation error on request parameters."},
        429: {"description": "Upstream rate limit exceeded."},
        502: {"description": "Upstream provider error or connection failure."},
        504: {"description": "Upstream request timed out."},
    },
)
def search_scholarly_papers(
    query: str = Query(..., description="Search query string"),
    categories: list[str] | None = Query(default=None, description="Optional arXiv categories"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=10, ge=1, le=100, description="Pagination limit"),
    sort_by: str = Query(default="relevance", description="Sort by: relevance, submitted_date, last_updated"),
    sort_order: str = Query(default="descending", description="Sort order: descending, ascending"),
    service: Any = Depends(get_discovery_service),
) -> DiscoverySearchResponseBody:
    """Search external scholarly papers with pagination and filters."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidRequestError("query must not be empty or whitespace-only")
    if sort_by not in ("relevance", "submitted_date", "last_updated"):
        raise InvalidRequestError(
            f"sort_by must be one of ('relevance', 'submitted_date', 'last_updated'), got {sort_by!r}"
        )
    if sort_order not in ("descending", "ascending"):
        raise InvalidRequestError(
            f"sort_order must be one of ('descending', 'ascending'), got {sort_order!r}"
        )

    cleaned_categories: list[str] = []
    if categories:
        for cat in categories:
            for part in cat.split(","):
                if part.strip():
                    cleaned_categories.append(part.strip())

    from nexora.discovery.models import ScholarlySearchQuery

    search_query = ScholarlySearchQuery(
        query=query.strip(),
        categories=tuple(cleaned_categories),
        offset=offset,
        limit=limit,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    result = service.search(search_query)
    return DiscoverySearchResponseBody(**result.to_dict())


@v1_router.get(
    "/discovery/paper/{external_id:path}",
    response_model=DiscoveryPaperResponseBody,
    status_code=200,
    summary="Get normalized metadata for a single external scholarly paper",
    description="Retrieves metadata for an arXiv paper by identifier. Rejects arbitrary URLs.",
    responses={
        200: {"description": "Paper metadata found.", "model": DiscoveryPaperResponseBody},
        400: {"description": "Malformed or unpermitted external identifier."},
        404: {"description": "Paper was not found on arXiv."},
        429: {"description": "Upstream rate limit exceeded."},
        502: {"description": "Upstream provider error or connection failure."},
        504: {"description": "Upstream request timed out."},
    },
)
def get_scholarly_paper_details(
    external_id: str,
    service: Any = Depends(get_discovery_service),
) -> DiscoveryPaperResponseBody:
    """Retrieve metadata for an arXiv paper by identifier."""
    paper = service.get_paper(external_id)
    return DiscoveryPaperResponseBody(**paper.to_dict())


@v1_router.post(
    "/discovery/import",
    response_model=DiscoveryImportResponseBody,
    status_code=200,
    summary="Explicitly download, deduplicate, and ingest a scholarly paper",
    description=(
        "Explicitly imports a paper from an external provider (arXiv) into a target collection. "
        "Performs deduplication: if already imported and indexed, returns early without re-downloading. "
        "Safely downloads the PDF via SafePDFDownloader, persists the paper record, and executes "
        "the document ingestion pipeline."
    ),
    responses={
        200: {"description": "Import result (imported or already_imported).", "model": DiscoveryImportResponseBody},
        400: {"description": "Invalid collection ID, malformed paper ID, or invalid PDF."},
        401: {"description": "Authentication required or invalid credentials."},
        403: {"description": "Access to target collection is forbidden."},
        404: {"description": "Collection or paper not found."},
        409: {"description": "Duplicate document conflict."},
        413: {"description": "PDF exceeds maximum allowed download size."},
        422: {"description": "Request validation failed."},
        429: {"description": "Upstream rate limit exceeded."},
        500: {"description": "Ingestion failed or internal server error."},
        502: {"description": "Upstream provider error or connection failure."},
        504: {"description": "Upstream request timed out."},
    },
)
def import_scholarly_paper(
    body: DiscoveryImportRequestBody,
    request: Request,
    service: Any = Depends(get_discovery_service),
    current_user: Any = Depends(get_current_user),
) -> DiscoveryImportResponseBody:
    """Explicitly import, deduplicate, and ingest an arXiv paper into a collection."""
    if current_user is not None:
        auth_service = get_auth_service(request)
        if auth_service is not None:
            auth_service.verify_collection_ownership(body.collection_id, current_user.id)

    result = service.import_paper(
        collection_id=body.collection_id,
        external_id=body.external_id,
        durable=body.durable,
    )
    return DiscoveryImportResponseBody(**result.to_dict())


@v1_router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailsResponseBody,
    status_code=200,
    summary="Get document details and indexing status",
    description="Retrieves metadata, file properties, and vector indexing status for a document. Enforces ownership authorization.",
    responses={
        200: {"description": "Document details returned successfully.", "model": DocumentDetailsResponseBody},
        400: {"description": "Invalid document UUID."},
        401: {"description": "Authentication required or invalid credentials."},
        403: {"description": "Access to this document is forbidden."},
        404: {"description": "Document was not found."},
    },
)
def get_document_details(
    document_id: str,
    request: Request,
    current_user: Any = Depends(get_current_user),
) -> DocumentDetailsResponseBody:
    """Get metadata and indexing status for a document."""
    try:
        doc_uuid = UUID(document_id.strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid document_id: must be a valid UUID.")

    auth_service = get_auth_service(request)
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise InvalidConfigurationError("session_factory is not configured")

    if current_user is None or auth_service is None:
        raise AuthenticationError("Authentication credentials are required or invalid.")

    doc, _, _, _ = auth_service.verify_document_ownership(doc_uuid, current_user.id)

    return DocumentDetailsResponseBody(
        document_id=str(doc.id),
        paper_id=str(doc.paper_id),
        filename=doc.filename,
        file_size=doc.file_size,
        mime_type=doc.mime_type,
        checksum=doc.checksum,
        indexing_status=str(doc.indexing_status.value if hasattr(doc.indexing_status, "value") else doc.indexing_status),
        indexing_attempts=doc.indexing_attempts or 0,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
    )


@v1_router.get(
    "/collections/{collection_id}",
    response_model=CollectionDetailsResponseBody,
    status_code=200,
    summary="Get collection metadata",
    description="Retrieves collection metadata and workspace association. Enforces ownership authorization.",
    responses={
        200: {"description": "Collection details returned successfully.", "model": CollectionDetailsResponseBody},
        400: {"description": "Invalid collection UUID."},
        401: {"description": "Authentication required or invalid credentials."},
        403: {"description": "Access to this collection is forbidden."},
        404: {"description": "Collection was not found."},
    },
)
def get_collection_details(
    collection_id: str,
    request: Request,
    current_user: Any = Depends(get_current_user),
) -> CollectionDetailsResponseBody:
    """Get metadata for a collection."""
    try:
        col_uuid = UUID(collection_id.strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid collection_id: must be a valid UUID.")

    auth_service = get_auth_service(request)
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise InvalidConfigurationError("session_factory is not configured")

    if current_user is None or auth_service is None:
        raise AuthenticationError("Authentication credentials are required or invalid.")

    col, _ = auth_service.verify_collection_ownership(col_uuid, current_user.id)

    return CollectionDetailsResponseBody(
        collection_id=str(col.id),
        workspace_id=str(col.workspace_id),
        name=col.name,
        description=col.description,
        created_at=col.created_at.isoformat() if col.created_at else None,
        updated_at=col.updated_at.isoformat() if col.updated_at else None,
    )


@v1_router.get(
    "/workspaces/{workspace_id}/collections",
    response_model=WorkspaceCollectionsResponseBody,
    status_code=200,
    summary="List collections in a workspace",
    description="Lists all collections belonging to a workspace. Enforces workspace ownership authorization.",
    responses={
        200: {"description": "Collections returned successfully.", "model": WorkspaceCollectionsResponseBody},
        400: {"description": "Invalid workspace UUID."},
        401: {"description": "Authentication required or invalid credentials."},
        403: {"description": "Access to this workspace is forbidden."},
        404: {"description": "Workspace was not found."},
    },
)
def list_workspace_collections(
    workspace_id: str,
    request: Request,
    current_user: Any = Depends(get_current_user),
) -> WorkspaceCollectionsResponseBody:
    """List collections in a workspace."""
    try:
        ws_uuid = UUID(workspace_id.strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid workspace_id: must be a valid UUID.")

    auth_service = get_auth_service(request)
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise InvalidConfigurationError("session_factory is not configured")

    if current_user is None or auth_service is None:
        raise AuthenticationError("Authentication credentials are required or invalid.")

    auth_service.verify_workspace_ownership(ws_uuid, current_user.id)

    from nexora.database.repository.collection import CollectionRepository

    with session_factory() as session:
        col_repo = CollectionRepository(session)
        collections = col_repo.list_by_workspace_id(ws_uuid)

        col_models = [
            CollectionDetailsResponseBody(
                collection_id=str(c.id),
                workspace_id=str(c.workspace_id),
                name=c.name,
                description=c.description,
                created_at=c.created_at.isoformat() if c.created_at else None,
                updated_at=c.updated_at.isoformat() if c.updated_at else None,
            )
            for c in collections
        ]

    return WorkspaceCollectionsResponseBody(
        workspace_id=str(ws_uuid),
        collections=col_models,
    )


@v1_router.post(
    "/agents/plan",
    response_model=AgentPlanResponseBody,
    status_code=200,
    summary="Generate an evidence-grounded research plan across 7 dimensions",
    description=(
        "Uses the autonomous Research Planner Agent to convert a research problem statement "
        "into a structured research plan across 7 dimensions. Can be grounded in an authorized collection."
    ),
    responses={
        200: {
            "description": "Research plan generated successfully.",
            "model": AgentPlanResponseBody,
        },
        400: {"description": "Malformed request, oversized goal, or invalid collection UUID."},
        401: {"description": "Authentication required to access research planner or private collections."},
        403: {"description": "Caller does not own or have access to the specified collection."},
        404: {"description": "Referenced collection was not found."},
        500: {"description": "Research plan generation failed."},
    },
)
def plan_research_endpoint(
    body: AgentPlanRequestBody,
    request: Request,
    planner: Any = Depends(get_planner_agent),
    current_user: Any = Depends(get_optional_user),
) -> AgentPlanResponseBody:
    enforce_auth = getattr(request.app.state, "enforce_auth", False)
    if enforce_auth and current_user is None:
        raise AuthenticationError("Authentication required to access the research planner.")
    if current_user is None and body.collection_id is not None:
        raise AuthenticationError("Authentication required to access private tenant collections.")

    auth_service = get_auth_service(request)
    session_factory = getattr(request.app.state, "session_factory", None)
    context = AgentContext(
        user_id=current_user.id if current_user is not None else None,
        workspace_id=getattr(current_user, "workspace_id", None) if current_user is not None else None,
        auth_enforced=enforce_auth or current_user is not None,
        session_factory=session_factory,
        auth_service=auth_service,
    )

    plan = planner.plan(
        goal=body.goal,
        collection_id=body.collection_id,
        context=context,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
    )
    resp = AgentPlanResponse.from_plan(plan)
    return AgentPlanResponseBody(**resp.to_dict())


@v1_router.post(
    "/agents/run",
    response_model=AgentRunResponseBody,
    status_code=200,
    summary="Execute a bounded autonomous research agent run",
    description=(
        "Executes a bounded iterative Reason-Act-Observe research agent loop with allowlisted tools, "
        "step limits, loop detection, and execution timeouts."
    ),
    responses={
        200: {
            "description": "Agent execution completed or reached safety boundary.",
            "model": AgentRunResponseBody,
        },
        400: {"description": "Malformed request, invalid parameters, or malformed action."},
        401: {"description": "Authentication required to run autonomous research agents."},
        403: {"description": "Operation violates tenant security boundaries."},
        500: {"description": "Agent execution failed due to an unrecoverable internal error."},
    },
)
def run_agent_endpoint(
    body: AgentRunRequestBody,
    request: Request,
    orchestrator: Any = Depends(get_agent_orchestrator),
    current_user: Any = Depends(get_optional_user),
) -> AgentRunResponseBody:
    enforce_auth = getattr(request.app.state, "enforce_auth", False)
    if enforce_auth and current_user is None:
        raise AuthenticationError("Authentication required to run autonomous research agents.")

    auth_service = get_auth_service(request)
    session_factory = getattr(request.app.state, "session_factory", None)
    context = AgentContext(
        user_id=current_user.id if current_user is not None else None,
        workspace_id=getattr(current_user, "workspace_id", None) if current_user is not None else None,
        auth_enforced=enforce_auth or current_user is not None,
        session_factory=session_factory,
        auth_service=auth_service,
    )

    try:
        result = orchestrator.run(
            goal=body.goal,
            context=context,
            max_steps=body.max_steps,
            timeout_seconds=body.timeout_seconds,
            model=body.model,
            temperature=body.temperature,
            max_output_tokens=body.max_output_tokens,
        )
    except AgentSecurityError as exc:
        raise ForbiddenError(str(exc)) from exc
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc
    except Exception as exc:
        raise AgentServiceError(f"Agent execution failed: {exc}") from exc

    resp = AgentRunResponse.from_result(result)
    return AgentRunResponseBody(**resp.to_dict())


# ===========================================================================
# Knowledge Graph Endpoints & Wire Models (Stage 5 Step 4.1)
# ===========================================================================

class CollectionGraphSummaryResponseBody(BaseModel):
    """Wire format for collection graph summary statistics."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(..., description="UUID of the collection.")
    total_entities: int = Field(..., ge=0, description="Total entity nodes in the collection graph.")
    total_relationships: int = Field(..., ge=0, description="Total relationship edges in the collection graph.")
    entity_counts_by_type: dict[str, int] = Field(default_factory=dict, description="Entity counts grouped by type.")
    relationship_counts_by_type: dict[str, int] = Field(default_factory=dict, description="Relationship counts grouped by type.")


class EntityTypeCountResponseBody(BaseModel):
    """Wire format for entity type counts."""

    model_config = ConfigDict(extra="forbid")

    entity_type: str = Field(..., description="Entity classification type.")
    count: int = Field(..., ge=0, description="Number of entities of this type.")


class RelationshipTypeCountResponseBody(BaseModel):
    """Wire format for relationship type counts."""

    model_config = ConfigDict(extra="forbid")

    relationship_type: str = Field(..., description="Relationship type name.")
    count: int = Field(..., ge=0, description="Number of relationships of this type.")


class AuthorCollaborationResponseBody(BaseModel):
    """Wire format for author co-authorship collaboration pairs."""

    model_config = ConfigDict(extra="forbid")

    author_id: str = Field(..., description="UUID of first author entity.")
    author_name: str = Field(..., description="Name of first author.")
    collaborator_id: str = Field(..., description="UUID of collaborator entity.")
    collaborator_name: str = Field(..., description="Name of collaborator.")
    shared_paper_ids: list[str] = Field(default_factory=list, description="UUIDs of shared research papers.")
    collaboration_count: int = Field(..., ge=1, description="Number of shared co-authored papers.")


class AuthorConnectivityResponseBody(BaseModel):
    """Wire format for author graph connectivity and degree centrality."""

    model_config = ConfigDict(extra="forbid")

    author_id: str = Field(..., description="UUID of author entity.")
    author_name: str = Field(..., description="Author name.")
    paper_count: int = Field(..., ge=0, description="Number of papers authored.")
    collaborator_count: int = Field(..., ge=0, description="Number of distinct co-authors.")
    total_collaborations: int = Field(..., ge=0, description="Total collaboration edges.")


class CollectionAnalyticsResponseBody(BaseModel):
    """Wire format for comprehensive collection graph analytics."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(..., description="UUID of the collection.")
    summary: CollectionGraphSummaryResponseBody = Field(..., description="Overall graph summary.")
    entity_distribution: list[EntityTypeCountResponseBody] = Field(default_factory=list, description="Entity distribution by type.")
    relationship_distribution: list[RelationshipTypeCountResponseBody] = Field(default_factory=list, description="Relationship distribution by type.")
    top_collaborations: list[AuthorCollaborationResponseBody] = Field(default_factory=list, description="Top author collaboration pairs.")
    author_connectivity: list[AuthorConnectivityResponseBody] = Field(default_factory=list, description="Author degree centrality metrics.")


class GraphEntityItemResponseBody(BaseModel):
    """Wire format for a single graph entity."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="UUID of the entity.")
    collection_id: str = Field(..., description="UUID of the owning collection.")
    name: str = Field(..., description="Entity name as extracted.")
    canonical_name: str = Field(..., description="Normalized canonical identifier.")
    entity_type: str = Field(..., description="Entity classification type.")
    description: str | None = Field(default=None, description="Entity description or summary.")
    aliases: list[str] = Field(default_factory=list, description="Alternative names and synonyms.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary entity metadata.")
    created_at: str | None = Field(default=None, description="Creation timestamp.")
    updated_at: str | None = Field(default=None, description="Last update timestamp.")


class GraphRelationshipItemResponseBody(BaseModel):
    """Wire format for a single graph relationship edge."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="UUID of the relationship.")
    collection_id: str = Field(..., description="UUID of the owning collection.")
    relationship_type: str = Field(..., description="Semantic relation type.")
    source_type: str = Field(..., description="Source node type ('entity' or 'paper').")
    source_id: str = Field(..., description="Source node UUID.")
    target_type: str = Field(..., description="Target node type ('entity' or 'paper').")
    target_id: str = Field(..., description="Target node UUID.")
    confidence: float | None = Field(default=None, description="Extraction confidence score.")
    evidence_text: str | None = Field(default=None, description="Grounding excerpt from source document.")
    chunk_id: str | None = Field(default=None, description="UUID of source document chunk.")
    page_number: int | None = Field(default=None, description="Page number of evidence.")
    paper_id: str | None = Field(default=None, description="UUID of grounding research paper.")
    extraction_method: str | None = Field(default=None, description="Extraction method used.")
    created_at: str | None = Field(default=None, description="Creation timestamp.")
    updated_at: str | None = Field(default=None, description="Last update timestamp.")


class GraphEntityListResponseBody(BaseModel):
    """Wire format for entity list and search results."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(..., description="UUID of the collection.")
    entities: list[GraphEntityItemResponseBody] = Field(default_factory=list, description="Entities matching query/pagination.")
    total_count: int = Field(..., ge=0, description="Total matching entities.")
    offset: int = Field(..., ge=0, description="Pagination offset.")
    limit: int = Field(..., ge=1, description="Pagination limit.")


class GraphEntityDetailResponseBody(BaseModel):
    """Wire format for entity details with direct relationships."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="UUID of the entity.")
    collection_id: str = Field(..., description="UUID of the owning collection.")
    name: str = Field(..., description="Entity name.")
    canonical_name: str = Field(..., description="Canonical name.")
    entity_type: str = Field(..., description="Entity classification type.")
    description: str | None = Field(default=None, description="Entity description.")
    aliases: list[str] = Field(default_factory=list, description="Alternative aliases.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Entity metadata.")
    created_at: str | None = Field(default=None, description="Creation timestamp.")
    updated_at: str | None = Field(default=None, description="Last update timestamp.")
    relationships: list[GraphRelationshipItemResponseBody] = Field(default_factory=list, description="Direct relationship edges.")
    relationship_count: int = Field(..., ge=0, description="Number of direct relationship edges returned.")


class GraphNodeResponseBody(BaseModel):
    """Wire format for a graph node in traversal responses."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="UUID of the node.")
    node_type: str = Field(..., description="Node type ('entity' or 'paper').")
    name: str = Field(..., description="Display name or paper title.")
    entity_type: str | None = Field(default=None, description="Entity type if node_type is 'entity'.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Node properties and metadata.")


class GraphEdgeResponseBody(BaseModel):
    """Wire format for a graph edge with complete evidence provenance."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="UUID of the edge.")
    relationship_type: str = Field(..., description="Semantic relation type.")
    source_type: str = Field(..., description="Source endpoint type.")
    source_id: str = Field(..., description="Source endpoint UUID.")
    target_type: str = Field(..., description="Target endpoint type.")
    target_id: str = Field(..., description="Target endpoint UUID.")
    confidence: float | None = Field(default=None, description="Confidence score.")
    evidence_text: str | None = Field(default=None, description="Grounding text excerpt.")
    chunk_id: str | None = Field(default=None, description="UUID of source chunk.")
    page_number: int | None = Field(default=None, description="Page number of evidence.")
    paper_id: str | None = Field(default=None, description="UUID of grounding research paper.")
    extraction_method: str | None = Field(default=None, description="Extraction method.")
    relationship_metadata: dict[str, Any] | None = Field(default=None, description="Relationship metadata.")


class GraphNeighborhoodRequestBody(BaseModel):
    """Request wire format for neighborhood exploration."""

    model_config = ConfigDict(strict=True, extra="forbid")

    node_id: str = Field(..., description="Seed node UUID.")
    node_type: str = Field(..., description="Seed node type ('entity' or 'paper').")
    max_depth: int = Field(default=1, ge=1, le=3, description="Exploration depth (1 to 3).")
    max_nodes: int = Field(default=50, ge=1, le=100, description="Maximum total nodes (1 to 100).")
    max_edges: int = Field(default=100, ge=1, le=200, description="Maximum total edges (1 to 200).")
    direction: str = Field(default="both", description="Traversal direction ('outgoing', 'incoming', 'both').")
    relationship_types: list[str] | None = Field(default=None, description="Optional relationship types filter.")
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Minimum edge confidence score.")
    include_unscored: bool = Field(default=True, description="Whether to include unscored edges.")


class GraphSubgraphResponseBody(BaseModel):
    """Response wire format for neighborhood exploration."""

    model_config = ConfigDict(extra="forbid")

    seed_id: str = Field(..., description="UUID of the seed node.")
    seed_type: str = Field(..., description="Type of the seed node.")
    depth: int = Field(..., ge=0, description="Actual traversal depth.")
    nodes: list[GraphNodeResponseBody] = Field(default_factory=list, description="Nodes in the explored subgraph.")
    edges: list[GraphEdgeResponseBody] = Field(default_factory=list, description="Edges in the explored subgraph.")
    is_truncated: bool = Field(..., description="Whether exploration was truncated by a budget.")
    truncation_reason: str | None = Field(default=None, description="Reason for truncation if truncated.")


class GraphPathsRequestBody(BaseModel):
    """Request wire format for pathfinding."""

    model_config = ConfigDict(strict=True, extra="forbid")

    source_id: str = Field(..., description="Source node UUID.")
    source_type: str = Field(..., description="Source node type ('entity' or 'paper').")
    target_id: str = Field(..., description="Target node UUID.")
    target_type: str = Field(..., description="Target node type ('entity' or 'paper').")
    max_depth: int = Field(default=2, ge=1, le=3, description="Maximum path depth (1 to 3).")
    max_paths: int = Field(default=5, ge=1, le=10, description="Maximum paths to discover (1 to 10).")
    direction: str = Field(default="outgoing", description="Traversal direction ('outgoing', 'incoming', 'both').")
    relationship_types: list[str] | None = Field(default=None, description="Optional relationship types filter.")
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Minimum confidence score.")
    include_unscored: bool = Field(default=False, description="Whether to include unscored edges.")


class GraphPathResponseBody(BaseModel):
    """Wire format for a single discovered graph path."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNodeResponseBody] = Field(..., description="Ordered sequence of nodes along the path.")
    edges: list[GraphEdgeResponseBody] = Field(..., description="Ordered sequence of edges along the path.")
    length: int = Field(..., ge=1, description="Number of hops along the path.")
    confidence_score: float | None = Field(default=None, description="Combined path confidence score.")
    is_partially_unscored: bool = Field(..., description="Whether any edge in the path lacked confidence.")


class GraphPathsResponseBody(BaseModel):
    """Response wire format for pathfinding."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., description="Source node UUID.")
    source_type: str = Field(..., description="Source node type.")
    target_id: str = Field(..., description="Target node UUID.")
    target_type: str = Field(..., description="Target node type.")
    paths: list[GraphPathResponseBody] = Field(default_factory=list, description="Discovered paths.")
    path_count: int = Field(..., ge=0, description="Total paths returned.")


class AuthorCollaborationsResponseBody(BaseModel):
    """Response wire format for author collaborations."""

    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(..., description="UUID of the collection.")
    collaborations: list[AuthorCollaborationResponseBody] = Field(default_factory=list, description="Author collaboration records.")
    total_count: int = Field(..., ge=0, description="Total collaboration pairs returned.")


def get_graph_traversal_service(request: Request) -> Any:
    """Dependency: returns or initializes the KnowledgeGraphTraversalService."""
    service = getattr(request.app.state, "graph_traversal_service", None)
    if service is None:
        session_factory = getattr(request.app.state, "session_factory", None)
        if session_factory is not None:
            from nexora.graph.traversal import KnowledgeGraphTraversalService

            service = KnowledgeGraphTraversalService(session_factory=session_factory)
            request.app.state.graph_traversal_service = service
        else:
            raise InvalidConfigurationError("session_factory is not configured for knowledge graph traversal")
    return service


def get_graph_analytics_service(request: Request) -> Any:
    """Dependency: returns or initializes the KnowledgeGraphAnalyticsService."""
    service = getattr(request.app.state, "graph_analytics_service", None)
    if service is None:
        session_factory = getattr(request.app.state, "session_factory", None)
        if session_factory is not None:
            from nexora.graph.analytics_service import KnowledgeGraphAnalyticsService

            service = KnowledgeGraphAnalyticsService(session_factory=session_factory)
            request.app.state.graph_analytics_service = service
        else:
            raise InvalidConfigurationError("session_factory is not configured for knowledge graph analytics")
    return service


def _verify_graph_collection_access(collection_id: str, request: Request, current_user: Any) -> tuple[UUID, Any]:
    """Verify caller is authenticated and owns the specified collection for graph access.

    Enforces authentication even if enforce_auth=False because graph data is tenant-private.
    Raises:
        InvalidRequestError (400) if collection_id is not a valid UUID.
        AuthenticationError (401) if caller is unauthenticated.
        ForbiddenError (403) if caller does not own the collection.
        DocumentNotFoundError (404) if collection does not exist.
    """
    try:
        col_uuid = UUID(str(collection_id).strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid collection_id: must be a valid UUID.")

    auth_service = get_auth_service(request)
    if current_user is None or auth_service is None:
        raise AuthenticationError("Authentication credentials are required to access collection knowledge graph data.")

    auth_service.verify_collection_ownership(col_uuid, current_user.id)
    return col_uuid, auth_service


# ---------------------------------------------------------------------------
# 1. GET /api/v1/collections/{collection_id}/graph/summary
# ---------------------------------------------------------------------------

@v1_router.get(
    "/collections/{collection_id}/graph/summary",
    response_model=CollectionGraphSummaryResponseBody,
    status_code=200,
    summary="Get collection graph summary statistics",
    description="Returns quantitative entity and relationship counts for an authorized collection graph.",
    responses={
        200: {"description": "Graph summary statistics returned successfully.", "model": CollectionGraphSummaryResponseBody},
        400: {"description": "Invalid collection UUID."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not own or have access to this collection."},
        404: {"description": "Referenced collection was not found."},
        500: {"description": "Failed to retrieve graph summary statistics."},
    },
)
def get_graph_summary_endpoint(
    collection_id: str,
    request: Request,
    current_user: Any = Depends(get_current_user),
    analytics_service: Any = Depends(get_graph_analytics_service),
) -> CollectionGraphSummaryResponseBody:
    col_uuid, _ = _verify_graph_collection_access(collection_id, request, current_user)
    try:
        summary = analytics_service.get_summary(col_uuid)
    except Exception as exc:
        raise GraphServiceError(f"Failed to retrieve graph summary: {exc}") from exc

    return CollectionGraphSummaryResponseBody(
        collection_id=str(summary.collection_id),
        total_entities=summary.total_entities,
        total_relationships=summary.total_relationships,
        entity_counts_by_type=summary.entity_counts_by_type,
        relationship_counts_by_type=summary.relationship_counts_by_type,
    )


# ---------------------------------------------------------------------------
# 2. GET /api/v1/collections/{collection_id}/graph/analytics
# ---------------------------------------------------------------------------

@v1_router.get(
    "/collections/{collection_id}/graph/analytics",
    response_model=CollectionAnalyticsResponseBody,
    status_code=200,
    summary="Get comprehensive collection graph analytics",
    description="Returns graph summary, entity/relationship distributions, and author collaboration metrics.",
    responses={
        200: {"description": "Graph analytics returned successfully.", "model": CollectionAnalyticsResponseBody},
        400: {"description": "Invalid parameters or collection UUID."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not own or have access to this collection."},
        404: {"description": "Referenced collection was not found."},
        500: {"description": "Failed to retrieve graph analytics."},
    },
)
def get_graph_analytics_endpoint(
    collection_id: str,
    request: Request,
    top_collaborations_limit: int = Query(default=20, ge=1, le=100, description="Max top collaboration pairs to return."),
    author_connectivity_limit: int = Query(default=50, ge=1, le=100, description="Max author connectivity metrics to return."),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0, description="Minimum edge confidence score for author metrics."),
    include_unscored: bool = Query(default=True, description="Whether to include unscored edges in collaboration counts."),
    current_user: Any = Depends(get_current_user),
    analytics_service: Any = Depends(get_graph_analytics_service),
) -> CollectionAnalyticsResponseBody:
    col_uuid, _ = _verify_graph_collection_access(collection_id, request, current_user)
    try:
        analytics = analytics_service.get_collection_analytics(
            col_uuid,
            top_collaborations_limit=top_collaborations_limit,
            author_connectivity_limit=author_connectivity_limit,
            min_confidence=min_confidence,
            include_unscored=include_unscored,
        )
    except Exception as exc:
        raise GraphServiceError(f"Failed to retrieve graph analytics: {exc}") from exc

    return CollectionAnalyticsResponseBody(
        collection_id=str(analytics.collection_id),
        summary=CollectionGraphSummaryResponseBody(
            collection_id=str(analytics.summary.collection_id),
            total_entities=analytics.summary.total_entities,
            total_relationships=analytics.summary.total_relationships,
            entity_counts_by_type=analytics.summary.entity_counts_by_type,
            relationship_counts_by_type=analytics.summary.relationship_counts_by_type,
        ),
        entity_distribution=[
            EntityTypeCountResponseBody(entity_type=e.entity_type, count=e.count)
            for e in analytics.entity_distribution
        ],
        relationship_distribution=[
            RelationshipTypeCountResponseBody(relationship_type=r.relationship_type, count=r.count)
            for r in analytics.relationship_distribution
        ],
        top_collaborations=[
            AuthorCollaborationResponseBody(
                author_id=str(c.author_id),
                author_name=c.author_name,
                collaborator_id=str(c.collaborator_id),
                collaborator_name=c.collaborator_name,
                shared_paper_ids=[str(p) for p in c.shared_paper_ids],
                collaboration_count=c.collaboration_count,
            )
            for c in analytics.top_collaborations
        ],
        author_connectivity=[
            AuthorConnectivityResponseBody(
                author_id=str(m.author_id),
                author_name=m.author_name,
                paper_count=m.paper_count,
                collaborator_count=m.collaborator_count,
                total_collaborations=m.total_collaborations,
            )
            for m in analytics.author_connectivity
        ],
    )


# ---------------------------------------------------------------------------
# 3. GET /api/v1/collections/{collection_id}/graph/entities
# ---------------------------------------------------------------------------

@v1_router.get(
    "/collections/{collection_id}/graph/entities",
    response_model=GraphEntityListResponseBody,
    status_code=200,
    summary="List or search graph entities in a collection",
    description="List or search graph entities belonging to an authorized collection.",
    responses={
        200: {"description": "Entities returned successfully.", "model": GraphEntityListResponseBody},
        400: {"description": "Invalid parameters or collection UUID."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not own or have access to this collection."},
        404: {"description": "Referenced collection was not found."},
        500: {"description": "Failed to list or search graph entities."},
    },
)
def list_or_search_graph_entities(
    collection_id: str,
    request: Request,
    query: str | None = Query(default=None, max_length=200, description="Optional entity name query substring to search."),
    entity_type: str | None = Query(default=None, max_length=50, description="Optional entity type filter."),
    offset: int = Query(default=0, ge=0, description="Pagination offset."),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum items to return (max 20 when searching)."),
    current_user: Any = Depends(get_current_user),
) -> GraphEntityListResponseBody:
    col_uuid, _ = _verify_graph_collection_access(collection_id, request, current_user)

    # Enforce maximum search limit of 20 matching repository specification
    if query is not None and query.strip():
        if limit > 20:
            raise InvalidRequestError("limit cannot exceed 20 when searching entities by query.")

    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise InvalidConfigurationError("session_factory is not configured")

    from nexora.database.repository.graph_entity import GraphEntityRepository

    with session_factory() as session:
        entity_repo = GraphEntityRepository(session)
        if query is not None and query.strip():
            raw_entities = entity_repo.search_entities(
                collection_id=col_uuid,
                query=query.strip(),
                entity_type=entity_type,
                limit=limit,
            )
            entities = list(raw_entities)
            total_count = len(entities)
        else:
            raw_entities = entity_repo.list_by_collection_id(
                col_uuid,
                entity_type=entity_type,
                offset=offset,
                limit=limit,
            )
            entities = list(raw_entities)
            if entity_type is not None:
                type_counts = entity_repo.count_by_type(col_uuid)
                total_count = type_counts.get(entity_type.strip().lower(), 0)
            else:
                total_count = entity_repo.count_by_collection(col_uuid)

        items = [
            GraphEntityItemResponseBody(
                id=str(e.id),
                collection_id=str(e.collection_id),
                name=e.name,
                canonical_name=e.canonical_name,
                entity_type=e.entity_type,
                description=e.description,
                aliases=list(e.aliases or []),
                metadata=dict(e.entity_metadata or {}),
                created_at=e.created_at.isoformat() if e.created_at else None,
                updated_at=e.updated_at.isoformat() if e.updated_at else None,
            )
            for e in entities
        ]

    return GraphEntityListResponseBody(
        collection_id=str(col_uuid),
        entities=items,
        total_count=total_count,
        offset=offset,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# 4. GET /api/v1/collections/{collection_id}/graph/entities/{entity_id}
# ---------------------------------------------------------------------------

@v1_router.get(
    "/collections/{collection_id}/graph/entities/{entity_id}",
    response_model=GraphEntityDetailResponseBody,
    status_code=200,
    summary="Get entity details and its direct relationships",
    description="Retrieves specific entity metadata and direct relationships within an authorized collection.",
    responses={
        200: {"description": "Entity details returned successfully.", "model": GraphEntityDetailResponseBody},
        400: {"description": "Invalid entity or collection UUID."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not own or have access to this collection."},
        404: {"description": "Referenced entity or collection was not found."},
        500: {"description": "Failed to retrieve entity details."},
    },
)
def get_graph_entity_detail(
    collection_id: str,
    entity_id: str,
    request: Request,
    current_user: Any = Depends(get_current_user),
) -> GraphEntityDetailResponseBody:
    col_uuid, _ = _verify_graph_collection_access(collection_id, request, current_user)
    try:
        ent_uuid = UUID(str(entity_id).strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid entity_id: must be a valid UUID.")

    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise InvalidConfigurationError("session_factory is not configured")

    from nexora.database.repository.graph_entity import GraphEntityRepository
    from nexora.database.repository.graph_relationship import GraphRelationshipRepository

    with session_factory() as session:
        entity_repo = GraphEntityRepository(session)
        rel_repo = GraphRelationshipRepository(session)

        entity = entity_repo.get_by_id_scoped(ent_uuid, collection_id=col_uuid)
        if entity is None:
            raise DocumentNotFoundError(f"Entity '{entity_id}' not found in collection '{collection_id}'.")

        relationships = rel_repo.list_by_entity_id(ent_uuid, collection_id=col_uuid, limit=100)
        rel_items = [
            GraphRelationshipItemResponseBody(
                id=str(r.id),
                collection_id=str(r.collection_id),
                relationship_type=r.relationship_type,
                source_type=r.source_type,
                source_id=str(r.source_id),
                target_type=r.target_type,
                target_id=str(r.target_id),
                confidence=r.confidence,
                evidence_text=r.evidence_text,
                chunk_id=str(r.chunk_id) if r.chunk_id else None,
                page_number=r.page_number,
                paper_id=str(r.paper_id) if r.paper_id else None,
                extraction_method=r.extraction_method,
                created_at=r.created_at.isoformat() if r.created_at else None,
                updated_at=r.updated_at.isoformat() if r.updated_at else None,
            )
            for r in relationships
        ]

        return GraphEntityDetailResponseBody(
            id=str(entity.id),
            collection_id=str(entity.collection_id),
            name=entity.name,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            description=entity.description,
            aliases=list(entity.aliases or []),
            metadata=dict(entity.entity_metadata or {}),
            created_at=entity.created_at.isoformat() if entity.created_at else None,
            updated_at=entity.updated_at.isoformat() if entity.updated_at else None,
            relationships=rel_items,
            relationship_count=len(rel_items),
        )


# ---------------------------------------------------------------------------
# 5. POST /api/v1/collections/{collection_id}/graph/neighborhood
# ---------------------------------------------------------------------------

@v1_router.post(
    "/collections/{collection_id}/graph/neighborhood",
    response_model=GraphSubgraphResponseBody,
    status_code=200,
    summary="Explore bounded neighborhood around a seed node",
    description="Explores 1-hop or bounded multi-hop neighborhood around an entity or paper within an authorized collection.",
    responses={
        200: {"description": "Neighborhood explored successfully.", "model": GraphSubgraphResponseBody},
        400: {"description": "Invalid parameters, seed type, or UUID."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not own or have access to this collection."},
        404: {"description": "Referenced seed node or collection was not found."},
        500: {"description": "Failed to explore graph neighborhood."},
    },
)
def explore_graph_neighborhood_endpoint(
    collection_id: str,
    body: GraphNeighborhoodRequestBody,
    request: Request,
    current_user: Any = Depends(get_current_user),
    traversal_service: Any = Depends(get_graph_traversal_service),
) -> GraphSubgraphResponseBody:
    col_uuid, _ = _verify_graph_collection_access(collection_id, request, current_user)
    try:
        node_uuid = UUID(str(body.node_id).strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid node_id: must be a valid UUID.")

    node_type = body.node_type.strip().lower()
    if node_type not in ("entity", "paper"):
        raise InvalidRequestError("node_type must be either 'entity' or 'paper'.")

    direction = body.direction.strip().lower()
    if direction not in ("outgoing", "incoming", "both"):
        raise InvalidRequestError("direction must be 'outgoing', 'incoming', or 'both'.")

    try:
        subgraph = traversal_service.explore_multihop(
            start_id=node_uuid,
            start_type=node_type,
            collection_id=col_uuid,
            max_depth=body.max_depth,
            max_nodes=body.max_nodes,
            max_edges=body.max_edges,
            direction=direction,
            relationship_types=body.relationship_types,
            min_confidence=body.min_confidence,
            include_unscored=body.include_unscored,
        )
    except (ValueError, TypeError) as exc:
        raise InvalidRequestError(str(exc)) from exc
    except Exception as exc:
        raise GraphServiceError(f"Failed to explore graph neighborhood: {exc}") from exc

    return GraphSubgraphResponseBody(
        seed_id=str(subgraph.seed_id),
        seed_type=subgraph.seed_type,
        depth=subgraph.depth,
        nodes=[
            GraphNodeResponseBody(
                id=str(n.id),
                node_type=n.node_type,
                name=n.name,
                entity_type=n.entity_type,
                metadata=dict(n.metadata or {}),
            )
            for n in subgraph.nodes
        ],
        edges=[
            GraphEdgeResponseBody(
                id=str(e.id),
                relationship_type=e.relationship_type,
                source_type=e.source_type,
                source_id=str(e.source_id),
                target_type=e.target_type,
                target_id=str(e.target_id),
                confidence=e.confidence,
                evidence_text=e.evidence_text,
                chunk_id=str(e.chunk_id) if e.chunk_id else None,
                page_number=e.page_number,
                paper_id=str(e.paper_id) if e.paper_id else None,
                extraction_method=e.extraction_method,
                relationship_metadata=dict(e.relationship_metadata) if e.relationship_metadata else None,
            )
            for e in subgraph.edges
        ],
        is_truncated=subgraph.is_truncated,
        truncation_reason=subgraph.truncation_reason,
    )


# ---------------------------------------------------------------------------
# 6. POST /api/v1/collections/{collection_id}/graph/paths
# ---------------------------------------------------------------------------

@v1_router.post(
    "/collections/{collection_id}/graph/paths",
    response_model=GraphPathsResponseBody,
    status_code=200,
    summary="Find connecting paths between two research nodes",
    description="Finds deterministic paths between source and target nodes with multiplicative confidence scoring.",
    responses={
        200: {"description": "Paths discovered successfully.", "model": GraphPathsResponseBody},
        400: {"description": "Invalid parameters, endpoint types, or UUIDs."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not own or have access to this collection."},
        404: {"description": "Referenced endpoints or collection was not found."},
        500: {"description": "Failed to discover graph paths."},
    },
)
def find_graph_paths_endpoint(
    collection_id: str,
    body: GraphPathsRequestBody,
    request: Request,
    current_user: Any = Depends(get_current_user),
    traversal_service: Any = Depends(get_graph_traversal_service),
) -> GraphPathsResponseBody:
    col_uuid, _ = _verify_graph_collection_access(collection_id, request, current_user)
    try:
        src_uuid = UUID(str(body.source_id).strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid source_id: must be a valid UUID.")
    try:
        tgt_uuid = UUID(str(body.target_id).strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid target_id: must be a valid UUID.")

    src_type = body.source_type.strip().lower()
    tgt_type = body.target_type.strip().lower()
    if src_type not in ("entity", "paper"):
        raise InvalidRequestError("source_type must be either 'entity' or 'paper'.")
    if tgt_type not in ("entity", "paper"):
        raise InvalidRequestError("target_type must be either 'entity' or 'paper'.")

    direction = body.direction.strip().lower()
    if direction not in ("outgoing", "incoming", "both"):
        raise InvalidRequestError("direction must be 'outgoing', 'incoming', or 'both'.")

    try:
        paths = traversal_service.find_paths(
            source_id=src_uuid,
            source_type=src_type,
            target_id=tgt_uuid,
            target_type=tgt_type,
            collection_id=col_uuid,
            max_depth=body.max_depth,
            max_paths=body.max_paths,
            direction=direction,
            relationship_types=body.relationship_types,
            min_confidence=body.min_confidence,
            include_unscored=body.include_unscored,
        )
    except (ValueError, TypeError) as exc:
        raise InvalidRequestError(str(exc)) from exc
    except Exception as exc:
        raise GraphServiceError(f"Failed to find graph paths: {exc}") from exc

    path_items = [
        GraphPathResponseBody(
            nodes=[
                GraphNodeResponseBody(
                    id=str(n.id),
                    node_type=n.node_type,
                    name=n.name,
                    entity_type=n.entity_type,
                    metadata=dict(n.metadata or {}),
                )
                for n in p.nodes
            ],
            edges=[
                GraphEdgeResponseBody(
                    id=str(e.id),
                    relationship_type=e.relationship_type,
                    source_type=e.source_type,
                    source_id=str(e.source_id),
                    target_type=e.target_type,
                    target_id=str(e.target_id),
                    confidence=e.confidence,
                    evidence_text=e.evidence_text,
                    chunk_id=str(e.chunk_id) if e.chunk_id else None,
                    page_number=e.page_number,
                    paper_id=str(e.paper_id) if e.paper_id else None,
                    extraction_method=e.extraction_method,
                    relationship_metadata=dict(e.relationship_metadata) if e.relationship_metadata else None,
                )
                for e in p.edges
            ],
            length=p.length,
            confidence_score=p.confidence_score,
            is_partially_unscored=p.is_partially_unscored,
        )
        for p in paths
    ]

    return GraphPathsResponseBody(
        source_id=str(src_uuid),
        source_type=src_type,
        target_id=str(tgt_uuid),
        target_type=tgt_type,
        paths=path_items,
        path_count=len(path_items),
    )


# ---------------------------------------------------------------------------
# 7. GET /api/v1/collections/{collection_id}/graph/collaborations
# ---------------------------------------------------------------------------

@v1_router.get(
    "/collections/{collection_id}/graph/collaborations",
    response_model=AuthorCollaborationsResponseBody,
    status_code=200,
    summary="Get author collaboration pairs in a collection",
    description="Returns aggregated co-authorship collaboration pairs within an authorized collection.",
    responses={
        200: {"description": "Collaborations returned successfully.", "model": AuthorCollaborationsResponseBody},
        400: {"description": "Invalid parameters or collection UUID."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not own or have access to this collection."},
        404: {"description": "Referenced collection was not found."},
        500: {"description": "Failed to retrieve author collaborations."},
    },
)
def get_author_collaborations_endpoint(
    collection_id: str,
    request: Request,
    author_id: str | None = Query(default=None, description="Optional author entity UUID filter."),
    min_shared_papers: int = Query(default=1, ge=1, description="Minimum shared papers between authors."),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0, description="Minimum relationship confidence score."),
    include_unscored: bool = Query(default=True, description="Whether to include unscored edges in collaboration counts."),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum collaboration records to return."),
    current_user: Any = Depends(get_current_user),
    analytics_service: Any = Depends(get_graph_analytics_service),
) -> AuthorCollaborationsResponseBody:
    col_uuid, _ = _verify_graph_collection_access(collection_id, request, current_user)
    author_uuid: UUID | None = None
    if author_id is not None:
        try:
            author_uuid = UUID(str(author_id).strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError("Invalid author_id: must be a valid UUID.")

    try:
        collaborations = analytics_service.get_author_collaborations(
            col_uuid,
            author_id=author_uuid,
            min_shared_papers=min_shared_papers,
            min_confidence=min_confidence,
            include_unscored=include_unscored,
            limit=limit,
        )
    except (ValueError, TypeError) as exc:
        raise InvalidRequestError(str(exc)) from exc
    except Exception as exc:
        raise GraphServiceError(f"Failed to retrieve author collaborations: {exc}") from exc

    items = [
        AuthorCollaborationResponseBody(
            author_id=str(c.author_id),
            author_name=c.author_name,
            collaborator_id=str(c.collaborator_id),
            collaborator_name=c.collaborator_name,
            shared_paper_ids=[str(p) for p in c.shared_paper_ids],
            collaboration_count=c.collaboration_count,
        )
        for c in collaborations
    ]

    return AuthorCollaborationsResponseBody(
        collection_id=str(col_uuid),
        collaborations=items,
        total_count=len(items),
    )


# ---------------------------------------------------------------------------
# 8. Graph Extraction Jobs Endpoints
# ---------------------------------------------------------------------------

@v1_router.get(
    "/graph/jobs/{job_id}",
    response_model=GraphExtractionJobResponseBody,
    status_code=200,
    summary="Get status of a graph extraction job",
    description="Returns lifecycle status, attempts, error details, and claim metadata for a graph extraction job.",
    responses={
        200: {"description": "Job details returned successfully.", "model": GraphExtractionJobResponseBody},
        400: {"description": "Invalid job UUID format."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not have access to this collection's job."},
        404: {"description": "Job not found."},
        500: {"description": "Database or unexpected error."},
    },
)
def get_graph_job_endpoint(
    job_id: str,
    request: Request,
    current_user: Any = Depends(get_current_user),
) -> GraphExtractionJobResponseBody:
    try:
        job_uuid = UUID(str(job_id).strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid job_id: must be a valid UUID.")

    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise InvalidConfigurationError("session_factory is not configured for graph jobs")

    from nexora.database.repository.graph_extraction_job import GraphExtractionJobRepository

    with session_factory() as session:
        job_repo = GraphExtractionJobRepository(session)
        job = job_repo.get_by_id(job_uuid)
        if job is None:
            raise RecordNotFoundError(f"Graph extraction job {job_uuid} not found.")

        _verify_graph_collection_access(str(job.collection_id), request, current_user)

        return GraphExtractionJobResponseBody(
            job_id=str(job.id),
            paper_id=str(job.paper_id),
            collection_id=str(job.collection_id),
            status=job.status.value if hasattr(job.status, "value") else str(job.status),
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            metadata_extracted=job.metadata_extracted,
            error_message=job.last_error,
            created_at=job.created_at.isoformat() if job.created_at else None,
            updated_at=job.updated_at.isoformat() if job.updated_at else None,
            claimed_by=job.claimed_by,
            lease_expires_at=job.lease_expires_at.isoformat() if job.lease_expires_at else None,
        )


@v1_router.post(
    "/graph/jobs/{job_id}/retry",
    response_model=GraphExtractionJobResponseBody,
    status_code=200,
    summary="Manually retry a failed graph extraction job",
    description="Resets a failed or stalled graph extraction job back to pending status for retry.",
    responses={
        200: {"description": "Job reset for retry successfully.", "model": GraphExtractionJobResponseBody},
        400: {"description": "Invalid job UUID format or job cannot be retried."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not have access to this collection's job."},
        404: {"description": "Job not found."},
        500: {"description": "Database or unexpected error."},
    },
)
def retry_graph_job_endpoint(
    job_id: str,
    request: Request,
    current_user: Any = Depends(get_current_user),
) -> GraphExtractionJobResponseBody:
    try:
        job_uuid = UUID(str(job_id).strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid job_id: must be a valid UUID.")

    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise InvalidConfigurationError("session_factory is not configured for graph jobs")

    from nexora.database.repository.graph_extraction_job import GraphExtractionJobRepository

    with session_factory() as session:
        job_repo = GraphExtractionJobRepository(session)
        job = job_repo.get_by_id(job_uuid)
        if job is None:
            raise RecordNotFoundError(f"Graph extraction job {job_uuid} not found.")

        _verify_graph_collection_access(str(job.collection_id), request, current_user)

        retried_job = job_repo.reset_job_for_retry(job_uuid)
        session.commit()

        return GraphExtractionJobResponseBody(
            job_id=str(retried_job.id),
            paper_id=str(retried_job.paper_id),
            collection_id=str(retried_job.collection_id),
            status=retried_job.status.value if hasattr(retried_job.status, "value") else str(retried_job.status),
            attempts=retried_job.attempts,
            max_attempts=retried_job.max_attempts,
            metadata_extracted=retried_job.metadata_extracted,
            error_message=retried_job.last_error,
            created_at=retried_job.created_at.isoformat() if retried_job.created_at else None,
            updated_at=retried_job.updated_at.isoformat() if retried_job.updated_at else None,
            claimed_by=retried_job.claimed_by,
            lease_expires_at=retried_job.lease_expires_at.isoformat() if retried_job.lease_expires_at else None,
        )


@v1_router.get(
    "/collections/{collection_id}/graph/jobs",
    response_model=list[GraphExtractionJobResponseBody],
    status_code=200,
    summary="List graph extraction jobs in a collection",
    description="Returns graph extraction jobs within an authorized collection, optionally filtered by paper or status.",
    responses={
        200: {"description": "List of jobs returned successfully."},
        400: {"description": "Invalid collection UUID format."},
        401: {"description": "Authentication credentials required."},
        403: {"description": "Caller does not have access to this collection."},
        500: {"description": "Database or unexpected error."},
    },
)
def list_graph_jobs_endpoint(
    collection_id: str,
    request: Request,
    paper_id: str | None = Query(default=None, description="Optional paper UUID filter."),
    status: str | None = Query(default=None, description="Optional status filter (e.g. pending, completed, failed)."),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum jobs to return."),
    current_user: Any = Depends(get_current_user),
) -> list[GraphExtractionJobResponseBody]:
    col_uuid, _ = _verify_graph_collection_access(collection_id, request, current_user)

    target_paper_uuid: UUID | None = None
    if paper_id is not None:
        try:
            target_paper_uuid = UUID(str(paper_id).strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError("Invalid paper_id: must be a valid UUID.")

    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise InvalidConfigurationError("session_factory is not configured for graph jobs")

    from nexora.database.models.graph_extraction_job import GraphExtractionJob
    from sqlalchemy import select

    with session_factory() as session:
        stmt = select(GraphExtractionJob).where(GraphExtractionJob.collection_id == col_uuid)
        if target_paper_uuid is not None:
            stmt = stmt.where(GraphExtractionJob.paper_id == target_paper_uuid)
        if status is not None and status.strip():
            stmt = stmt.where(GraphExtractionJob.status == status.strip().lower())

        stmt = stmt.order_by(GraphExtractionJob.created_at.desc()).limit(limit)
        jobs = session.scalars(stmt).all()

        return [
            GraphExtractionJobResponseBody(
                job_id=str(j.id),
                paper_id=str(j.paper_id),
                collection_id=str(j.collection_id),
                status=j.status.value if hasattr(j.status, "value") else str(j.status),
                attempts=j.attempts,
                max_attempts=j.max_attempts,
                metadata_extracted=j.metadata_extracted,
                error_message=j.last_error,
                created_at=j.created_at.isoformat() if j.created_at else None,
                updated_at=j.updated_at.isoformat() if j.updated_at else None,
                claimed_by=j.claimed_by,
                lease_expires_at=j.lease_expires_at.isoformat() if j.lease_expires_at else None,
            )
            for j in jobs
        ]
