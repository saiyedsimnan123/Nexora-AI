"""HTTP endpoints. Adapts HTTP requests to application services; no business logic."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from nexora.api.errors import (
    FileTooLargeError,
    InvalidConfigurationError,
    InvalidRequestError,
    ResearchServiceError,
)
from nexora.api.models import (
    DocumentIngestResponse,
    PaperAnalysisRequest,
    PaperAnalysisResponse,
    PaperComparisonRequest,
    PaperComparisonResponse,
    ResearchGapsRequest,
    ResearchGapsResponse,
    ResearchRequest,
    ResearchVerifyRequest,
    ResearchVerifyResponse,
)
from nexora.api.service import ResearchApplicationService


class ResearchRequestBody(BaseModel):
    """Wire format. Strict types only; semantic rules live in ResearchRequest."""

    model_config = ConfigDict(strict=True, extra="forbid")

    query: str
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


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


def get_rag_pipeline(request: Request) -> Any:
    """Dependency: the RAG pipeline injected into the app by create_app()."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        from nexora.rag.pipeline import create_offline_rag_pipeline

        pipeline = create_offline_rag_pipeline()
        request.app.state.rag_pipeline = pipeline
    return pipeline




health_router = APIRouter()
v1_router = APIRouter(prefix="/api/v1")


@health_router.get("/health")
def health() -> dict[str, str]:
    """Liveness only; touches no dependency."""
    return {"status": "ok"}


@v1_router.post("/research", response_model=ResearchResponseBody)
def research(
    body: ResearchRequestBody,
    service: ResearchApplicationService = Depends(get_research_service),
) -> ResearchResponseBody:
    request = ResearchRequest(
        query=body.query,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
    )
    return ResearchResponseBody(**service.ask(request).to_dict())


@v1_router.post(
    "/research/query",
    response_model=RAGQueryResponseBody,
    summary="Query research papers with grounded RAG",
    description="Retrieve research evidence and generate an attributed answer with verifiable citations.",
)
def research_rag_query(
    body: RAGQueryRequestBody,
    pipeline: Any = Depends(get_rag_pipeline),
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

    try:
        response = pipeline.answer(
            request=body.query,
            retrieval_mode=body.retrieval_mode,
            top_k=body.top_k,
            min_score=body.min_score,
            temperature=body.temperature,
            max_output_tokens=body.max_output_tokens,
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
    pipeline: Any = Depends(get_rag_pipeline),
) -> RAGQueryResponseBody:
    return research_rag_query(body, pipeline=pipeline)


@v1_router.post(
    "/research/verify",
    response_model=ResearchVerifyResponseBody,
    status_code=200,
    summary="Verify research answer claims against cited evidence",
    description="Verifies claims in a generated or supplied research answer against retrieved evidence context.",
)
def research_verify_endpoint(
    body: ResearchVerifyRequestBody,
    service: ResearchApplicationService = Depends(get_research_service),
) -> ResearchVerifyResponseBody:
    req = ResearchVerifyRequest(
        query=body.query,
        text=body.text,
        model=body.model,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
        expected_paper_id=body.expected_paper_id,
    )
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
    service: Any = Depends(get_analysis_service),
) -> PaperAnalysisResponseBody:
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
    service: Any = Depends(get_analysis_service),
) -> PaperAnalysisResponseBody:
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
    service: Any = Depends(get_comparison_service),
) -> PaperComparisonResponseBody:
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
    "/research/gaps",
    response_model=ResearchGapsResponseBody,
    status_code=200,
    summary="Discover evidence-backed research gaps across papers or a collection",
    description="Discovers author-documented and AI-inferred candidate research gaps across specified papers or a collection with traceable evidence.",
)
def discover_research_gaps_endpoint(
    body: ResearchGapsRequestBody,
    service: Any = Depends(get_gap_service),
) -> ResearchGapsResponseBody:
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
    service: Any = Depends(get_gap_service),
) -> ResearchGapsResponseBody:
    req = ResearchGapsRequest(paper_ids=[paper_id])
    assert req.paper_ids is not None
    result = service.discover_gaps_for_papers(req.paper_ids)
    resp = ResearchGapsResponse.from_report(result)
    return ResearchGapsResponseBody(**resp.to_dict())



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
) -> DocumentIngestResponseBody:
    """Upload and ingest a research paper PDF into relational storage and vector index."""
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise InvalidRequestError("paper_id is required.")

    try:
        paper_uuid = UUID(paper_id.strip())
    except (ValueError, AttributeError):
        raise InvalidRequestError("Invalid paper_id: must be a valid UUID.")

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
