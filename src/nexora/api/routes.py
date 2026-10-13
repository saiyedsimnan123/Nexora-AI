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
)
from nexora.api.models import DocumentIngestResponse, ResearchRequest
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


def get_research_service(request: Request) -> ResearchApplicationService:
    """Dependency: the research service injected into the app by create_app()."""
    return request.app.state.research_service


def get_ingestion_service(request: Request) -> Any:
    """Dependency: the document ingestion service injected into the app by create_app()."""
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise InvalidConfigurationError("ingestion_service is not configured")
    return service


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
