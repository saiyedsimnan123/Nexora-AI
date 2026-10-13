"""HTTP endpoints. Adapts HTTP requests to application services; no business logic."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, ConfigDict

from nexora.api.errors import (
    FileTooLargeError,
    InvalidConfigurationError,
    InvalidRequestError,
)
from nexora.api.models import ResearchRequest
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

    document_id: str
    paper_id: str
    chunks_count: int
    indexed_chunks_count: int


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


@v1_router.post("/documents/ingest", response_model=DocumentIngestResponseBody)
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
            tmp_file.flush()
        except Exception:
            tmp_file.close()
            if tmp_path.exists():
                os.unlink(tmp_path)
            raise

    if total_bytes == 0:
        if tmp_path.exists():
            os.unlink(tmp_path)
        raise InvalidRequestError("Uploaded file is empty.")

    try:
        pipeline = getattr(request.app.state, "document_pipeline", None)

        if hasattr(service, "process_and_ingest"):
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

        return DocumentIngestResponseBody(
            document_id=str(result.document_id),
            paper_id=str(result.paper_id),
            chunks_count=result.chunks_count,
            indexed_chunks_count=result.indexed_chunks_count,
        )
    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)
