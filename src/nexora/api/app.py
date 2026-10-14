"""FastAPI application factory and centralized error handling."""

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from nexora import __version__
from nexora.api.errors import (
    AnalysisServiceError,
    ComparisonServiceError,
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidConfigurationError,
    InvalidRequestError,
    ResearchServiceError,
)
from nexora.api.routes import health_router, v1_router
from nexora.database.repository.base import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from nexora.document.ingestion import VectorIndexingError


def _error_response(
    status_code: int, code: str, message: str, details: list[dict[str, Any]] | None = None
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error})


async def _invalid_request(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(400, "invalid_request", "Invalid research request.")


async def _file_too_large(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(413, "file_too_large", "Uploaded file exceeds maximum allowed size.")


async def _not_found(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(404, "not_found", "The requested resource was not found.")


async def _conflict(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(409, "conflict", "The resource already exists.")


async def _indexing_failed(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "indexing_failed", "Document indexing failed.")


async def _research_failed(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "research_failed", "The research request could not be completed.")


async def _analysis_failed(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "analysis_failed", "The paper analysis could not be completed.")


async def _comparison_failed(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "comparison_failed", "The paper comparison could not be completed.")


async def _misconfigured(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "server_misconfigured", "The service is not configured correctly.")


async def _validation_failed(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        {"loc": [str(part) for part in err.get("loc", ())], "type": str(err.get("type", ""))}
        for err in exc.errors()
    ]
    return _error_response(422, "validation_error", "Request validation failed.", details)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    try:
        phrase = HTTPStatus(exc.status_code).phrase
    except ValueError:
        phrase = "HTTP error"
    return _error_response(exc.status_code, "http_error", phrase)


async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "internal_error", "An internal error occurred.")


def create_app(
    research_service: Any,
    *,
    ingestion_service: Any = None,
    analysis_service: Any = None,
    comparison_service: Any = None,
    document_pipeline: Any = None,
    session_factory: Any = None,
    max_upload_size_bytes: int = 50 * 1024 * 1024,
) -> FastAPI:
    """Build the app around injected research, ingestion, analysis, and comparison services."""
    if not callable(getattr(research_service, "ask", None)):
        raise InvalidConfigurationError("research_service must provide ask(request)")

    app = FastAPI(
        title="Nexora AI",
        description="Research intelligence API: evidence-grounded answers to research questions.",
        version=__version__,
    )
    app.state.research_service = research_service
    app.state.ingestion_service = ingestion_service
    app.state.analysis_service = analysis_service
    app.state.comparison_service = comparison_service
    app.state.document_pipeline = document_pipeline
    app.state.session_factory = session_factory
    app.state.max_upload_size_bytes = max_upload_size_bytes

    app.include_router(health_router)
    app.include_router(v1_router)

    app.add_exception_handler(FileTooLargeError, _file_too_large)
    app.add_exception_handler(DocumentNotFoundError, _not_found)
    app.add_exception_handler(RecordNotFoundError, _not_found)
    app.add_exception_handler(DuplicateRecordError, _conflict)
    app.add_exception_handler(VectorIndexingError, _indexing_failed)
    app.add_exception_handler(InvalidRequestError, _invalid_request)
    app.add_exception_handler(ResearchServiceError, _research_failed)
    app.add_exception_handler(AnalysisServiceError, _analysis_failed)
    app.add_exception_handler(ComparisonServiceError, _comparison_failed)
    app.add_exception_handler(InvalidConfigurationError, _misconfigured)
    app.add_exception_handler(RequestValidationError, _validation_failed)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected)
    return app
