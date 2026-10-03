"""FastAPI application factory and centralized error handling."""

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from nexora import __version__
from nexora.api.errors import (
    InvalidConfigurationError,
    InvalidRequestError,
    ResearchServiceError,
)
from nexora.api.routes import health_router, v1_router


def _error_response(
    status_code: int, code: str, message: str, details: list[dict[str, Any]] | None = None
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error})


async def _invalid_request(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(400, "invalid_request", "Invalid research request.")


async def _research_failed(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "research_failed", "The research request could not be completed.")


async def _misconfigured(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "server_misconfigured", "The service is not configured correctly.")


async def _validation_failed(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Only locations and error types: never echo the submitted input.
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


def create_app(research_service: Any) -> FastAPI:
    """Build the app around an injected ResearchApplicationService.

    Creates no clients and performs no network calls.
    """
    if not callable(getattr(research_service, "ask", None)):
        raise InvalidConfigurationError("research_service must provide ask(request)")
    app = FastAPI(
        title="Nexora AI",
        description="Research intelligence API: evidence-grounded answers to research questions.",
        version=__version__,
    )
    app.state.research_service = research_service
    app.include_router(health_router)
    app.include_router(v1_router)
    app.add_exception_handler(InvalidRequestError, _invalid_request)
    app.add_exception_handler(ResearchServiceError, _research_failed)
    app.add_exception_handler(InvalidConfigurationError, _misconfigured)
    app.add_exception_handler(RequestValidationError, _validation_failed)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected)
    return app
