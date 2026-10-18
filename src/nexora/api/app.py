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
    AuthenticationError,
    ComparisonServiceError,
    DocumentNotFoundError,
    FileTooLargeError,
    ForbiddenError,
    InvalidConfigurationError,
    InvalidRequestError,
    ResearchServiceError,
    VerificationServiceError,
    GapDiscoveryServiceError,
)
from nexora.api.routes import health_router, v1_router
from nexora.database.repository.base import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from nexora.discovery.errors import (
    DiscoveryConnectionError,
    DiscoveryError,
    DiscoveryRateLimitError,
    DiscoveryResponseError,
    DiscoveryTimeoutError,
    InvalidIdentifierError,
    InvalidPDFError,
    OversizedPDFError,
    PDFNotFoundError,
    PaperNotFoundError,
    SSRFProtectionError,
)
from nexora.document.ingestion import VectorIndexingError


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error}, headers=headers)


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


async def _verification_failed(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "verification_failed", "The research verification could not be completed.")


async def _gap_discovery_failed(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "gap_discovery_failed", "The research gap discovery could not be completed.")


async def _misconfigured(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "server_misconfigured", "The service is not configured correctly.")


async def _validation_failed(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        {"loc": [str(part) for part in err.get("loc", ())], "type": str(err.get("type", ""))}
        for err in exc.errors()
    ]
    return _error_response(422, "validation_error", "Request validation failed.", details)


async def _unauthorized(request: Request, exc: Exception) -> JSONResponse:
    message = str(exc) if str(exc) else "Authentication credentials are required or invalid."
    return _error_response(
        401, "unauthorized", message, headers={"WWW-Authenticate": "Bearer"}
    )


async def _forbidden(request: Request, exc: Exception) -> JSONResponse:
    message = str(exc) if str(exc) else "Access to the requested resource is forbidden."
    return _error_response(403, "forbidden", message)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    try:
        phrase = HTTPStatus(exc.status_code).phrase
    except ValueError:
        phrase = "HTTP error"
    headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
    return _error_response(exc.status_code, "http_error", phrase, headers=headers)


async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(500, "internal_error", "An internal error occurred.")


async def _rate_limited(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(
        429, "rate_limited", "Upstream discovery provider rate limit exceeded. Please retry later."
    )


async def _gateway_timeout(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(504, "gateway_timeout", "Upstream discovery provider request timed out.")


async def _bad_gateway(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(502, "bad_gateway", "Upstream discovery provider error or unreachable.")


async def _discovery_failed(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(
        500, "discovery_failed", "The scholarly discovery request could not be completed."
    )


def create_app(
    research_service: Any,
    *,
    rag_pipeline: Any = None,
    ingestion_service: Any = None,
    analysis_service: Any = None,
    comparison_service: Any = None,
    gap_service: Any = None,
    discovery_service: Any = None,
    document_pipeline: Any = None,
    session_factory: Any = None,
    auth_service: Any = None,
    enforce_auth: bool | None = None,
    cors_config: Any = None,
    rate_limit_config: Any = None,
    rate_limit_storage: Any = None,
    max_upload_size_bytes: int = 50 * 1024 * 1024,
) -> FastAPI:
    """Build the app around injected research, ingestion, analysis, comparison, gap, discovery, and auth services."""
    if not callable(getattr(research_service, "ask", None)):
        raise InvalidConfigurationError("research_service must provide ask(request)")

    if enforce_auth is None:
        enforce_auth = auth_service is not None

    if enforce_auth and auth_service is None and session_factory is not None:
        from nexora.auth.service import AuthenticationService
        auth_service = AuthenticationService(session_factory)

    if enforce_auth and auth_service is None and session_factory is None:
        raise InvalidConfigurationError(
            "Authentication is enforced but no session_factory or auth_service is configured."
        )

    app = FastAPI(
        title="Nexora AI",
        description=(
            "Research intelligence API: evidence-grounded answers to research questions "
            "with full citation attribution, lexical BM25, and hybrid retrieval."
        ),
        version=__version__,
    )
    app.state.research_service = research_service
    app.state.rag_pipeline = rag_pipeline
    app.state.ingestion_service = ingestion_service
    app.state.analysis_service = analysis_service
    app.state.comparison_service = comparison_service
    app.state.gap_service = gap_service
    app.state.discovery_service = discovery_service
    app.state.document_pipeline = document_pipeline
    app.state.session_factory = session_factory
    app.state.auth_service = auth_service
    app.state.enforce_auth = enforce_auth
    app.state.cors_config = cors_config
    app.state.rate_limit_config = rate_limit_config
    app.state.max_upload_size_bytes = max_upload_size_bytes

    # 1. Rate Limit Middleware (runs after CORS, before endpoints)
    if rate_limit_config is not None and getattr(rate_limit_config, "enabled", False):
        from nexora.api.rate_limit import RateLimitMiddleware

        app.add_middleware(
            RateLimitMiddleware,
            config=rate_limit_config,
            storage=rate_limit_storage,
        )

    # 2. CORS Middleware (outermost: handles preflight OPTIONS and adds CORS headers to all responses)
    if (
        cors_config is not None
        and getattr(cors_config, "enabled", False)
        and getattr(cors_config, "allow_origins", None)
    ):
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_config.allow_origins),
            allow_credentials=cors_config.allow_credentials,
            allow_methods=list(cors_config.allow_methods),
            allow_headers=list(cors_config.allow_headers),
            max_age=cors_config.max_age,
        )

    app.include_router(health_router)
    app.include_router(v1_router)

    app.add_exception_handler(AuthenticationError, _unauthorized)
    app.add_exception_handler(ForbiddenError, _forbidden)
    app.add_exception_handler(FileTooLargeError, _file_too_large)
    app.add_exception_handler(OversizedPDFError, _file_too_large)
    app.add_exception_handler(DocumentNotFoundError, _not_found)
    app.add_exception_handler(RecordNotFoundError, _not_found)
    app.add_exception_handler(PaperNotFoundError, _not_found)
    app.add_exception_handler(PDFNotFoundError, _not_found)
    app.add_exception_handler(DuplicateRecordError, _conflict)
    app.add_exception_handler(VectorIndexingError, _indexing_failed)
    app.add_exception_handler(InvalidRequestError, _invalid_request)
    app.add_exception_handler(InvalidIdentifierError, _invalid_request)
    app.add_exception_handler(InvalidPDFError, _invalid_request)
    app.add_exception_handler(SSRFProtectionError, _invalid_request)
    app.add_exception_handler(DiscoveryRateLimitError, _rate_limited)
    app.add_exception_handler(DiscoveryTimeoutError, _gateway_timeout)
    app.add_exception_handler(DiscoveryConnectionError, _bad_gateway)
    app.add_exception_handler(DiscoveryResponseError, _bad_gateway)
    app.add_exception_handler(DiscoveryError, _discovery_failed)
    app.add_exception_handler(ResearchServiceError, _research_failed)
    app.add_exception_handler(AnalysisServiceError, _analysis_failed)
    app.add_exception_handler(ComparisonServiceError, _comparison_failed)
    app.add_exception_handler(VerificationServiceError, _verification_failed)
    app.add_exception_handler(GapDiscoveryServiceError, _gap_discovery_failed)
    app.add_exception_handler(InvalidConfigurationError, _misconfigured)
    app.add_exception_handler(RequestValidationError, _validation_failed)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected)
    return app


def create_offline_app(*, rag_pipeline: Any = None) -> FastAPI:
    """Convenience factory to create a fully offline FastAPI app.

    Uses an in-memory RAG pipeline backed by the curated research paper benchmark.
    Requires zero API keys, external servers, or network calls.
    """
    from nexora.rag.pipeline import create_offline_rag_pipeline

    if rag_pipeline is None:
        rag_pipeline = create_offline_rag_pipeline()

    class _OfflineResearchService:
        def __init__(self, pipeline: Any) -> None:
            self._pipeline = pipeline

        def ask(self, request: Any) -> Any:
            from nexora.research.models import ResearchAnswer

            query_text = request.query if hasattr(request, "query") else str(request)
            resp = self._pipeline.answer(query_text)
            return ResearchAnswer(
                text=resp.answer,
                query=query_text,
                model=resp.model,
                usage=resp.usage,
                evidence_count=resp.evidence_count,
            )

    return create_app(
        _OfflineResearchService(rag_pipeline),
        rag_pipeline=rag_pipeline,
        enforce_auth=False,
    )


class _LazyASGIApp:
    """Lazy ASGI application that initializes create_offline_app on first request."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    def _get_app(self) -> FastAPI:
        if self._app is None:
            self._app = create_offline_app()
        return self._app

    @property
    def state(self) -> Any:
        return self._get_app().state

    @property
    def openapi_schema(self) -> Any:
        return self._get_app().openapi_schema

    def openapi(self) -> dict[str, Any]:
        return self._get_app().openapi()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self._get_app()(scope, receive, send)


app = _LazyASGIApp()


