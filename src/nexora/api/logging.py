"""Request correlation ID, structured JSON logging, and sensitive data redaction.

Provides:
- Context variable for request correlation tracking across threads and coroutines.
- ASGI RequestIDMiddleware validating incoming X-Request-ID headers and ensuring all
  HTTP responses (including errors) echo the correlation ID.
- StructuredJsonFormatter producing consistent machine-readable JSON log records
  with timestamp, level, logger, request ID, and message.
- Sensitive data redactor preventing authorization tokens, credentials, and passwords
  from leaking into log streams.
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
import json
import logging
import re
import uuid
from typing import Any, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Context variable to hold request correlation ID for current task / coroutine
_REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

# Header names
REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"

# Strict validation pattern for external request IDs (alphanumeric, hyphens, underscores, dots; 1-64 chars)
_SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")

# Patterns for redacting sensitive credentials in logs
_SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{8,}", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"((?:x-api-key|api[-_]?key|token|secret|password|passwd|key_hash)[\s:=]+)(?:['\"]?)[^\s,'\"&]{4,}(?:['\"]?)", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(postgres(?:ql)?(?:\+\w+)?://[^:]+:)[^@]+(@)", re.IGNORECASE), r"\1[REDACTED]\2"),
    (re.compile(r"(redis(?:s)?://(?:[^:]*:)?)[^@]+(@)", re.IGNORECASE), r"\1[REDACTED]\2"),
    (re.compile(r"\b(sk-[a-zA-Z0-9_\-]{20,})\b", re.IGNORECASE), r"[REDACTED]"),
]


def get_request_id() -> str:
    """Return the current request ID from context, or an empty string if not in a request."""
    return _REQUEST_ID_CTX.get()


def set_request_id(request_id: str) -> contextvars.Token[str]:
    """Set the request ID context variable."""
    return _REQUEST_ID_CTX.set(request_id)


def reset_request_id(token: contextvars.Token[str]) -> None:
    """Reset the request ID context variable."""
    _REQUEST_ID_CTX.reset(token)


def generate_request_id() -> str:
    """Generate a safe, unique random request ID."""
    return f"req_{uuid.uuid4().hex}"


def validate_request_id(value: str | None) -> str | None:
    """Validate incoming request ID format and length.

    Returns the sanitized string if valid; returns None if missing or unsafe.
    """
    if not value or not isinstance(value, str):
        return None
    val = value.strip()
    if not val or len(val) > 64 or not _SAFE_REQUEST_ID_REGEX.match(val):
        return None
    return val


def redact_sensitive_data(text: str) -> str:
    """Redact passwords, Bearer tokens, API keys, and connection strings from text."""
    if not text or not isinstance(text, str):
        return str(text)
    redacted = text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class StructuredJsonFormatter(logging.Formatter):
    """Logging formatter emitting single-line structured JSON logs with correlation IDs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        # Base structured fields
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        raw_message = record.getMessage()
        sanitized_message = redact_sensitive_data(raw_message)

        req_id = getattr(record, "request_id", None) or get_request_id()

        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "severity": record.levelname,
            "logger": record.name,
            "message": sanitized_message,
        }

        if req_id:
            payload["request_id"] = req_id

        if record.exc_info:
            payload["exception"] = redact_sensitive_data(self.formatException(record.exc_info))

        return json.dumps(payload, ensure_ascii=False)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for request ID propagation, generation, and response header injection."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        # Check incoming headers for an existing request/correlation ID
        incoming_id = request.headers.get(REQUEST_ID_HEADER) or request.headers.get(CORRELATION_ID_HEADER)
        valid_id = validate_request_id(incoming_id)

        # Generate a safe random ID if absent or invalid
        request_id = valid_id if valid_id is not None else generate_request_id()

        # Store in request.state for route handlers and set in contextvar for logging
        request.state.request_id = request_id
        token = set_request_id(request_id)

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            # For unhandled exceptions bubbling past exception handlers, return sanitized 500
            # with the correlation ID attached
            logger.error("Unhandled exception processing request: %s", exc, exc_info=True)
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "An internal error occurred.",
                    }
                },
            )
        finally:
            reset_request_id(token)

        # Ensure the response header always carries the correlation ID
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def setup_logging(level: str | int = "INFO", json_format: bool = True) -> None:
    """Configure the root or application logger with StructuredJsonFormatter."""
    root_logger = logging.getLogger()
    log_level = getattr(logging, str(level).upper(), logging.INFO) if isinstance(level, str) else level
    root_logger.setLevel(log_level)

    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))

    # Replace existing handlers on root
    root_logger.handlers = [handler]
