"""Tests for request correlation ID, structured JSON logging, and sensitive data redaction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from nexora.api.logging import (
    CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    RequestIDMiddleware,
    StructuredJsonFormatter,
    generate_request_id,
    get_request_id,
    redact_sensitive_data,
    set_request_id,
    validate_request_id,
)


def _create_test_app() -> FastAPI:
    """Build a minimal test app with RequestIDMiddleware."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ok")
    def ok_endpoint(request: Request) -> dict[str, str]:
        ctx_id = get_request_id()
        state_id = getattr(request.state, "request_id", None)
        return {"ctx_id": ctx_id, "state_id": state_id}

    @app.get("/client-error")
    def client_error_endpoint() -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "bad_request"})

    @app.get("/unhandled-crash")
    def crash_endpoint() -> None:
        raise RuntimeError("Simulated unhandled route failure")

    return app


# ---------------------------------------------------------------------------
# 1. Request ID Generation and Validation
# ---------------------------------------------------------------------------


def test_generate_request_id_format() -> None:
    req_id = generate_request_id()
    assert req_id.startswith("req_")
    assert len(req_id) == 36  # "req_" + 32-char hex UUID
    assert re.match(r"^req_[0-9a-f]{32}$", req_id)


@pytest.mark.parametrize(
    "valid_id",
    [
        "req_12345",
        "abc-def-123",
        "trace.id_999",
        "A" * 64,
        "valid-request-id.123_abc",
    ],
)
def test_validate_request_id_valid_values(valid_id: str) -> None:
    assert validate_request_id(valid_id) == valid_id


@pytest.mark.parametrize(
    "invalid_id",
    [
        None,
        "",
        "   ",
        "A" * 65,  # Exceeds 64 characters
        "req with spaces",
        "req<script>alert(1)</script>",
        "req; DROP TABLE users;--",
        "req\nnewline",
        "req\x00nullbyte",
        "req/slash",
    ],
)
def test_validate_request_id_invalid_values(invalid_id: Any) -> None:
    assert validate_request_id(invalid_id) is None


# ---------------------------------------------------------------------------
# 2. Middleware & Response Headers
# ---------------------------------------------------------------------------


def test_middleware_generates_id_when_none_provided() -> None:
    app = _create_test_app()
    client = TestClient(app)

    response = client.get("/ok")
    assert response.status_code == 200
    req_id = response.headers.get(REQUEST_ID_HEADER)
    assert req_id is not None
    assert req_id.startswith("req_")

    data = response.json()
    assert data["ctx_id"] == req_id
    assert data["state_id"] == req_id


def test_middleware_preserves_valid_incoming_id() -> None:
    app = _create_test_app()
    client = TestClient(app)

    custom_id = "trace-client-abc-12345"
    response = client.get("/ok", headers={REQUEST_ID_HEADER: custom_id})
    assert response.status_code == 200
    assert response.headers.get(REQUEST_ID_HEADER) == custom_id

    data = response.json()
    assert data["ctx_id"] == custom_id
    assert data["state_id"] == custom_id


def test_middleware_accepts_correlation_id_header_fallback() -> None:
    app = _create_test_app()
    client = TestClient(app)

    custom_id = "corr-client-999"
    response = client.get("/ok", headers={CORRELATION_ID_HEADER: custom_id})
    assert response.status_code == 200
    assert response.headers.get(REQUEST_ID_HEADER) == custom_id

    data = response.json()
    assert data["ctx_id"] == custom_id


def test_middleware_rejects_unsafe_incoming_id_and_generates_safe_one() -> None:
    app = _create_test_app()
    client = TestClient(app)

    unsafe_id = "<invalid>\nheader\x00value"
    response = client.get("/ok", headers={REQUEST_ID_HEADER: unsafe_id})
    assert response.status_code == 200
    req_id = response.headers.get(REQUEST_ID_HEADER)
    assert req_id is not None
    assert req_id != unsafe_id
    assert req_id.startswith("req_")


def test_middleware_echoes_request_id_on_client_error() -> None:
    app = _create_test_app()
    client = TestClient(app)

    custom_id = "trace-400-test"
    response = client.get("/client-error", headers={REQUEST_ID_HEADER: custom_id})
    assert response.status_code == 400
    assert response.headers.get(REQUEST_ID_HEADER) == custom_id


def test_middleware_catches_unhandled_crash_and_returns_500_with_request_id() -> None:
    app = _create_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    custom_id = "trace-crash-test"
    response = client.get("/unhandled-crash", headers={REQUEST_ID_HEADER: custom_id})
    assert response.status_code == 500
    assert response.headers.get(REQUEST_ID_HEADER) == custom_id
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "internal_error"


def test_contextvar_is_reset_after_request() -> None:
    app = _create_test_app()
    client = TestClient(app)

    assert get_request_id() == ""
    response = client.get("/ok")
    assert response.status_code == 200
    # Must be reset back to empty after request completes
    assert get_request_id() == ""


# ---------------------------------------------------------------------------
# 3. Sensitive Data Redaction
# ---------------------------------------------------------------------------


def test_redact_sensitive_bearer_token() -> None:
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token"
    redacted = redact_sensitive_data(text)
    assert "eyJhbGci" not in redacted
    assert "Authorization: Bearer [REDACTED]" in redacted


def test_redact_sensitive_api_keys() -> None:
    cases = [
        ("X-API-Key: my_super_secret_api_key_12345", "X-API-Key: [REDACTED]"),
        ("api_key='abc123secret'", "api_key=[REDACTED]"),
        ("secret: 'my-confidential-token-value'", "secret: [REDACTED]"),
        ("password: 'MyVerySecretPassword123!'", "password: [REDACTED]"),
    ]
    for raw, expected_substr in cases:
        redacted = redact_sensitive_data(raw)
        assert expected_substr in redacted


def test_redact_openai_sk_keys() -> None:
    text = "Using OpenAI key sk-proj-1234567890abcdef1234567890 for embeddings"
    redacted = redact_sensitive_data(text)
    assert "sk-proj-1234567890abcdef" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_database_and_redis_connection_strings() -> None:
    pg_url = "Connecting to postgresql+psycopg://dbuser:supersecretpass@db.example.com:5432/nexora"
    redacted_pg = redact_sensitive_data(pg_url)
    assert "supersecretpass" not in redacted_pg
    assert "postgresql+psycopg://dbuser:[REDACTED]@db.example.com:5432/nexora" in redacted_pg

    redis_url = "Connecting to redis://:mypassword@cache.example.com:6379/0"
    redacted_redis = redact_sensitive_data(redis_url)
    assert "mypassword" not in redacted_redis
    assert "redis://:[REDACTED]@cache.example.com:6379/0" in redacted_redis


# ---------------------------------------------------------------------------
# 4. Structured JSON Logging Formatter
# ---------------------------------------------------------------------------


def test_structured_json_formatter_fields() -> None:
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="nexora.api.service",
        level=logging.INFO,
        pathname="service.py",
        lineno=42,
        msg="Processing query for tenant %s",
        args=("tenant_1",),
        exc_info=None,
    )

    token = set_request_id("req_test_log_123")
    try:
        formatted = formatter.format(record)
        data = json.loads(formatted)
        assert data["level"] == "INFO"
        assert data["severity"] == "INFO"
        assert data["logger"] == "nexora.api.service"
        assert data["message"] == "Processing query for tenant tenant_1"
        assert data["request_id"] == "req_test_log_123"
        assert "timestamp" in data
    finally:
        from nexora.api.logging import reset_request_id
        reset_request_id(token)


def test_structured_json_formatter_redacts_credentials_in_message() -> None:
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="nexora.auth",
        level=logging.WARNING,
        pathname="auth.py",
        lineno=10,
        msg="Failed login with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 and password: secret_password_here",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert "secret_password_here" not in data["message"]
    assert "eyJhbGci" not in data["message"]
    assert "[REDACTED]" in data["message"]


def test_structured_json_formatter_formats_and_redacts_exceptions() -> None:
    formatter = StructuredJsonFormatter()
    try:
        raise ValueError("Failed connecting to redis://:leaked_pw@127.0.0.1:6379/0")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="nexora.db",
        level=logging.ERROR,
        pathname="db.py",
        lineno=50,
        msg="Unhandled database error",
        args=(),
        exc_info=exc_info,
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert data["level"] == "ERROR"
    assert "exception" in data
    assert "leaked_pw" not in data["exception"]
    assert "[REDACTED]" in data["exception"]
