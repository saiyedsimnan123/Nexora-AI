"""Tests for application lifespan, graceful shutdown, and resource cleanup (Milestone 36)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from nexora.api.app import _run_shutdown_cleanup, create_app


def _make_dummy_research_service() -> Any:
    service = MagicMock()
    service.ask.return_value = {"answer": "mocked"}
    return service


# ---------------------------------------------------------------------------
# 1. Database Connection Pool Disposal
# ---------------------------------------------------------------------------


def test_lifespan_disposes_database_engine_on_shutdown() -> None:
    mock_engine = MagicMock()
    mock_session_factory = MagicMock()
    mock_session_factory.bind = mock_engine

    app = create_app(
        _make_dummy_research_service(),
        session_factory=mock_session_factory,
        enforce_auth=False,
    )

    with TestClient(app):
        # App is running
        mock_engine.dispose.assert_not_called()

    # After lifespan exit (shutdown)
    mock_engine.dispose.assert_called_once()


def test_lifespan_disposes_engine_from_session_factory_kw() -> None:
    mock_engine = MagicMock()
    mock_session_factory = MagicMock(spec=[])
    mock_session_factory.kw = {"bind": mock_engine}

    app = create_app(
        _make_dummy_research_service(),
        session_factory=mock_session_factory,
        enforce_auth=False,
    )

    with TestClient(app):
        pass

    mock_engine.dispose.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Vector Store & Qdrant Client Closure
# ---------------------------------------------------------------------------


def test_lifespan_closes_qdrant_client_on_shutdown() -> None:
    mock_qdrant = MagicMock()
    app = create_app(
        _make_dummy_research_service(),
        vector_client=mock_qdrant,
        enforce_auth=False,
    )

    with TestClient(app):
        mock_qdrant.close.assert_not_called()

    mock_qdrant.close.assert_called_once()


def test_lifespan_closes_vector_store_from_ingestion_service() -> None:
    mock_vs = MagicMock()
    mock_ingestion = MagicMock()
    mock_ingestion._vector_store = mock_vs

    app = create_app(
        _make_dummy_research_service(),
        ingestion_service=mock_ingestion,
        enforce_auth=False,
    )

    with TestClient(app):
        pass

    mock_vs.close.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Rate Limit Storage Closure
# ---------------------------------------------------------------------------


def test_lifespan_closes_rate_limit_storage_on_shutdown() -> None:
    mock_storage = MagicMock()
    app = create_app(
        _make_dummy_research_service(),
        rate_limit_storage=mock_storage,
        enforce_auth=False,
    )

    with TestClient(app):
        mock_storage.close.assert_not_called()

    mock_storage.close.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Discovery HTTP Clients Closure
# ---------------------------------------------------------------------------


def test_lifespan_closes_discovery_provider_and_downloader_on_shutdown() -> None:
    mock_provider = MagicMock()
    mock_downloader = MagicMock()
    mock_discovery = MagicMock()
    mock_discovery._provider = mock_provider
    mock_discovery._downloader = mock_downloader

    app = create_app(
        _make_dummy_research_service(),
        discovery_service=mock_discovery,
        enforce_auth=False,
    )

    with TestClient(app):
        mock_provider.close.assert_not_called()
        mock_downloader.close.assert_not_called()

    mock_provider.close.assert_called_once()
    mock_downloader.close.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Custom Shutdown Hooks
# ---------------------------------------------------------------------------


def test_lifespan_executes_registered_shutdown_hooks() -> None:
    hook1 = MagicMock()
    hook2 = MagicMock()

    app = create_app(_make_dummy_research_service(), enforce_auth=False)
    app.state.shutdown_hooks.append(hook1)
    app.state.shutdown_hooks.append(hook2)

    with TestClient(app):
        hook1.assert_not_called()
        hook2.assert_not_called()

    hook1.assert_called_once()
    hook2.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Fault Isolation / Exception Resilience
# ---------------------------------------------------------------------------


def test_lifespan_cleanup_resilience_when_subsystem_raises() -> None:
    # Engine raises on dispose
    failing_engine = MagicMock()
    failing_engine.dispose.side_effect = RuntimeError("Database connection pool drop error")
    mock_session_factory = MagicMock()
    mock_session_factory.bind = failing_engine

    # Qdrant client raises on close
    failing_qdrant = MagicMock()
    failing_qdrant.close.side_effect = ConnectionResetError("Qdrant connection reset")

    # Redis storage succeeds
    healthy_storage = MagicMock()

    # Hook 1 raises, Hook 2 succeeds
    failing_hook = MagicMock(side_effect=ValueError("Hook exception"))
    healthy_hook = MagicMock()

    app = create_app(
        _make_dummy_research_service(),
        session_factory=mock_session_factory,
        vector_client=failing_qdrant,
        rate_limit_storage=healthy_storage,
        enforce_auth=False,
    )
    app.state.shutdown_hooks.append(failing_hook)
    app.state.shutdown_hooks.append(healthy_hook)

    # Lifespan must not crash or abort remaining cleanups
    with TestClient(app):
        pass

    failing_engine.dispose.assert_called_once()
    failing_qdrant.close.assert_called_once()
    healthy_storage.close.assert_called_once()
    failing_hook.assert_called_once()
    healthy_hook.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Repeated Cleanup Idempotency & Absent Resources
# ---------------------------------------------------------------------------


def test_repeated_shutdown_cleanup_is_safe_and_idempotent() -> None:
    mock_engine = MagicMock()
    mock_session_factory = MagicMock()
    mock_session_factory.bind = mock_engine

    app = create_app(
        _make_dummy_research_service(),
        session_factory=mock_session_factory,
        enforce_auth=False,
    )

    # Manually invoke cleanup twice
    _run_shutdown_cleanup(app)
    _run_shutdown_cleanup(app)

    assert mock_engine.dispose.call_count == 2


def test_lifespan_with_no_optional_resources_succeeds_cleanly() -> None:
    app = create_app(
        _make_dummy_research_service(),
        session_factory=None,
        vector_store=None,
        vector_client=None,
        rate_limit_storage=None,
        discovery_service=None,
        enforce_auth=False,
    )

    with TestClient(app):
        pass
    # No error raised on clean exit
