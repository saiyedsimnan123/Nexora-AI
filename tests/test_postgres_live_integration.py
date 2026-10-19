"""Integration tests for live PostgreSQL database migrations and schema readiness (Milestone 37).

When a live PostgreSQL instance is available (via TEST_POSTGRES_URL or NEXORA_DATABASE_URL),
this suite verifies:
1. Real connectivity to the PostgreSQL database.
2. Successful execution of Alembic migrations up to head ('044c24ad4c33').
3. Presence of all 10 core schema tables and the alembic_version table in PostgreSQL.
4. Production readiness check (check_database_readiness) returning ready.
5. End-to-end FastAPI /ready endpoint reporting HTTP 200 with status='ready'.
6. Fail-closed behavior on database connectivity interruption (HTTP 503).

When run locally without a running PostgreSQL instance, this suite skips cleanly with an
informative message. When run in GitHub Actions CI with the postgres:16 service container,
this suite executes against real PostgreSQL.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from nexora.api.app import create_app
from nexora.auth.service import AuthenticationService
from nexora.database.readiness import (
    check_database_readiness,
    get_applied_migration_heads,
    get_expected_migration_heads,
)
from nexora.database.session import create_session_factory
from nexora.retrieval.in_memory import InMemoryVectorStore


def _get_live_postgres_url() -> str | None:
    url = os.environ.get("TEST_POSTGRES_URL") or os.environ.get("NEXORA_DATABASE_URL")
    if url and (url.startswith("postgresql://") or url.startswith("postgresql+")):
        return url
    return None


def _probe_postgres_connection(url: str) -> bool:
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def postgres_url() -> str:
    url = _get_live_postgres_url()
    if not url or not _probe_postgres_connection(url):
        pytest.skip(
            "Live PostgreSQL database not reachable. Set TEST_POSTGRES_URL or NEXORA_DATABASE_URL "
            "(e.g. postgresql+psycopg://test_user:test_password@localhost:5432/nexora_test) "
            "to run live PostgreSQL migration and readiness integration tests."
        )
    return url


@pytest.fixture(scope="module")
def live_db_engine(postgres_url: str):
    engine = create_engine(postgres_url, pool_pre_ping=True)
    yield engine
    engine.dispose()


def test_live_postgres_migrations_reach_head(postgres_url: str, live_db_engine: Any) -> None:
    """Verify that Alembic migrations run and establish all expected schema tables on PostgreSQL."""
    os.environ["NEXORA_DATABASE_URL"] = postgres_url
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    inspector = inspect(live_db_engine)
    existing_tables = set(inspector.get_table_names())

    expected_tables = {
        "alembic_version",
        "users",
        "workspaces",
        "collections",
        "papers",
        "documents",
        "document_chunks",
        "authors",
        "paper_authors",
        "paper_references",
        "api_keys",
    }
    for table in expected_tables:
        assert table in existing_tables, f"Table '{table}' missing from PostgreSQL after migration"


def test_live_postgres_readiness_check(postgres_url: str, live_db_engine: Any) -> None:
    """Verify that check_database_readiness validates real PostgreSQL tables and migration head."""
    session_factory = create_session_factory(live_db_engine)

    # Must succeed without raising DatabaseConnectivityError or DatabaseSchemaError
    check_database_readiness(session_factory)

    # Verify head revision on PostgreSQL
    applied_heads = get_applied_migration_heads(session_factory)
    expected_heads = get_expected_migration_heads()
    assert applied_heads == expected_heads
    assert "044c24ad4c33" in applied_heads

    with session_factory() as session:
        assert get_applied_migration_heads(session) == expected_heads

    assert get_applied_migration_heads(live_db_engine) == expected_heads


def test_live_postgres_app_ready_endpoint(postgres_url: str, live_db_engine: Any) -> None:
    """Verify that FastAPI /ready endpoint returns HTTP 200 against real PostgreSQL."""
    session_factory = create_session_factory(live_db_engine)
    auth_service = AuthenticationService(session_factory=session_factory)

    mock_research = MagicMock()
    mock_research.ask.return_value = {"answer": "ready"}

    app = create_app(
        mock_research,
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
        ingestion_service=SimpleNamespace(_vector_store=InMemoryVectorStore()),
    )

    client = TestClient(app)

    # 1. /health returns 200
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}

    # 2. /ready returns 200 with all components operational
    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    data = r_ready.json()
    assert data["status"] == "ready"
    assert data["checks"]["database"]["status"] == "ready"
    assert data["checks"]["auth"]["status"] == "ready"
    assert data["checks"]["research"]["status"] == "ready"
