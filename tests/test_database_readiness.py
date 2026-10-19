"""Focused regression tests for database schema readiness and migration verification (Milestone 33)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from nexora.api.app import create_app
from nexora.auth.bootstrap import bootstrap_user_and_key
from nexora.auth.service import AuthenticationService
from nexora.database.models.base import Base
import nexora.database.models  # Register all models
from nexora.database.readiness import (
    DatabaseConnectivityError,
    DatabaseSchemaError,
    check_database_readiness,
    get_applied_migration_heads,
    get_expected_migration_heads,
)
from nexora.database.session import create_session_factory
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk, SearchResult

SECRET_PASSWORD = "test-supersecret-db-password-12345"


class FakeResearchService:
    def ask(self, request: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "text": "Answer text",
                "query": getattr(request, "query", ""),
                "model": "fake-model",
                "evidence_count": 0,
            }
        )


def _build_test_app(
    session_factory: Any,
    *,
    vector_store: Any = None,
    enforce_auth: bool = True,
) -> Any:
    research_service = FakeResearchService()
    auth_service = AuthenticationService(session_factory=session_factory) if session_factory else None
    store = vector_store or InMemoryVectorStore()
    return create_app(
        research_service=research_service,
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=enforce_auth,
        ingestion_service=SimpleNamespace(_vector_store=store),
    )


def test_get_expected_migration_heads_returns_current_head() -> None:
    heads = get_expected_migration_heads()
    assert isinstance(heads, tuple)
    assert len(heads) >= 1
    assert "044c24ad4c33" in heads


def test_get_applied_migration_heads_across_all_target_types() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # 1. When table is missing, all return empty tuple
    session_factory = create_session_factory(engine)
    assert get_applied_migration_heads(None) == ()
    assert get_applied_migration_heads(engine) == ()
    assert get_applied_migration_heads(session_factory) == ()
    with session_factory() as session:
        assert get_applied_migration_heads(session) == ()
    with engine.connect() as conn:
        assert get_applied_migration_heads(conn) == ()

    # 2. When table is empty, all return empty tuple
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
    assert get_applied_migration_heads(engine) == ()
    assert get_applied_migration_heads(session_factory) == ()
    with session_factory() as session:
        assert get_applied_migration_heads(session) == ()

    # 3. When revision is populated, all return the expected tuple
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('044c24ad4c33')"))
    assert get_applied_migration_heads(session_factory) == ("044c24ad4c33",)
    assert get_applied_migration_heads(engine) == ("044c24ad4c33",)
    with session_factory() as session:
        assert get_applied_migration_heads(session) == ("044c24ad4c33",)
    with engine.connect() as conn:
        assert get_applied_migration_heads(conn) == ("044c24ad4c33",)


def test_correctly_migrated_database_is_reported_ready() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('044c24ad4c33')"))

    session_factory = create_session_factory(engine)

    # Unit level check
    check_database_readiness(session_factory)

    # API /ready check
    app = _build_test_app(session_factory)
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["database"]["status"] == "ready"


def test_outdated_migration_revision_returns_503() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    # Seed with previous revision 044c24ad4c32 instead of head 044c24ad4c33
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('044c24ad4c32')"))

    session_factory = create_session_factory(engine)

    # Unit level check raises DatabaseSchemaError
    with pytest.raises(DatabaseSchemaError) as exc_info:
        check_database_readiness(session_factory)
    assert "does not match expected application head" in str(exc_info.value)

    # API /ready returns 503
    app = _build_test_app(session_factory)
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["checks"]["database"]["status"] == "unavailable"
    assert data["checks"]["database"]["error"] == "DatabaseSchemaError"


def test_missing_migration_metadata_returns_503() -> None:
    # Tables created without alembic_version table
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)

    with pytest.raises(DatabaseSchemaError) as exc_info:
        check_database_readiness(session_factory)
    assert "alembic_version" in str(exc_info.value)
    assert "missing" in str(exc_info.value).lower()

    app = _build_test_app(session_factory)
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["checks"]["database"]["status"] == "unavailable"
    assert data["checks"]["database"]["error"] == "DatabaseSchemaError"


def test_empty_migration_metadata_returns_503() -> None:
    # alembic_version table exists but has 0 rows
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))

    session_factory = create_session_factory(engine)

    with pytest.raises(DatabaseSchemaError) as exc_info:
        check_database_readiness(session_factory)
    assert "contains no applied revisions" in str(exc_info.value)

    app = _build_test_app(session_factory)
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["checks"]["database"]["status"] == "unavailable"
    assert data["checks"]["database"]["error"] == "DatabaseSchemaError"


def test_missing_core_tables_returns_503() -> None:
    # alembic_version is at head, but core table (e.g. api_keys) was dropped
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('044c24ad4c33')"))
        conn.execute(text("DROP TABLE api_keys"))

    session_factory = create_session_factory(engine)

    with pytest.raises(DatabaseSchemaError) as exc_info:
        check_database_readiness(session_factory)
    assert "Required database tables are missing" in str(exc_info.value)
    assert "api_keys" in str(exc_info.value)

    app = _build_test_app(session_factory)
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["checks"]["database"]["status"] == "unavailable"
    assert data["checks"]["database"]["error"] == "DatabaseSchemaError"


def test_database_connectivity_failure_returns_503() -> None:
    # 1. Missing session factory raises DatabaseConnectivityError
    with pytest.raises(DatabaseConnectivityError):
        check_database_readiness(None)

    # 2. Connection failure raises original exception (e.g. RuntimeError or OperationalError)
    def broken_factory() -> Any:
        class BrokenSession:
            def __enter__(self) -> Any:
                raise RuntimeError("Connection to PostgreSQL failed: connection refused")

            def __exit__(self, *args: Any) -> None:
                pass

        return BrokenSession()

    with pytest.raises(RuntimeError):
        check_database_readiness(broken_factory)

    app = _build_test_app(broken_factory)
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["checks"]["database"]["status"] == "unavailable"
    assert data["checks"]["database"]["error"] == "RuntimeError"


def test_readiness_errors_do_not_disclose_credentials_or_raw_sql() -> None:
    def secret_leaking_factory() -> Any:
        class LeakySession:
            def __enter__(self) -> Any:
                # Simulates lower-level DB driver error embedding connection string with password
                raise RuntimeError(
                    f"OperationalError: password authentication failed for user 'app' "
                    f"password='{SECRET_PASSWORD}' host='10.0.0.5' db='nexora_prod'"
                )

            def __exit__(self, *args: Any) -> None:
                pass

        return LeakySession()

    app = _build_test_app(secret_leaking_factory)
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 503

    # Raw response body inspection
    response_text = response.text
    assert SECRET_PASSWORD not in response_text
    assert "10.0.0.5" not in response_text
    assert "password authentication failed" not in response_text

    # Header inspection
    for header_name, header_value in response.headers.items():
        assert SECRET_PASSWORD not in header_value


def test_existing_qdrant_readiness_checks_continue_to_work() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('044c24ad4c33')"))
    session_factory = create_session_factory(engine)

    class BrokenVectorStore:
        def check_readiness(self) -> None:
            from nexora.retrieval.qdrant import QdrantStoreError
            raise QdrantStoreError("Qdrant collection vector dimension 768 != 1536")

    app = _build_test_app(session_factory, vector_store=BrokenVectorStore())
    client = TestClient(app)

    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    # Database is ready
    assert data["checks"]["database"]["status"] == "ready"
    # Vector store is unavailable with QdrantStoreError
    assert data["checks"]["vector_store"]["status"] == "unavailable"
    assert data["checks"]["vector_store"]["error"] == "QdrantStoreError"


def test_existing_health_check_remains_operational_during_database_failure() -> None:
    # Liveness probe (/health) must remain 200 OK even if database is completely broken
    def broken_factory() -> Any:
        raise RuntimeError("Database down")

    app = _build_test_app(broken_factory)
    client = TestClient(app)

    # Readiness is 503
    assert client.get("/ready").status_code == 503
    # Liveness is 200
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}


def test_authentication_and_tenant_isolation_remain_unchanged() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('044c24ad4c33')"))
    session_factory = create_session_factory(engine)

    # Bootstrap users
    user1, ws1, _, raw_key1 = bootstrap_user_and_key(
        session_factory, email="alice@test.com", workspace_name="Alice Lab", key_name="Key 1"
    )
    user2, ws2, _, raw_key2 = bootstrap_user_and_key(
        session_factory, email="bob@test.com", workspace_name="Bob Lab", key_name="Key 2"
    )

    from nexora.database.models.collection import Collection
    with session_factory() as sess:
        col1 = Collection(workspace_id=ws1.id, name="Alice Collection")
        sess.add(col1)
        sess.commit()
        sess.refresh(col1)
        col1_id = str(col1.id)

    app = _build_test_app(session_factory)
    client = TestClient(app)

    # 1. Unauthenticated request -> 401
    resp_unauth = client.post("/api/v1/research", json={"query": "test"})
    assert resp_unauth.status_code == 401

    # 2. Authenticated user 2 querying user 1's collection -> 403 Forbidden
    resp_cross = client.post(
        "/api/v1/research",
        json={"query": "test", "collection_id": col1_id},
        headers={"Authorization": f"Bearer {raw_key2}"},
    )
    assert resp_cross.status_code == 403

    # 3. Authenticated user 1 querying own collection -> 200 OK
    resp_valid = client.post(
        "/api/v1/research",
        json={"query": "test", "collection_id": col1_id},
        headers={"Authorization": f"Bearer {raw_key1}"},
    )
    assert resp_valid.status_code == 200


def test_real_alembic_upgrade_and_downgrade_cycles_with_readiness(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from alembic import command
    from alembic.config import Config

    db_path = tmp_path / "test_alembic_readiness.db"
    monkeypatch.setenv("NEXORA_DATABASE_URL", f"sqlite:///{db_path}")
    alembic_cfg = Config("alembic.ini")

    # 1. Upgrade to head
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    session_factory = create_session_factory(engine)

    # Must pass readiness
    check_database_readiness(session_factory)
    app = _build_test_app(session_factory)
    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"

    # 2. Downgrade by one revision (to 044c24ad4c32)
    command.downgrade(alembic_cfg, "044c24ad4c32")

    # Must fail readiness
    with pytest.raises(DatabaseSchemaError):
        check_database_readiness(session_factory)

    resp_down = client.get("/ready")
    assert resp_down.status_code == 503
    assert resp_down.json()["checks"]["database"]["status"] == "unavailable"
    assert resp_down.json()["checks"]["database"]["error"] == "DatabaseSchemaError"

    # 3. Re-upgrade to head
    command.upgrade(alembic_cfg, "head")
    resp_up = client.get("/ready")
    assert resp_up.status_code == 200
    assert resp_up.json()["status"] == "ready"


def test_migration_discovery_missing_directory_raises_schema_error(tmp_path: Any) -> None:
    non_existent = tmp_path / "missing_migrations"
    with pytest.raises(DatabaseSchemaError) as exc_info:
        get_expected_migration_heads(non_existent)
    assert "does not exist" in str(exc_info.value)


def test_migration_discovery_empty_or_broken_heads_raises_schema_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from alembic.script import ScriptDirectory

    monkeypatch.setattr(ScriptDirectory, "get_heads", lambda self: [])
    with pytest.raises(DatabaseSchemaError) as exc_info:
        get_expected_migration_heads()
    assert "No migration heads found" in str(exc_info.value)


def test_migration_discovery_script_exception_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from alembic.script import ScriptDirectory

    def broken_get_heads(self: Any) -> Any:
        raise RuntimeError("SyntaxError in revision script 044c24ad4c34")

    monkeypatch.setattr(ScriptDirectory, "get_heads", broken_get_heads)

    with pytest.raises(DatabaseSchemaError) as exc_info:
        get_expected_migration_heads()
    assert "Failed to determine expected migration heads" in str(exc_info.value)


def test_readiness_fails_closed_503_when_migration_discovery_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # Database is migrated to '044c24ad4c33', but migration discovery fails
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('044c24ad4c33')"))
    session_factory = create_session_factory(engine)

    def failing_heads(*args: Any, **kwargs: Any) -> Any:
        raise DatabaseSchemaError("Failed to discover expected migration heads")

    monkeypatch.setattr("nexora.database.readiness.get_expected_migration_heads", failing_heads)

    # Unit-level check must raise DatabaseSchemaError, NOT succeed with a fallback head
    with pytest.raises(DatabaseSchemaError):
        check_database_readiness(session_factory)

    # API /ready must return 503 unavailable with sanitized error name
    app = _build_test_app(session_factory)
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["checks"]["database"]["status"] == "unavailable"
    assert data["checks"]["database"]["error"] == "DatabaseSchemaError"
