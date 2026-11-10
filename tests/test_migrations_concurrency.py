"""Regression tests for concurrent database migrations and partial schema recovery.

Verifies:
1. Concurrent threads attempting apply_pending_migrations against a shared file DB
   serialize safely and arrive cleanly at head 044c24ad4c36 without conflicts.
2. PostgreSQL dialect acquires pg_advisory_xact_lock(4815162342044) before reading
   alembic_version/running DDL and skips Alembic upgrade when already at head under lock.
3. Partial research_jobs schema (e.g., table present without error_message) is cleanly
   reconciled without dropping tables or losing existing rows.
4. Repeated calls to apply_pending_migrations at head skip Alembic execution.
5. _LazyProductionASGIApp initializes once under lock and skips migrations on /health.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, inspect, text

from nexora.database.migrations.runner import (
    NEXORA_MIGRATION_LOCK_KEY,
    apply_pending_migrations,
)
from nexora.database.readiness import (
    get_applied_migration_heads,
    get_expected_migration_heads,
)


def _migrations_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "src"
        / "nexora"
        / "database"
        / "migrations"
    )


def test_concurrent_migration_attempts_single_flight(tmp_path: Path) -> None:
    """Concurrent migration attempts on a shared DB upgrade cleanly without collision."""
    db_file = tmp_path / "concurrent_migrations.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    # Initialize database at revision 044c24ad4c35 (just before research_jobs)
    eng = create_engine(db_url, connect_args={"check_same_thread": False})
    try:
        with eng.connect() as conn:
            cfg = Config()
            cfg.set_main_option("script_location", str(_migrations_dir()).replace("%", "%%"))
            cfg.attributes["connection"] = conn
            cfg.attributes["advisory_lock_held"] = True
            with conn.begin():
                command.upgrade(cfg, "044c24ad4c35")
    finally:
        eng.dispose()

    # Launch 6 concurrent migration attempts
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(apply_pending_migrations, db_url) for _ in range(6)]
        results = [fut.result(timeout=30) for fut in futures]

    # Exactly 1 worker should have executed the upgrade; the rest skipped
    assert results.count(True) == 1
    assert results.count(False) == 5

    verify_eng = create_engine(db_url)
    try:
        expected_heads = get_expected_migration_heads()
        assert get_applied_migration_heads(verify_eng) == expected_heads
        with verify_eng.connect() as conn:
            insp = inspect(conn)
            assert insp.has_table("research_jobs")
            cols = {c["name"] for c in insp.get_columns("research_jobs")}
            assert "error_message" in cols
    finally:
        verify_eng.dispose()


def test_postgresql_advisory_lock_used_and_skips_when_already_migrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PostgreSQL path issues pg_advisory_xact_lock and skips upgrade if at head under lock."""
    executed_sql: list[str] = []

    class FakeTx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    class FakePostgresConn:
        dialect = SimpleNamespace(name="postgresql")

        def __init__(self, precheck_done: list[bool]) -> None:
            self._precheck_done = precheck_done

        def execute(self, stmt, *args, **kwargs):
            sql = str(stmt)
            executed_sql.append(sql)
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))

        def begin(self):
            return FakeTx()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    class FakePostgresEngine:
        url = "postgresql://user:pass@localhost:5432/nexora_test"

        def __init__(self) -> None:
            self.calls = 0

        def connect(self):
            self.calls += 1
            return FakePostgresConn([self.calls > 1])

        def dispose(self):
            pass

    fake_engine = FakePostgresEngine()

    # Simulate: pre-check reports behind head (dirty cold start); while waiting for advisory lock,
    # another cold start commits 044c24ad4c36 -> under lock, _is_schema_at_expected_head returns True!
    check_calls = 0

    def fake_head_check(conn, expected_heads):
        nonlocal check_calls
        check_calls += 1
        return check_calls >= 2

    monkeypatch.setattr(
        "nexora.database.migrations.runner._is_schema_at_expected_head",
        fake_head_check,
    )
    upgrade_spy = MagicMock()
    monkeypatch.setattr("alembic.command.upgrade", upgrade_spy)

    ran = apply_pending_migrations(engine=fake_engine)
    assert ran is False
    assert any(
        f"pg_advisory_xact_lock({NEXORA_MIGRATION_LOCK_KEY})" in sql
        for sql in executed_sql
    )
    upgrade_spy.assert_not_called()


def test_partial_research_jobs_table_recovery_preserves_rows(tmp_path: Path) -> None:
    """Partial research_jobs table missing error_message reconciles without losing existing rows."""
    db_file = tmp_path / "partial_state.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    eng = create_engine(db_url)
    try:
        with eng.connect() as conn:
            cfg = Config()
            cfg.set_main_option("script_location", str(_migrations_dir()).replace("%", "%%"))
            cfg.attributes["connection"] = conn
            cfg.attributes["advisory_lock_held"] = True
            with conn.begin():
                command.upgrade(cfg, "044c24ad4c35")
                conn.execute(
                    text(
                        "CREATE TABLE research_jobs ("
                        "id BLOB PRIMARY KEY, "
                        "query TEXT NOT NULL, "
                        "status VARCHAR(32) NOT NULL DEFAULT 'queued'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO research_jobs (id, query) "
                        "VALUES (X'0102030405060708090a0b0c0d0e0f10', 'existing row preserved')"
                    )
                )

        ran = apply_pending_migrations(db_url)
        assert ran is True

        expected_heads = get_expected_migration_heads()
        assert get_applied_migration_heads(eng) == expected_heads

        with eng.connect() as conn:
            cols = {c["name"] for c in inspect(conn).get_columns("research_jobs")}
            assert "error_message" in cols
            assert "guest_session_id" in cols
            assert "attempts" in cols
            rows = conn.execute(text("SELECT query, status FROM research_jobs")).fetchall()
            assert rows == [("existing row preserved", "queued")]
    finally:
        eng.dispose()


def test_lazy_production_app_concurrent_health_and_api_single_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/health must not trigger migrations, and concurrent ASGI calls migrate at most once."""
    from nexora.server import _LazyProductionASGIApp

    migration_calls: list[str] = []

    def fake_run_migrations(env):
        migration_calls.append("migrated")
        return True

    mock_engine = SimpleNamespace(dispose=lambda: None)
    mock_factory = MagicMock()
    monkeypatch.setattr(
        "nexora.database.connection.create_database_engine",
        lambda cfg: mock_engine,
    )
    monkeypatch.setattr(
        "nexora.database.session.create_session_factory",
        lambda eng: mock_factory,
    )
    monkeypatch.setattr("nexora.server._run_startup_migrations_once", fake_run_migrations)
    monkeypatch.setenv("NEXORA_DATABASE_URL", "postgresql://user:pass@localhost:5432/nexora_test")
    monkeypatch.setenv("NEXORA_LLM_API_KEY", "test-key")
    monkeypatch.setenv("NEXORA_LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("NEXORA_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("NEXORA_VECTOR_STORE", "memory")
    monkeypatch.setenv("VERCEL_ENV", "preview")

    lazy_app = _LazyProductionASGIApp()
    client = TestClient(lazy_app)

    # 1. /health liveness must return 200 OK without triggering migrations
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json() == {"status": "ok"}
    assert migration_calls == []

    # 2. First real API request runs startup migrations once
    guest_res = client.post("/api/v1/auth/guest-session")
    assert guest_res.status_code == 200
    assert len(migration_calls) == 1

    # 3. Subsequent API & health requests do NOT re-run startup migrations
    assert client.post("/api/v1/auth/guest-session").status_code == 200
    assert client.get("/health").status_code == 200
    assert len(migration_calls) == 1
