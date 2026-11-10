"""Alembic migration runner for programmatic execution.

Provides single-flight in-process locking and cross-instance PostgreSQL advisory
locking so concurrent cold starts (e.g., Vercel serverless functions handling
concurrent /health and /api/... requests) never execute conflicting DDL.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, pool, text
from sqlalchemy.engine import Connection, Engine

from nexora.database.config import DatabaseConfig
from nexora.database.readiness import (
    EXPECTED_CORE_TABLES,
    get_applied_migration_heads,
    get_expected_migration_heads,
)

logger = logging.getLogger(__name__)

# Must match NEXORA_MIGRATION_LOCK_KEY in migrations/env.py
NEXORA_MIGRATION_LOCK_KEY = 4815162342044
_MIGRATION_LOCK = threading.Lock()


def _acquire_advisory_lock(conn: Connection) -> None:
    """Acquire transaction-scoped PostgreSQL advisory lock if on PostgreSQL."""
    if conn.dialect.name == "postgresql":
        logger.debug("Acquiring PostgreSQL advisory lock %s for migrations", NEXORA_MIGRATION_LOCK_KEY)
        conn.execute(text(f"SELECT pg_advisory_xact_lock({NEXORA_MIGRATION_LOCK_KEY})"))


def _is_schema_at_expected_head(
    conn: Connection,
    expected_heads: tuple[str, ...],
) -> bool:
    """Return True only when alembic_version matches head and required tables/columns exist."""
    applied_heads = get_applied_migration_heads(conn)
    if applied_heads != expected_heads:
        return False

    inspector = inspect(conn)
    actual_tables = {name.lower() for name in inspector.get_table_names()}
    if EXPECTED_CORE_TABLES - actual_tables:
        return False

    if "research_jobs" in actual_tables:
        cols = {col["name"].lower() for col in inspector.get_columns("research_jobs")}
        if "error_message" not in cols:
            return False

    return True


def _reconcile_partial_research_jobs_table(conn: Connection) -> None:
    """Safely reconcile missing research_jobs table/columns without dropping data."""
    from importlib.util import module_from_spec, spec_from_file_location

    mig_path = (
        Path(__file__).resolve().parent
        / "versions"
        / "044c24ad4c36_add_research_jobs_table.py"
    )
    spec = spec_from_file_location("nexora_mig_044c24ad4c36", mig_path)
    if spec is None or spec.loader is None:
        return
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    mig_ctx = MigrationContext.configure(conn)
    with Operations.context(mig_ctx):
        mod.upgrade()


def apply_pending_migrations(
    db_url: str | None = None,
    *,
    engine: Engine | None = None,
    force: bool = False,
) -> bool:
    """Safely apply any pending Alembic migrations up to head.

    Uses both an in-process mutex and a PostgreSQL transaction-scoped advisory
    lock (`pg_advisory_xact_lock`) so concurrent serverless cold starts serialize
    safely and subsequent instances immediately skip DDL once at head.

    Args:
        db_url: SQLAlchemy database URL. Reads from ``DatabaseConfig.from_env()`` if omitted.
        engine: Optional pre-built SQLAlchemy Engine (useful for tests).
        force: Run migration verification/upgrade even if pre-check matches head.

    Returns:
        ``True`` if migrations or schema reconciliation executed, ``False`` if skipped.
    """
    if engine is None:
        if db_url is None:
            db_config = DatabaseConfig.from_env()
            target_url = db_config.url
        else:
            target_url = db_url
            if target_url.startswith("postgres://"):
                target_url = "postgresql://" + target_url[len("postgres://"):]
    else:
        target_url = str(engine.url)

    migrations_dir = Path(__file__).resolve().parent
    expected_heads = get_expected_migration_heads(migrations_dir)

    with _MIGRATION_LOCK:
        owns_engine = engine is None
        active_engine: Engine = (
            engine
            if engine is not None
            else create_engine(target_url, poolclass=pool.NullPool)
        )

        try:
            # 1. Fast read-only pre-check (avoids taking advisory lock on warm paths)
            if not force:
                try:
                    with active_engine.connect() as check_conn:
                        if _is_schema_at_expected_head(check_conn, expected_heads):
                            logger.debug(
                                "Database already at expected migration head %s; skipping.",
                                expected_heads,
                            )
                            return False
                except Exception as precheck_exc:
                    logger.debug("Pre-check did not confirm head (%s); entering locked migration path.", precheck_exc)

            # 2. Acquire cross-instance PostgreSQL advisory lock inside explicit transaction
            with active_engine.connect() as conn:
                with conn.begin():
                    _acquire_advisory_lock(conn)

                    # Re-check under advisory lock (another cold-start worker may have just committed!)
                    if not force and _is_schema_at_expected_head(conn, expected_heads):
                        logger.info(
                            "Database migration head %s already applied by another worker; skipping.",
                            expected_heads,
                        )
                        return False

                    applied_heads = get_applied_migration_heads(conn)
                    if applied_heads == expected_heads:
                        # Head matches yet table/column reconciliation is needed (partial DB state)
                        logger.warning(
                            "Migration head is %s but schema verification failed; reconciling partial state cleanly.",
                            applied_heads,
                        )
                        _reconcile_partial_research_jobs_table(conn)
                        return True

                    cfg = Config()
                    cfg.set_main_option("script_location", str(migrations_dir).replace("%", "%%"))
                    cfg.set_main_option("sqlalchemy.url", target_url.replace("%", "%%"))
                    cfg.attributes["sqlalchemy_url"] = target_url
                    cfg.attributes["connection"] = conn
                    cfg.attributes["advisory_lock_held"] = True

                    logger.info("Applying pending database migrations up to head...")
                    command.upgrade(cfg, "head")
                    logger.info("Database migrations applied cleanly up to %s.", expected_heads)
                    return True
        finally:
            if owns_engine:
                active_engine.dispose()
