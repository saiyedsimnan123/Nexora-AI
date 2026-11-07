"""Alembic migration runner for programmatic execution."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

from nexora.database.config import DatabaseConfig

logger = logging.getLogger(__name__)


def apply_pending_migrations(db_url: str | None = None) -> None:
    """Safely apply any pending Alembic migrations up to head.

    If db_url is None, reads from DatabaseConfig.from_env().
    Constructs an in-memory Alembic Config targeting the migrations directory,
    ensuring it runs in any execution environment (local or serverless).
    """
    if db_url is None:
        db_config = DatabaseConfig.from_env()
        target_url = db_config.url
    else:
        target_url = db_url

    migrations_dir = Path(__file__).resolve().parent
    cfg = Config()
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", target_url)

    logger.info("Checking and applying pending database migrations up to head...")
    command.upgrade(cfg, "head")
    logger.info("Database migrations check complete.")
