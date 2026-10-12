import pytest
from sqlalchemy import create_engine
from alembic.config import Config
from alembic import command
import os
import sqlite3

def test_migrations():
    # Setup a fresh in-memory sqlite db or temp file for alembic
    db_path = "test_migrations.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Run alembic upgrade head
    os.environ["NEXORA_DATABASE_URL"] = f"sqlite:///{db_path}"
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    # Verify tables exist
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]

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
        "paper_references"
    }

    for t in expected_tables:
        assert t in tables, f"Missing table {t}"

    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)
