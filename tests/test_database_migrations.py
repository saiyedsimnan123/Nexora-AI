import pytest
from sqlalchemy import create_engine
from alembic.config import Config
from alembic import command
import os
import sqlite3

def test_migrations(tmp_path, monkeypatch):
    # Setup a fresh in-memory sqlite db or temp file for alembic
    db_path = tmp_path / "test_migrations.db"

    # Run alembic upgrade head
    monkeypatch.setenv("NEXORA_DATABASE_URL", f"sqlite:///{db_path}")
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    # Verify tables exist
    conn = sqlite3.connect(str(db_path))
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

    cursor.execute("PRAGMA table_info(documents);")
    doc_columns = {r[1] for r in cursor.fetchall()}
    assert "indexing_status" in doc_columns
    assert "indexing_error" in doc_columns
    assert "indexed_at" in doc_columns
    assert "indexing_attempts" in doc_columns
    assert "claimed_at" in doc_columns
    assert "claimed_by" in doc_columns

    conn.close()
