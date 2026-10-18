import os
import sqlite3
import uuid
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine


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
        "paper_references",
        "api_keys",
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

    cursor.execute("PRAGMA table_info(api_keys);")
    api_key_columns = {r[1] for r in cursor.fetchall()}
    assert "id" in api_key_columns
    assert "user_id" in api_key_columns
    assert "name" in api_key_columns
    assert "key_hash" in api_key_columns
    assert "key_prefix" in api_key_columns
    assert "expires_at" in api_key_columns
    assert "revoked_at" in api_key_columns
    assert "last_used_at" in api_key_columns
    assert "created_at" in api_key_columns
    assert "updated_at" in api_key_columns

    cursor.execute("PRAGMA index_list(api_keys);")
    index_names = {r[1] for r in cursor.fetchall()}
    assert any("ix_api_keys_key_hash" in name for name in index_names)
    assert any("ix_api_keys_user_id" in name for name in index_names)

    conn.close()


def test_migrations_downgrade_and_reupgrade_cycle(tmp_path, monkeypatch):
    """Verify clean downgrade of api_keys migration and re-upgrade to head."""
    db_path = tmp_path / "test_downgrade.db"
    monkeypatch.setenv("NEXORA_DATABASE_URL", f"sqlite:///{db_path}")
    alembic_cfg = Config("alembic.ini")

    # 1. Upgrade to head
    command.upgrade(alembic_cfg, "head")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_head = [r[0] for r in cursor.fetchall()]
    assert "api_keys" in tables_head

    # 2. Downgrade to 044c24ad4c32 (the revision before api_keys)
    command.downgrade(alembic_cfg, "044c24ad4c32")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_down = [r[0] for r in cursor.fetchall()]
    assert "api_keys" not in tables_down
    assert "users" in tables_down
    assert "documents" in tables_down

    # Documents columns from 044c24ad4c32 still intact
    cursor.execute("PRAGMA table_info(documents);")
    doc_cols = {r[1] for r in cursor.fetchall()}
    assert "claimed_at" in doc_cols
    assert "claimed_by" in doc_cols

    # 3. Re-upgrade to head
    command.upgrade(alembic_cfg, "head")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_reup = [r[0] for r in cursor.fetchall()]
    assert "api_keys" in tables_reup

    conn.close()


def test_migration_preserves_existing_data(tmp_path, monkeypatch):
    """Verify upgrading a pre-existing database from 044c24ad4c32 to head preserves all existing data."""
    db_path = tmp_path / "test_data_preservation.db"
    monkeypatch.setenv("NEXORA_DATABASE_URL", f"sqlite:///{db_path}")
    alembic_cfg = Config("alembic.ini")

    # Upgrade only to 044c24ad4c32 (before api_keys table was introduced)
    command.upgrade(alembic_cfg, "044c24ad4c32")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    user_id = str(uuid.uuid4()).replace("-", "")
    workspace_id = str(uuid.uuid4()).replace("-", "")
    collection_id = str(uuid.uuid4()).replace("-", "")
    paper_id = str(uuid.uuid4()).replace("-", "")
    doc_id = str(uuid.uuid4()).replace("-", "")

    # Insert existing data at 044c24ad4c32
    cursor.execute(
        "INSERT INTO users (id, email) VALUES (?, ?)",
        (user_id, "existing_user@example.com"),
    )
    cursor.execute(
        "INSERT INTO workspaces (id, user_id, name) VALUES (?, ?, ?)",
        (workspace_id, user_id, "Existing Workspace"),
    )
    cursor.execute(
        "INSERT INTO collections (id, workspace_id, name) VALUES (?, ?, ?)",
        (collection_id, workspace_id, "Existing Collection"),
    )
    cursor.execute(
        "INSERT INTO papers (id, collection_id, title) VALUES (?, ?, ?)",
        (paper_id, collection_id, "Existing Research Paper"),
    )
    cursor.execute(
        "INSERT INTO documents (id, paper_id, filename, file_size, mime_type, checksum, storage_key, indexing_status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (doc_id, paper_id, "paper.pdf", 1024, "application/pdf", "a" * 64, "storage/paper.pdf", "completed"),
    )
    conn.commit()
    conn.close()

    # Now apply the 044c24ad4c33 migration (upgrade head)
    command.upgrade(alembic_cfg, "head")

    # Verify all pre-existing records survived intact
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    assert cursor.fetchone()[0] == "existing_user@example.com"

    cursor.execute("SELECT name FROM workspaces WHERE id = ?", (workspace_id,))
    assert cursor.fetchone()[0] == "Existing Workspace"

    cursor.execute("SELECT name FROM collections WHERE id = ?", (collection_id,))
    assert cursor.fetchone()[0] == "Existing Collection"

    cursor.execute("SELECT title FROM papers WHERE id = ?", (paper_id,))
    assert cursor.fetchone()[0] == "Existing Research Paper"

    cursor.execute("SELECT filename, indexing_status FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    assert row[0] == "paper.pdf"
    assert row[1] == "completed"

    # Now insert an api_key referencing the pre-existing user
    key_id = str(uuid.uuid4()).replace("-", "")
    cursor.execute(
        "INSERT INTO api_keys (id, user_id, name, key_hash, key_prefix) VALUES (?, ?, ?, ?, ?)",
        (key_id, user_id, "Existing User Key", "hash_12345", "nexora_ab12"),
    )
    conn.commit()

    cursor.execute("SELECT name, key_prefix FROM api_keys WHERE id = ?", (key_id,))
    key_row = cursor.fetchone()
    assert key_row[0] == "Existing User Key"
    assert key_row[1] == "nexora_ab12"

    conn.close()


def test_alembic_linear_history_and_revisions():
    """Verify Alembic history forms a strict linear sequence with 044c24ad4c33 as head."""
    from alembic.script import ScriptDirectory

    alembic_cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(alembic_cfg)

    heads = script.get_heads()
    assert heads == ["044c24ad4c33"], f"Expected single head '044c24ad4c33', got {heads}"

    revisions = [rev.revision for rev in script.walk_revisions()]
    assert revisions == [
        "044c24ad4c33",
        "044c24ad4c32",
        "044c24ad4c31",
        "044c24ad4c30",
    ]
    assert "e456789a0123" not in revisions
