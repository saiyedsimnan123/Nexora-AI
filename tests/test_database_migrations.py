"""Tests for Alembic database migrations."""

import os
from pathlib import Path
from unittest import mock

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import MetaData, create_engine, inspect

from nexora.database.models import Base


@pytest.fixture
def alembic_config(tmp_path):
    """Provides a valid Alembic Config pointing to the project's alembic.ini."""
    # Find the repository root assuming tests/ is at the root
    project_root = Path(__file__).parent.parent
    ini_path = project_root / "alembic.ini"

    config = Config(str(ini_path))

    # We must use a file-based SQLite database so the engine and Alembic context
    # connect to the exact same database.
    db_url = f"sqlite:///{tmp_path}/test.db"
    config.set_main_option("sqlalchemy.url", db_url)

    return config


def test_alembic_config_discovery(alembic_config):
    """Test that Alembic can load the env.py and find migrations."""
    # The script_location should be relative to the ini file
    script_location = alembic_config.get_main_option("script_location")
    assert script_location is not None
    assert "src/nexora/database/migrations" in script_location


def test_migration_upgrade_and_downgrade(alembic_config):
    """Test full upgrade to head and downgrade to base."""
    # Use the config's URL to create a dedicated engine
    db_url = alembic_config.get_main_option("sqlalchemy.url")
    engine = create_engine(db_url)

    # Upgrade to head
    command.upgrade(alembic_config, "head")

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    expected_tables = {
        "alembic_version",
        "authors",
        "users",
        "workspaces",
        "collections",
        "papers",
        "documents",
        "paper_authors",
        "paper_references",
        "document_chunks"
    }
    assert tables == expected_tables, f"Missing or extra tables: {tables ^ expected_tables}"

    # Downgrade to base
    command.downgrade(alembic_config, "base")

    inspector = inspect(engine)
    tables_after_downgrade = set(inspector.get_table_names())
    # Should only be left with alembic_version (which Alembic manages)
    assert tables_after_downgrade == {"alembic_version"}


def test_schema_drift_after_upgrade(alembic_config, tmp_path):
    """Verify that after migrating to head, autogenerate detects no drift."""

    # We must use a file-based SQLite database for autogenerate so different connections see the same tables
    db_url = f"sqlite:///{tmp_path}/drift.db"
    alembic_config.set_main_option("sqlalchemy.url", db_url)

    # Run the migrations to build the schema
    command.upgrade(alembic_config, "head")

    # Capture the output of autogenerate using a custom env hook or intercepting the context
    from alembic.migration import MigrationContext
    from alembic.autogenerate import compare_metadata

    engine = create_engine(db_url)
    with engine.connect() as connection:
        # We need to test drift against the tables we want, not the ones added by test fixtures elsewhere.
        target_tables = {
            "authors", "users", "workspaces", "collections", "papers", "documents",
            "paper_authors", "paper_references", "document_chunks"
        }

        def include_name(name, type_, parent_names):
            if type_ == "table" and name not in target_tables:
                return False
            return True

        context = MigrationContext.configure(
            connection,
            opts={"include_name": include_name}
        )

        # We filter the metadata object itself to only include models in our real scope
        filtered_metadata = MetaData()
        for t_name, t_table in Base.metadata.tables.items():
            if t_name in target_tables:
                t_table.to_metadata(filtered_metadata)

        # Compare actual database schema with the filtered ORM models
        diff = compare_metadata(context, filtered_metadata)

        # We expect absolutely no differences
        assert not diff, f"Schema drift detected: {diff}"
