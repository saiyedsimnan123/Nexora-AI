from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, text

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from nexora.database.models.base import Base
from nexora.database.config import DatabaseConfig
import nexora.database.models  # Import to register all models

target_metadata = Base.metadata

# Dedicated signed 64-bit advisory lock key for Nexora schema migrations.
NEXORA_MIGRATION_LOCK_KEY = 4815162342044
_DEFAULT_PLACEHOLDER_URL = "driver://user:pass@localhost/dbname"


def _resolve_sqlalchemy_url() -> str | None:
    attr_url = config.attributes.get("sqlalchemy_url")
    if attr_url:
        return str(attr_url)
    main_url = config.get_main_option("sqlalchemy.url")
    if main_url and main_url != _DEFAULT_PLACEHOLDER_URL:
        return main_url
    try:
        return DatabaseConfig.from_env().url
    except Exception:
        return os.environ.get("NEXORA_DATABASE_URL") or main_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = _resolve_sqlalchemy_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with PostgreSQL advisory lock serialization."""
    external_connection = config.attributes.get("connection")
    if external_connection is not None:
        context.configure(
            connection=external_connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            if (
                external_connection.dialect.name == "postgresql"
                and not config.attributes.get("advisory_lock_held", False)
            ):
                external_connection.execute(
                    text(f"SELECT pg_advisory_xact_lock({NEXORA_MIGRATION_LOCK_KEY})")
                )
            context.run_migrations()
        return

    url = _resolve_sqlalchemy_url()
    configuration = config.get_section(config.config_ini_section, {})
    if url:
        configuration["sqlalchemy.url"] = url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            if connection.dialect.name == "postgresql":
                connection.execute(
                    text(f"SELECT pg_advisory_xact_lock({NEXORA_MIGRATION_LOCK_KEY})")
                )
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
