"""SQLAlchemy engine factory built on DatabaseConfig.

Creating an Engine is lazy: no connection is opened, no SQL is run and no
table is created here. The caller owns the returned Engine and should call
``engine.dispose()`` when finished. No driver is imported by this module;
SQLAlchemy loads the dialect's driver when the Engine is created.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from nexora.database.config import DatabaseConfig, DatabaseConfigurationError


def create_database_engine(config: DatabaseConfig) -> Engine:
    """Create an Engine from ``config`` using its pool and echo settings.

    All pool options are passed straight to SQLAlchemy. If the configured
    dialect/pool cannot accept one of them, SQLAlchemy raises (it is never
    silently dropped). SQLAlchemy errors are not wrapped.
    """
    if not isinstance(config, DatabaseConfig):
        raise DatabaseConfigurationError("config must be a DatabaseConfig")
    return create_engine(
        config.url.strip(),
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_timeout=config.pool_timeout,
        pool_recycle=config.pool_recycle,
        echo=config.echo,
    )
