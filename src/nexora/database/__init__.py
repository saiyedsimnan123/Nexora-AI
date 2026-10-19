"""Nexora database layer: configuration, engine, sessions and transactions."""

from nexora.database.config import DatabaseConfig, DatabaseConfigurationError
from nexora.database.connection import create_database_engine
from nexora.database.models import Base
from nexora.database.readiness import (
    DatabaseConnectivityError,
    DatabaseReadinessError,
    DatabaseSchemaError,
    check_database_readiness,
)
from nexora.database.session import (
    create_database_session,
    create_session_factory,
    database_transaction,
)

__all__ = [
    "Base",
    "DatabaseConfig",
    "DatabaseConfigurationError",
    "DatabaseConnectivityError",
    "DatabaseReadinessError",
    "DatabaseSchemaError",
    "check_database_readiness",
    "create_database_engine",
    "create_database_session",
    "create_session_factory",
    "database_transaction",
]
