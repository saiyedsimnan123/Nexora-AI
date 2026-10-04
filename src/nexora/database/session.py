"""Session factory, session creation and the transaction boundary.

This module only defines functions: nothing runs at import time, no Engine,
Session or sessionmaker is created at module level, and nothing here connects
to the database (SQLAlchemy connects lazily, on first use).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from nexora.database.config import DatabaseConfigurationError


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a reusable factory of independent Sessions bound to ``engine``.

    Sessions use ``expire_on_commit=False`` so loaded data stays readable after
    commit. SQLAlchemy 2.x sessions never autocommit. The caller owns the
    factory and the Engine.
    """
    if not isinstance(engine, Engine):
        raise DatabaseConfigurationError("engine must be a SQLAlchemy Engine")
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_database_session(session_factory: sessionmaker[Session]) -> Session:
    """Create a new, independent Session (it does not connect until used)."""
    if not isinstance(session_factory, sessionmaker):
        raise DatabaseConfigurationError("session_factory must be a SQLAlchemy sessionmaker")
    return session_factory()


@contextmanager
def database_transaction(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Run a block in one transaction.

    Commits on success. On any exception it rolls back and re-raises the
    original exception unchanged. The Session is always closed. An invalid
    factory is rejected before any database work.
    """
    session = create_database_session(session_factory)
    try:
        yield session
        session.commit()
    except BaseException:  # also rolls back on cancellation; always re-raised
        session.rollback()
        raise
    finally:
        session.close()
