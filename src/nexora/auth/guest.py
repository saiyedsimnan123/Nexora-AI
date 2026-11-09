"""Guest access service and environment controls for Vercel Preview.

Provides secure, sandboxed guest identities strictly restricted to non-production
Preview environments. Production environments strictly prohibit guest access.
"""

from __future__ import annotations

import os
from typing import Any, Mapping
from uuid import UUID

from nexora.auth.models import AuthenticatedUser

# Deterministic namespace IDs for Preview guest tenant
GUEST_USER_ID = UUID("00000000-0000-4000-a000-000000000001")
GUEST_WORKSPACE_ID = UUID("00000000-0000-4000-a000-000000000002")
GUEST_COLLECTION_ID = UUID("00000000-0000-4000-a000-000000000003")
GUEST_EMAIL = "guest@preview.nexora.ai"
DEFAULT_GUEST_TOKEN = "nx_guest_preview_session"
GUEST_TOKEN_PREFIX = "nx_guest_"


def is_guest_access_allowed(environ: Mapping[str, str] | None = None) -> bool:
    """Check if guest access is permitted in the current runtime environment.

    Security Rules:
    1. STRICTLY PROHIBITED in production (VERCEL_ENV == 'production').
    2. ALLOWED in Vercel Preview (VERCEL_ENV == 'preview').
    3. ALLOWED when explicitly enabled via NEXORA_GUEST_ACCESS_ENABLED=true
       provided VERCEL_ENV is not 'production'.
    """
    env = os.environ if environ is None else environ
    vercel_env = (
        env.get("VERCEL_ENV") or os.environ.get("VERCEL_ENV") or ""
    ).strip().lower()

    # Rule 1: Production is ALWAYS forbidden
    if vercel_env == "production":
        return False

    # Rule 2: Vercel Preview is allowed
    if vercel_env == "preview":
        return True

    # Rule 3: Explicit configuration flag (for local/testing preview)
    guest_flag = (
        env.get("NEXORA_GUEST_ACCESS_ENABLED")
        or os.environ.get("NEXORA_GUEST_ACCESS_ENABLED")
        or ""
    ).strip().lower()
    if guest_flag in ("true", "1", "yes", "enabled"):
        return True

    return False


def is_guest_token(raw_token: str | None) -> bool:
    """Check if a raw token represents a guest access token."""
    if not raw_token or not isinstance(raw_token, str):
        return False
    cleaned = raw_token.strip().lower()
    return (
        cleaned.startswith(GUEST_TOKEN_PREFIX)
        or cleaned in ("guest_preview_session", "guest_preview", "guest_session")
        or cleaned.startswith("guest_")
    )


def get_guest_authenticated_user() -> AuthenticatedUser:
    """Return in-memory AuthenticatedUser representation for the guest user."""
    return AuthenticatedUser(
        id=GUEST_USER_ID,
        email=GUEST_EMAIL,
        api_key_id=None,
        api_key_prefix="nx_guest",
    )


_PROVISIONED_SESSIONS: set[Any] = set()


def ensure_guest_tenant(session: Any) -> AuthenticatedUser:
    """Ensure the guest User, Workspace, and Collection exist in PostgreSQL."""
    engine = getattr(session, "bind", None)
    if engine is not None and engine in _PROVISIONED_SESSIONS:
        return get_guest_authenticated_user()

    from nexora.database.models.collection import Collection
    from nexora.database.models.user import User
    from nexora.database.models.workspace import Workspace
    from nexora.database.repository.collection import CollectionRepository
    from nexora.database.repository.user import UserRepository
    from nexora.database.repository.workspace import WorkspaceRepository

    user_repo = UserRepository(session)
    user = user_repo.get_by_id(GUEST_USER_ID)
    if user is None:
        user = User(
            id=GUEST_USER_ID,
            email=GUEST_EMAIL,
        )
        session.add(user)
        session.flush()

    ws_repo = WorkspaceRepository(session)
    workspace = ws_repo.get_by_id(GUEST_WORKSPACE_ID)
    if workspace is None:
        workspace = Workspace(
            id=GUEST_WORKSPACE_ID,
            user_id=GUEST_USER_ID,
            name="Guest Preview Workspace",
            description="Sandboxed preview workspace for unauthenticated guest testing.",
        )
        session.add(workspace)
        session.flush()

    col_repo = CollectionRepository(session)
    collection = col_repo.get_by_id(GUEST_COLLECTION_ID)
    if collection is None:
        collection = Collection(
            id=GUEST_COLLECTION_ID,
            workspace_id=GUEST_WORKSPACE_ID,
            name="Guest Research Collection",
            description="Default collection for guest research papers and queries.",
        )
        session.add(collection)
        session.flush()

    session.commit()
    if engine is not None:
        _PROVISIONED_SESSIONS.add(engine)

    return get_guest_authenticated_user()
