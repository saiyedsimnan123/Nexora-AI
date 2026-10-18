"""Bootstrap utilities for initial user, workspace, and API key provisioning."""

from __future__ import annotations

import argparse
import datetime
import sys
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from nexora.database.config import DatabaseConfig
from nexora.database.connection import create_database_engine
from nexora.database.models.api_key import ApiKey
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository
from nexora.database.session import create_session_factory


def bootstrap_user_and_key(
    session_factory: Any,
    *,
    email: str,
    workspace_name: str = "Default Workspace",
    key_name: str = "Initial API Key",
    expires_at: datetime.datetime | None = None,
) -> tuple[User, Workspace, ApiKey, str]:
    """Bootstrap a user, their primary workspace, and a new API key.

    Returns:
        tuple of (User, Workspace, ApiKey, raw_secret_token).
        The raw_secret_token is returned ONLY here and must be stored securely.
    """
    clean_email = email.strip().lower()
    if not clean_email or "@" not in clean_email:
        raise ValueError(f"Invalid email address: {email!r}")

    with session_factory() as session:
        user_repo = UserRepository(session)
        user = user_repo.get_by_email(clean_email)
        if user is None:
            user = user_repo.create(email=clean_email)
            session.flush()

        ws_repo = WorkspaceRepository(session)
        workspaces = ws_repo.list_by_user_id(user.id)
        if workspaces:
            workspace = workspaces[0]
        else:
            workspace = ws_repo.create(user_id=user.id, name=workspace_name)
            session.flush()

        key_repo = ApiKeyRepository(session)
        api_key, raw_secret = key_repo.generate_key(
            user_id=user.id,
            name=key_name,
            expires_at=expires_at,
        )
        session.commit()

        # Refresh instances before leaving session
        session.refresh(user)
        session.refresh(workspace)
        session.refresh(api_key)

        return user, workspace, api_key, raw_secret


def main(args: list[str] | None = None) -> int:
    """CLI entrypoint for bootstrapping users and keys."""
    parser = argparse.ArgumentParser(description="Bootstrap Nexora AI user and API key")
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument("--workspace", default="Default Workspace", help="Workspace name")
    parser.add_argument("--key-name", default="Admin API Key", help="API key descriptor")
    parsed = parser.parse_args(args)

    db_config = DatabaseConfig.from_env()
    engine = create_database_engine(db_config)
    session_factory = create_session_factory(engine)

    try:
        user, ws, key, raw_token = bootstrap_user_and_key(
            session_factory,
            email=parsed.email,
            workspace_name=parsed.workspace,
            key_name=parsed.key_name,
        )
        print("=" * 60)
        print("Nexora AI Credential Bootstrapped Successfully")
        print("=" * 60)
        print(f"User ID:        {user.id}")
        print(f"Email:          {user.email}")
        print(f"Workspace ID:   {ws.id} ({ws.name})")
        print(f"API Key ID:     {key.id}")
        print(f"Key Prefix:     {key.key_prefix}")
        print("-" * 60)
        print(f"RAW API KEY:    {raw_token}")
        print("-" * 60)
        print("Store this token securely in your environment as NEXORA_API_KEY.")
        print("This secret will NOT be displayed again.")
        print("=" * 60)
        return 0
    except Exception as exc:
        print(f"Error bootstrapping user: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
