"""ApiKey repository: secure generation, lookup, and revocation of API keys."""

from __future__ import annotations

import datetime
import hashlib
import secrets
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select

from nexora.database.models.api_key import ApiKey
from nexora.database.repository.base import BaseRepository

DEFAULT_KEY_PREFIX = "nx_live_"


class ApiKeyRepository(BaseRepository[ApiKey]):
    """Repository for ApiKey models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, ApiKey)

    def generate_key(
        self,
        *,
        user_id: UUID,
        name: str = "Default Key",
        expires_at: datetime.datetime | None = None,
        prefix: str = DEFAULT_KEY_PREFIX,
    ) -> tuple[ApiKey, str]:
        """Generate a cryptographically secure random API key, store its hash, and return (record, plaintext_key).

        The plaintext key is ONLY returned once at generation time. Only its SHA-256
        hash is stored in the database.
        """
        raw_secret = f"{prefix}{secrets.token_urlsafe(32)}"
        key_prefix = raw_secret[:14]
        key_hash = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()

        record = self._create(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            expires_at=expires_at,
        )
        return record, raw_secret

    def get_active_by_raw_key(self, raw_key: str) -> ApiKey | None:
        """Find an active, unrevoked, unexpired ApiKey matching raw_key."""
        if not isinstance(raw_key, str) or not raw_key.strip():
            return None

        key_hash = hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()
        stmt = select(ApiKey).where(ApiKey.key_hash == key_hash)
        record = self._session.scalars(stmt).first()
        if record is None:
            return None

        # Timing-safe double check against hash
        if not secrets.compare_digest(record.key_hash, key_hash):
            return None

        # Check revocation
        if record.revoked_at is not None:
            return None

        # Check expiration
        if record.expires_at is not None:
            now = datetime.datetime.now(datetime.timezone.utc)
            exp = record.expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=datetime.timezone.utc)
            if exp < now:
                return None

        return record

    def touch_last_used(self, key_id: UUID) -> None:
        """Update last_used_at timestamp for key."""
        now = datetime.datetime.now(datetime.timezone.utc)
        self._update(key_id, last_used_at=now)

    def revoke(self, key_id: UUID) -> bool:
        """Revoke an API key immediately. Returns True if found and revoked."""
        record = self.get_by_id(key_id)
        if record is None:
            return False
        now = datetime.datetime.now(datetime.timezone.utc)
        self._update(key_id, revoked_at=now)
        return True

    def list_by_user_id(
        self,
        user_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[ApiKey]:
        """List all API keys belonging to a user."""
        stmt = (
            select(ApiKey)
            .where(ApiKey.user_id == user_id)
            .order_by(ApiKey.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()
