"""Authentication domain models."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class AuthenticatedUser:
    """Authenticated user context resolved from verified API credentials."""

    id: UUID
    email: str
    api_key_id: UUID
    api_key_prefix: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": str(self.id),
            "email": self.email,
            "api_key_id": str(self.api_key_id),
            "api_key_prefix": self.api_key_prefix,
        }
