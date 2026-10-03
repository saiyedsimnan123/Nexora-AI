"""Provider-neutral LLM message and response models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

SUPPORTED_ROLES = ("system", "user", "assistant")
_SECRET_MARKERS = ("api_key", "apikey", "authorization", "secret", "password")


@dataclass(frozen=True)
class LLMMessage:
    """One chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str):
            raise TypeError("role must be a string")
        if self.role not in SUPPORTED_ROLES:
            raise ValueError(f"role must be one of {SUPPORTED_ROLES}")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")


def _check_no_secret_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            if not isinstance(key, str):
                raise TypeError("raw keys must be strings")
            if any(marker in key.lower() for marker in _SECRET_MARKERS):
                raise ValueError("raw must not contain credential-like keys")
            _check_no_secret_keys(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            _check_no_secret_keys(inner)


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral generation result.

    ``usage`` maps token-count names (e.g. ``input_tokens``) to non-negative
    integers; providers may supply any subset. ``raw`` holds small
    provider-specific details (e.g. a response id) and must not contain secrets.
    """

    text: str
    model: str
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.model, str):
            raise TypeError("model must be a string")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.usage is not None:
            if not isinstance(self.usage, dict):
                raise TypeError("usage must be a dict or None")
            for key, count in self.usage.items():
                if not isinstance(key, str):
                    raise TypeError("usage keys must be strings")
                if isinstance(count, bool) or not isinstance(count, int):
                    raise TypeError("usage values must be integers")
                if count < 0:
                    raise ValueError("usage values must be non-negative")
            object.__setattr__(self, "usage", dict(self.usage))
        if self.raw is not None:
            if not isinstance(self.raw, dict):
                raise TypeError("raw must be a dict or None")
            _check_no_secret_keys(self.raw)
            object.__setattr__(self, "raw", copy.deepcopy(self.raw))
