"""Research answer data model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from nexora.llm.models import _check_no_secret_keys


@dataclass(frozen=True)
class ResearchAnswer:
    """An LLM answer grounded in an evidence context (data only)."""

    text: str
    query: str
    model: str
    usage: dict[str, int] | None = None
    evidence_count: int = 0
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.model, str):
            raise TypeError("model must be a string")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if isinstance(self.evidence_count, bool) or not isinstance(self.evidence_count, int):
            raise TypeError("evidence_count must be an integer")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")
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
