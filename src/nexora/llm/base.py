"""LLM provider protocol (no provider SDK imports allowed here)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nexora.llm.models import LLMMessage, LLMResponse


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider call fails (not for invalid input)."""


@runtime_checkable
class LLMProvider(Protocol):
    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        ...
