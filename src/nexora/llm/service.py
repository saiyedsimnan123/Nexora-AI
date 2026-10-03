"""LLM application boundary and provider factory."""

from __future__ import annotations

from typing import Any

from nexora.llm.base import LLMProvider
from nexora.llm.config import (
    LLMConfig,
    validate_max_output_tokens,
    validate_temperature,
)
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.openai_provider import OpenAIProvider


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Build the provider named in ``config`` (no network calls)."""
    if not isinstance(config, LLMConfig):
        raise TypeError("config must be an LLMConfig")
    if config.provider.strip().lower() == "openai":
        return OpenAIProvider(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider!r}")


class LLMService:
    """Validates requests and delegates to an injected LLMProvider."""

    def __init__(self, provider: LLMProvider) -> None:
        if not isinstance(provider, LLMProvider):
            raise TypeError("provider must implement LLMProvider")
        self._provider = provider

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        if not messages:
            raise ValueError("messages must not be empty")
        if not all(isinstance(m, LLMMessage) for m in messages):
            raise TypeError("messages must contain only LLMMessage objects")
        if model is not None:
            if not isinstance(model, str):
                raise TypeError("model must be a string or None")
            if not model.strip():
                raise ValueError("model must not be empty")
        validate_temperature(temperature)
        validate_max_output_tokens(max_output_tokens)
        return self._provider.generate(
            list(messages),
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
