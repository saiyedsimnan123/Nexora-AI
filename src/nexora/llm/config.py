"""LLM configuration. Environment is read only inside LLMConfig.from_env()."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0


def validate_temperature(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("temperature must be a number")
    if not math.isfinite(value) or not MIN_TEMPERATURE <= value <= MAX_TEMPERATURE:
        raise ValueError(
            f"temperature must be finite and between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
        )
    return value


def validate_max_output_tokens(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_output_tokens must be an integer")
    if value <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    return value


@dataclass(frozen=True)
class LLMConfig:
    model: str
    provider: str = "openai"
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    timeout: float = 30.0
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("provider", "model"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string or None")
        if self.base_url is not None:
            if not isinstance(self.base_url, str):
                raise TypeError("base_url must be a string or None")
            parsed = urlparse(self.base_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("base_url must be an http or https URL")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("timeout must be a number")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        validate_temperature(self.temperature)
        validate_max_output_tokens(self.max_output_tokens)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LLMConfig:
        env = os.environ if environ is None else environ

        def get(name: str) -> str | None:
            value = env.get(f"NEXORA_LLM_{name}")
            return value.strip() if value and value.strip() else None

        def number(name: str, convert: Any) -> Any:
            raw = get(name)
            if raw is None:
                return None
            try:
                return convert(raw)
            except ValueError:
                raise ValueError(f"NEXORA_LLM_{name} is not a valid number") from None

        kwargs: dict[str, Any] = {
            "model": get("MODEL") or "",
            "api_key": get("API_KEY"),
            "base_url": get("BASE_URL"),
            "temperature": number("TEMPERATURE", float),
            "max_output_tokens": number("MAX_OUTPUT_TOKENS", int),
        }
        if get("PROVIDER"):
            kwargs["provider"] = get("PROVIDER")
        timeout = number("TIMEOUT", float)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return cls(**kwargs)
