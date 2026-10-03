"""OpenAI provider using the Responses API. The only module that touches the SDK."""

from __future__ import annotations

from typing import Any

from nexora.llm.base import LLMProviderError
from nexora.llm.config import (
    LLMConfig,
    validate_max_output_tokens,
    validate_temperature,
)
from nexora.llm.models import LLMMessage, LLMResponse

_USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


class OpenAIProvider:
    """Implements LLMProvider on top of ``client.responses.create``.

    The client is injectable; if absent it is built lazily on first use
    (construction itself never touches the network or the SDK).
    """

    def __init__(self, config: LLMConfig, *, client: Any = None) -> None:
        if not isinstance(config, LLMConfig):
            raise TypeError("config must be an LLMConfig")
        self._config = config
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # imported lazily

            self._client = OpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout,
            )
        return self._client

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self._validate_request(messages, model, temperature, max_output_tokens)
        params: dict[str, Any] = {
            "model": model if model is not None else self._config.model,
            "input": [{"role": m.role, "content": m.content} for m in messages],
        }
        temp = temperature if temperature is not None else self._config.temperature
        limit = (
            max_output_tokens
            if max_output_tokens is not None
            else self._config.max_output_tokens
        )
        if temp is not None:
            params["temperature"] = temp
        if limit is not None:
            params["max_output_tokens"] = limit

        try:
            response = self._get_client().responses.create(**params)
        except Exception as exc:
            raise LLMProviderError(
                f"OpenAI request failed ({type(exc).__name__})"
            ) from exc
        return self._convert(response, params["model"])

    @staticmethod
    def _validate_request(
        messages: Any,
        model: Any,
        temperature: Any,
        max_output_tokens: Any,
    ) -> None:
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

    @staticmethod
    def _convert(response: Any, requested_model: str) -> LLMResponse:
        text = getattr(response, "output_text", None)
        if not isinstance(text, str):
            raise LLMProviderError("OpenAI response did not contain output text")
        usage_obj = getattr(response, "usage", None)
        usage = {}
        for name in _USAGE_FIELDS:
            value = getattr(usage_obj, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[name] = value
        raw = {}
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str):
            raw["response_id"] = response_id
        model_name = getattr(response, "model", None)
        if not isinstance(model_name, str) or not model_name.strip():
            model_name = requested_model
        return LLMResponse(
            text=text, model=model_name, usage=usage or None, raw=raw or None
        )
