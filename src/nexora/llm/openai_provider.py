"""OpenAI provider using the Responses API. The only module that touches the SDK."""

from __future__ import annotations

import re
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
    """Unified OpenAI provider supporting Chat Completions and Responses APIs.

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
            try:
                from openai import OpenAI  # imported lazily

                self._client = OpenAI(
                    api_key=self._config.api_key,
                    base_url=self._config.base_url,
                    timeout=self._config.timeout,
                    max_retries=2,
                )
            except Exception as exc:
                raise self._wrap_error(exc) from exc
        return self._client

    def _sanitize_message(self, message: str) -> str:
        clean = message
        if self._config.api_key:
            clean = clean.replace(self._config.api_key, "[REDACTED]")
        clean = re.sub(r"sk-[a-zA-Z0-9_\-]{20,}", "[REDACTED]", clean)
        clean = re.sub(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", "Bearer [REDACTED]", clean)
        return clean

    def _wrap_error(self, exc: Exception) -> LLMProviderError:
        raw_msg = str(exc).strip()
        sanitized = self._sanitize_message(raw_msg)
        exc_type = type(exc).__name__

        if "Authentication" in exc_type:
            detail = f"OpenAI authentication failed ({exc_type}): {sanitized}"
        elif "RateLimit" in exc_type:
            detail = f"OpenAI rate limit or quota exceeded ({exc_type}): {sanitized}"
        elif "Timeout" in exc_type:
            detail = f"OpenAI request timed out ({exc_type}): {sanitized}"
        elif "Connection" in exc_type:
            detail = f"OpenAI connection error ({exc_type}): {sanitized}"
        elif sanitized and sanitized != exc_type:
            detail = f"OpenAI request failed ({exc_type}): {sanitized}"
        else:
            detail = f"OpenAI request failed ({exc_type})"

        return LLMProviderError(detail)

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self._validate_request(messages, model, temperature, max_output_tokens)
        effective_model = model if model is not None else self._config.model
        temp = temperature if temperature is not None else self._config.temperature
        limit = (
            max_output_tokens
            if max_output_tokens is not None
            else self._config.max_output_tokens
        )

        client = self._get_client()

        # 1. Dedicated test doubles or pure Responses API clients
        if hasattr(client, "responses") and not hasattr(client, "chat"):
            params: dict[str, Any] = {
                "model": effective_model,
                "input": [{"role": m.role, "content": m.content} for m in messages],
            }
            if temp is not None:
                params["temperature"] = temp
            if limit is not None:
                params["max_output_tokens"] = limit
            try:
                response = client.responses.create(**params)
            except Exception as exc:
                raise self._wrap_error(exc) from exc
            return self._convert(response, effective_model)

        # 2. Standard chat completions (primary for real OpenAI clients)
        chat_params: dict[str, Any] = {
            "model": effective_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if temp is not None:
            chat_params["temperature"] = temp
        if limit is not None:
            # Modern reasoning models require max_completion_tokens
            if "o1" in effective_model.lower() or "o3" in effective_model.lower():
                chat_params["max_completion_tokens"] = limit
            else:
                chat_params["max_tokens"] = limit

        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            try:
                response = client.chat.completions.create(**chat_params)
            except Exception as exc:
                raise self._wrap_error(exc) from exc
            return self._convert(response, effective_model)

        # 3. Fallback to responses if chat.completions is absent
        if hasattr(client, "responses"):
            params = {
                "model": effective_model,
                "input": [{"role": m.role, "content": m.content} for m in messages],
            }
            if temp is not None:
                params["temperature"] = temp
            if limit is not None:
                params["max_output_tokens"] = limit
            try:
                response = client.responses.create(**params)
            except Exception as exc:
                raise self._wrap_error(exc) from exc
            return self._convert(response, effective_model)

        raise LLMProviderError("OpenAI client provides neither chat.completions nor responses API")

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
        if text is None:
            choices = getattr(response, "choices", None)
            if choices and len(choices) > 0:
                first = choices[0]
                msg = getattr(first, "message", None)
                if msg is not None:
                    text = getattr(msg, "content", None)
                elif isinstance(first, dict):
                    text = first.get("message", {}).get("content")

        if not isinstance(text, str):
            raise LLMProviderError("OpenAI response did not contain output text")

        usage_obj = getattr(response, "usage", None)
        usage = {}
        if usage_obj is not None:
            # Map input/output/total tokens cleanly across Responses and Chat APIs
            inp = getattr(usage_obj, "input_tokens", None)
            if inp is None:
                inp = getattr(usage_obj, "prompt_tokens", None)
            if isinstance(inp, int) and not isinstance(inp, bool) and inp >= 0:
                usage["input_tokens"] = inp

            out = getattr(usage_obj, "output_tokens", None)
            if out is None:
                out = getattr(usage_obj, "completion_tokens", None)
            if isinstance(out, int) and not isinstance(out, bool) and out >= 0:
                usage["output_tokens"] = out

            tot = getattr(usage_obj, "total_tokens", None)
            if isinstance(tot, int) and not isinstance(tot, bool) and tot >= 0:
                usage["total_tokens"] = tot

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
