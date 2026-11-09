"""Google Gemini LLM provider for Nexora AI.

Implements the provider-independent ``LLMProvider`` protocol using the official
Google Gemini REST API (``generateContent`` endpoint).

Key design principles:
- Safe, offline initialization: constructing a provider never makes network calls.
- Secret hygiene: credentials, API keys, and auth headers are never logged, formatted
  in repr, or exposed in exception messages.
- Testability: accepts an injected client/transport for 100% offline, deterministic
  unit testing without network dependencies or paid credentials.
- Free-tier compatibility: defaults to ``gemini-2.5-flash`` with standard token usage
  tracking and structured JSON output support.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from typing import Any

import httpx

from nexora.llm.base import LLMProviderError
from nexora.llm.config import (
    LLMConfig,
    validate_max_output_tokens,
    validate_temperature,
)
from nexora.llm.models import LLMMessage, LLMResponse

DEFAULT_GEMINI_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"

ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_NEXORA_GEMINI_API_KEY = "NEXORA_GEMINI_API_KEY"
ENV_LLM_API_KEY = "NEXORA_LLM_API_KEY"


def sanitize_secret_text(text: str, extra_secrets: Sequence[str | None] = ()) -> str:
    """Sanitize any API keys or secret tokens from ``text`` to prevent leaks."""
    if not text:
        return ""
    result = text
    keys_to_scrub: set[str] = set()
    for s in extra_secrets:
        if s and len(str(s).strip()) >= 4:
            s_str = str(s).strip()
            clean_s = s_str.strip("'\"")
            keys_to_scrub.add(s_str)
            if len(clean_s) >= 4:
                keys_to_scrub.add(clean_s)

    for env_var in (
        ENV_GEMINI_API_KEY,
        ENV_NEXORA_GEMINI_API_KEY,
        ENV_LLM_API_KEY,
        "OPENAI_API_KEY",
        "GEMINI_KEY",
    ):
        val = os.getenv(env_var)
        if val and len(val.strip()) >= 4:
            val_str = val.strip()
            clean_val = val_str.strip("'\"")
            keys_to_scrub.add(val_str)
            if len(clean_val) >= 4:
                keys_to_scrub.add(clean_val)

    for key in sorted(keys_to_scrub, key=len, reverse=True):
        result = result.replace(key, "[REDACTED]")

    result = re.sub(
        r"x-goog-api-key['\":\s=]+[a-zA-Z0-9_\-]+",
        "x-goog-api-key: [REDACTED]",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"([?&]key=)[^&\s]+",
        r"\1[REDACTED]",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"(Bearer\s+)[a-zA-Z0-9_\-\.]{10,}",
        r"\1[REDACTED]",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED]", result)
    result = re.sub(r"sk-[a-zA-Z0-9_\-]{20,}", "[REDACTED]", result)

    return result


def _extract_response_error_details(
    resp: Any, extra_secrets: Sequence[str | None] = ()
) -> str:
    """Extract and sanitize descriptive error details from an HTTP response object."""
    if resp is None:
        return ""

    data: Any = None
    try:
        if callable(getattr(resp, "json", None)):
            data = resp.json()
        elif hasattr(resp, "data") and isinstance(resp.data, dict):
            data = resp.data
    except Exception:
        data = None

    if isinstance(data, dict):
        error_obj = data.get("error")
        if isinstance(error_obj, dict):
            msg = str(error_obj.get("message") or "").strip()
            status = str(error_obj.get("status") or "").strip()
            details = error_obj.get("details")

            reasons: list[str] = []
            if isinstance(details, list):
                for item in details:
                    if isinstance(item, dict):
                        if item.get("reason"):
                            reasons.append(str(item["reason"]))
                        if item.get("fieldViolations") and isinstance(
                            item["fieldViolations"], list
                        ):
                            for fv in item["fieldViolations"]:
                                if isinstance(fv, dict) and fv.get("description"):
                                    reasons.append(str(fv["description"]))

            detail_parts: list[str] = []
            if msg:
                detail_parts.append(msg)
            elif status:
                detail_parts.append(f"Status {status}")

            if reasons:
                seen: set[str] = set()
                deduped = [r for r in reasons if not (r in seen or seen.add(r))]
                detail_parts.append(f"(reason: {', '.join(deduped)})")

            if detail_parts:
                combined = " ".join(detail_parts)
                return sanitize_secret_text(combined, extra_secrets=extra_secrets)

    raw_text = getattr(resp, "text", "")
    if isinstance(raw_text, str) and raw_text.strip():
        collapsed = " ".join(raw_text.strip().split())[:300]
        return sanitize_secret_text(collapsed, extra_secrets=extra_secrets)

    return ""


def _format_actionable_error(exc: Exception, api_key: str | None = None) -> str:
    """Format a sanitized, actionable error message from an exception."""
    raw_msg = sanitize_secret_text(str(exc), extra_secrets=[api_key])
    exc_type = type(exc).__name__

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        sanitized_cause = sanitize_secret_text(str(cause), extra_secrets=[api_key])
        if sanitized_cause and sanitized_cause not in raw_msg:
            raw_msg = f"{raw_msg}: {sanitized_cause}"

    resp = getattr(exc, "response", None) or (
        getattr(cause, "response", None) if cause else None
    )
    if resp is not None:
        resp_detail = _extract_response_error_details(resp, extra_secrets=[api_key])
        if resp_detail and resp_detail not in raw_msg:
            raw_msg = f"{raw_msg}: {resp_detail}"

    status_code = getattr(resp, "status_code", None) if resp is not None else None
    if status_code is None and "HTTP " in raw_msg:
        match = re.search(r"HTTP\s+(\d{3})", raw_msg)
        if match:
            try:
                status_code = int(match.group(1))
            except ValueError:
                pass

    if status_code == 400:
        if "API_KEY_INVALID" in raw_msg or "api key not valid" in raw_msg.lower():
            return (
                f"Gemini API key rejected (HTTP 400): invalid API key. "
                f"Verify or regenerate your API key at https://aistudio.google.com/app/apikey. "
                f"Details: {raw_msg}"
            )
        return (
            f"Gemini API request rejected (HTTP 400): check model name and request parameters. "
            f"Details: {raw_msg}"
        )
    if status_code == 401:
        return f"Gemini authentication failed (HTTP 401): invalid or missing API key. Details: {raw_msg}"
    if status_code == 403:
        return f"Gemini permission denied (HTTP 403): verify API key permissions, API activation, and project quota. Details: {raw_msg}"
    if status_code == 404:
        return f"Gemini model or endpoint not found (HTTP 404): check configured model name. Details: {raw_msg}"
    if status_code == 429:
        return f"Gemini rate limit or quota exceeded (HTTP 429): retry after backoff. Details: {raw_msg}"
    if status_code is not None and status_code >= 500:
        return f"Gemini upstream service error (HTTP {status_code}): Google Generative Language service error. Details: {raw_msg}"
    if "ConnectError" in exc_type or "ConnectionError" in exc_type or (
        cause and ("ConnectError" in type(cause).__name__ or "ConnectionError" in type(cause).__name__)
    ):
        return f"Gemini network connection error: unable to reach API endpoint. Details: {raw_msg}"
    if "Timeout" in exc_type or (cause and "Timeout" in type(cause).__name__):
        return f"Gemini request timed out: API did not respond within timeout. Details: {raw_msg}"

    return f"Gemini generation failed ({exc_type}): {raw_msg}"


class GeminiProvider:
    """Google Gemini LLM provider implementing the ``LLMProvider`` protocol.

    Communicates with Google Gemini via REST APIs (``generateContent``).
    Accepts an injected client for offline testing.
    """

    provider: str = "gemini"
    name: str = "gemini"

    def __init__(self, config: LLMConfig, *, client: Any = None) -> None:
        if not isinstance(config, LLMConfig):
            raise TypeError("config must be an LLMConfig")
        self._config = config

        # Determine effective model
        raw_model = (config.model or "").strip()
        clean_model = raw_model.removeprefix("models/")
        if not raw_model or raw_model in ("gpt-4o-mini", "default"):
            self._model = DEFAULT_GEMINI_LLM_MODEL
        else:
            self._model = clean_model

        raw_base = (
            config.base_url
            or os.getenv("GEMINI_BASE_URL")
            or os.getenv("NEXORA_LLM_BASE_URL")
            or DEFAULT_GEMINI_BASE_URL
        )
        self._base_url = raw_base.rstrip("/")

        api_key = (
            config.api_key
            or os.getenv(ENV_GEMINI_API_KEY)
            or os.getenv(ENV_NEXORA_GEMINI_API_KEY)
            or os.getenv(ENV_LLM_API_KEY)
        )
        self._api_key = str(api_key).strip().strip("'\"") if api_key and str(api_key).strip().strip("'\"") else None

        self._client = client
        self._owns_client = False

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self._model!r}, base_url={self._base_url!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise LLMProviderError(
                    "Gemini API key is required; set config.api_key or GEMINI_API_KEY environment variable"
                )
            self._client = httpx.Client(base_url=self._base_url, timeout=self._config.timeout)
            self._owns_client = True
        return self._client

    @staticmethod
    def _validate_request(
        messages: list[LLMMessage],
        model: str | None,
        temperature: float | None,
        max_output_tokens: int | None,
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

    def _build_payload(
        self,
        messages: list[LLMMessage],
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> dict[str, Any]:
        """Convert provider-neutral LLM messages to Gemini REST API payload."""
        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append({"text": msg.content})
            else:
                gemini_role = "model" if msg.role == "assistant" else "user"
                # If last content has same role, combine parts
                if contents and contents[-1]["role"] == gemini_role:
                    contents[-1]["parts"].append({"text": msg.content})
                else:
                    contents.append({
                        "role": gemini_role,
                        "parts": [{"text": msg.content}],
                    })

        # Gemini requires at least one content item
        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})

        payload: dict[str, Any] = {"contents": contents}

        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        elif self._config.temperature is not None:
            generation_config["temperature"] = self._config.temperature

        if max_output_tokens is not None:
            generation_config["maxOutputTokens"] = max_output_tokens
        elif self._config.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = self._config.max_output_tokens

        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text using Google Gemini API.

        Args:
            messages: List of chat messages.
            model: Optional model override.
            temperature: Optional sampling temperature override (0.0 to 2.0).
            max_output_tokens: Optional token limit override.

        Returns:
            Provider-neutral LLMResponse.

        Raises:
            TypeError: For wrong argument types.
            ValueError: For empty messages or invalid parameters.
            LLMProviderError: If the API call fails or credentials are rejected.
        """
        self._validate_request(messages, model, temperature, max_output_tokens)
        effective_model = (model or self._model).removeprefix("models/")

        payload = self._build_payload(messages, temperature, max_output_tokens)
        path = f"/v1beta/models/{effective_model}:generateContent"

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key

        client = self._get_client()

        try:
            if hasattr(client, "post"):
                response = client.post(path, json=payload, headers=headers)
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                if callable(getattr(response, "json", None)):
                    data = response.json()
                else:
                    data = getattr(response, "data", response)
            elif hasattr(client, "generate"):
                # Direct test double
                return client.generate(
                    messages,
                    model=effective_model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            else:
                raise LLMProviderError("Client does not provide a callable 'post' or 'generate' method")
        except LLMProviderError:
            raise
        except Exception as exc:
            error_message = _format_actionable_error(exc, api_key=self._api_key)
            raise LLMProviderError(error_message) from exc

        return self._parse_response(data, effective_model)

    def _parse_response(self, data: Any, model: str) -> LLMResponse:
        """Parse Gemini REST response JSON into LLMResponse."""
        if not isinstance(data, dict):
            raise LLMProviderError(f"Gemini response must be a JSON object, got {type(data).__name__}")

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or len(candidates) == 0:
            # Check for prompt feedback block
            feedback = data.get("promptFeedback")
            if isinstance(feedback, dict) and feedback.get("blockReason"):
                reason = feedback.get("blockReason")
                raise LLMProviderError(f"Gemini generation blocked: {reason}")
            raise LLMProviderError("Gemini response contained no candidates")

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise LLMProviderError("Gemini candidate must be an object")

        content = candidate.get("content")
        text_parts: list[str] = []
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict) and "text" in p:
                        text_parts.append(str(p["text"]))

        output_text = "".join(text_parts)

        # Parse token usage
        usage: dict[str, int] | None = None
        usage_meta = data.get("usageMetadata")
        if isinstance(usage_meta, dict):
            prompt_tokens = usage_meta.get("promptTokenCount")
            candidates_tokens = usage_meta.get("candidatesTokenCount")
            total_tokens = usage_meta.get("totalTokenCount")
            usage_dict: dict[str, int] = {}
            if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                usage_dict["input_tokens"] = prompt_tokens
            if isinstance(candidates_tokens, int) and candidates_tokens >= 0:
                usage_dict["output_tokens"] = candidates_tokens
            if isinstance(total_tokens, int) and total_tokens >= 0:
                usage_dict["total_tokens"] = total_tokens
            if usage_dict:
                usage = usage_dict

        # Provider-specific details (raw) without credentials
        raw: dict[str, Any] = {}
        finish_reason = candidate.get("finishReason")
        if finish_reason:
            raw["finish_reason"] = str(finish_reason)

        return LLMResponse(
            text=output_text,
            model=model,
            usage=usage,
            raw=raw if raw else None,
        )
