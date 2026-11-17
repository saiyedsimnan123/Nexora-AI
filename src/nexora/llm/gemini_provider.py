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

import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from nexora.llm.base import LLMProviderError
from nexora.llm.config import (
    LLMConfig,
    validate_max_output_tokens,
    validate_temperature,
)
from nexora.llm.models import LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-2.0-flash"
SUPPORTED_GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-flash-lite",
)
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"

TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_NEXORA_GEMINI_API_KEY = "NEXORA_GEMINI_API_KEY"
ENV_LLM_API_KEY = "NEXORA_LLM_API_KEY"


def is_transient_gemini_error(exc: Exception) -> bool:
    """Return True if the error is transient and eligible for backoff retry and model fallback."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        status_code = getattr(resp, "status_code", None)
        if status_code in TRANSIENT_STATUS_CODES:
            return True
        if status_code in (400, 401, 403, 404):
            return False

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        cause_resp = getattr(cause, "response", None)
        if cause_resp is not None:
            cause_status = getattr(cause_resp, "status_code", None)
            if cause_status in TRANSIENT_STATUS_CODES:
                return True
            if cause_status in (400, 401, 403, 404):
                return False

    exc_type = type(exc).__name__
    cause_type = type(cause).__name__ if cause else ""
    if any(
        t in exc_type or t in cause_type
        for t in (
            "Timeout",
            "ConnectError",
            "ConnectionError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "NetworkError",
            "RemoteProtocolError",
        )
    ):
        return True

    msg = str(exc).lower()
    if any(perm in msg for perm in ("400", "401", "403", "404", "api_key_invalid", "permission denied", "not found")):
        return False
    if any(m in msg for m in ("503", "429", "502", "504", "high demand", "rate limit", "temporarily unavailable", "timed out")):
        return True

    return False


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
    result = re.sub(r"AIza[0-9A-Za-z\-_]{8,}", "[REDACTED]", result)
    result = re.sub(r"sk-[a-zA-Z0-9_\-]{16,}", "[REDACTED]", result)
    result = re.sub(r"nx_live_[a-zA-Z0-9_\-]{16,}", "[REDACTED]", result)

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

    def __init__(
        self,
        config: LLMConfig,
        *,
        client: Any = None,
        fallback_model: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        max_retry_delay: float | None = None,
        max_retry_duration: float | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
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

        # Determine fallback model
        raw_fb = (
            fallback_model
            or config.fallback_model
            or os.getenv("NEXORA_GEMINI_FALLBACK_MODEL")
            or os.getenv("GEMINI_FALLBACK_MODEL")
        )
        if raw_fb and raw_fb.strip():
            clean_fb = raw_fb.strip().removeprefix("models/")
            self._fallback_model = clean_fb if clean_fb != self._model else None
        else:
            if self._model != DEFAULT_GEMINI_FALLBACK_MODEL:
                self._fallback_model = DEFAULT_GEMINI_FALLBACK_MODEL
            else:
                self._fallback_model = "gemini-1.5-flash"

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

        # Retry configuration: fast defaults for test doubles, realistic bounded defaults for live clients
        is_test_client = client is not None
        default_max_retries = 2
        default_delay = 0.01 if is_test_client else 0.5
        default_max_delay = 0.05 if is_test_client else 4.0
        default_max_duration = 1.0 if is_test_client else 15.0

        resolved_retries = max_retries if max_retries is not None else config.max_retries
        if resolved_retries is None:
            raw_r = os.getenv("NEXORA_LLM_MAX_RETRIES") or os.getenv("GEMINI_MAX_RETRIES")
            if raw_r and raw_r.strip():
                try:
                    resolved_retries = int(raw_r.strip())
                except ValueError:
                    resolved_retries = default_max_retries
            else:
                resolved_retries = default_max_retries
        self._max_retries = max(0, resolved_retries)

        resolved_delay = retry_delay if retry_delay is not None else config.retry_delay
        if resolved_delay is None:
            raw_d = os.getenv("NEXORA_LLM_RETRY_DELAY") or os.getenv("GEMINI_RETRY_DELAY")
            if raw_d and raw_d.strip():
                try:
                    resolved_delay = float(raw_d.strip())
                except ValueError:
                    resolved_delay = default_delay
            else:
                resolved_delay = default_delay
        self._retry_delay = max(0.0, float(resolved_delay))

        resolved_max_delay = max_retry_delay if max_retry_delay is not None else config.max_retry_delay
        self._max_retry_delay = max(0.0, float(resolved_max_delay if resolved_max_delay is not None else default_max_delay))

        resolved_duration = max_retry_duration if max_retry_duration is not None else config.max_retry_duration
        self._max_retry_duration = max(0.0, float(resolved_duration if resolved_duration is not None else default_max_duration))

        self._sleep_fn = sleep_fn or time.sleep

    @property
    def model(self) -> str:
        return self._model

    @property
    def fallback_model(self) -> str | None:
        return self._fallback_model

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def retry_delay(self) -> float:
        return self._retry_delay

    @property
    def max_retry_delay(self) -> float:
        return self._max_retry_delay

    @property
    def max_retry_duration(self) -> float:
        return self._max_retry_duration

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with full jitter."""
        base = min(self._max_retry_delay, self._retry_delay * (2.0 ** attempt))
        if base <= 0:
            return 0.0
        return random.uniform(0.5 * base, base)

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self._model!r}, base_url={self._base_url!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise LLMProviderError(
                    "Gemini API key is required; set config.api_key or GEMINI_API_KEY environment variable"
                )
            read_timeout = max(float(self._config.timeout), 60.0)
            timeout_config = httpx.Timeout(
                timeout=read_timeout,
                connect=15.0,
                read=read_timeout,
                write=30.0,
                pool=15.0,
            )
            self._client = httpx.Client(base_url=self._base_url, timeout=timeout_config)
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

        models_to_try: list[str] = [effective_model]
        if self._fallback_model and self._fallback_model != effective_model:
            models_to_try.append(self._fallback_model)

        start_time = time.monotonic()
        last_exception: Exception | None = None
        max_attempts_per_model = self._max_retries + 1

        for model_idx, target_model in enumerate(models_to_try):
            path = f"/v1beta/models/{target_model}:generateContent"
            is_fallback = (model_idx > 0)

            for attempt in range(max_attempts_per_model):
                elapsed = time.monotonic() - start_time
                if elapsed >= self._max_retry_duration and (model_idx > 0 or attempt > 0):
                    logger.warning(
                        "Gemini retry deadline exceeded (%.2fs >= %.2fs). Terminating retry loop.",
                        elapsed,
                        self._max_retry_duration,
                    )
                    break

                try:
                    if hasattr(client, "post"):
                        response = client.post(path, json=payload, headers=headers)
                        if hasattr(response, "raise_for_status"):
                            response.raise_for_status()
                        if callable(getattr(response, "json", None)):
                            data = response.json()
                        else:
                            data = getattr(response, "data", response)
                        parsed = self._parse_response(data, target_model)
                        if is_fallback and parsed.raw is not None:
                            parsed.raw["fallback_from"] = effective_model
                        return parsed

                    elif hasattr(client, "generate"):
                        response = client.generate(
                            messages,
                            model=target_model,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                        )
                        if is_fallback and getattr(response, "raw", None) is not None:
                            response.raw["fallback_from"] = effective_model
                        return response
                    else:
                        raise LLMProviderError("Client does not provide a callable 'post' or 'generate' method")

                except LLMProviderError:
                    raise
                except Exception as exc:
                    last_exception = exc

                    # Fail fast on non-transient errors (400, 401, 403, 404, etc.)
                    if not is_transient_gemini_error(exc):
                        error_message = _format_actionable_error(exc, api_key=self._api_key)
                        raise LLMProviderError(error_message) from exc

                    is_last_attempt = (attempt == max_attempts_per_model - 1)
                    has_fallback = (model_idx < len(models_to_try) - 1)

                    if not is_last_attempt:
                        delay = self._calculate_backoff_delay(attempt)
                        if (time.monotonic() - start_time + delay) > self._max_retry_duration:
                            logger.warning(
                                "Gemini retry delay (%.3fs) would exceed max retry duration (%.2fs).",
                                delay,
                                self._max_retry_duration,
                            )
                            if has_fallback:
                                break
                            else:
                                break

                        logger.info(
                            "Gemini model %s transient error (%s). Retrying in %.3fs (attempt %d/%d)...",
                            target_model,
                            type(exc).__name__,
                            delay,
                            attempt + 1,
                            max_attempts_per_model,
                        )
                        self._sleep_fn(delay)
                        continue
                    else:
                        if has_fallback:
                            next_model = models_to_try[model_idx + 1]
                            logger.warning(
                                "Gemini model %s exhausted %d attempts due to transient error: %s. "
                                "Falling back to supported Gemini model %s.",
                                target_model,
                                max_attempts_per_model,
                                _format_actionable_error(exc, api_key=self._api_key),
                                next_model,
                            )
                            break

        if last_exception is not None:
            base_err = _format_actionable_error(last_exception, api_key=self._api_key)
            if len(models_to_try) > 1:
                detail = (
                    f"{base_err} (Primary model '{effective_model}' and fallback model "
                    f"'{self._fallback_model}' were both attempted with bounded retries and exhausted)"
                )
                raise LLMProviderError(detail) from last_exception
            raise LLMProviderError(base_err) from last_exception

        raise LLMProviderError("Gemini generation failed: retries exhausted without a response")

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
