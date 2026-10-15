"""Google Gemini semantic embedding provider for Nexora AI (Milestone 12C-38).

``GeminiEmbeddingProvider`` implements the provider-independent
``EmbeddingProvider`` interface using the official Google Gemini REST API
(``embedContent`` and ``batchEmbedContents`` endpoints).

Key design principles:
- Safe, offline initialization: constructing a provider never makes network calls.
- Secret hygiene: credentials and auth headers are never logged, formatted in repr,
  or exposed in exception messages.
- Testability: accepts an injected client/transport for completely offline,
  deterministic unit testing without network dependencies or paid credentials.
- Strict validation: embeddings must be non-empty sequences of finite floats matching
  the configured vector dimension.
- Truthful status: never falls back to synthetic or hash embeddings.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from nexora.embedding.base import validate_texts
from nexora.embedding.config import ENV_API_KEY, EmbeddingConfig

ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_NEXORA_GEMINI_API_KEY = "NEXORA_GEMINI_API_KEY"

DEFAULT_GEMINI_MODEL = "gemini-embedding-2"
DEFAULT_GEMINI_DIMENSION = 768
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_BATCH_SIZE = 100

LEGACY_MODEL_ALIASES: dict[str, str] = {
    "text-embedding-004": "gemini-embedding-2",
    "models/text-embedding-004": "gemini-embedding-2",
    "gemini-embedding-001": "gemini-embedding-2",
    "models/gemini-embedding-001": "gemini-embedding-2",
    "embedding-001": "gemini-embedding-2",
    "models/embedding-001": "gemini-embedding-2",
}


def sanitize_secret_text(text: str, extra_secrets: Sequence[str | None] = ()) -> str:
    """Sanitize any API keys or secret tokens from ``text`` to prevent leaks."""
    if not text:
        return ""
    result = text
    # 1. Scrub extra secrets and environment variables
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
        ENV_API_KEY,
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
        result = result.replace(key, "***REDACTED***")

    # 2. Scrub regex patterns
    result = re.sub(
        r"x-goog-api-key['\":\s=]+[a-zA-Z0-9_\-]+",
        "x-goog-api-key: ***REDACTED***",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"([?&]key=)[^&\s]+",
        r"\1***REDACTED***",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"(Bearer\s+)[a-zA-Z0-9_\-]+",
        r"\1***REDACTED***",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"AIza[0-9A-Za-z-_]{35}", "***REDACTED***", result)

    return result


def _extract_response_error_details(
    resp: Any,
    extra_secrets: Sequence[str | None] = (),
) -> str:
    """Extract and sanitize descriptive error details from an HTTP response object.

    Handles Google Generative Language JSON error formats:
    - ``error.message``
    - ``error.status``
    - ``error.details[].reason`` (e.g. ``API_KEY_INVALID``)
    - ``error.details[].fieldViolations[].description``
    - Fallback to sanitized response text.

    Guarantees secrets are never leaked.
    """
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
                        if item.get("fieldViolations") and isinstance(item["fieldViolations"], list):
                            for fv in item["fieldViolations"]:
                                if isinstance(fv, dict) and fv.get("description"):
                                    reasons.append(str(fv["description"]))

            detail_parts: list[str] = []
            if msg:
                detail_parts.append(msg)
            elif status:
                detail_parts.append(f"Status {status}")

            if reasons:
                seen = set()
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

    resp = getattr(exc, "response", None) or (getattr(cause, "response", None) if cause else None)
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
        return f"Gemini API request rejected (HTTP 400): check model name and request parameters. Details: {raw_msg}"
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
    if "ConnectError" in exc_type or "ConnectionError" in exc_type or (cause and ("ConnectError" in type(cause).__name__ or "ConnectionError" in type(cause).__name__)):
        return f"Gemini network connection error: unable to reach API endpoint. Details: {raw_msg}"
    if "Timeout" in exc_type or (cause and "Timeout" in type(cause).__name__):
        return f"Gemini request timed out: API did not respond within timeout. Details: {raw_msg}"

    return f"Gemini readiness check failed ({exc_type}): {raw_msg}"


def is_simulated_client(client: Any) -> bool:
    """Determine whether a client is a mock, fake, or synthetic test double."""
    if client is None:
        return False
    if getattr(client, "is_simulated", False) or getattr(client, "is_synthetic", False):
        return True
    cls_name = type(client).__name__
    if any(k in cls_name for k in ("Mock", "Fake", "Stub", "MagicMock", "TestDouble")):
        return True
    if hasattr(client, "calls") and isinstance(getattr(client, "calls"), list):
        return True
    if not isinstance(client, httpx.Client):
        return True
    transport = getattr(client, "_transport", None)
    if transport is not None:
        transport_cls = type(transport).__name__
        if any(k in transport_cls for k in ("Mock", "Fake", "Stub", "MagicMock")):
            return True
    return False


@dataclass(frozen=True)
class GeminiReadinessReport:
    """Readiness report for the Google Gemini embedding provider.

    Attributes:
        status: High-level status: 'ready', 'simulated', 'unavailable', or 'failed'.
        is_ready: True if the provider is operational and produced a valid probe vector.
        is_live: True ONLY if a genuine live network API call succeeded with real credentials.
                 Never True for simulated, mock, or synthetic test doubles.
        is_simulated: True if a mock, fake, or synthetic client/provider was detected.
        model: Effective Gemini embedding model name (e.g. 'gemini-embedding-2').
        dimension: Configured output vector dimension (e.g. 768).
        sample_vector_dimension: Actual length of the returned probe embedding vector, if obtained.
        configuration_valid: True if provider configuration parameters are valid.
        error: Sanitized error or limitation message if not ready, or None if successful.
        probe_text: Text used for the readiness probe.
        latency_ms: Round-trip latency in milliseconds, if the probe was executed.
        timestamp: ISO 8601 UTC timestamp of the check.
    """

    status: str  # "ready" | "simulated" | "unavailable" | "failed"
    is_ready: bool
    is_live: bool
    is_simulated: bool
    model: str
    dimension: int
    sample_vector_dimension: int | None = None
    configuration_valid: bool = True
    error: str | None = None
    probe_text: str = "health check"
    latency_ms: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert readiness report to a JSON-serializable dictionary."""
        return {
            "status": self.status,
            "is_ready": self.is_ready,
            "is_live": self.is_live,
            "is_simulated": self.is_simulated,
            "model": self.model,
            "dimension": self.dimension,
            "sample_vector_dimension": self.sample_vector_dimension,
            "configuration_valid": self.configuration_valid,
            "error": self.error,
            "probe_text": self.probe_text,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


class GeminiEmbeddingProviderError(RuntimeError):
    """Raised when the Gemini embedding provider fails or cannot produce valid vectors.

    Error messages are sanitized to prevent credential leakage.
    """


def _validate_embedding(
    embedding: Any, expected_dimension: int | None, where: str = ""
) -> list[float]:
    """Validate one embedding and return it as ``list[float]``."""
    if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
        raise GeminiEmbeddingProviderError(
            f"Gemini embedding{where} must be a sequence of numbers, "
            f"got {type(embedding).__name__}"
        )
    if len(embedding) == 0:
        raise GeminiEmbeddingProviderError(f"Gemini returned an empty embedding{where}")

    vector: list[float] = []
    for index, value in enumerate(embedding):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GeminiEmbeddingProviderError(
                f"Gemini embedding{where} has a non-numeric value "
                f"({type(value).__name__}) at index {index}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise GeminiEmbeddingProviderError(
                f"Gemini embedding{where} has a non-finite value at index {index}"
            )
        vector.append(number)

    if expected_dimension is not None and len(vector) != expected_dimension:
        raise GeminiEmbeddingProviderError(
            f"Gemini embedding{where} has {len(vector)} dimensions, "
            f"expected {expected_dimension}"
        )
    return vector


class GeminiEmbeddingProvider:
    """``EmbeddingProvider`` backed by Google Gemini embedding REST APIs.

    Attributes:
        provider: Provider identifier string ("gemini").
        name: Provider name alias ("gemini").
        model: Configured Gemini embedding model name.
        dimension: Configured embedding vector dimension (defaults to 768).
    """

    provider: str = "gemini"
    name: str = "gemini"

    def __init__(self, config: EmbeddingConfig, client: Any | None = None) -> None:
        if not isinstance(config, EmbeddingConfig):
            raise TypeError(
                f"config must be an EmbeddingConfig, got {type(config).__name__}"
            )
        self._config = config

        # Determine model: normalize default/fallback names and migrate legacy aliases
        raw_model = (config.model or "").strip()
        clean_raw = raw_model.removeprefix("models/")
        if not raw_model or raw_model in ("hash-sha256", "text-embedding-3-small", "default"):
            self._model = DEFAULT_GEMINI_MODEL
        elif raw_model in LEGACY_MODEL_ALIASES:
            self._model = LEGACY_MODEL_ALIASES[raw_model]
        elif clean_raw in LEGACY_MODEL_ALIASES:
            self._model = LEGACY_MODEL_ALIASES[clean_raw]
        else:
            self._model = raw_model

        self._dimension = config.dimension if config.dimension is not None else DEFAULT_GEMINI_DIMENSION
        self._batch_size = config.batch_size if config.batch_size is not None else DEFAULT_GEMINI_BATCH_SIZE

        raw_base = config.base_url or os.getenv("GEMINI_BASE_URL") or os.getenv("NEXORA_EMBEDDING_BASE_URL") or DEFAULT_GEMINI_BASE_URL
        self._base_url = raw_base.rstrip("/")

        # Clean model name for URL paths (e.g. models/gemini-embedding-2 -> gemini-embedding-2)
        self._clean_model = self._model.removeprefix("models/")

        self._owns_client = False
        self._is_simulated_override: bool | None = None
        if client is not None:
            self._client = client
            key = (
                config.api_key
                or os.getenv(ENV_GEMINI_API_KEY)
                or os.getenv(ENV_NEXORA_GEMINI_API_KEY)
                or os.getenv(ENV_API_KEY)
            )
            self._api_key = str(key).strip().strip("'\"") if key and str(key).strip().strip("'\"") else None
            return

        api_key = (
            config.api_key
            or os.getenv(ENV_GEMINI_API_KEY)
            or os.getenv(ENV_NEXORA_GEMINI_API_KEY)
            or os.getenv(ENV_API_KEY)
        )
        cleaned_key = str(api_key).strip().strip("'\"") if api_key else ""
        if not cleaned_key:
            raise GeminiEmbeddingProviderError(
                f"An API key is required to create a Gemini client; "
                f"set config.api_key or the {ENV_GEMINI_API_KEY} environment variable"
            )

        self._api_key = cleaned_key
        self._client = httpx.Client(base_url=self._base_url, timeout=30.0)
        self._owns_client = True

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_simulated(self) -> bool:
        """Whether this provider is backed by a simulated or mock test double."""
        if self._is_simulated_override is not None:
            return self._is_simulated_override
        return is_simulated_client(self._client)

    @is_simulated.setter
    def is_simulated(self, value: bool) -> None:
        self._is_simulated_override = bool(value)

    @property
    def is_live(self) -> bool:
        """Whether this provider is backed by a genuine live client with credentials."""
        if self.is_simulated:
            return False
        if not self._api_key or not str(self._api_key).strip():
            return False
        if self._client is None or not isinstance(self._client, httpx.Client):
            return False
        return True

    def __repr__(self) -> str:
        """Safe representation containing zero secrets."""
        return (
            f"GeminiEmbeddingProvider(model={self._model!r}, "
            f"dimension={self._dimension}, "
            f"base_url={self._base_url!r}, "
            f"is_simulated={self.is_simulated})"
        )

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send an HTTP POST request to the Gemini API and return parsed JSON."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key

        try:
            if hasattr(self._client, "post"):
                response = self._client.post(path, json=payload, headers=headers)
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                if callable(getattr(response, "json", None)):
                    return response.json()
                return getattr(response, "data", response)
            raise GeminiEmbeddingProviderError("Client does not provide a callable 'post' method")
        except GeminiEmbeddingProviderError:
            raise
        except Exception as error:
            status_info = ""
            error_body_detail = ""
            resp = getattr(error, "response", None)
            if resp is not None:
                if hasattr(resp, "status_code"):
                    status_info = f" (HTTP {resp.status_code})"
                error_body_detail = _extract_response_error_details(
                    resp, extra_secrets=[self._api_key]
                )
            detail_suffix = f": {error_body_detail}" if error_body_detail else ""
            raise GeminiEmbeddingProviderError(
                f"Gemini embedding request failed ({type(error).__name__}){status_info}{detail_suffix}"
            ) from error

    def embed_text(self, text: str) -> list[float]:
        """Return the Gemini embedding vector for ``text``.

        Args:
            text: Non-empty string to embed.

        Returns:
            The embedding as a list of finite floats.

        Raises:
            TypeError: If ``text`` is not a str.
            ValueError: If ``text`` is empty or whitespace-only.
            GeminiEmbeddingProviderError: If the API request fails or returns an invalid vector.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        path = f"/v1beta/models/{self._clean_model}:embedContent"
        payload: dict[str, Any] = {
            "model": f"models/{self._clean_model}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self._dimension,
        }

        response_data = self._request(path, payload)
        if not isinstance(response_data, dict):
            raise GeminiEmbeddingProviderError(
                f"Gemini response must be a dict, got {type(response_data).__name__}"
            )

        embedding_obj = response_data.get("embedding")
        if not isinstance(embedding_obj, dict):
            raise GeminiEmbeddingProviderError("Gemini response did not contain an 'embedding' object")

        values = embedding_obj.get("values")
        if values is None:
            raise GeminiEmbeddingProviderError("Gemini embedding did not contain 'values'")

        return _validate_embedding(values, self._dimension)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text in the same order as ``texts``.

        Batches requests according to ``batch_size`` (default up to 100).
        All-or-nothing: if any batch fails, an exception is raised.

        Args:
            texts: List of non-empty strings. An empty list returns [].

        Returns:
            List of embedding vectors.

        Raises:
            TypeError: If ``texts`` is not a list or contains non-strings.
            ValueError: If an element is empty or whitespace-only.
            GeminiEmbeddingProviderError: If the request fails or vectors are malformed.
        """
        validate_texts(texts)
        if not texts:
            return []

        path = f"/v1beta/models/{self._clean_model}:batchEmbedContents"
        results: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start : start + self._batch_size]
            req_items: list[dict[str, Any]] = []
            for t in chunk:
                item: dict[str, Any] = {
                    "model": f"models/{self._clean_model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self._dimension,
                }
                req_items.append(item)

            payload = {
                "model": f"models/{self._clean_model}",
                "requests": req_items,
            }
            response_data = self._request(path, payload)

            if not isinstance(response_data, dict):
                raise GeminiEmbeddingProviderError(
                    f"Gemini batch response must be a dict, got {type(response_data).__name__}"
                )

            raw_embeddings = response_data.get("embeddings")
            if not isinstance(raw_embeddings, list):
                raise GeminiEmbeddingProviderError(
                    "Gemini batch response did not contain an 'embeddings' list"
                )

            if len(raw_embeddings) != len(chunk):
                raise GeminiEmbeddingProviderError(
                    f"Gemini returned {len(raw_embeddings)} embeddings for {len(chunk)} batch inputs"
                )

            for index, emb_item in enumerate(raw_embeddings):
                if not isinstance(emb_item, dict):
                    raise GeminiEmbeddingProviderError(
                        f"Gemini batch embedding item {index} is not an object"
                    )
                values = emb_item.get("values")
                if values is None:
                    raise GeminiEmbeddingProviderError(
                        f"Gemini batch embedding item {index} did not contain 'values'"
                    )
                results.append(
                    _validate_embedding(values, self._dimension, f" for input {start + index}")
                )

        return results

    def close(self) -> None:
        """Close the underlying client if owned by this provider."""
        if self._owns_client and self._client is not None:
            if hasattr(self._client, "close"):
                self._client.close()

    def check_readiness(self, probe_text: str = "health check") -> GeminiReadinessReport:
        """Perform a safe, explicit readiness check and smoke test.

        Returns a ``GeminiReadinessReport`` capturing operational status,
        model, dimension, vector validation, latency, and sanitized error info.

        Guarantees:
        - Never raises unhandled exceptions.
        - Never falls back to synthetic or hash embeddings.
        - Never reports a mock, fake, or synthetic provider as live (``is_live=False``).
        - Completely sanitizes any secrets or API keys from error messages.
        """
        if not isinstance(probe_text, str) or not probe_text.strip():
            probe_text = "health check"

        simulated = self.is_simulated

        # Check credentials if not simulated
        if not simulated and (not self._api_key or not str(self._api_key).strip()):
            return GeminiReadinessReport(
                status="unavailable",
                is_ready=False,
                is_live=False,
                is_simulated=False,
                model=self._model,
                dimension=self._dimension,
                sample_vector_dimension=None,
                configuration_valid=True,
                error=(
                    f"External Gemini credentials are unavailable. Set {ENV_GEMINI_API_KEY} "
                    f"or {ENV_NEXORA_GEMINI_API_KEY}."
                ),
                probe_text=probe_text,
                latency_ms=None,
            )

        start_time = time.perf_counter()
        try:
            vector = self.embed_text(probe_text)
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            sanitized_err = _format_actionable_error(exc, self._api_key)
            return GeminiReadinessReport(
                status="failed",
                is_ready=False,
                is_live=False,
                is_simulated=simulated,
                model=self._model,
                dimension=self._dimension,
                sample_vector_dimension=None,
                configuration_valid=True,
                error=sanitized_err,
                probe_text=probe_text,
                latency_ms=latency_ms,
            )

        # Validate returned vector
        if not isinstance(vector, list) or len(vector) != self._dimension:
            return GeminiReadinessReport(
                status="failed",
                is_ready=False,
                is_live=False,
                is_simulated=simulated,
                model=self._model,
                dimension=self._dimension,
                sample_vector_dimension=len(vector) if isinstance(vector, (list, tuple)) else None,
                configuration_valid=True,
                error=(
                    f"Vector dimension mismatch: expected {self._dimension}, "
                    f"got {len(vector) if isinstance(vector, (list, tuple)) else type(vector).__name__}"
                ),
                probe_text=probe_text,
                latency_ms=latency_ms,
            )

        if simulated:
            return GeminiReadinessReport(
                status="simulated",
                is_ready=True,
                is_live=False,
                is_simulated=True,
                model=self._model,
                dimension=self._dimension,
                sample_vector_dimension=len(vector),
                configuration_valid=True,
                error=None,
                probe_text=probe_text,
                latency_ms=latency_ms,
            )

        return GeminiReadinessReport(
            status="ready",
            is_ready=True,
            is_live=True,
            is_simulated=False,
            model=self._model,
            dimension=self._dimension,
            sample_vector_dimension=len(vector),
            configuration_valid=True,
            error=None,
            probe_text=probe_text,
            latency_ms=latency_ms,
        )

    def __enter__(self) -> GeminiEmbeddingProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def check_gemini_readiness(
    target: GeminiEmbeddingProvider | EmbeddingConfig | Mapping[str, Any] | None = None,
    probe_text: str = "health check",
) -> GeminiReadinessReport:
    """Safe, high-level readiness check for Google Gemini embedding provider.

    Accepts an existing ``GeminiEmbeddingProvider``, an ``EmbeddingConfig``,
    a config dict, or ``None`` (which loads environment configuration).

    Returns a ``GeminiReadinessReport`` without throwing unhandled exceptions.
    """
    if isinstance(target, GeminiEmbeddingProvider):
        return target.check_readiness(probe_text=probe_text)

    # If target is None, load from environment
    if target is None:
        api_key = (
            os.getenv(ENV_GEMINI_API_KEY)
            or os.getenv(ENV_NEXORA_GEMINI_API_KEY)
            or os.getenv(ENV_API_KEY)
        )
        model = os.getenv("NEXORA_EMBEDDING_MODEL") or DEFAULT_GEMINI_MODEL
        dim_str = os.getenv("NEXORA_EMBEDDING_DIMENSION")
        try:
            dim = int(dim_str) if dim_str and dim_str.strip().isdigit() else DEFAULT_GEMINI_DIMENSION
        except Exception:
            dim = DEFAULT_GEMINI_DIMENSION

        if not api_key or not str(api_key).strip():
            return GeminiReadinessReport(
                status="unavailable",
                is_ready=False,
                is_live=False,
                is_simulated=False,
                model=model,
                dimension=dim,
                configuration_valid=True,
                error=(
                    f"External Gemini credentials are unavailable. Set {ENV_GEMINI_API_KEY} "
                    f"or {ENV_NEXORA_GEMINI_API_KEY}."
                ),
                probe_text=probe_text,
            )

        try:
            config = EmbeddingConfig(
                provider="gemini",
                model=model,
                api_key=api_key,
                base_url=os.getenv("GEMINI_BASE_URL") or os.getenv("NEXORA_EMBEDDING_BASE_URL"),
                dimension=dim,
            )
        except Exception as exc:
            return GeminiReadinessReport(
                status="failed",
                is_ready=False,
                is_live=False,
                is_simulated=False,
                model=model,
                dimension=dim,
                configuration_valid=False,
                error=f"Gemini configuration error: {sanitize_secret_text(str(exc))}",
                probe_text=probe_text,
            )

        try:
            provider = GeminiEmbeddingProvider(config)
            return provider.check_readiness(probe_text=probe_text)
        except Exception as exc:
            return GeminiReadinessReport(
                status="failed",
                is_ready=False,
                is_live=False,
                is_simulated=False,
                model=model,
                dimension=dim,
                configuration_valid=True,
                error=f"Failed to initialize Gemini provider: {sanitize_secret_text(str(exc))}",
                probe_text=probe_text,
            )

    # If target is EmbeddingConfig
    if isinstance(target, EmbeddingConfig):
        model = target.model if target.model not in ("hash-sha256", "text-embedding-3-small", "default") else DEFAULT_GEMINI_MODEL
        dim = target.dimension if target.dimension is not None else DEFAULT_GEMINI_DIMENSION

        if not target.has_api_key:
            env_key = os.getenv(ENV_GEMINI_API_KEY) or os.getenv(ENV_NEXORA_GEMINI_API_KEY) or os.getenv(ENV_API_KEY)
            if not env_key or not str(env_key).strip():
                return GeminiReadinessReport(
                    status="unavailable",
                    is_ready=False,
                    is_live=False,
                    is_simulated=False,
                    model=model,
                    dimension=dim,
                    configuration_valid=True,
                    error=(
                        f"External Gemini credentials are unavailable. Set api_key or "
                        f"{ENV_GEMINI_API_KEY}."
                    ),
                    probe_text=probe_text,
                )

        try:
            provider = GeminiEmbeddingProvider(target)
            return provider.check_readiness(probe_text=probe_text)
        except Exception as exc:
            return GeminiReadinessReport(
                status="failed",
                is_ready=False,
                is_live=False,
                is_simulated=False,
                model=model,
                dimension=dim,
                configuration_valid=True,
                error=f"Failed to initialize Gemini provider: {sanitize_secret_text(str(exc))}",
                probe_text=probe_text,
            )

    # If target is Mapping (dict)
    if isinstance(target, Mapping):
        try:
            model = str(target.get("model", DEFAULT_GEMINI_MODEL))
            dim = target.get("dimension", DEFAULT_GEMINI_DIMENSION)
            cfg = EmbeddingConfig(
                provider="gemini",
                model=model,
                api_key=target.get("api_key"),
                base_url=target.get("base_url"),
                dimension=dim,
                batch_size=target.get("batch_size"),
            )
            return check_gemini_readiness(cfg, probe_text=probe_text)
        except Exception as exc:
            raw_dim = target.get("dimension")
            effective_dim = raw_dim if isinstance(raw_dim, int) and raw_dim > 0 else DEFAULT_GEMINI_DIMENSION
            return GeminiReadinessReport(
                status="failed",
                is_ready=False,
                is_live=False,
                is_simulated=False,
                model=str(target.get("model", DEFAULT_GEMINI_MODEL)),
                dimension=effective_dim,
                configuration_valid=False,
                error=f"Invalid Gemini configuration dictionary: {sanitize_secret_text(str(exc))}",
                probe_text=probe_text,
            )

    return GeminiReadinessReport(
        status="failed",
        is_ready=False,
        is_live=False,
        is_simulated=False,
        model=DEFAULT_GEMINI_MODEL,
        dimension=DEFAULT_GEMINI_DIMENSION,
        configuration_valid=False,
        error=f"Expected GeminiEmbeddingProvider, EmbeddingConfig, or dict, got {type(target).__name__}",
        probe_text=probe_text,
    )
