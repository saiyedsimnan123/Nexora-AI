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
from collections.abc import Sequence
from typing import Any

import httpx

from nexora.embedding.base import validate_texts
from nexora.embedding.config import ENV_API_KEY, EmbeddingConfig

ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_NEXORA_GEMINI_API_KEY = "NEXORA_GEMINI_API_KEY"

DEFAULT_GEMINI_MODEL = "text-embedding-004"
DEFAULT_GEMINI_DIMENSION = 768
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_BATCH_SIZE = 100


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

        # Determine model: normalize default/fallback names
        model = config.model
        if not model or model in ("hash-sha256", "text-embedding-3-small", "default"):
            self._model = DEFAULT_GEMINI_MODEL
        else:
            self._model = model

        self._dimension = config.dimension if config.dimension is not None else DEFAULT_GEMINI_DIMENSION
        self._batch_size = config.batch_size if config.batch_size is not None else DEFAULT_GEMINI_BATCH_SIZE

        raw_base = config.base_url or os.getenv("GEMINI_BASE_URL") or os.getenv("NEXORA_EMBEDDING_BASE_URL") or DEFAULT_GEMINI_BASE_URL
        self._base_url = raw_base.rstrip("/")

        # Clean model name for URL paths (e.g. models/text-embedding-004 -> text-embedding-004)
        self._clean_model = self._model.removeprefix("models/")

        self._owns_client = False
        if client is not None:
            self._client = client
            self._api_key: str | None = config.api_key
            return

        api_key = (
            config.api_key
            or os.getenv(ENV_GEMINI_API_KEY)
            or os.getenv(ENV_NEXORA_GEMINI_API_KEY)
            or os.getenv(ENV_API_KEY)
        )
        if not api_key or not str(api_key).strip():
            raise GeminiEmbeddingProviderError(
                f"An API key is required to create a Gemini client; "
                f"set config.api_key or the {ENV_GEMINI_API_KEY} environment variable"
            )

        self._api_key = str(api_key).strip()
        self._client = httpx.Client(base_url=self._base_url, timeout=30.0)
        self._owns_client = True

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

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
            resp = getattr(error, "response", None)
            if resp is not None and hasattr(resp, "status_code"):
                status_info = f" (HTTP {resp.status_code})"
            raise GeminiEmbeddingProviderError(
                f"Gemini embedding request failed ({type(error).__name__}){status_info}"
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
        }
        if self._config.dimension is not None:
            payload["outputDimensionality"] = self._config.dimension

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
                }
                if self._config.dimension is not None:
                    item["outputDimensionality"] = self._config.dimension
                req_items.append(item)

            payload = {"requests": req_items}
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

    def __enter__(self) -> GeminiEmbeddingProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
