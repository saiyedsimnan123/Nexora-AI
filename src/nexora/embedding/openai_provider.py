"""OpenAI embedding provider for Nexora AI.

``OpenAIEmbeddingProvider`` implements the provider-independent
``EmbeddingProvider`` interface using the official OpenAI Python SDK. The SDK
is imported lazily, only when a real client has to be created, so importing
this module never needs the SDK, never touches the network and never reads
secrets. Tests (or other code) can inject any client with the same
``client.embeddings.create(...)`` shape.
"""

import math
from collections.abc import Sequence
from typing import Any

from nexora.embedding.config import ENV_API_KEY, EmbeddingConfig

# Models that accept the ``dimensions`` request parameter. OpenAI documents it
# as supported by "text-embedding-3 and later" models; older models such as
# text-embedding-ada-002 reject it. Extend this tuple for newer model families.
_DIMENSIONS_SUPPORTED_PREFIXES: tuple[str, ...] = ("text-embedding-3",)


class EmbeddingProviderError(RuntimeError):
    """Raised when the OpenAI embedding provider cannot produce a valid vector.

    Messages describe what went wrong without including API keys, request
    text, or raw SDK response objects. When the failure came from the client,
    the original exception is available as ``__cause__``.
    """


def _create_openai_client(api_key: str, base_url: str | None) -> Any:
    """Create an official OpenAI SDK client (no network call is made)."""
    try:
        import openai
    except ImportError as error:
        raise EmbeddingProviderError(
            "The 'openai' package is required for OpenAIEmbeddingProvider; "
            "install it with: pip install openai"
        ) from error

    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    return openai.OpenAI(api_key=api_key)


def _to_vector(response: Any, expected_dimension: int | None) -> list[float]:
    """Convert an embeddings response into a validated ``list[float]``."""
    try:
        embedding = response.data[0].embedding
    except (AttributeError, IndexError, TypeError, KeyError) as error:
        raise EmbeddingProviderError(
            "OpenAI response did not contain an embedding"
        ) from error

    if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
        raise EmbeddingProviderError(
            f"OpenAI embedding must be a sequence of numbers, "
            f"got {type(embedding).__name__}"
        )
    if len(embedding) == 0:
        raise EmbeddingProviderError("OpenAI returned an empty embedding")

    vector: list[float] = []
    for index, value in enumerate(embedding):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingProviderError(
                f"OpenAI embedding has a non-numeric value "
                f"({type(value).__name__}) at index {index}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingProviderError(
                f"OpenAI embedding has a non-finite value at index {index}"
            )
        vector.append(number)

    if expected_dimension is not None and len(vector) != expected_dimension:
        raise EmbeddingProviderError(
            f"OpenAI embedding has {len(vector)} dimensions, "
            f"expected {expected_dimension}"
        )
    return vector


class OpenAIEmbeddingProvider:
    """``EmbeddingProvider`` backed by the OpenAI embeddings API.

    The model, API key, base URL and dimension all come from ``EmbeddingConfig``;
    nothing is hard-coded. Construction never makes a network request.

    Args:
        config: Embedding configuration. ``config.api_key`` is required unless
            ``client`` is supplied. ``config.base_url``, when set, is passed to
            the SDK client. ``config.dimension``, when set, is sent to models
            that support it and is always used to validate the result.
        client: Optional pre-built client exposing ``embeddings.create(...)``.
            When given it is used as-is and no OpenAI client is created.

    Raises:
        TypeError: If ``config`` is not an ``EmbeddingConfig``.
        EmbeddingProviderError: If no client is supplied and the API key is
            missing or the OpenAI SDK is not installed.
    """

    def __init__(self, config: EmbeddingConfig, client: Any | None = None) -> None:
        if not isinstance(config, EmbeddingConfig):
            raise TypeError(
                f"config must be an EmbeddingConfig, got {type(config).__name__}"
            )
        self._config = config

        if client is not None:
            self._client = client
            return

        if not config.has_api_key:
            raise EmbeddingProviderError(
                f"An API key is required to create an OpenAI client; "
                f"set config.api_key or the {ENV_API_KEY} environment variable"
            )
        self._client = _create_openai_client(config.api_key, config.base_url)

    def embed_text(self, text: str) -> list[float]:
        """Return the OpenAI embedding vector for ``text``.

        The text is sent exactly as given (it is not modified).

        Args:
            text: Non-empty text to embed.

        Returns:
            The embedding as a list of finite floats.

        Raises:
            TypeError: If ``text`` is not a str.
            ValueError: If ``text`` is empty or whitespace-only.
            EmbeddingProviderError: If the request fails or the response is
                not a valid embedding of the configured dimension.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        request: dict[str, Any] = {
            "model": self._config.model,
            "input": text,
            "encoding_format": "float",
        }
        if self._config.dimension is not None and self._config.model.startswith(
            _DIMENSIONS_SUPPORTED_PREFIXES
        ):
            request["dimensions"] = self._config.dimension

        try:
            response = self._client.embeddings.create(**request)
        except Exception as error:
            raise EmbeddingProviderError(
                f"OpenAI embedding request failed ({type(error).__name__})"
            ) from error

        return _to_vector(response, self._config.dimension)
