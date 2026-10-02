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

from nexora.embedding.base import validate_texts
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


def _validate_embedding(
    embedding: Any, expected_dimension: int | None, where: str = ""
) -> list[float]:
    """Validate one embedding and return it as ``list[float]``.

    ``where`` is an optional message suffix such as `` for input 2``.
    """
    if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
        raise EmbeddingProviderError(
            f"OpenAI embedding{where} must be a sequence of numbers, "
            f"got {type(embedding).__name__}"
        )
    if len(embedding) == 0:
        raise EmbeddingProviderError(f"OpenAI returned an empty embedding{where}")

    vector: list[float] = []
    for index, value in enumerate(embedding):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingProviderError(
                f"OpenAI embedding{where} has a non-numeric value "
                f"({type(value).__name__}) at index {index}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingProviderError(
                f"OpenAI embedding{where} has a non-finite value at index {index}"
            )
        vector.append(number)

    if expected_dimension is not None and len(vector) != expected_dimension:
        raise EmbeddingProviderError(
            f"OpenAI embedding{where} has {len(vector)} dimensions, "
            f"expected {expected_dimension}"
        )
    return vector


def _to_vector(response: Any, expected_dimension: int | None) -> list[float]:
    """Convert a single-input embeddings response into a validated vector."""
    try:
        embedding = response.data[0].embedding
    except (AttributeError, IndexError, TypeError, KeyError) as error:
        raise EmbeddingProviderError(
            "OpenAI response did not contain an embedding"
        ) from error
    return _validate_embedding(embedding, expected_dimension)


def _in_input_order(items: list[Any]) -> list[Any]:
    """Return response items in input order, using ``index`` when provided.

    * No item has an index: the response order is used as-is.
    * Every item has an integer index: items are placed by index, and the
      indexes must be exactly ``0..n-1`` (no duplicates, gaps or extras).
    * Anything else (some indexes missing, non-integer indexes) is rejected
      rather than guessed at.
    """
    indexes = [getattr(item, "index", None) for item in items]

    if all(index is None for index in indexes):
        return items
    if any(index is None for index in indexes):
        raise EmbeddingProviderError("OpenAI response indexes are incomplete")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indexes):
        raise EmbeddingProviderError("OpenAI response contains a non-integer index")
    if sorted(indexes) != list(range(len(items))):
        raise EmbeddingProviderError(
            "OpenAI response indexes are duplicated, missing or out of range"
        )

    ordered: list[Any] = [None] * len(items)
    for item, index in zip(items, indexes):
        ordered[index] = item
    return ordered


def _to_vectors(
    response: Any, expected_count: int, expected_dimension: int | None
) -> list[list[float]]:
    """Convert a batch embeddings response into validated vectors in input order."""
    try:
        items = list(response.data)
    except (AttributeError, TypeError) as error:
        raise EmbeddingProviderError(
            "OpenAI response did not contain embeddings"
        ) from error

    if len(items) != expected_count:
        raise EmbeddingProviderError(
            f"OpenAI returned {len(items)} embeddings for {expected_count} inputs"
        )

    vectors: list[list[float]] = []
    for position, item in enumerate(_in_input_order(items)):
        try:
            embedding = item.embedding
        except AttributeError as error:
            raise EmbeddingProviderError(
                f"OpenAI response item for input {position} "
                f"did not contain an embedding"
            ) from error
        vectors.append(
            _validate_embedding(embedding, expected_dimension, f" for input {position}")
        )
    return vectors


class OpenAIEmbeddingProvider:
    """``EmbeddingProvider`` backed by the OpenAI embeddings API.

    The model, API key, base URL, dimension and batch size all come from
    ``EmbeddingConfig``; nothing is hard-coded. Construction never makes a
    network request.

    Args:
        config: Embedding configuration. ``config.api_key`` is required unless
            ``client`` is supplied. ``config.base_url``, when set, is passed to
            the SDK client. ``config.dimension``, when set, is sent to models
            that support it and is always used to validate results.
            ``config.batch_size``, when set, caps the number of texts per
            request in ``embed_texts``.
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

        # Request fields shared by every call, computed once.
        self._base_request: dict[str, Any] = {
            "model": config.model,
            "encoding_format": "float",
        }
        if config.dimension is not None and config.model.startswith(
            _DIMENSIONS_SUPPORTED_PREFIXES
        ):
            self._base_request["dimensions"] = config.dimension

        if client is not None:
            self._client = client
            return

        if not config.has_api_key:
            raise EmbeddingProviderError(
                f"An API key is required to create an OpenAI client; "
                f"set config.api_key or the {ENV_API_KEY} environment variable"
            )
        self._client = _create_openai_client(config.api_key, config.base_url)

    def _request(self, input_value: str | list[str]) -> Any:
        """Send one embeddings request, wrapping any client failure."""
        request = {**self._base_request, "input": input_value}
        try:
            return self._client.embeddings.create(**request)
        except Exception as error:
            raise EmbeddingProviderError(
                f"OpenAI embedding request failed ({type(error).__name__})"
            ) from error

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

        response = self._request(text)
        return _to_vector(response, self._config.dimension)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text, in the same order as ``texts``.

        Texts are sent as lists in a single embeddings request, or in several
        requests of at most ``config.batch_size`` texts each when that is set.
        Results are re-ordered by the ``index`` the API returns (when present),
        so ``result[i]`` always belongs to ``texts[i]``. Texts are sent
        unmodified. The call is all-or-nothing: if any request or response is
        invalid, an error is raised and no partial result is returned.

        Note: OpenAI limits the size of a single request, so for large lists
        set ``config.batch_size``.

        Args:
            texts: A list of non-empty strings. An empty list returns ``[]``
                without making any request.

        Raises:
            TypeError: If ``texts`` is not a list or an element is not a str.
            ValueError: If an element is empty or whitespace-only.
            EmbeddingProviderError: If a request fails or a response is
                malformed (wrong count, bad indexes, bad or wrong-sized vectors).
        """
        validate_texts(texts)
        if not texts:
            return []

        batch_size = self._config.batch_size
        if batch_size is None:
            batch_size = len(texts)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._request(batch)
            vectors.extend(_to_vectors(response, len(batch), self._config.dimension))
        return vectors
