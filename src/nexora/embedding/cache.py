"""Embedding cache abstraction for Nexora AI.

Computing an embedding can be slow and, for API providers, costs money. The
same text always maps to the same vector for a given provider and model, so
re-computing it is wasteful. An ``EmbeddingCache`` stores vectors under string
keys so later code can skip repeated work.

This module defines only the cache interface and a simple in-memory
implementation. It is independent of any embedding provider: deciding what the
keys are and when to read or write the cache belongs to the layer that uses it
(a later step).
"""

import math
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingCache(Protocol):
    """Interface for a key-to-vector embedding cache.

    Implementations must never let callers mutate cached data through the
    lists they pass in or get back.
    """

    def get(self, key: str) -> list[float] | None:
        """Return the cached vector for ``key``, or ``None`` on a cache miss."""
        ...

    def set(self, key: str, vector: list[float]) -> None:
        """Store ``vector`` under ``key``, replacing any existing entry."""
        ...

    def delete(self, key: str) -> None:
        """Remove ``key`` from the cache. Deleting a missing key is a no-op."""
        ...

    def clear(self) -> None:
        """Remove every entry from the cache."""
        ...


def _validate_key(key: object) -> None:
    if not isinstance(key, str):
        raise TypeError(f"key must be a str, got {type(key).__name__}")
    if not key.strip():
        raise ValueError("key must not be empty or whitespace-only")


def _validated_copy(vector: object) -> list[float]:
    """Validate ``vector`` and return a new ``list[float]`` copy of it."""
    if not isinstance(vector, list):
        raise TypeError(f"vector must be a list, got {type(vector).__name__}")
    if not vector:
        raise ValueError("vector must not be empty")

    copy: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"vector[{index}] must be a number, got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"vector[{index}] must be finite")
        copy.append(number)
    return copy


class InMemoryEmbeddingCache:
    """Thread-safe, in-memory ``EmbeddingCache``.

    Intended for development, tests and single-process use. Entries live only
    as long as the process, are not shared between processes, and are never
    evicted automatically (there is no size limit or TTL yet). Production
    deployments with several processes or servers can later provide another
    ``EmbeddingCache`` implementation (for example a shared network cache)
    without changing the code that depends on the interface.

    Keys are used exactly as given (no normalization). Vectors are validated,
    stored as floats, and copied on the way in and on the way out, so callers
    can never change cached data by mutating their own lists.
    """

    def __init__(self) -> None:
        self._entries: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> list[float] | None:
        """Return a copy of the vector stored under ``key``, or ``None``.

        Raises:
            TypeError: If ``key`` is not a str.
            ValueError: If ``key`` is empty or whitespace-only.
        """
        _validate_key(key)
        with self._lock:
            stored = self._entries.get(key)
        # Stored lists are never mutated (set() replaces them), so copying
        # outside the lock is safe.
        return None if stored is None else list(stored)

    def set(self, key: str, vector: list[float]) -> None:
        """Store a copy of ``vector`` under ``key``, replacing any old entry.

        Raises:
            TypeError: If ``key`` is not a str, ``vector`` is not a list, or a
                vector element is not a number (bools are rejected).
            ValueError: If ``key`` is blank, or ``vector`` is empty or contains
                NaN or infinity.
        """
        _validate_key(key)
        stored = _validated_copy(vector)
        with self._lock:
            self._entries[key] = stored

    def delete(self, key: str) -> None:
        """Remove ``key`` if present; deleting a missing key is safe.

        Raises:
            TypeError: If ``key`` is not a str.
            ValueError: If ``key`` is empty or whitespace-only.
        """
        _validate_key(key)
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        """Remove every entry."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._entries)
