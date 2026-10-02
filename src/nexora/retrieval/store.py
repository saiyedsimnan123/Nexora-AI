"""Vector store contract for Nexora's retrieval layer.

``VectorStore`` exists so that the retrieval layer depends on a small,
infrastructure-independent abstraction instead of on a specific vector
database. Anything that stores ``VectorRecord`` objects and can answer
similarity queries with a ``SearchResult`` satisfies the contract
structurally, without inheriting from it.

Concrete implementations (an in-memory store and a Qdrant-backed store) are
added in later milestones. Qdrant is intentionally not implemented in
Step 11B, and this module must stay free of infrastructure dependencies.
"""

from typing import Protocol, runtime_checkable

from nexora.retrieval.models import SearchResult, VectorRecord

__all__ = ["VectorStore"]


@runtime_checkable
class VectorStore(Protocol):
    """Contract for storing vector records and searching them by similarity."""

    def upsert(self, records: list[VectorRecord]) -> None:
        """Store the given records, replacing any existing records with the same id."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Delete the records with the given ids."""
        ...

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
    ) -> SearchResult:
        """Return up to ``limit`` records most similar to ``vector``."""
        ...

    def count(self) -> int:
        """Return the number of stored vector records."""
        ...
