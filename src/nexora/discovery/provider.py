"""Protocol definition for scholarly discovery and metadata providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nexora.discovery.models import (
    ScholarlyPaperMetadata,
    ScholarlySearchQuery,
    ScholarlySearchResult,
)


@runtime_checkable
class ScholarlySourceProvider(Protocol):
    """Protocol for external scholarly discovery and metadata providers."""

    @property
    def provider_name(self) -> str:
        """Normalized unique identifier for the provider (e.g. 'arxiv')."""
        ...

    def search(self, query: ScholarlySearchQuery) -> ScholarlySearchResult:
        """Search for papers matching query criteria.

        Args:
            query: The search criteria, category filters, and pagination parameters.

        Returns:
            A ScholarlySearchResult containing normalized paper metadata items.

        Raises:
            DiscoveryRateLimitError: If provider rate limits are exceeded.
            DiscoveryTimeoutError: If the request times out.
            DiscoveryConnectionError: If the provider is unreachable.
            DiscoveryResponseError: If the provider returns an HTTP 5xx or invalid XML.
        """
        ...

    def get_paper(self, external_id: str) -> ScholarlyPaperMetadata | None:
        """Retrieve detailed metadata for a single paper by its external identifier.

        Args:
            external_id: The provider-specific identifier (e.g. '2106.09685').

        Returns:
            The ScholarlyPaperMetadata if found, or None if the paper does not exist.

        Raises:
            InvalidIdentifierError: If the external identifier is malformed.
            DiscoveryRateLimitError: If provider rate limits are exceeded.
            DiscoveryTimeoutError: If the request times out.
            DiscoveryConnectionError: If the provider is unreachable.
            DiscoveryResponseError: If the provider returns an HTTP 5xx or invalid XML.
        """
        ...
