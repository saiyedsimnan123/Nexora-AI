"""Knowledge graph analytics application service (Stage 5 Step 3.5).

Provides a cohesive, collection-scoped, read-only analytics service for
summary metrics, entity/relationship distributions, author collaboration
networks, and author graph connectivity.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Generator, Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from nexora.database.repository.graph_entity import GraphEntityRepository
from nexora.database.repository.graph_relationship import GraphRelationshipRepository
from nexora.graph.analytics import (
    AuthorCollaborationRecord,
    AuthorConnectivityMetrics,
    AuthorPaperConnection,
    CollectionAnalyticsResult,
    CollectionGraphSummary,
    EntityTypeCount,
    RelationshipTypeCount,
)

logger = logging.getLogger(__name__)


class KnowledgeGraphAnalyticsService:
    """Cohesive application service for collection-scoped knowledge graph analytics.

    Delegates aggregation execution directly to database repositories without
    duplicating SQL queries or adding an unnecessary second aggregation layer.
    """

    def __init__(self, *, session_factory: Any = None) -> None:
        """Initialize the analytics service with an optional session factory."""
        self._session_factory = session_factory

    @contextmanager
    def _session_scope(self, session: Session | None) -> Generator[Session, None, None]:
        """Provide a database session, managing cleanup if the session is service-owned.

        Caller-owned sessions (passed directly) are never committed, rolled back, or closed.
        Service-owned sessions (created via session_factory) are closed upon context exit.
        """
        if session is not None:
            yield session
        elif self._session_factory is not None:
            owned_session = self._session_factory()
            try:
                yield owned_session
            finally:
                owned_session.close()
        else:
            raise ValueError(
                "KnowledgeGraphAnalyticsService requires an active database session or session_factory."
            )

    @staticmethod
    def _validate_uuid(name: str, value: Any) -> UUID:
        """Validate that a parameter is a UUID instance."""
        if not isinstance(value, UUID):
            raise TypeError(f"{name} must be a UUID, got {type(value).__name__}")
        return value

    @staticmethod
    def _validate_uuid_list(name: str, items: Sequence[UUID] | None) -> list[UUID] | None:
        """Validate an optional sequence of UUID instances."""
        if items is None:
            return None
        validated: list[UUID] = []
        for idx, item in enumerate(items):
            if not isinstance(item, UUID):
                raise TypeError(f"{name}[{idx}] must be a UUID, got {type(item).__name__}")
            if item not in validated:
                validated.append(item)
        return validated

    @staticmethod
    def _validate_confidence(min_confidence: float | None) -> float | None:
        """Validate confidence threshold bounds [0.0, 1.0]."""
        if min_confidence is None:
            return None
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise TypeError(f"min_confidence must be a float, got {type(min_confidence).__name__}")
        val = float(min_confidence)
        if val < 0.0 or val > 1.0:
            raise ValueError(f"min_confidence must be between 0.0 and 1.0, got {min_confidence}")
        return val

    @staticmethod
    def _validate_int(name: str, value: Any, min_val: int) -> int:
        """Validate that an integer parameter satisfies minimum bounds."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        if value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        return value

    def get_summary(
        self,
        collection_id: UUID,
        *,
        session: Session | None = None,
    ) -> CollectionGraphSummary:
        """Retrieve overall summary metrics for an authorized collection.

        Args:
            collection_id: UUID of the collection.
            session: Optional caller-owned database session.

        Returns:
            CollectionGraphSummary containing entity and relationship counts and distributions.
        """
        self._validate_uuid("collection_id", collection_id)
        with self._session_scope(session) as db_sess:
            entity_repo = GraphEntityRepository(db_sess)
            rel_repo = GraphRelationshipRepository(db_sess)
            return rel_repo.get_graph_summary(collection_id, entity_repo=entity_repo)

    def get_entity_distribution(
        self,
        collection_id: UUID,
        *,
        session: Session | None = None,
    ) -> list[EntityTypeCount]:
        """Retrieve entity counts grouped by entity_type for an authorized collection.

        Args:
            collection_id: UUID of the collection.
            session: Optional caller-owned database session.

        Returns:
            list[EntityTypeCount]: Deterministically ordered entity type counts.
        """
        self._validate_uuid("collection_id", collection_id)
        with self._session_scope(session) as db_sess:
            entity_repo = GraphEntityRepository(db_sess)
            return entity_repo.get_entity_type_counts(collection_id)

    def get_relationship_distribution(
        self,
        collection_id: UUID,
        *,
        session: Session | None = None,
    ) -> list[RelationshipTypeCount]:
        """Retrieve relationship counts grouped by relationship_type for an authorized collection.

        Args:
            collection_id: UUID of the collection.
            session: Optional caller-owned database session.

        Returns:
            list[RelationshipTypeCount]: Deterministically ordered relationship type counts.
        """
        self._validate_uuid("collection_id", collection_id)
        with self._session_scope(session) as db_sess:
            rel_repo = GraphRelationshipRepository(db_sess)
            return rel_repo.get_relationship_type_counts(collection_id)

    def get_author_collaborations(
        self,
        collection_id: UUID,
        *,
        author_id: UUID | None = None,
        min_shared_papers: int = 1,
        min_confidence: float | None = None,
        include_unscored: bool = True,
        limit: int = 100,
        session: Session | None = None,
    ) -> list[AuthorCollaborationRecord]:
        """Retrieve pairwise author collaboration records based on shared papers.

        Args:
            collection_id: UUID of the authorized collection.
            author_id: Optional collection-scoped GraphEntity ID to filter collaborations involving this author.
            min_shared_papers: Minimum number of co-authored papers (default 1).
            min_confidence: Optional minimum confidence score (0.0 to 1.0).
            include_unscored: Whether to include unscored relationships if min_confidence is active.
            limit: Maximum records to return.
            session: Optional caller-owned database session.

        Returns:
            list[AuthorCollaborationRecord]: Deterministically ordered author collaboration pairs.
        """
        self._validate_uuid("collection_id", collection_id)
        if author_id is not None:
            self._validate_uuid("author_id", author_id)
        self._validate_int("min_shared_papers", min_shared_papers, 1)
        self._validate_confidence(min_confidence)
        if not isinstance(include_unscored, bool):
            raise TypeError(f"include_unscored must be a bool, got {type(include_unscored).__name__}")
        self._validate_int("limit", limit, 1)

        with self._session_scope(session) as db_sess:
            rel_repo = GraphRelationshipRepository(db_sess)
            return rel_repo.aggregate_author_collaborations(
                collection_id,
                author_id=author_id,
                min_shared_papers=min_shared_papers,
                min_confidence=min_confidence,
                include_unscored=include_unscored,
                limit=limit,
            )

    def get_author_connectivity(
        self,
        collection_id: UUID,
        *,
        author_ids: Sequence[UUID] | None = None,
        min_confidence: float | None = None,
        include_unscored: bool = True,
        offset: int = 0,
        limit: int = 100,
        session: Session | None = None,
    ) -> list[AuthorConnectivityMetrics]:
        """Retrieve degree and connectivity metrics for author graph entities in a collection.

        Args:
            collection_id: UUID of the authorized collection.
            author_ids: Optional list of collection-scoped author GraphEntity IDs to filter.
            min_confidence: Optional minimum confidence score.
            include_unscored: Whether to include unscored relationships if min_confidence is active.
            offset: Non-negative pagination offset.
            limit: Positive pagination limit.
            session: Optional caller-owned database session.

        Returns:
            list[AuthorConnectivityMetrics]: Deterministically ordered author connectivity metrics.
        """
        self._validate_uuid("collection_id", collection_id)
        clean_authors = self._validate_uuid_list("author_ids", author_ids)
        self._validate_confidence(min_confidence)
        if not isinstance(include_unscored, bool):
            raise TypeError(f"include_unscored must be a bool, got {type(include_unscored).__name__}")
        self._validate_int("offset", offset, 0)
        self._validate_int("limit", limit, 1)

        with self._session_scope(session) as db_sess:
            rel_repo = GraphRelationshipRepository(db_sess)
            return rel_repo.get_author_connectivity_metrics(
                collection_id,
                author_ids=clean_authors,
                min_confidence=min_confidence,
                include_unscored=include_unscored,
                offset=offset,
                limit=limit,
            )

    def get_author_paper_connections(
        self,
        collection_id: UUID,
        *,
        author_ids: Sequence[UUID] | None = None,
        paper_ids: Sequence[UUID] | None = None,
        relationship_types: Sequence[str] | None = None,
        min_confidence: float | None = None,
        include_unscored: bool = True,
        offset: int = 0,
        limit: int = 100,
        session: Session | None = None,
    ) -> list[AuthorPaperConnection]:
        """Retrieve verified authorship connections between author graph entities and papers.

        Note on Provenance:
            When multi-row relationships connect the same (author, paper) endpoint pair,
            the underlying repository aggregation selects MAX(confidence) and MAX(evidence_text)
            independently. Confidence and evidence text can originate from different relationship rows.

        Args:
            collection_id: UUID of authorized collection.
            author_ids: Optional author GraphEntity IDs to filter.
            paper_ids: Optional Paper IDs to filter.
            relationship_types: Optional relationship types (default: 'authored_by').
            min_confidence: Optional minimum confidence score.
            include_unscored: Whether to include unscored relationships when min_confidence is set.
            offset: Non-negative pagination offset.
            limit: Positive pagination limit.
            session: Optional caller-owned database session.

        Returns:
            list[AuthorPaperConnection]: Deterministically ordered author-paper connection records.
        """
        self._validate_uuid("collection_id", collection_id)
        clean_authors = self._validate_uuid_list("author_ids", author_ids)
        clean_papers = self._validate_uuid_list("paper_ids", paper_ids)
        self._validate_confidence(min_confidence)
        if not isinstance(include_unscored, bool):
            raise TypeError(f"include_unscored must be a bool, got {type(include_unscored).__name__}")
        self._validate_int("offset", offset, 0)
        self._validate_int("limit", limit, 1)

        with self._session_scope(session) as db_sess:
            rel_repo = GraphRelationshipRepository(db_sess)
            return rel_repo.list_author_paper_connections(
                collection_id,
                author_ids=clean_authors,
                paper_ids=clean_papers,
                relationship_types=relationship_types,
                min_confidence=min_confidence,
                include_unscored=include_unscored,
                offset=offset,
                limit=limit,
            )

    def get_collection_analytics(
        self,
        collection_id: UUID,
        *,
        top_collaborations_limit: int = 20,
        author_connectivity_limit: int = 50,
        min_confidence: float | None = None,
        include_unscored: bool = True,
        session: Session | None = None,
    ) -> CollectionAnalyticsResult:
        """Compose a unified analytics result for an authorized collection.

        Coordinates repository lookups in a single session scope without
        issuing duplicate queries.

        Args:
            collection_id: UUID of the collection.
            top_collaborations_limit: Maximum top collaboration records to include.
            author_connectivity_limit: Maximum author connectivity records to include.
            min_confidence: Optional minimum confidence threshold.
            include_unscored: Whether to include unscored edges when threshold is active.
            session: Optional caller-owned database session.

        Returns:
            CollectionAnalyticsResult: Composed typed analytics result.
        """
        self._validate_uuid("collection_id", collection_id)
        self._validate_int("top_collaborations_limit", top_collaborations_limit, 1)
        self._validate_int("author_connectivity_limit", author_connectivity_limit, 1)
        self._validate_confidence(min_confidence)
        if not isinstance(include_unscored, bool):
            raise TypeError(f"include_unscored must be a bool, got {type(include_unscored).__name__}")

        with self._session_scope(session) as db_sess:
            entity_repo = GraphEntityRepository(db_sess)
            rel_repo = GraphRelationshipRepository(db_sess)

            summary = rel_repo.get_graph_summary(collection_id, entity_repo=entity_repo)
            entity_dist = entity_repo.get_entity_type_counts(collection_id)
            rel_dist = rel_repo.get_relationship_type_counts(collection_id)
            collabs = rel_repo.aggregate_author_collaborations(
                collection_id,
                min_confidence=min_confidence,
                include_unscored=include_unscored,
                limit=top_collaborations_limit,
            )
            connectivity = rel_repo.get_author_connectivity_metrics(
                collection_id,
                min_confidence=min_confidence,
                include_unscored=include_unscored,
                offset=0,
                limit=author_connectivity_limit,
            )

            return CollectionAnalyticsResult(
                collection_id=collection_id,
                summary=summary,
                entity_distribution=entity_dist,
                relationship_distribution=rel_dist,
                top_collaborations=collabs,
                author_connectivity=connectivity,
            )
