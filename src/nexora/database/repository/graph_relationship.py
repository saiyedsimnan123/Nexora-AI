"""Repository for GraphRelationship models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.graph_entity import GraphEntity
from nexora.database.models.graph_relationship import (
    GraphEndpointType,
    GraphRelationship,
)
from nexora.database.models.paper import Paper
from nexora.database.repository.base import (
    BaseRepository,
    ConstraintError,
    DuplicateRecordError,
    RecordNotFoundError,
)
from nexora.database.repository.graph_entity import GraphEntityRepository

if TYPE_CHECKING:
    from nexora.graph.analytics import (
        AuthorCollaborationRecord,
        AuthorConnectivityMetrics,
        AuthorPaperConnection,
        CollectionGraphSummary,
        RelationshipTypeCount,
    )

_UNSET = object()


class GraphRelationshipRepository(BaseRepository[GraphRelationship]):
    """Repository for collection-scoped, evidence-grounded knowledge graph relationships."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, GraphRelationship)

    def create(
        self,
        *,
        collection_id: UUID,
        relationship_type: str,
        source_type: str,
        source_id: UUID,
        target_type: str,
        target_id: UUID,
        paper_id: UUID | None = None,
        chunk_id: UUID | None = None,
        page_number: int | None = None,
        evidence_text: str | None = None,
        confidence: float | None = None,
        extraction_method: str | None = None,
        relationship_metadata: dict[str, Any] | None = None,
        validate_endpoints: bool = True,
    ) -> GraphRelationship:
        """Create a new typed relationship in an authorized collection with optional evidence."""
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
        if not isinstance(source_id, UUID):
            raise TypeError(f"source_id must be a UUID, got {type(source_id).__name__}")
        if not isinstance(target_id, UUID):
            raise TypeError(f"target_id must be a UUID, got {type(target_id).__name__}")

        src_type = source_type.strip().lower()
        tgt_type = target_type.strip().lower()
        rel_type = relationship_type.strip().lower()

        if src_type == tgt_type and source_id == target_id:
            raise ConstraintError(
                "Self-referencing relationship prohibited: source and target endpoints must be distinct."
            )

        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise TypeError(f"confidence must be a float, got {type(confidence).__name__}")
            if float(confidence) < 0.0 or float(confidence) > 1.0:
                raise ConstraintError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        if page_number is not None:
            if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
                raise ConstraintError(f"page_number must be an integer >= 1, got {page_number}")

        if evidence_text is not None:
            if not isinstance(evidence_text, str) or not evidence_text.strip():
                raise ConstraintError("evidence_text cannot be blank if provided")

        if validate_endpoints:
            # 1. Validate Source Endpoint exists and belongs to collection_id
            if src_type == GraphEndpointType.PAPER.value or src_type == "paper":
                paper = self._session.get(Paper, source_id)
                if paper is None:
                    raise RecordNotFoundError(f"Source paper with id {source_id} not found")
                if paper.collection_id != collection_id:
                    raise ConstraintError(
                        f"Cross-collection relationship prohibited: Source paper {source_id} "
                        f"belongs to collection {paper.collection_id}, not {collection_id}"
                    )
            elif src_type == GraphEndpointType.ENTITY.value or src_type == "entity":
                entity = self._session.get(GraphEntity, source_id)
                if entity is None:
                    raise RecordNotFoundError(f"Source entity with id {source_id} not found")
                if entity.collection_id != collection_id:
                    raise ConstraintError(
                        f"Cross-collection relationship prohibited: Source entity {source_id} "
                        f"belongs to collection {entity.collection_id}, not {collection_id}"
                    )

            # 2. Validate Target Endpoint exists and belongs to collection_id
            if tgt_type == GraphEndpointType.PAPER.value or tgt_type == "paper":
                paper = self._session.get(Paper, target_id)
                if paper is None:
                    raise RecordNotFoundError(f"Target paper with id {target_id} not found")
                if paper.collection_id != collection_id:
                    raise ConstraintError(
                        f"Cross-collection relationship prohibited: Target paper {target_id} "
                        f"belongs to collection {paper.collection_id}, not {collection_id}"
                    )
            elif tgt_type == GraphEndpointType.ENTITY.value or tgt_type == "entity":
                entity = self._session.get(GraphEntity, target_id)
                if entity is None:
                    raise RecordNotFoundError(f"Target entity with id {target_id} not found")
                if entity.collection_id != collection_id:
                    raise ConstraintError(
                        f"Cross-collection relationship prohibited: Target entity {target_id} "
                        f"belongs to collection {entity.collection_id}, not {collection_id}"
                    )

            # 3. Validate Evidence Provenance if specified
            if paper_id is not None:
                ev_paper = self._session.get(Paper, paper_id)
                if ev_paper is None:
                    raise RecordNotFoundError(f"Evidence paper with id {paper_id} not found")
                if ev_paper.collection_id != collection_id:
                    raise ConstraintError(
                        f"Cross-collection evidence provenance prohibited: Evidence paper {paper_id} "
                        f"belongs to collection {ev_paper.collection_id}, not {collection_id}"
                    )

            if chunk_id is not None:
                ev_chunk = self._session.get(DocumentChunkRecord, chunk_id)
                if ev_chunk is None:
                    raise RecordNotFoundError(f"Evidence chunk with id {chunk_id} not found")
                doc = self._session.get(Document, ev_chunk.document_id)
                if doc is None:
                    raise RecordNotFoundError(f"Document {ev_chunk.document_id} for chunk not found")
                doc_paper = self._session.get(Paper, doc.paper_id)
                if doc_paper is None:
                    raise RecordNotFoundError(f"Paper {doc.paper_id} for document not found")
                if doc_paper.collection_id != collection_id:
                    raise ConstraintError(
                        f"Cross-collection evidence provenance prohibited: Evidence chunk belongs to "
                        f"paper in collection {doc_paper.collection_id}, not {collection_id}"
                    )
                if paper_id is not None and doc.paper_id != paper_id:
                    raise ConstraintError(
                        f"Evidence chunk belongs to paper {doc.paper_id}, not matching specified paper_id {paper_id}"
                    )

        # Set specific cascading foreign keys based on endpoint types
        source_paper_id = source_id if (src_type == "paper") else None
        source_entity_id = source_id if (src_type == "entity") else None
        target_paper_id = target_id if (tgt_type == "paper") else None
        target_entity_id = target_id if (tgt_type == "entity") else None

        return self._create(
            collection_id=collection_id,
            relationship_type=rel_type,
            source_type=src_type,
            source_id=source_id,
            target_type=tgt_type,
            target_id=target_id,
            source_paper_id=source_paper_id,
            source_entity_id=source_entity_id,
            target_paper_id=target_paper_id,
            target_entity_id=target_entity_id,
            paper_id=paper_id,
            chunk_id=chunk_id,
            page_number=page_number,
            evidence_text=evidence_text,
            confidence=confidence,
            extraction_method=extraction_method,
            relationship_metadata=relationship_metadata,
        )

    def _enrich_relationship(
        self,
        rel: GraphRelationship,
        *,
        evidence_text: str | None = None,
        chunk_id: UUID | None = None,
        page_number: int | None = None,
        paper_id: UUID | None = None,
        confidence: float | None = None,
        extraction_method: str | None = None,
        relationship_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Enrich existing relationship if stronger evidence or higher confidence is provided."""
        changed = False
        if evidence_text and not rel.evidence_text:
            rel.evidence_text = evidence_text
            changed = True
        if chunk_id and not rel.chunk_id:
            rel.chunk_id = chunk_id
            changed = True
        if page_number and not rel.page_number:
            rel.page_number = page_number
            changed = True
        if paper_id and not rel.paper_id:
            rel.paper_id = paper_id
            changed = True
        if extraction_method and not rel.extraction_method:
            rel.extraction_method = extraction_method
            changed = True
        if confidence is not None:
            if rel.confidence is None or confidence > rel.confidence:
                rel.confidence = confidence
                changed = True
        if relationship_metadata and not rel.relationship_metadata:
            rel.relationship_metadata = relationship_metadata
            changed = True
        if changed:
            self._session.flush()
        return changed

    def get_or_create(
        self,
        *,
        collection_id: UUID,
        relationship_type: str,
        source_type: str,
        source_id: UUID,
        target_type: str,
        target_id: UUID,
        paper_id: UUID | None = None,
        chunk_id: UUID | None = None,
        page_number: int | None = None,
        evidence_text: str | None = None,
        confidence: float | None = None,
        extraction_method: str | None = None,
        relationship_metadata: dict[str, Any] | None = None,
        validate_endpoints: bool = True,
    ) -> tuple[GraphRelationship, bool]:
        """Idempotently retrieve or create a typed relationship in an authorized collection.

        If the relationship already exists and new grounded evidence or higher confidence is supplied,
        the existing relationship is enriched.

        Returns:
            tuple[GraphRelationship, bool]: (relationship, created_flag)
        """
        src_type = source_type.strip().lower()
        tgt_type = target_type.strip().lower()
        rel_type = relationship_type.strip().lower()

        stmt = select(GraphRelationship).where(
            GraphRelationship.collection_id == collection_id,
            GraphRelationship.relationship_type == rel_type,
            GraphRelationship.source_type == src_type,
            GraphRelationship.source_id == source_id,
            GraphRelationship.target_type == tgt_type,
            GraphRelationship.target_id == target_id,
        )
        existing = self._session.scalars(stmt).first()
        if existing is not None:
            self._enrich_relationship(
                existing,
                evidence_text=evidence_text,
                chunk_id=chunk_id,
                page_number=page_number,
                paper_id=paper_id,
                confidence=confidence,
                extraction_method=extraction_method,
                relationship_metadata=relationship_metadata,
            )
            return existing, False

        try:
            created = self.create(
                collection_id=collection_id,
                relationship_type=rel_type,
                source_type=src_type,
                source_id=source_id,
                target_type=tgt_type,
                target_id=target_id,
                paper_id=paper_id,
                chunk_id=chunk_id,
                page_number=page_number,
                evidence_text=evidence_text,
                confidence=confidence,
                extraction_method=extraction_method,
                relationship_metadata=relationship_metadata,
                validate_endpoints=validate_endpoints,
            )
            return created, True
        except DuplicateRecordError:
            recovered = self._session.scalars(stmt).first()
            if recovered is not None:
                self._enrich_relationship(
                    recovered,
                    evidence_text=evidence_text,
                    chunk_id=chunk_id,
                    page_number=page_number,
                    paper_id=paper_id,
                    confidence=confidence,
                    extraction_method=extraction_method,
                    relationship_metadata=relationship_metadata,
                )
                return recovered, False
            raise

    def get_by_id_scoped(self, id: UUID, *, collection_id: UUID) -> GraphRelationship | None:
        """Get a relationship by ID strictly scoped to an authorized collection."""
        if not isinstance(id, UUID):
            raise TypeError(f"id must be a UUID, got {type(id).__name__}")
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        stmt = select(GraphRelationship).where(
            GraphRelationship.id == id,
            GraphRelationship.collection_id == collection_id,
        )
        return self._session.scalars(stmt).first()

    def get_by_id_scoped_or_raise(self, id: UUID, *, collection_id: UUID) -> GraphRelationship:
        """Get a relationship by ID scoped to collection, or raise RecordNotFoundError."""
        rel = self.get_by_id_scoped(id, collection_id=collection_id)
        if rel is None:
            raise RecordNotFoundError(
                f"GraphRelationship with id {id} not found in collection {collection_id}"
            )
        return rel

    def list_by_collection_id(
        self,
        collection_id: UUID,
        *,
        relationship_type: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[GraphRelationship]:
        """List relationships for an authorized collection with deterministic pagination."""
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = select(GraphRelationship).where(GraphRelationship.collection_id == collection_id)
        if relationship_type is not None:
            stmt = stmt.where(GraphRelationship.relationship_type == relationship_type.strip().lower())

        stmt = (
            stmt.order_by(
                GraphRelationship.relationship_type,
                GraphRelationship.created_at,
                GraphRelationship.id,
            )
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def list_by_paper_id(
        self,
        paper_id: UUID,
        *,
        collection_id: UUID | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[GraphRelationship]:
        """List relationships associated with a paper (as source, target, or asserting paper)."""
        if not isinstance(paper_id, UUID):
            raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        paper_filter = or_(
            GraphRelationship.source_paper_id == paper_id,
            GraphRelationship.target_paper_id == paper_id,
            GraphRelationship.paper_id == paper_id,
        )
        stmt = select(GraphRelationship).where(paper_filter)

        if collection_id is not None:
            if not isinstance(collection_id, UUID):
                raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
            stmt = stmt.where(GraphRelationship.collection_id == collection_id)

        stmt = (
            stmt.order_by(
                GraphRelationship.relationship_type,
                GraphRelationship.created_at,
                GraphRelationship.id,
            )
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def list_by_entity_id(
        self,
        entity_id: UUID,
        *,
        collection_id: UUID | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[GraphRelationship]:
        """List relationships associated with an entity (as source or target)."""
        if not isinstance(entity_id, UUID):
            raise TypeError(f"entity_id must be a UUID, got {type(entity_id).__name__}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        entity_filter = or_(
            GraphRelationship.source_entity_id == entity_id,
            GraphRelationship.target_entity_id == entity_id,
        )
        stmt = select(GraphRelationship).where(entity_filter)

        if collection_id is not None:
            if not isinstance(collection_id, UUID):
                raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
            stmt = stmt.where(GraphRelationship.collection_id == collection_id)

        stmt = (
            stmt.order_by(
                GraphRelationship.relationship_type,
                GraphRelationship.created_at,
                GraphRelationship.id,
            )
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def list_topological_edges(
        self,
        collection_id: UUID,
        *,
        paper_ids: Sequence[UUID] | None = None,
        entity_ids: Sequence[UUID] | None = None,
        direction: str = "both",
        relationship_types: Sequence[str] | None = None,
        min_confidence: float | None = None,
        include_unscored: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[GraphRelationship]:
        """Batch retrieve topological edges connecting paper and/or entity endpoints in an authorized collection.

        Strictly queries genuine topological endpoints (source_paper_id, target_paper_id,
        source_entity_id, target_entity_id) and NEVER treats asserting paper_id as an endpoint.

        Args:
            collection_id: UUID of the authorized collection.
            paper_ids: Optional list of Paper UUIDs to query edges for.
            entity_ids: Optional list of GraphEntity UUIDs to query edges for.
            direction: 'outgoing', 'incoming', 'both', or 'undirected'.
            relationship_types: Optional list of relationship type strings to filter by.
            min_confidence: Optional minimum confidence score (0.0 to 1.0).
            include_unscored: If True (default), includes edges with NULL confidence when min_confidence is set.
            offset: Non-negative pagination offset.
            limit: Positive pagination limit.

        Returns:
            Sequence[GraphRelationship]: Matching relationships in deterministic order.
        """
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        if not isinstance(direction, str):
            raise TypeError(f"direction must be a str, got {type(direction).__name__}")
        clean_dir = direction.strip().lower()
        if clean_dir not in ("outgoing", "incoming", "both", "undirected"):
            raise ValueError(
                f"direction must be 'outgoing', 'incoming', 'both', or 'undirected', got {direction!r}"
            )
        if clean_dir == "undirected":
            clean_dir = "both"

        if min_confidence is not None:
            if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
                raise TypeError(f"min_confidence must be a float, got {type(min_confidence).__name__}")
            if float(min_confidence) < 0.0 or float(min_confidence) > 1.0:
                raise ValueError(f"min_confidence must be between 0.0 and 1.0, got {min_confidence}")

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        p_list: list[UUID] = []
        if paper_ids:
            for p in paper_ids:
                if not isinstance(p, UUID):
                    raise TypeError(f"paper_ids items must be UUIDs, got {type(p).__name__}")
                if p not in p_list:
                    p_list.append(p)

        e_list: list[UUID] = []
        if entity_ids:
            for e in entity_ids:
                if not isinstance(e, UUID):
                    raise TypeError(f"entity_ids items must be UUIDs, got {type(e).__name__}")
                if e not in e_list:
                    e_list.append(e)

        if not p_list and not e_list:
            return ()

        endpoint_conditions: list[Any] = []
        if p_list:
            if clean_dir in ("outgoing", "both"):
                endpoint_conditions.append(GraphRelationship.source_paper_id.in_(p_list))
            if clean_dir in ("incoming", "both"):
                endpoint_conditions.append(GraphRelationship.target_paper_id.in_(p_list))

        if e_list:
            if clean_dir in ("outgoing", "both"):
                endpoint_conditions.append(GraphRelationship.source_entity_id.in_(e_list))
            if clean_dir in ("incoming", "both"):
                endpoint_conditions.append(GraphRelationship.target_entity_id.in_(e_list))

        if not endpoint_conditions:
            return ()

        stmt = select(GraphRelationship).where(
            GraphRelationship.collection_id == collection_id,
            or_(*endpoint_conditions),
        )

        if relationship_types:
            clean_types = [
                r.strip().lower() for r in relationship_types if isinstance(r, str) and r.strip()
            ]
            if clean_types:
                stmt = stmt.where(GraphRelationship.relationship_type.in_(clean_types))

        if min_confidence is not None:
            conf_val = float(min_confidence)
            if include_unscored:
                stmt = stmt.where(
                    or_(
                        GraphRelationship.confidence.is_(None),
                        GraphRelationship.confidence >= conf_val,
                    )
                )
            else:
                stmt = stmt.where(
                    and_(
                        GraphRelationship.confidence.is_not(None),
                        GraphRelationship.confidence >= conf_val,
                    )
                )

        stmt = (
            stmt.order_by(
                GraphRelationship.confidence.desc().nulls_last(),
                GraphRelationship.relationship_type.asc(),
                GraphRelationship.id.asc(),
            )
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def update_scoped(
        self,
        id: UUID,
        *,
        collection_id: UUID,
        confidence: float | Any = _UNSET,
        evidence_text: str | Any = _UNSET,
        page_number: int | Any = _UNSET,
        extraction_method: str | Any = _UNSET,
        relationship_metadata: dict[str, Any] | Any = _UNSET,
        paper_id: UUID | Any = _UNSET,
        chunk_id: UUID | Any = _UNSET,
    ) -> GraphRelationship:
        """Update relationship evidence/confidence strictly within an authorized collection."""
        rel = self.get_by_id_scoped_or_raise(id, collection_id=collection_id)

        kwargs: dict[str, Any] = {}
        if confidence is not _UNSET:
            if confidence is not None:
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                    raise TypeError(f"confidence must be a float, got {type(confidence).__name__}")
                if float(confidence) < 0.0 or float(confidence) > 1.0:
                    raise ConstraintError(f"confidence must be between 0.0 and 1.0, got {confidence}")
            kwargs["confidence"] = confidence

        if page_number is not _UNSET:
            if page_number is not None:
                if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
                    raise ConstraintError(f"page_number must be an integer >= 1, got {page_number}")
            kwargs["page_number"] = page_number

        if evidence_text is not _UNSET:
            if evidence_text is not None:
                if not isinstance(evidence_text, str) or not evidence_text.strip():
                    raise ConstraintError("evidence_text cannot be blank if provided")
            kwargs["evidence_text"] = evidence_text

        if extraction_method is not _UNSET:
            kwargs["extraction_method"] = extraction_method

        if relationship_metadata is not _UNSET:
            kwargs["relationship_metadata"] = relationship_metadata

        if paper_id is not _UNSET:
            if paper_id is not None:
                if not isinstance(paper_id, UUID):
                    raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")
                p = self._session.get(Paper, paper_id)
                if p is None:
                    raise RecordNotFoundError(f"Evidence paper {paper_id} not found")
                if p.collection_id != collection_id:
                    raise ConstraintError(
                        f"Cross-collection evidence provenance prohibited: Evidence paper {paper_id} "
                        f"belongs to collection {p.collection_id}, not {collection_id}"
                    )
            kwargs["paper_id"] = paper_id

        if chunk_id is not _UNSET:
            if chunk_id is not None:
                if not isinstance(chunk_id, UUID):
                    raise TypeError(f"chunk_id must be a UUID, got {type(chunk_id).__name__}")
                c = self._session.get(DocumentChunkRecord, chunk_id)
                if c is None:
                    raise RecordNotFoundError(f"Evidence chunk {chunk_id} not found")
                doc = self._session.get(Document, c.document_id)
                if doc is None:
                    raise RecordNotFoundError(f"Document for chunk {chunk_id} not found")
                doc_paper = self._session.get(Paper, doc.paper_id)
                if doc_paper is None:
                    raise RecordNotFoundError(f"Paper for chunk {chunk_id} not found")
                if doc_paper.collection_id != collection_id:
                    raise ConstraintError(
                        f"Cross-collection evidence provenance prohibited: Evidence chunk belongs to "
                        f"paper in collection {doc_paper.collection_id}, not {collection_id}"
                    )
                effective_paper_id = kwargs.get("paper_id", rel.paper_id)
                if effective_paper_id is not None and doc.paper_id != effective_paper_id:
                    raise ConstraintError(
                        f"Evidence chunk belongs to paper {doc.paper_id}, not matching specified paper_id {effective_paper_id}"
                    )
            kwargs["chunk_id"] = chunk_id

        return self._update(id, **kwargs)

    def delete_scoped(self, id: UUID, *, collection_id: UUID) -> None:
        """Delete a relationship strictly within its authorized collection."""
        self.get_by_id_scoped_or_raise(id, collection_id=collection_id)
        self.delete(id)

    def count_by_collection(self, collection_id: UUID) -> int:
        """Return the total number of graph relationships in an authorized collection."""
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        stmt = select(func.count(GraphRelationship.id)).where(GraphRelationship.collection_id == collection_id)
        return int(self._session.scalar(stmt) or 0)

    def count_by_type(self, collection_id: UUID) -> dict[str, int]:
        """Return relationship counts grouped by relationship_type in an authorized collection.

        Returns:
            dict[str, int]: Mapping of relationship_type to count, ordered by relationship_type ASC.
        """
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        stmt = (
            select(GraphRelationship.relationship_type, func.count(GraphRelationship.id))
            .where(GraphRelationship.collection_id == collection_id)
            .group_by(GraphRelationship.relationship_type)
            .order_by(GraphRelationship.relationship_type.asc())
        )
        return {str(row[0]): int(row[1]) for row in self._session.execute(stmt)}

    def get_relationship_type_counts(self, collection_id: UUID) -> list[RelationshipTypeCount]:
        """Return relationship counts grouped by relationship_type as typed models."""
        from nexora.graph.analytics import RelationshipTypeCount

        counts = self.count_by_type(collection_id)
        return [
            RelationshipTypeCount(relationship_type=rtype, count=cnt)
            for rtype, cnt in counts.items()
        ]

    def get_graph_summary(
        self,
        collection_id: UUID,
        *,
        entity_repo: GraphEntityRepository | None = None,
    ) -> CollectionGraphSummary:
        """Return an overall summary of entities and relationships for an authorized collection."""
        from nexora.graph.analytics import CollectionGraphSummary

        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        e_repo = entity_repo or GraphEntityRepository(self._session)
        total_entities = e_repo.count_by_collection(collection_id)
        entity_counts = e_repo.count_by_type(collection_id)
        total_rels = self.count_by_collection(collection_id)
        rel_counts = self.count_by_type(collection_id)

        return CollectionGraphSummary(
            collection_id=collection_id,
            total_entities=total_entities,
            total_relationships=total_rels,
            entity_counts_by_type=entity_counts,
            relationship_counts_by_type=rel_counts,
        )

    def list_author_paper_connections(
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
    ) -> list[AuthorPaperConnection]:
        """Aggregate connections between author graph entities and papers in an authorized collection.

        Queries genuine endpoint columns (source_paper_id, target_entity_id, etc.) and
        NEVER treats asserting paper_id as an endpoint. Deduplicates distinct (author, paper)
        pairs and returns results with deterministic ordering.
        """
        from nexora.graph.analytics import AuthorPaperConnection

        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        if min_confidence is not None:
            if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
                raise TypeError(f"min_confidence must be a float, got {type(min_confidence).__name__}")
            if float(min_confidence) < 0.0 or float(min_confidence) > 1.0:
                raise ValueError(f"min_confidence must be between 0.0 and 1.0, got {min_confidence}")

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        clean_author_ids: list[UUID] = []
        if author_ids is not None:
            for aid in author_ids:
                if not isinstance(aid, UUID):
                    raise TypeError(f"author_ids items must be UUIDs, got {type(aid).__name__}")
                if aid not in clean_author_ids:
                    clean_author_ids.append(aid)
            if not clean_author_ids:
                return []

        clean_paper_ids: list[UUID] = []
        if paper_ids is not None:
            for pid in paper_ids:
                if not isinstance(pid, UUID):
                    raise TypeError(f"paper_ids items must be UUIDs, got {type(pid).__name__}")
                if pid not in clean_paper_ids:
                    clean_paper_ids.append(pid)
            if not clean_paper_ids:
                return []

        # Standard direction: Paper -> Author
        stmt1 = (
            select(
                GraphEntity.id.label("author_id"),
                GraphEntity.name.label("author_name"),
                Paper.id.label("paper_id"),
                Paper.title.label("paper_title"),
                func.max(GraphRelationship.confidence).label("confidence"),
                func.max(GraphRelationship.evidence_text).label("evidence_text"),
            )
            .select_from(GraphRelationship)
            .join(
                Paper,
                and_(
                    Paper.id == GraphRelationship.source_paper_id,
                    Paper.collection_id == collection_id,
                ),
            )
            .join(
                GraphEntity,
                and_(
                    GraphEntity.id == GraphRelationship.target_entity_id,
                    GraphEntity.collection_id == collection_id,
                    GraphEntity.entity_type == "author",
                ),
            )
            .where(
                GraphRelationship.collection_id == collection_id,
            )
        )

        # Inverse direction: Author -> Paper
        stmt2 = (
            select(
                GraphEntity.id.label("author_id"),
                GraphEntity.name.label("author_name"),
                Paper.id.label("paper_id"),
                Paper.title.label("paper_title"),
                func.max(GraphRelationship.confidence).label("confidence"),
                func.max(GraphRelationship.evidence_text).label("evidence_text"),
            )
            .select_from(GraphRelationship)
            .join(
                Paper,
                and_(
                    Paper.id == GraphRelationship.target_paper_id,
                    Paper.collection_id == collection_id,
                ),
            )
            .join(
                GraphEntity,
                and_(
                    GraphEntity.id == GraphRelationship.source_entity_id,
                    GraphEntity.collection_id == collection_id,
                    GraphEntity.entity_type == "author",
                ),
            )
            .where(
                GraphRelationship.collection_id == collection_id,
            )
        )

        # Apply relationship types filter (defaults to "authored_by")
        target_types = relationship_types if relationship_types is not None else ("authored_by",)
        clean_rel_types = [
            r.strip().lower() for r in target_types if isinstance(r, str) and r.strip()
        ]
        if clean_rel_types:
            stmt1 = stmt1.where(GraphRelationship.relationship_type.in_(clean_rel_types))
            stmt2 = stmt2.where(GraphRelationship.relationship_type.in_(clean_rel_types))

        # Confidence filter
        if min_confidence is not None:
            conf_val = float(min_confidence)
            if include_unscored:
                conf_filter = or_(
                    GraphRelationship.confidence.is_(None),
                    GraphRelationship.confidence >= conf_val,
                )
            else:
                conf_filter = and_(
                    GraphRelationship.confidence.is_not(None),
                    GraphRelationship.confidence >= conf_val,
                )
            stmt1 = stmt1.where(conf_filter)
            stmt2 = stmt2.where(conf_filter)

        # Author IDs filter
        if clean_author_ids:
            stmt1 = stmt1.where(GraphEntity.id.in_(clean_author_ids))
            stmt2 = stmt2.where(GraphEntity.id.in_(clean_author_ids))

        # Paper IDs filter
        if clean_paper_ids:
            stmt1 = stmt1.where(Paper.id.in_(clean_paper_ids))
            stmt2 = stmt2.where(Paper.id.in_(clean_paper_ids))

        # Group by
        stmt1 = stmt1.group_by(GraphEntity.id, GraphEntity.name, Paper.id, Paper.title)
        stmt2 = stmt2.group_by(GraphEntity.id, GraphEntity.name, Paper.id, Paper.title)

        combined = stmt1.union(stmt2).subquery()

        # Final select deduplicating across union and ordering deterministically
        final_stmt = (
            select(
                combined.c.author_id,
                combined.c.author_name,
                combined.c.paper_id,
                combined.c.paper_title,
                func.max(combined.c.confidence).label("confidence"),
                func.max(combined.c.evidence_text).label("evidence_text"),
            )
            .group_by(
                combined.c.author_id,
                combined.c.author_name,
                combined.c.paper_id,
                combined.c.paper_title,
            )
            .order_by(
                combined.c.author_name.asc(),
                combined.c.author_id.asc(),
                combined.c.paper_title.asc(),
                combined.c.paper_id.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        rows = self._session.execute(final_stmt).all()
        return [
            AuthorPaperConnection(
                author_id=row[0],
                author_name=row[1],
                paper_id=row[2],
                paper_title=row[3],
                confidence=float(row[4]) if row[4] is not None else None,
                evidence_text=row[5],
            )
            for row in rows
        ]

    def aggregate_author_collaborations(
        self,
        collection_id: UUID,
        *,
        author_id: UUID | None = None,
        min_shared_papers: int = 1,
        min_confidence: float | None = None,
        include_unscored: bool = True,
        limit: int = 100,
    ) -> list[AuthorCollaborationRecord]:
        """Aggregate author collaboration records based on shared papers in an authorized collection.

        Args:
            collection_id: UUID of authorized collection.
            author_id: Optional author UUID to filter collaborations involving this author.
            min_shared_papers: Minimum number of co-authored papers (default 1).
            min_confidence: Optional minimum relationship confidence.
            include_unscored: Whether to include unscored relationships if min_confidence is set.
            limit: Maximum number of collaboration records to return.

        Returns:
            list[AuthorCollaborationRecord]: Deterministically ordered collaboration records.
        """
        from nexora.graph.analytics import AuthorCollaborationRecord

        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
        if author_id is not None and not isinstance(author_id, UUID):
            raise TypeError(f"author_id must be a UUID, got {type(author_id).__name__}")
        if isinstance(min_shared_papers, bool) or not isinstance(min_shared_papers, int) or min_shared_papers < 1:
            raise ValueError("min_shared_papers must be an integer >= 1")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        # Fetch all distinct author-paper connections in this collection
        connections = self.list_author_paper_connections(
            collection_id=collection_id,
            min_confidence=min_confidence,
            include_unscored=include_unscored,
            offset=0,
            limit=50000,
        )
        if not connections:
            return []

        # Map paper_id -> list of (author_id, author_name)
        paper_authors: dict[UUID, list[tuple[UUID, str]]] = {}
        author_names: dict[UUID, str] = {}
        for conn in connections:
            author_names[conn.author_id] = conn.author_name
            paper_authors.setdefault(conn.paper_id, []).append((conn.author_id, conn.author_name))

        # Collect shared papers per pair
        pair_shared_papers: dict[tuple[UUID, UUID], list[UUID]] = {}

        for pid, authors in paper_authors.items():
            if len(authors) < 2:
                continue
            unique_authors = list({a_id: a_name for a_id, a_name in authors}.items())
            num_authors = len(unique_authors)
            for i in range(num_authors):
                for j in range(i + 1, num_authors):
                    a1_id, a1_name = unique_authors[i]
                    a2_id, a2_name = unique_authors[j]

                    if author_id is not None:
                        if a1_id == author_id:
                            pair_key = (a1_id, a2_id)
                        elif a2_id == author_id:
                            pair_key = (a2_id, a1_id)
                        else:
                            continue
                    else:
                        pair_key = (a1_id, a2_id) if a1_id < a2_id else (a2_id, a1_id)

                    pair_shared_papers.setdefault(pair_key, []).append(pid)

        # Build records
        records: list[AuthorCollaborationRecord] = []
        for (a1_id, a2_id), pids in pair_shared_papers.items():
            unique_pids = sorted(set(pids))
            if len(unique_pids) < min_shared_papers:
                continue
            records.append(
                AuthorCollaborationRecord(
                    author_id=a1_id,
                    author_name=author_names[a1_id],
                    collaborator_id=a2_id,
                    collaborator_name=author_names[a2_id],
                    shared_paper_ids=unique_pids,
                    collaboration_count=len(unique_pids),
                )
            )

        # Deterministic sorting:
        # collaboration_count DESC, author_name ASC, collaborator_name ASC, author_id ASC, collaborator_id ASC
        records.sort(
            key=lambda r: (
                -r.collaboration_count,
                r.author_name.lower(),
                r.collaborator_name.lower(),
                str(r.author_id),
                str(r.collaborator_id),
            )
        )
        return records[:limit]

    def get_author_connectivity_metrics(
        self,
        collection_id: UUID,
        *,
        author_ids: Sequence[UUID] | None = None,
        min_confidence: float | None = None,
        include_unscored: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> list[AuthorConnectivityMetrics]:
        """Produce connectivity and degree metrics for authors within an authorized collection."""
        from nexora.graph.analytics import AuthorConnectivityMetrics

        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        clean_author_ids: list[UUID] = []
        if author_ids is not None:
            for aid in author_ids:
                if not isinstance(aid, UUID):
                    raise TypeError(f"author_ids items must be UUIDs, got {type(aid).__name__}")
                if aid not in clean_author_ids:
                    clean_author_ids.append(aid)
            if not clean_author_ids:
                return []

        # 1. Fetch relevant author entities in the collection
        author_stmt = select(GraphEntity.id, GraphEntity.name).where(
            GraphEntity.collection_id == collection_id,
            GraphEntity.entity_type == "author",
        )
        if clean_author_ids:
            author_stmt = author_stmt.where(GraphEntity.id.in_(clean_author_ids))
        author_stmt = author_stmt.order_by(GraphEntity.name.asc(), GraphEntity.id.asc())

        author_rows = self._session.execute(author_stmt).all()
        if not author_rows:
            return []

        # 2. Fetch all author-paper connections for this collection
        connections = self.list_author_paper_connections(
            collection_id=collection_id,
            min_confidence=min_confidence,
            include_unscored=include_unscored,
            offset=0,
            limit=50000,
        )

        # 3. Build lookup maps: author_id -> set of paper_ids, paper_id -> set of author_ids
        author_papers: dict[UUID, set[UUID]] = {}
        paper_authors: dict[UUID, set[UUID]] = {}

        for conn in connections:
            author_papers.setdefault(conn.author_id, set()).add(conn.paper_id)
            paper_authors.setdefault(conn.paper_id, set()).add(conn.author_id)

        # 4. Compute metrics for each author entity
        metrics: list[AuthorConnectivityMetrics] = []
        for aid, aname in author_rows:
            pids = author_papers.get(aid, set())
            paper_count = len(pids)

            collaborators: set[UUID] = set()
            total_collabs = 0
            for pid in pids:
                co_authors = paper_authors.get(pid, set()) - {aid}
                collaborators.update(co_authors)
                total_collabs += len(co_authors)

            metrics.append(
                AuthorConnectivityMetrics(
                    author_id=aid,
                    author_name=aname,
                    paper_count=paper_count,
                    collaborator_count=len(collaborators),
                    total_collaborations=total_collabs,
                )
            )

        # 5. Deterministic sorting:
        # collaborator_count DESC, total_collaborations DESC, paper_count DESC, author_name ASC, author_id ASC
        metrics.sort(
            key=lambda m: (
                -m.collaborator_count,
                -m.total_collaborations,
                -m.paper_count,
                m.author_name.lower(),
                str(m.author_id),
            )
        )

        return metrics[offset : offset + limit]
