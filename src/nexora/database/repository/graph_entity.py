"""Repository for GraphEntity models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from nexora.database.models.graph_entity import (
    GraphEntity,
    normalize_entity_name,
)
from nexora.database.repository.base import (
    BaseRepository,
    DuplicateRecordError,
    RecordNotFoundError,
)

if TYPE_CHECKING:
    from nexora.graph.analytics import EntityTypeCount

_UNSET = object()


class GraphEntityRepository(BaseRepository[GraphEntity]):
    """Repository for collection-scoped canonical knowledge graph entities."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, GraphEntity)

    def create(
        self,
        *,
        collection_id: UUID,
        name: str,
        entity_type: str,
        description: str | None = None,
        aliases: list[str] | None = None,
        entity_metadata: dict[str, Any] | None = None,
    ) -> GraphEntity:
        """Create a new graph entity in an authorized collection."""
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        canonical = normalize_entity_name(name)
        cleaned_type = entity_type.strip().lower()

        return self._create(
            collection_id=collection_id,
            name=name.strip(),
            canonical_name=canonical,
            entity_type=cleaned_type,
            description=description,
            aliases=aliases,
            entity_metadata=entity_metadata,
        )

    def get_or_create(
        self,
        *,
        collection_id: UUID,
        name: str,
        entity_type: str,
        description: str | None = None,
        aliases: list[str] | None = None,
        entity_metadata: dict[str, Any] | None = None,
    ) -> tuple[GraphEntity, bool]:
        """Idempotently retrieve or create a canonical entity within a collection.

        Returns:
            tuple[GraphEntity, bool]: (entity, created_flag)
        """
        canonical = normalize_entity_name(name)
        cleaned_type = entity_type.strip().lower()

        existing = self.get_by_canonical_name(
            collection_id=collection_id,
            entity_type=cleaned_type,
            canonical_name=canonical,
        )
        if existing is not None:
            # Update description / aliases if provided and currently missing
            changed = False
            if description and not existing.description:
                existing.description = description
                changed = True
            if aliases:
                current_aliases = set(existing.aliases or [])
                new_aliases = current_aliases.union(aliases)
                if len(new_aliases) > len(current_aliases):
                    existing.aliases = sorted(new_aliases)
                    changed = True
            if changed:
                self._session.flush()
            return existing, False

        try:
            created = self.create(
                collection_id=collection_id,
                name=name,
                entity_type=cleaned_type,
                description=description,
                aliases=aliases,
                entity_metadata=entity_metadata,
            )
            return created, True
        except DuplicateRecordError:
            # Concurrent insert or race condition: re-fetch
            recovered = self.get_by_canonical_name(
                collection_id=collection_id,
                entity_type=cleaned_type,
                canonical_name=canonical,
            )
            if recovered is not None:
                return recovered, False
            raise

    def get_by_id_scoped(self, id: UUID, *, collection_id: UUID) -> GraphEntity | None:
        """Get an entity by ID strictly scoped to an authorized collection."""
        if not isinstance(id, UUID):
            raise TypeError(f"id must be a UUID, got {type(id).__name__}")
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        stmt = select(GraphEntity).where(
            GraphEntity.id == id,
            GraphEntity.collection_id == collection_id,
        )
        return self._session.scalars(stmt).first()

    def get_by_id_scoped_or_raise(self, id: UUID, *, collection_id: UUID) -> GraphEntity:
        """Get an entity by ID scoped to collection, or raise RecordNotFoundError."""
        entity = self.get_by_id_scoped(id, collection_id=collection_id)
        if entity is None:
            raise RecordNotFoundError(
                f"GraphEntity with id {id} not found in collection {collection_id}"
            )
        return entity

    def get_by_canonical_name(
        self,
        *,
        collection_id: UUID,
        entity_type: str,
        canonical_name: str,
    ) -> GraphEntity | None:
        """Retrieve an entity by canonical name and type within a collection."""
        stmt = select(GraphEntity).where(
            GraphEntity.collection_id == collection_id,
            GraphEntity.entity_type == entity_type.strip().lower(),
            GraphEntity.canonical_name == canonical_name.strip().lower(),
        )
        return self._session.scalars(stmt).first()

    def list_by_collection_id(
        self,
        collection_id: UUID,
        *,
        entity_type: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[GraphEntity]:
        """List entities for an authorized collection with deterministic pagination."""
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = select(GraphEntity).where(GraphEntity.collection_id == collection_id)
        if entity_type is not None:
            stmt = stmt.where(GraphEntity.entity_type == entity_type.strip().lower())

        stmt = (
            stmt.order_by(GraphEntity.canonical_name, GraphEntity.id)
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def search_entities(
        self,
        *,
        collection_id: UUID,
        query: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> Sequence[GraphEntity]:
        """Search entities by name substring or canonical match within an authorized collection."""
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
        if not isinstance(query, str) or not query.strip():
            return ()
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        normalized = normalize_entity_name(query)
        stmt = select(GraphEntity).where(GraphEntity.collection_id == collection_id)
        if entity_type is not None:
            stmt = stmt.where(GraphEntity.entity_type == entity_type.strip().lower())

        search_filter = or_(
            GraphEntity.canonical_name.contains(normalized),
            GraphEntity.name.ilike(f"%{query.strip()}%"),
        )
        stmt = stmt.where(search_filter).order_by(GraphEntity.canonical_name, GraphEntity.id).limit(limit)
        return self._session.scalars(stmt).all()

    def update_scoped(
        self,
        id: UUID,
        *,
        collection_id: UUID,
        name: str | Any = _UNSET,
        description: str | Any = _UNSET,
        aliases: list[str] | Any = _UNSET,
        entity_metadata: dict[str, Any] | Any = _UNSET,
    ) -> GraphEntity:
        """Update an entity strictly within its authorized collection."""
        # Ensure the entity belongs to this collection
        self.get_by_id_scoped_or_raise(id, collection_id=collection_id)

        kwargs: dict[str, Any] = {}
        if name is not _UNSET:
            kwargs["name"] = name.strip()
            kwargs["canonical_name"] = normalize_entity_name(name)
        if description is not _UNSET:
            kwargs["description"] = description
        if aliases is not _UNSET:
            kwargs["aliases"] = aliases
        if entity_metadata is not _UNSET:
            kwargs["entity_metadata"] = entity_metadata

        return self._update(id, **kwargs)

    def delete_scoped(self, id: UUID, *, collection_id: UUID) -> None:
        """Delete an entity strictly within its authorized collection."""
        self.get_by_id_scoped_or_raise(id, collection_id=collection_id)
        self.delete(id)

    def count_by_collection(self, collection_id: UUID) -> int:
        """Return the total number of graph entities in an authorized collection."""
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        stmt = select(func.count(GraphEntity.id)).where(GraphEntity.collection_id == collection_id)
        return int(self._session.scalar(stmt) or 0)

    def count_by_type(self, collection_id: UUID) -> dict[str, int]:
        """Return entity counts grouped by entity_type in an authorized collection.

        Returns:
            dict[str, int]: Mapping of entity_type to count, ordered by entity_type ASC.
        """
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        stmt = (
            select(GraphEntity.entity_type, func.count(GraphEntity.id))
            .where(GraphEntity.collection_id == collection_id)
            .group_by(GraphEntity.entity_type)
            .order_by(GraphEntity.entity_type.asc())
        )
        return {str(row[0]): int(row[1]) for row in self._session.execute(stmt)}

    def get_entity_type_counts(self, collection_id: UUID) -> list[EntityTypeCount]:
        """Return entity counts grouped by entity_type as typed models."""
        from nexora.graph.analytics import EntityTypeCount

        counts = self.count_by_type(collection_id)
        return [
            EntityTypeCount(entity_type=etype, count=cnt)
            for etype, cnt in counts.items()
        ]
