"""Knowledge graph traversal service (Stage 5 Step 3.2).

Provides bounded multi-hop breadth-first search (BFS), 1-hop neighborhood exploration,
and deterministic pathfinding over collection-scoped knowledge graph topologies.
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from nexora.database.models.graph_entity import (
    GraphEntity,
)
from nexora.database.models.graph_relationship import (
    GraphEndpointType,
    GraphRelationship,
    VALID_ENDPOINT_TYPES,
)
from nexora.database.models.paper import Paper
from nexora.database.repository.base import RecordNotFoundError
from nexora.database.repository.graph_relationship import GraphRelationshipRepository
from nexora.graph.traversal_models import (
    GraphEdge,
    GraphNode,
    GraphPath,
    GraphSubgraph,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_DEPTH = 2
HARD_MAX_DEPTH = 3

DEFAULT_MAX_NODES = 50
HARD_MAX_NODES = 100

DEFAULT_MAX_EDGES = 100
HARD_MAX_EDGES = 200

DEFAULT_MAX_PATHS = 5
HARD_MAX_PATHS = 10


class KnowledgeGraphTraversalService:
    """Application service for collection-scoped knowledge graph traversal and pathfinding."""

    def __init__(
        self,
        *,
        session_factory: Any = None,
        default_max_depth: int = DEFAULT_MAX_DEPTH,
        default_max_nodes: int = DEFAULT_MAX_NODES,
        default_max_edges: int = DEFAULT_MAX_EDGES,
        default_max_paths: int = DEFAULT_MAX_PATHS,
    ) -> None:
        self._validate_int_bounds("default_max_depth", default_max_depth, 1, HARD_MAX_DEPTH)
        self._validate_int_bounds("default_max_nodes", default_max_nodes, 1, HARD_MAX_NODES)
        self._validate_int_bounds("default_max_edges", default_max_edges, 1, HARD_MAX_EDGES)
        self._validate_int_bounds("default_max_paths", default_max_paths, 1, HARD_MAX_PATHS)

        self._session_factory = session_factory
        self._default_max_depth = default_max_depth
        self._default_max_nodes = default_max_nodes
        self._default_max_edges = default_max_edges
        self._default_max_paths = default_max_paths

    @staticmethod
    def _validate_int_bounds(name: str, value: Any, min_val: int, max_val: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        if value < min_val or value > max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
        return value

    @staticmethod
    def _validate_direction(direction: str) -> str:
        if not isinstance(direction, str):
            raise TypeError(f"direction must be a str, got {type(direction).__name__}")
        clean = direction.strip().lower()
        if clean not in ("outgoing", "incoming", "both", "undirected"):
            raise ValueError(
                f"direction must be 'outgoing', 'incoming', 'both', or 'undirected', got {direction!r}"
            )
        return "both" if clean == "undirected" else clean

    @staticmethod
    def _validate_endpoint_type(name: str, node_type: str) -> str:
        if not isinstance(node_type, str):
            raise TypeError(f"{name} must be a str, got {type(node_type).__name__}")
        clean = node_type.strip().lower()
        if clean not in VALID_ENDPOINT_TYPES:
            raise ValueError(
                f"Invalid {name} '{node_type}'. Permitted types: {sorted(VALID_ENDPOINT_TYPES)}"
            )
        return clean

    @staticmethod
    def _validate_confidence(min_confidence: float | None) -> float | None:
        if min_confidence is None:
            return None
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise TypeError(f"min_confidence must be a float, got {type(min_confidence).__name__}")
        val = float(min_confidence)
        if val < 0.0 or val > 1.0:
            raise ValueError(f"min_confidence must be between 0.0 and 1.0, got {min_confidence}")
        return val

    def _resolve_single_node(
        self,
        session: Session,
        node_type: str,
        node_id: UUID,
        collection_id: UUID,
    ) -> GraphNode:
        """Resolve a single GraphNode ensuring collection isolation, or raise RecordNotFoundError."""
        if not isinstance(node_id, UUID):
            raise TypeError(f"node_id must be a UUID, got {type(node_id).__name__}")

        if node_type == GraphEndpointType.PAPER.value:
            paper = session.get(Paper, node_id)
            if paper is None or paper.collection_id != collection_id:
                raise RecordNotFoundError(
                    f"Paper {node_id} not found in collection {collection_id}"
                )
            return GraphNode.from_paper(paper)
        elif node_type == GraphEndpointType.ENTITY.value:
            stmt = select(GraphEntity).where(
                GraphEntity.id == node_id,
                GraphEntity.collection_id == collection_id,
            )
            entity = session.scalars(stmt).first()
            if entity is None:
                raise RecordNotFoundError(
                    f"GraphEntity {node_id} not found in collection {collection_id}"
                )
            return GraphNode.from_entity(entity)
        else:
            raise ValueError(f"Unsupported node_type '{node_type}'")

    def _batch_resolve_nodes(
        self,
        session: Session,
        node_keys: set[tuple[str, UUID]],
        collection_id: UUID,
    ) -> dict[tuple[str, UUID], GraphNode]:
        """Batch retrieve and construct GraphNode instances within an authorized collection."""
        resolved: dict[tuple[str, UUID], GraphNode] = {}
        if not node_keys:
            return resolved

        paper_ids = [nid for ntype, nid in node_keys if ntype == GraphEndpointType.PAPER.value]
        entity_ids = [nid for ntype, nid in node_keys if ntype == GraphEndpointType.ENTITY.value]

        if paper_ids:
            stmt = select(Paper).where(
                Paper.collection_id == collection_id,
                Paper.id.in_(paper_ids),
            )
            for paper in session.scalars(stmt).all():
                resolved[(GraphEndpointType.PAPER.value, paper.id)] = GraphNode.from_paper(paper)

        if entity_ids:
            stmt = select(GraphEntity).where(
                GraphEntity.collection_id == collection_id,
                GraphEntity.id.in_(entity_ids),
            )
            for entity in session.scalars(stmt).all():
                resolved[(GraphEndpointType.ENTITY.value, entity.id)] = GraphNode.from_entity(entity)

        return resolved

    def get_neighborhood(
        self,
        node_id: UUID,
        node_type: str,
        collection_id: UUID,
        *,
        direction: str = "both",
        relationship_types: Sequence[str] | None = None,
        min_confidence: float | None = None,
        include_unscored: bool = False,
        limit: int = 50,
        max_nodes: int | None = None,
        max_edges: int | None = None,
        session: Session | None = None,
    ) -> GraphSubgraph:
        """Single-hop neighborhood exploration around a seed node."""
        eff_nodes = max_nodes if max_nodes is not None else limit
        eff_edges = max_edges if max_edges is not None else self._default_max_edges
        return self.explore_multihop(
            start_id=node_id,
            start_type=node_type,
            collection_id=collection_id,
            max_depth=1,
            max_nodes=eff_nodes,
            max_edges=eff_edges,
            direction=direction,
            relationship_types=relationship_types,
            min_confidence=min_confidence,
            include_unscored=include_unscored,
            session=session,
        )

    def explore_multihop(
        self,
        start_id: UUID,
        start_type: str,
        collection_id: UUID,
        *,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        max_edges: int | None = None,
        direction: str = "both",
        relationship_types: Sequence[str] | None = None,
        min_confidence: float | None = None,
        include_unscored: bool = False,
        session: Session | None = None,
    ) -> GraphSubgraph:
        """Bounded multi-hop breadth-first search traversal with cycle breaking and deterministic ordering.

        Depth-boundary semantics:
            Reaching `max_depth` indicates completion of the requested search radius.
            `is_truncated` remains False and `truncation_reason` is None when traversal explores all reachable
            nodes within `max_depth`, distinguishing radius completion from resource-budget truncation
            (`max_nodes_reached` or `max_edges_reached`). No lookahead queries are issued beyond `max_depth`.
        """
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        eff_max_depth = self._validate_int_bounds(
            "max_depth",
            max_depth if max_depth is not None else self._default_max_depth,
            1,
            HARD_MAX_DEPTH,
        )
        eff_max_nodes = self._validate_int_bounds(
            "max_nodes",
            max_nodes if max_nodes is not None else self._default_max_nodes,
            1,
            HARD_MAX_NODES,
        )
        eff_max_edges = self._validate_int_bounds(
            "max_edges",
            max_edges if max_edges is not None else self._default_max_edges,
            1,
            HARD_MAX_EDGES,
        )

        clean_dir = self._validate_direction(direction)
        clean_type = self._validate_endpoint_type("start_type", start_type)
        conf_filter = self._validate_confidence(min_confidence)

        if not isinstance(include_unscored, bool):
            raise TypeError(f"include_unscored must be a bool, got {type(include_unscored).__name__}")

        clean_rel_types: list[str] | None = None
        if relationship_types is not None:
            clean_rel_types = [
                r.strip().lower() for r in relationship_types if isinstance(r, str) and r.strip()
            ]
            if not clean_rel_types:
                # Specified relationship_types filter has no valid elements -> matches 0 edges
                owns_sess = False
                db_sess = session
                if db_sess is None:
                    if self._session_factory is not None:
                        db_sess = self._session_factory()
                        owns_sess = True
                    else:
                        raise ValueError(
                            "KnowledgeGraphTraversalService requires an active database session or session_factory."
                        )
                try:
                    seed_node = self._resolve_single_node(db_sess, clean_type, start_id, collection_id)
                    return GraphSubgraph(
                        seed_id=start_id,
                        seed_type=clean_type,
                        depth=0,
                        nodes=[seed_node],
                        edges=[],
                        is_truncated=False,
                    )
                finally:
                    if owns_sess:
                        db_sess.close()

        owns_session = False
        db_session = session
        if db_session is None:
            if self._session_factory is not None:
                db_session = self._session_factory()
                owns_session = True
            else:
                raise ValueError(
                    "KnowledgeGraphTraversalService requires an active database session or session_factory."
                )

        try:
            rel_repo = GraphRelationshipRepository(db_session)
            seed_node = self._resolve_single_node(db_session, clean_type, start_id, collection_id)

            visited_nodes: dict[tuple[str, UUID], GraphNode] = {seed_node.node_key: seed_node}
            collected_edges: list[GraphEdge] = []
            edge_ids_seen: set[UUID] = set()
            current_frontier: list[tuple[str, UUID]] = [seed_node.node_key]

            is_truncated = False
            truncation_reason: str | None = None
            actual_depth_reached = 0

            for hop in range(1, eff_max_depth + 1):
                if not current_frontier or is_truncated:
                    break

                # Deterministically sort frontier before querying
                current_frontier.sort(key=lambda k: (k[0], str(k[1])))

                frontier_papers = [nid for ntype, nid in current_frontier if ntype == GraphEndpointType.PAPER.value]
                frontier_entities = [nid for ntype, nid in current_frontier if ntype == GraphEndpointType.ENTITY.value]

                remaining_edges = eff_max_edges - len(collected_edges)
                if remaining_edges <= 0:
                    is_truncated = True
                    truncation_reason = "max_edges_reached"
                    break

                rel_records = rel_repo.list_topological_edges(
                    collection_id=collection_id,
                    paper_ids=frontier_papers if frontier_papers else None,
                    entity_ids=frontier_entities if frontier_entities else None,
                    direction=clean_dir,
                    relationship_types=clean_rel_types,
                    min_confidence=conf_filter,
                    include_unscored=include_unscored,
                    limit=remaining_edges + 1,
                )

                if len(rel_records) > remaining_edges:
                    rel_records = rel_records[:remaining_edges]
                    is_truncated = True
                    truncation_reason = "max_edges_reached"

                frontier_set = set(current_frontier)
                hop_new_node_keys: set[tuple[str, UUID]] = set()
                hop_edges: list[GraphEdge] = []

                for rel in rel_records:
                    if rel.id in edge_ids_seen:
                        continue

                    edge = GraphEdge.from_orm(rel)

                    # Determine neighbor endpoint(s) relative to current frontier
                    if clean_dir == "outgoing":
                        if edge.source_key not in frontier_set:
                            continue
                        neighbor_key = edge.target_key
                    elif clean_dir == "incoming":
                        if edge.target_key not in frontier_set:
                            continue
                        neighbor_key = edge.source_key
                    else:  # both / undirected
                        if edge.source_key in frontier_set:
                            neighbor_key = edge.target_key
                        elif edge.target_key in frontier_set:
                            neighbor_key = edge.source_key
                        else:
                            continue

                    # If neighbor is a new node not yet visited
                    if neighbor_key not in visited_nodes and neighbor_key not in hop_new_node_keys:
                        # Check node budget
                        if len(visited_nodes) + len(hop_new_node_keys) >= eff_max_nodes:
                            is_truncated = True
                            truncation_reason = "max_nodes_reached"
                            break  # Stop expanding new nodes or edges pointing to them
                        hop_new_node_keys.add(neighbor_key)

                    edge_ids_seen.add(edge.id)
                    hop_edges.append(edge)

                # Batch resolve discovered neighbor nodes for this hop
                if hop_new_node_keys:
                    resolved_nodes = self._batch_resolve_nodes(db_session, hop_new_node_keys, collection_id)
                    for k, node in resolved_nodes.items():
                        visited_nodes[k] = node

                # Only collect edges whose both endpoints are in visited_nodes
                for edge in hop_edges:
                    if edge.source_key in visited_nodes and edge.target_key in visited_nodes:
                        collected_edges.append(edge)

                if hop_edges:
                    actual_depth_reached = hop

                # Set next frontier to newly discovered valid nodes
                current_frontier = [k for k in hop_new_node_keys if k in visited_nodes]

            # Deterministic ordering for returned subgraph
            seed_key = (clean_type, start_id)
            all_nodes = list(visited_nodes.values())
            all_nodes.sort(
                key=lambda n: (
                    0 if n.node_key == seed_key else 1,
                    n.node_type,
                    n.name,
                    str(n.id),
                )
            )

            collected_edges.sort(
                key=lambda e: (
                    -(e.confidence if e.confidence is not None else -1.0),
                    e.relationship_type,
                    str(e.id),
                )
            )

            return GraphSubgraph(
                seed_id=start_id,
                seed_type=clean_type,
                depth=actual_depth_reached,
                nodes=all_nodes,
                edges=collected_edges,
                is_truncated=is_truncated,
                truncation_reason=truncation_reason,
            )
        finally:
            if owns_session:
                db_session.close()

    def find_paths(
        self,
        source_id: UUID,
        source_type: str,
        target_id: UUID,
        target_type: str,
        collection_id: UUID,
        *,
        max_depth: int | None = None,
        max_paths: int | None = None,
        direction: str = "outgoing",
        relationship_types: Sequence[str] | None = None,
        min_confidence: float | None = None,
        include_unscored: bool = False,
        session: Session | None = None,
    ) -> list[GraphPath]:
        """Find deterministic paths connecting two nodes within collection and budget boundaries.

        Path-budget semantics:
            Returns at most `max_paths` shortest, highest-confidence paths deterministically.
            Per the Stage 5 Step 3.1 specification, `find_paths` returns a `list[GraphPath]` without an outer
            wrapper model, and therefore list-level truncation metadata is not emitted.
        """
        if not isinstance(source_id, UUID):
            raise TypeError(f"source_id must be a UUID, got {type(source_id).__name__}")
        if not isinstance(target_id, UUID):
            raise TypeError(f"target_id must be a UUID, got {type(target_id).__name__}")
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        src_type = self._validate_endpoint_type("source_type", source_type)
        tgt_type = self._validate_endpoint_type("target_type", target_type)
        clean_dir = self._validate_direction(direction)
        conf_filter = self._validate_confidence(min_confidence)

        eff_max_depth = self._validate_int_bounds(
            "max_depth",
            max_depth if max_depth is not None else HARD_MAX_DEPTH,
            1,
            HARD_MAX_DEPTH,
        )
        eff_max_paths = self._validate_int_bounds(
            "max_paths",
            max_paths if max_paths is not None else self._default_max_paths,
            1,
            HARD_MAX_PATHS,
        )

        if not isinstance(include_unscored, bool):
            raise TypeError(f"include_unscored must be a bool, got {type(include_unscored).__name__}")

        clean_rel_types: list[str] | None = None
        if relationship_types is not None:
            clean_rel_types = [
                r.strip().lower() for r in relationship_types if isinstance(r, str) and r.strip()
            ]
            if not clean_rel_types:
                return []

        owns_session = False
        db_session = session
        if db_session is None:
            if self._session_factory is not None:
                db_session = self._session_factory()
                owns_session = True
            else:
                raise ValueError(
                    "KnowledgeGraphTraversalService requires an active database session or session_factory."
                )

        try:
            rel_repo = GraphRelationshipRepository(db_session)
            source_node = self._resolve_single_node(db_session, src_type, source_id, collection_id)
            target_node = self._resolve_single_node(db_session, tgt_type, target_id, collection_id)

            source_key = (src_type, source_id)
            target_key = (tgt_type, target_id)

            if source_key == target_key:
                return [
                    GraphPath(
                        nodes=[source_node],
                        edges=[],
                        length=0,
                        confidence_score=1.0,
                        is_partially_unscored=False,
                    )
                ]

            queue: deque[tuple[tuple[str, UUID], list[GraphEdge], list[tuple[str, UUID]]]] = deque(
                [(source_key, [], [source_key])]
            )
            found_paths_raw: list[tuple[list[GraphEdge], list[tuple[str, UUID]]]] = []
            edges_cache: dict[tuple[str, UUID], list[GraphRelationship]] = {}

            while queue and len(found_paths_raw) < eff_max_paths:
                curr_key, curr_edges, curr_nodes = queue.popleft()

                if len(curr_edges) >= eff_max_depth:
                    continue

                curr_type, curr_id = curr_key
                if curr_key not in edges_cache:
                    p_ids = [curr_id] if curr_type == GraphEndpointType.PAPER.value else None
                    e_ids = [curr_id] if curr_type == GraphEndpointType.ENTITY.value else None
                    rel_records = rel_repo.list_topological_edges(
                        collection_id=collection_id,
                        paper_ids=p_ids,
                        entity_ids=e_ids,
                        direction=clean_dir,
                        relationship_types=clean_rel_types,
                        min_confidence=conf_filter,
                        include_unscored=include_unscored,
                        limit=HARD_MAX_EDGES,
                    )
                    edges_cache[curr_key] = list(rel_records)
                else:
                    rel_records = edges_cache[curr_key]

                for rel in rel_records:
                    edge = GraphEdge.from_orm(rel)

                    if clean_dir == "outgoing":
                        if edge.source_key != curr_key:
                            continue
                        next_key = edge.target_key
                    elif clean_dir == "incoming":
                        if edge.target_key != curr_key:
                            continue
                        next_key = edge.source_key
                    else:  # both / undirected
                        if edge.source_key == curr_key:
                            next_key = edge.target_key
                        elif edge.target_key == curr_key:
                            next_key = edge.source_key
                        else:
                            continue

                    # Path cycle prevention: cannot revisit any node in current path
                    if next_key in curr_nodes:
                        continue

                    new_edges = curr_edges + [edge]
                    new_nodes = curr_nodes + [next_key]

                    if next_key == target_key:
                        found_paths_raw.append((new_edges, new_nodes))
                        if len(found_paths_raw) >= eff_max_paths:
                            break
                    else:
                        queue.append((next_key, new_edges, new_nodes))

            # Batch resolve all nodes across found paths
            all_node_keys = {k for _, n_keys in found_paths_raw for k in n_keys}
            node_map = self._batch_resolve_nodes(db_session, all_node_keys, collection_id)

            paths: list[GraphPath] = []
            for p_edges, p_nodes in found_paths_raw:
                # Ensure all path nodes were resolved
                if not all(k in node_map for k in p_nodes):
                    continue

                has_unscored = any(e.confidence is None for e in p_edges)
                if has_unscored:
                    conf_score = None
                    partially_unscored = True
                else:
                    score = 1.0
                    for e in p_edges:
                        if e.confidence is not None:
                            score *= e.confidence
                    conf_score = round(score, 6)
                    partially_unscored = False

                path_obj = GraphPath(
                    nodes=[node_map[k] for k in p_nodes],
                    edges=p_edges,
                    length=len(p_edges),
                    confidence_score=conf_score,
                    is_partially_unscored=partially_unscored,
                )
                paths.append(path_obj)

            paths.sort(
                key=lambda p: (
                    p.length,
                    -(p.confidence_score if p.confidence_score is not None else -1.0),
                    tuple((n.node_type, str(n.id)) for n in p.nodes),
                )
            )
            return paths[:eff_max_paths]
        finally:
            if owns_session:
                db_session.close()
