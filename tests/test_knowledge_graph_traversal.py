"""Tests for Knowledge Graph Traversal Service (Stage 5 Step 3.2).

Covers all 22 required test scenarios:
1. One-hop outgoing traversal.
2. One-hop incoming traversal.
3. Both-direction traversal.
4. Multi-hop BFS traversal.
5. Maximum depth enforcement.
6. Maximum node budget.
7. Maximum edge budget.
8. Maximum path budget.
9. Relationship-type filtering.
10. Minimum-confidence filtering.
11. Excluding unscored edges by default when a confidence threshold is active.
12. Including unscored edges when explicitly requested.
13. Compound paper/entity identity collision safety.
14. Cycle prevention.
15. Deterministic ordering.
16. Collection isolation.
17. Exclusion of asserting-paper provenance from topology.
18. Preservation of edge evidence provenance.
19. Path confidence when every edge is scored.
20. Path confidence when any edge is unscored.
21. Caller-owned session remains open and is not committed or rolled back.
22. Accurate truncation metadata.

Plus supporting tests for:
- Service-owned session lifecycle.
- Parameter validation and hard caps.
- Pathfinding when no path exists or source == target.
- Empty relationship_types filter.
"""

from __future__ import annotations

import datetime
from uuid import UUID, uuid4
from unittest.mock import patch
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from nexora.database.models import (
    Base,
    Collection,
    Document,
    DocumentChunkRecord,
    GraphEntity,
    GraphRelationship,
    IndexingStatus,
    Paper,
    User,
    Workspace,
)
from nexora.database.repository import (
    GraphEntityRepository,
    GraphRelationshipRepository,
)
from nexora.database.repository.base import RecordNotFoundError
from nexora.graph.traversal import (
    HARD_MAX_DEPTH,
    HARD_MAX_EDGES,
    HARD_MAX_NODES,
    HARD_MAX_PATHS,
    KnowledgeGraphTraversalService,
)
from nexora.graph.traversal_models import (
    GraphEdge,
    GraphNode,
    GraphPath,
    GraphSubgraph,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def engine():
    """SQLite in-memory engine with PRAGMA foreign_keys=ON and StaticPool."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def session_factory(engine):
    return lambda: Session(engine)


@pytest.fixture
def graph_data(session):
    """Seed test knowledge graph with papers, entities, chunks, and relationships."""
    user = User(email="traversal_test@nexora.ai")
    session.add(user)
    session.flush()

    ws = Workspace(user_id=user.id, name="Traversal Lab")
    session.add(ws)
    session.flush()

    col_a = Collection(workspace_id=ws.id, name="Collection A")
    col_b = Collection(workspace_id=ws.id, name="Collection B")
    session.add_all([col_a, col_b])
    session.flush()

    p1 = Paper(
        collection_id=col_a.id,
        title="Paper 1: Foundations of Transformers",
        doi="10.1000/p1",
        publication_date=datetime.date(2017, 6, 1),
    )
    p2 = Paper(
        collection_id=col_a.id,
        title="Paper 2: Transformer Extensions",
        doi="10.1000/p2",
        publication_date=datetime.date(2019, 1, 1),
    )
    p3 = Paper(
        collection_id=col_a.id,
        title="Paper 3: Efficient Attention",
        doi="10.1000/p3",
        publication_date=datetime.date(2020, 5, 1),
    )
    p_b = Paper(
        collection_id=col_b.id,
        title="Paper B: Image Recognition in Col B",
        publication_date=datetime.date(2015, 1, 1),
    )
    session.add_all([p1, p2, p3, p_b])
    session.flush()

    doc1 = Document(
        paper_id=p1.id,
        filename="p1.pdf",
        mime_type="application/pdf",
        file_size=2048,
        checksum="b" * 64,
        storage_key="docs/p1.pdf",
        indexing_status=IndexingStatus.COMPLETED.value,
    )
    session.add(doc1)
    session.flush()

    chunk1 = DocumentChunkRecord(
        document_id=doc1.id,
        chunk_index=0,
        text="We propose the Transformer model based on multi-head self-attention.",
    )
    session.add(chunk1)
    session.flush()

    ent_repo = GraphEntityRepository(session)
    m1 = ent_repo.create(
        collection_id=col_a.id,
        name="Transformer Architecture",
        entity_type="method",
        description="Attention based model.",
    )
    m2 = ent_repo.create(
        collection_id=col_a.id,
        name="Self-Attention",
        entity_type="concept",
    )
    m3 = ent_repo.create(
        collection_id=col_a.id,
        name="FlashAttention",
        entity_type="method",
    )
    c1 = ent_repo.create(
        collection_id=col_a.id,
        name="Linear Complexity",
        entity_type="concept",
    )
    c2 = ent_repo.create(
        collection_id=col_a.id,
        name="Sequence Modeling",
        entity_type="concept",
    )
    a1 = ent_repo.create(
        collection_id=col_a.id,
        name="Ashish Vaswani",
        entity_type="author",
    )
    d1 = ent_repo.create(
        collection_id=col_a.id,
        name="WMT 2014 English-German",
        entity_type="dataset",
    )

    # Collection B entity
    mb = ent_repo.create(
        collection_id=col_b.id,
        name="Residual Connections",
        entity_type="method",
    )

    # Collision test: Paper and Entity with the EXACT same UUID
    shared_uuid = uuid4()
    p_shared = Paper(
        id=shared_uuid,
        collection_id=col_a.id,
        title="Shared UUID Paper",
    )
    e_shared = GraphEntity(
        id=shared_uuid,
        collection_id=col_a.id,
        name="Shared UUID Entity",
        canonical_name="shared uuid entity",
        entity_type="concept",
    )
    session.add_all([p_shared, e_shared])
    session.flush()

    rel_repo = GraphRelationshipRepository(session)

    # Edges in Col A
    # r1: P1 -> proposes -> M1 (confidence=0.95, grounded chunk evidence, page=2)
    r1 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="proposes",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=m1.id,
        paper_id=p1.id,
        chunk_id=chunk1.id,
        page_number=2,
        evidence_text="We propose the Transformer model based on multi-head self-attention.",
        confidence=0.95,
        extraction_method="llm_extraction",
    )

    # r2: M1 -> relates_to -> M2 (confidence=0.90, paper_id=p1.id - asserting paper!)
    r2 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="relates_to",
        source_type="entity",
        source_id=m1.id,
        target_type="entity",
        target_id=m2.id,
        paper_id=p1.id,  # Asserting paper provenance, NOT an endpoint
        confidence=0.90,
    )

    # r3: M2 -> relates_to -> C1 (confidence=0.85)
    r3 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="relates_to",
        source_type="entity",
        source_id=m2.id,
        target_type="entity",
        target_id=c1.id,
        confidence=0.85,
    )

    # r4: C1 -> relates_to -> C2 (confidence=0.80)
    r4 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="relates_to",
        source_type="entity",
        source_id=c1.id,
        target_type="entity",
        target_id=c2.id,
        confidence=0.80,
    )

    # r5: P2 -> cites -> P1 (confidence=1.0)
    r5 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="cites",
        source_type="paper",
        source_id=p2.id,
        target_type="paper",
        target_id=p1.id,
        confidence=1.0,
    )

    # r6: P1 -> authored_by -> A1 (confidence=1.0)
    r6 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a1.id,
        confidence=1.0,
    )

    # r7: P1 -> uses_dataset -> D1 (confidence=None - unscored!)
    r7 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="uses_dataset",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=d1.id,
        confidence=None,
    )

    # r8: Cycle edge: M2 -> relates_to -> M3 (confidence=0.75)
    r8 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="relates_to",
        source_type="entity",
        source_id=m2.id,
        target_type="entity",
        target_id=m3.id,
        confidence=0.75,
    )

    # r9: Cycle closing edge: M3 -> relates_to -> M1 (confidence=0.70)
    r9 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="relates_to",
        source_type="entity",
        source_id=m3.id,
        target_type="entity",
        target_id=m1.id,
        confidence=0.70,
    )

    # r10: Direct alternate edge: P1 -> relates_to -> C1 (confidence=0.50)
    r10 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="relates_to",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=c1.id,
        confidence=0.50,
    )

    # Collision test edges:
    # r_p_shared: P_shared -> cites -> P1
    r_p_shared = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="cites",
        source_type="paper",
        source_id=p_shared.id,
        target_type="paper",
        target_id=p1.id,
        confidence=0.99,
    )

    # r_e_shared: E_shared -> relates_to -> M1
    r_e_shared = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="relates_to",
        source_type="entity",
        source_id=e_shared.id,
        target_type="entity",
        target_id=m1.id,
        confidence=0.91,
    )

    # Edge in Col B: PB -> proposes -> MB
    rb = rel_repo.create(
        collection_id=col_b.id,
        relationship_type="proposes",
        source_type="paper",
        source_id=p_b.id,
        target_type="entity",
        target_id=mb.id,
        confidence=0.99,
    )

    session.commit()

    return {
        "col_a": col_a,
        "col_b": col_b,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p_b": p_b,
        "p_shared": p_shared,
        "e_shared": e_shared,
        "shared_uuid": shared_uuid,
        "chunk1": chunk1,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "c1": c1,
        "c2": c2,
        "a1": a1,
        "d1": d1,
        "mb": mb,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r4": r4,
        "r5": r5,
        "r6": r6,
        "r7": r7,
        "r8": r8,
        "r9": r9,
        "r10": r10,
        "r_p_shared": r_p_shared,
        "r_e_shared": r_e_shared,
        "rb": rb,
    }


# ============================================================================
# Required Test Scenarios (1 through 22)
# ============================================================================

def test_1_one_hop_outgoing_traversal(session, graph_data):
    """1. One-hop outgoing traversal strictly returns outgoing edges and neighbor targets."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    subgraph = service.get_neighborhood(
        p1.id,
        "paper",
        col.id,
        direction="outgoing",
        session=session,
    )

    assert subgraph.seed_key == ("paper", p1.id)
    assert subgraph.depth == 1
    # P1 outgoing edges: r1 (proposes M1), r6 (authored_by A1), r7 (uses_dataset D1), r10 (relates_to C1)
    edge_types = {e.relationship_type for e in subgraph.edges}
    assert edge_types == {"proposes", "authored_by", "uses_dataset", "relates_to"}
    assert all(e.source_key == ("paper", p1.id) for e in subgraph.edges)

    # Target nodes are included in subgraph.nodes
    node_keys = subgraph.node_keys
    assert ("paper", p1.id) in node_keys
    assert ("entity", graph_data["m1"].id) in node_keys
    assert ("entity", graph_data["a1"].id) in node_keys
    assert ("entity", graph_data["d1"].id) in node_keys
    assert ("entity", graph_data["c1"].id) in node_keys
    # P2 (cites P1) is incoming, must NOT be present
    assert ("paper", graph_data["p2"].id) not in node_keys


def test_2_one_hop_incoming_traversal(session, graph_data):
    """2. One-hop incoming traversal strictly returns incoming edges and neighbor sources."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    subgraph = service.get_neighborhood(
        p1.id,
        "paper",
        col.id,
        direction="incoming",
        session=session,
    )

    assert subgraph.seed_key == ("paper", p1.id)
    # Incoming to P1: r5 (P2 cites P1), r_p_shared (P_shared cites P1)
    assert len(subgraph.edges) == 2
    assert all(e.target_key == ("paper", p1.id) for e in subgraph.edges)
    assert all(e.relationship_type == "cites" for e in subgraph.edges)

    node_keys = subgraph.node_keys
    assert ("paper", p1.id) in node_keys
    assert ("paper", graph_data["p2"].id) in node_keys
    assert ("paper", graph_data["p_shared"].id) in node_keys
    # Outgoing targets (M1, A1, D1) must NOT be present
    assert ("entity", graph_data["m1"].id) not in node_keys


def test_3_both_direction_traversal(session, graph_data):
    """3. Both-direction traversal discovers both incoming and outgoing edges."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    subgraph = service.get_neighborhood(
        p1.id,
        "paper",
        col.id,
        direction="both",
        session=session,
    )

    assert subgraph.seed_key == ("paper", p1.id)
    edge_types = {e.relationship_type for e in subgraph.edges}
    assert "cites" in edge_types  # incoming
    assert "proposes" in edge_types  # outgoing
    assert ("paper", graph_data["p2"].id) in subgraph.node_keys
    assert ("entity", graph_data["m1"].id) in subgraph.node_keys


def test_4_multihop_bfs_traversal(session, graph_data):
    """4. Multi-hop BFS traversal expands frontier level by level."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    # 2 hops outgoing from P1:
    # Hop 1: P1 -> M1, P1 -> C1, P1 -> A1, P1 -> D1
    # Hop 2 from M1: M1 -> M2
    # Hop 2 from C1: C1 -> C2
    subgraph = service.explore_multihop(
        p1.id,
        "paper",
        col.id,
        max_depth=2,
        direction="outgoing",
        session=session,
    )

    assert subgraph.depth == 2
    node_keys = subgraph.node_keys
    assert ("paper", p1.id) in node_keys
    assert ("entity", graph_data["m1"].id) in node_keys
    assert ("entity", graph_data["m2"].id) in node_keys  # 2nd hop from M1
    assert ("entity", graph_data["c2"].id) in node_keys  # 2nd hop from C1


def test_5_max_depth_enforcement(session, graph_data):
    """5. Maximum depth enforcement strictly bounds exploration and rejects invalid depth."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    # Chain: P1 -> M1 -> M2 -> C1 -> C2 (4 hops)
    # With max_depth=1: only M1, C1, A1, D1 reached
    sg1 = service.explore_multihop(p1.id, "paper", col.id, max_depth=1, direction="outgoing", session=session)
    assert sg1.depth == 1
    assert ("entity", graph_data["m2"].id) not in sg1.node_keys
    assert ("entity", graph_data["c2"].id) not in sg1.node_keys

    # With max_depth=2: M2 and C2 reached, but not further
    sg2 = service.explore_multihop(p1.id, "paper", col.id, max_depth=2, direction="outgoing", session=session)
    assert sg2.depth == 2
    assert ("entity", graph_data["m2"].id) in sg2.node_keys

    # Reject depth > 3 (HARD_MAX_DEPTH) or < 1
    with pytest.raises(ValueError, match="max_depth must be between 1 and 3"):
        service.explore_multihop(p1.id, "paper", col.id, max_depth=4, session=session)

    with pytest.raises(ValueError, match="max_depth must be between 1 and 3"):
        service.explore_multihop(p1.id, "paper", col.id, max_depth=0, session=session)


def test_6_max_node_budget(session, graph_data):
    """6. Maximum node budget immediately halts expansion when reached."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    # P1 has 4 outgoing neighbors. Setting max_nodes=3 permits seed + 2 neighbors.
    subgraph = service.explore_multihop(
        p1.id,
        "paper",
        col.id,
        max_depth=2,
        max_nodes=3,
        direction="outgoing",
        session=session,
    )

    assert len(subgraph.nodes) <= 3
    assert subgraph.is_truncated is True
    assert subgraph.truncation_reason == "max_nodes_reached"
    # Seed is always preserved
    assert subgraph.nodes[0].node_key == ("paper", p1.id)


def test_7_max_edge_budget(session, graph_data):
    """7. Maximum edge budget halts edge collection when limit is hit."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    # P1 has at least 4 outgoing edges. Setting max_edges=2 collects only 2 edges.
    subgraph = service.explore_multihop(
        p1.id,
        "paper",
        col.id,
        max_depth=2,
        max_edges=2,
        direction="outgoing",
        session=session,
    )

    assert len(subgraph.edges) == 2
    assert subgraph.is_truncated is True
    assert subgraph.truncation_reason == "max_edges_reached"


def test_8_max_path_budget(session, graph_data):
    """8. Maximum path budget limits returned paths and validates path limits."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]
    c1 = graph_data["c1"]

    # There are at least 2 paths from P1 to C1:
    # 1. P1 -> C1 (direct via r10, length 1)
    # 2. P1 -> M1 -> M2 -> C1 (length 3)
    paths = service.find_paths(
        p1.id, "paper",
        c1.id, "entity",
        col.id,
        max_depth=3,
        max_paths=1,
        direction="outgoing",
        session=session,
    )

    assert len(paths) == 1
    assert paths[0].length == 1

    # Requesting 2 paths returns both
    paths_2 = service.find_paths(
        p1.id, "paper",
        c1.id, "entity",
        col.id,
        max_depth=3,
        max_paths=2,
        direction="outgoing",
        session=session,
    )
    assert len(paths_2) == 2

    # Reject max_paths > 10 (HARD_MAX_PATHS) or < 1
    with pytest.raises(ValueError, match="max_paths must be between 1 and 10"):
        service.find_paths(p1.id, "paper", c1.id, "entity", col.id, max_paths=11, session=session)

    with pytest.raises(ValueError, match="max_paths must be between 1 and 10"):
        service.find_paths(p1.id, "paper", c1.id, "entity", col.id, max_paths=0, session=session)


def test_9_relationship_type_filtering(session, graph_data):
    """9. Configurable relationship-type filtering strictly restricts traversed edge types."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    subgraph = service.get_neighborhood(
        p1.id,
        "paper",
        col.id,
        relationship_types=["proposes"],
        direction="outgoing",
        session=session,
    )

    assert len(subgraph.edges) == 1
    assert subgraph.edges[0].relationship_type == "proposes"
    assert subgraph.edges[0].target_key == ("entity", graph_data["m1"].id)


def test_10_min_confidence_filtering(session, graph_data):
    """10. Minimum-confidence filtering excludes edges below threshold."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    # P1 outgoing edges: r1=0.95, r6=1.0, r10=0.50, r7=None
    # With min_confidence=0.90, r10 (0.50) is excluded
    subgraph = service.get_neighborhood(
        p1.id,
        "paper",
        col.id,
        min_confidence=0.90,
        direction="outgoing",
        session=session,
    )

    assert all(e.confidence is not None and e.confidence >= 0.90 for e in subgraph.edges)
    edge_ids = {e.id for e in subgraph.edges}
    assert graph_data["r10"].id not in edge_ids


def test_11_excluding_unscored_edges_by_default_when_threshold_active(session, graph_data):
    """11. Excluding unscored edges (confidence=None) by default when a confidence threshold is active."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    # r7 has confidence=None (uses_dataset D1)
    subgraph = service.get_neighborhood(
        p1.id,
        "paper",
        col.id,
        min_confidence=0.50,
        direction="outgoing",
        # include_unscored defaults to False
        session=session,
    )

    edge_ids = {e.id for e in subgraph.edges}
    assert graph_data["r7"].id not in edge_ids
    assert all(e.confidence is not None for e in subgraph.edges)


def test_12_including_unscored_edges_when_explicitly_requested(session, graph_data):
    """12. Including unscored edges when explicitly requested (include_unscored=True)."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    subgraph = service.get_neighborhood(
        p1.id,
        "paper",
        col.id,
        min_confidence=0.90,
        include_unscored=True,
        direction="outgoing",
        session=session,
    )

    edge_ids = {e.id for e in subgraph.edges}
    assert graph_data["r7"].id in edge_ids
    unscored = [e for e in subgraph.edges if e.id == graph_data["r7"].id][0]
    assert unscored.confidence is None


def test_13_compound_paper_entity_identity_collision_safety(session, graph_data):
    """13. Compound identity (node_type, UUID) prevents collision between Paper and Entity sharing identical UUID."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    shared_uuid = graph_data["shared_uuid"]

    # 1. Traversal from Paper endpoint
    sg_paper = service.get_neighborhood(
        shared_uuid,
        "paper",
        col.id,
        direction="outgoing",
        session=session,
    )
    assert sg_paper.seed_key == ("paper", shared_uuid)
    assert len(sg_paper.edges) == 1
    assert sg_paper.edges[0].id == graph_data["r_p_shared"].id
    assert sg_paper.edges[0].target_key == ("paper", graph_data["p1"].id)

    # 2. Traversal from Entity endpoint
    sg_entity = service.get_neighborhood(
        shared_uuid,
        "entity",
        col.id,
        direction="outgoing",
        session=session,
    )
    assert sg_entity.seed_key == ("entity", shared_uuid)
    assert len(sg_entity.edges) == 1
    assert sg_entity.edges[0].id == graph_data["r_e_shared"].id
    assert sg_entity.edges[0].target_key == ("entity", graph_data["m1"].id)


def test_14_cycle_prevention(session, graph_data):
    """14. Cycle prevention prevents infinite traversal in cyclic graphs."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    m1 = graph_data["m1"]

    # Graph contains cycle: M1 -> M2 -> M3 -> M1
    # Multi-hop BFS with max_depth=3 must complete without infinite loop
    subgraph = service.explore_multihop(
        m1.id,
        "entity",
        col.id,
        max_depth=3,
        direction="outgoing",
        session=session,
    )

    assert subgraph.is_truncated is False
    # All 3 nodes in cycle should be visited once
    node_ids = {n.id for n in subgraph.nodes if n.node_type == "entity"}
    assert graph_data["m1"].id in node_ids
    assert graph_data["m2"].id in node_ids
    assert graph_data["m3"].id in node_ids

    # Path search must not repeat nodes
    paths = service.find_paths(
        m1.id, "entity",
        graph_data["m3"].id, "entity",
        col.id,
        max_depth=3,
        direction="outgoing",
        session=session,
    )
    assert len(paths) >= 1
    for p in paths:
        # Check no duplicated node in any single path
        assert len(p.nodes) == len(set(p.nodes))


def test_15_deterministic_ordering(session, graph_data):
    """15. Deterministic ordering: repeated identical queries against unchanged data produce identical outputs."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    sg1 = service.explore_multihop(p1.id, "paper", col.id, max_depth=2, direction="both", session=session)
    sg2 = service.explore_multihop(p1.id, "paper", col.id, max_depth=2, direction="both", session=session)

    assert [n.node_key for n in sg1.nodes] == [n.node_key for n in sg2.nodes]
    assert [e.id for e in sg1.edges] == [e.id for e in sg2.edges]

    p1_paths = service.find_paths(p1.id, "paper", graph_data["c1"].id, "entity", col.id, session=session)
    p2_paths = service.find_paths(p1.id, "paper", graph_data["c1"].id, "entity", col.id, session=session)
    assert [p.path_keys for p in p1_paths] == [p.path_keys for p in p2_paths]


def test_16_collection_isolation(session, graph_data):
    """16. Strict collection isolation prevents cross-tenant access and returns RecordNotFoundError."""
    service = KnowledgeGraphTraversalService()
    col_a = graph_data["col_a"]
    col_b = graph_data["col_b"]
    p1 = graph_data["p1"]
    p_b = graph_data["p_b"]

    # Attempting to query Paper from Col B under Col A must raise RecordNotFoundError
    with pytest.raises(RecordNotFoundError, match="not found in collection"):
        service.get_neighborhood(p_b.id, "paper", col_a.id, session=session)

    # Pathfinding across collection boundaries must raise RecordNotFoundError
    with pytest.raises(RecordNotFoundError, match="not found in collection"):
        service.find_paths(p1.id, "paper", p_b.id, "paper", col_a.id, session=session)

    with pytest.raises(RecordNotFoundError, match="not found in collection"):
        service.find_paths(p1.id, "paper", p_b.id, "paper", col_b.id, session=session)


def test_17_exclusion_of_asserting_paper_from_topology(session, graph_data):
    """17. CRITICAL: Asserting paper provenance (paper_id) is NOT treated as a topological edge."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]
    r2 = graph_data["r2"]  # M1 -> relates_to -> M2 (paper_id = p1.id)

    # Querying all topological edges for Paper P1 (both directions)
    subgraph = service.get_neighborhood(p1.id, "paper", col.id, direction="both", session=session)
    edge_ids = {e.id for e in subgraph.edges}

    # r2 was asserted by P1, but P1 is NOT an endpoint. It MUST NOT be present!
    assert r2.id not in edge_ids

    # Querying M1 DOES include r2
    m1_sg = service.get_neighborhood(graph_data["m1"].id, "entity", col.id, direction="both", session=session)
    m1_edge_ids = {e.id for e in m1_sg.edges}
    assert r2.id in m1_edge_ids


def test_18_preservation_of_edge_evidence_provenance(session, graph_data):
    """18. Evidence provenance (chunk_id, page_number, evidence_text, paper_id) is fully preserved."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    subgraph = service.get_neighborhood(p1.id, "paper", col.id, direction="outgoing", session=session)
    r1_edge = [e for e in subgraph.edges if e.id == graph_data["r1"].id][0]

    assert r1_edge.chunk_id == graph_data["chunk1"].id
    assert r1_edge.page_number == 2
    assert "We propose the Transformer model" in (r1_edge.evidence_text or "")
    assert r1_edge.paper_id == p1.id
    assert r1_edge.extraction_method == "llm_extraction"
    assert r1_edge.has_evidence is True


def test_19_path_confidence_when_every_edge_scored(session, graph_data):
    """19. Path confidence is the exact product of edge confidences when every edge is scored."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]
    c1 = graph_data["c1"]

    # Path: P1 -> M1 (0.95) -> M2 (0.90) -> C1 (0.85)
    # Expected confidence = 0.95 * 0.90 * 0.85 = 0.72675
    paths = service.find_paths(
        p1.id, "paper",
        c1.id, "entity",
        col.id,
        relationship_types=["proposes", "relates_to"],
        min_confidence=0.80,
        direction="outgoing",
        session=session,
    )

    three_hop = [p for p in paths if p.length == 3]
    assert len(three_hop) >= 1
    target_path = three_hop[0]
    expected_score = round(0.95 * 0.90 * 0.85, 6)
    assert target_path.confidence_score == expected_score
    assert target_path.is_partially_unscored is False


def test_20_path_confidence_when_any_edge_unscored(session, graph_data):
    """20. Path confidence remains None and is_partially_unscored=True when any edge is unscored."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]
    d1 = graph_data["d1"]

    # P1 -> uses_dataset -> D1 has confidence=None
    paths = service.find_paths(
        p1.id, "paper",
        d1.id, "entity",
        col.id,
        direction="outgoing",
        include_unscored=True,
        session=session,
    )

    assert len(paths) >= 1
    unscored_path = paths[0]
    assert unscored_path.confidence_score is None
    assert unscored_path.is_partially_unscored is True


def test_21_caller_owned_session_invariance(session, graph_data):
    """21. Caller-owned session remains open, active, and is never committed or rolled back."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    with patch.object(session, "commit", wraps=session.commit) as mock_commit, \
         patch.object(session, "rollback", wraps=session.rollback) as mock_rollback, \
         patch.object(session, "close", wraps=session.close) as mock_close:

        # Create uncommitted entity in the session
        uncommitted = GraphEntity(
            id=uuid4(),
            collection_id=col.id,
            name="Uncommitted Probe",
            canonical_name="uncommitted probe",
            entity_type="concept",
        )
        session.add(uncommitted)

        # Execute traversal operations
        sg = service.get_neighborhood(p1.id, "paper", col.id, session=session)
        assert len(sg.nodes) > 0

        paths = service.find_paths(p1.id, "paper", graph_data["c1"].id, "entity", col.id, session=session)
        assert isinstance(paths, list)

        # Invariance check: session is still active and methods were never called by service
        assert session.is_active is True
        mock_commit.assert_not_called()
        mock_rollback.assert_not_called()
        mock_close.assert_not_called()

        # The entity was not committed, caller rollback successfully discards it
        session.rollback()
        assert session.get(GraphEntity, uncommitted.id) is None


def test_22_accurate_truncation_metadata(session, graph_data):
    """22. Accurate truncation metadata reporting is_truncated and truncation_reason."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    # 1. Truncated by node budget
    sg_node_trunc = service.explore_multihop(
        p1.id, "paper", col.id,
        max_depth=2,
        max_nodes=2,
        direction="outgoing",
        session=session,
    )
    assert sg_node_trunc.is_truncated is True
    assert sg_node_trunc.truncation_reason == "max_nodes_reached"
    assert len(sg_node_trunc.nodes) <= 2

    # 2. Truncated by edge budget
    sg_edge_trunc = service.explore_multihop(
        p1.id, "paper", col.id,
        max_depth=2,
        max_edges=1,
        direction="outgoing",
        session=session,
    )
    assert sg_edge_trunc.is_truncated is True
    assert sg_edge_trunc.truncation_reason == "max_edges_reached"
    assert len(sg_edge_trunc.edges) <= 1

    # 3. Not truncated when budgets are sufficient
    sg_clean = service.explore_multihop(
        p1.id, "paper", col.id,
        max_depth=1,
        max_nodes=50,
        max_edges=100,
        direction="outgoing",
        session=session,
    )
    assert sg_clean.is_truncated is False
    assert sg_clean.truncation_reason is None


# ============================================================================
# Additional Lifecycle, Validation, and Boundary Tests
# ============================================================================

def test_service_owned_session_lifecycle(session_factory, graph_data):
    """Verify service-owned session is created and cleanly closed in finally block."""
    service = KnowledgeGraphTraversalService(session_factory=session_factory)
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    # Calling with session=None exercises the service-owned session lifecycle
    sg = service.get_neighborhood(p1.id, "paper", col.id, session=None)
    assert len(sg.nodes) > 0

    paths = service.find_paths(p1.id, "paper", graph_data["c1"].id, "entity", col.id, session=None)
    assert len(paths) > 0


def test_missing_session_and_factory_raises_value_error(graph_data):
    """Calling without active session or session_factory raises ValueError."""
    service = KnowledgeGraphTraversalService(session_factory=None)
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    with pytest.raises(ValueError, match="requires an active database session or session_factory"):
        service.get_neighborhood(p1.id, "paper", col.id, session=None)


def test_parameter_validation_and_hard_caps(session, graph_data):
    """Constructor and methods validate bounds and reject invalid values."""
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    # Constructor hard caps
    with pytest.raises(ValueError, match="default_max_depth must be between 1 and 3"):
        KnowledgeGraphTraversalService(default_max_depth=4)

    with pytest.raises(ValueError, match="default_max_nodes must be between 1 and 100"):
        KnowledgeGraphTraversalService(default_max_nodes=101)

    with pytest.raises(ValueError, match="default_max_edges must be between 1 and 200"):
        KnowledgeGraphTraversalService(default_max_edges=201)

    with pytest.raises(ValueError, match="default_max_paths must be between 1 and 10"):
        KnowledgeGraphTraversalService(default_max_paths=11)

    service = KnowledgeGraphTraversalService()

    # Invalid direction
    with pytest.raises(ValueError, match="direction must be"):
        service.get_neighborhood(p1.id, "paper", col.id, direction="diagonal", session=session)

    # Invalid endpoint type
    with pytest.raises(ValueError, match="Invalid start_type"):
        service.explore_multihop(p1.id, "invalid_type", col.id, session=session)

    # Invalid confidence
    with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
        service.explore_multihop(p1.id, "paper", col.id, min_confidence=1.5, session=session)


def test_find_paths_source_equals_target(session, graph_data):
    """find_paths when source == target returns path of length 0."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    paths = service.find_paths(p1.id, "paper", p1.id, "paper", col.id, session=session)
    assert len(paths) == 1
    assert paths[0].length == 0
    assert paths[0].start_node.id == p1.id
    assert paths[0].confidence_score == 1.0


def test_find_paths_no_path_exists(session, graph_data):
    """find_paths returns empty list when no path connects the nodes."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p3 = graph_data["p3"]  # Isolated paper in col_a with no edges
    c1 = graph_data["c1"]

    paths = service.find_paths(p3.id, "paper", c1.id, "entity", col.id, session=session)
    assert paths == []


def test_empty_relationship_types_filter(session, graph_data):
    """Passing empty relationship_types filter returns empty subgraph/paths."""
    service = KnowledgeGraphTraversalService()
    col = graph_data["col_a"]
    p1 = graph_data["p1"]

    sg = service.get_neighborhood(p1.id, "paper", col.id, relationship_types=[], session=session)
    assert len(sg.edges) == 0
    assert len(sg.nodes) == 1

    paths = service.find_paths(p1.id, "paper", graph_data["c1"].id, "entity", col.id, relationship_types=[], session=session)
    assert paths == []


# ============================================================================
# Step 3.3 Test Hardening Suite
# ============================================================================

def test_step_3_3_depth_boundary_completion_vs_resource_truncation(session):
    """Scenario A: Verify reaching max_depth indicates search radius completion, distinguishing from resource budget truncation."""
    user = User(email="depth_test@nexora.ai")
    session.add(user)
    session.flush()

    ws = Workspace(user_id=user.id, name="Depth Semantics Lab")
    session.add(ws)
    session.flush()

    col = Collection(workspace_id=ws.id, name="Depth Boundary Collection")
    session.add(col)
    session.flush()

    # Create a 4-hop chain: P0 -> E1 -> E2 -> E3 -> E4
    p0 = Paper(collection_id=col.id, title="Root Paper")
    session.add(p0)
    session.flush()

    ent_repo = GraphEntityRepository(session)
    e1 = ent_repo.create(collection_id=col.id, name="Node 1", entity_type="concept")
    e2 = ent_repo.create(collection_id=col.id, name="Node 2", entity_type="concept")
    e3 = ent_repo.create(collection_id=col.id, name="Node 3", entity_type="concept")
    e4 = ent_repo.create(collection_id=col.id, name="Node 4", entity_type="concept")

    rel_repo = GraphRelationshipRepository(session)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="paper", source_id=p0.id, target_type="entity", target_id=e1.id)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=e1.id, target_type="entity", target_id=e2.id)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=e2.id, target_type="entity", target_id=e3.id)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=e3.id, target_type="entity", target_id=e4.id)
    session.commit()

    service = KnowledgeGraphTraversalService()

    # 1. Traversal bounded by max_depth=2:
    # Must return all reachable nodes within radius 2 (P0, E1, E2).
    # Reaching max_depth completes the requested radius: is_truncated MUST remain False.
    sg_d2 = service.explore_multihop(p0.id, "paper", col.id, max_depth=2, direction="outgoing", session=session)
    assert sg_d2.depth == 2
    assert len(sg_d2.nodes) == 3
    assert {n.id for n in sg_d2.nodes} == {p0.id, e1.id, e2.id}
    assert sg_d2.is_truncated is False
    assert sg_d2.truncation_reason is None

    # 2. Traversal bounded by max_depth=3:
    # Must return all reachable nodes within radius 3 (P0, E1, E2, E3).
    sg_d3 = service.explore_multihop(p0.id, "paper", col.id, max_depth=3, direction="outgoing", session=session)
    assert sg_d3.depth == 3
    assert len(sg_d3.nodes) == 4
    assert {n.id for n in sg_d3.nodes} == {p0.id, e1.id, e2.id, e3.id}
    assert sg_d3.is_truncated is False
    assert sg_d3.truncation_reason is None

    # 3. Contrast with resource-budget truncation at the same depth:
    # max_nodes=2 cuts off expansion before all nodes within depth 2 can be collected.
    sg_node_trunc = service.explore_multihop(p0.id, "paper", col.id, max_depth=2, max_nodes=2, direction="outgoing", session=session)
    assert len(sg_node_trunc.nodes) == 2
    assert sg_node_trunc.is_truncated is True
    assert sg_node_trunc.truncation_reason == "max_nodes_reached"

    # max_edges=1 cuts off expansion by edge budget
    sg_edge_trunc = service.explore_multihop(p0.id, "paper", col.id, max_depth=2, max_edges=1, direction="outgoing", session=session)
    assert len(sg_edge_trunc.edges) == 1
    assert sg_edge_trunc.is_truncated is True
    assert sg_edge_trunc.truncation_reason == "max_edges_reached"


def test_step_3_3_path_budget_saturation_and_list_api_limitation(session):
    """Scenario B: Verify path budget limits, deterministic selection, and document current list return API limitation."""
    user = User(email="path_budget@nexora.ai")
    session.add(user)
    session.flush()

    ws = Workspace(user_id=user.id, name="Path Budget Lab")
    session.add(ws)
    session.flush()

    col = Collection(workspace_id=ws.id, name="Path Budget Collection")
    session.add(col)
    session.flush()

    ent_repo = GraphEntityRepository(session)
    source = ent_repo.create(collection_id=col.id, name="Source Node", entity_type="concept")
    target = ent_repo.create(collection_id=col.id, name="Target Node", entity_type="concept")

    # 4 intermediate nodes to form 4 distinct paths of length 2
    # Confidences: Route 1 = 0.95*0.95 (0.9025), Route 2 = 0.90*0.90 (0.81), Route 3 = 0.85*0.85 (0.7225), Route 4 = 0.80*0.80 (0.64)
    m1 = ent_repo.create(collection_id=col.id, name="Mid 1", entity_type="concept")
    m2 = ent_repo.create(collection_id=col.id, name="Mid 2", entity_type="concept")
    m3 = ent_repo.create(collection_id=col.id, name="Mid 3", entity_type="concept")
    m4 = ent_repo.create(collection_id=col.id, name="Mid 4", entity_type="concept")

    rel_repo = GraphRelationshipRepository(session)
    # Route 1
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=source.id, target_type="entity", target_id=m1.id, confidence=0.95)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=m1.id, target_type="entity", target_id=target.id, confidence=0.95)
    # Route 2
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=source.id, target_type="entity", target_id=m2.id, confidence=0.90)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=m2.id, target_type="entity", target_id=target.id, confidence=0.90)
    # Route 3
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=source.id, target_type="entity", target_id=m3.id, confidence=0.85)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=m3.id, target_type="entity", target_id=target.id, confidence=0.85)
    # Route 4
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=source.id, target_type="entity", target_id=m4.id, confidence=0.80)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=m4.id, target_type="entity", target_id=target.id, confidence=0.80)
    session.commit()

    service = KnowledgeGraphTraversalService()

    # 1. Request max_paths=2 when 4 valid paths exist
    paths_bounded = service.find_paths(
        source.id, "entity",
        target.id, "entity",
        col.id,
        max_paths=2,
        direction="outgoing",
        session=session,
    )

    # Strictly bounds to max_paths
    assert len(paths_bounded) == 2

    # Deterministic selection: top 2 highest confidence paths selected
    assert paths_bounded[0].nodes[1].id == m1.id
    assert paths_bounded[0].confidence_score == round(0.95 * 0.95, 6)
    assert paths_bounded[1].nodes[1].id == m2.id
    assert paths_bounded[1].confidence_score == round(0.90 * 0.90, 6)

    # 2. Document the current Stage 5 API limitation:
    # find_paths returns a Python list; lists do not expose is_truncated metadata.
    assert isinstance(paths_bounded, list)
    assert not hasattr(paths_bounded, "is_truncated")
    assert not hasattr(paths_bounded, "truncation_reason")

    # 3. Requesting max_paths=4 returns all 4 paths in deterministic confidence order
    paths_all = service.find_paths(
        source.id, "entity",
        target.id, "entity",
        col.id,
        max_paths=4,
        direction="outgoing",
        session=session,
    )
    assert len(paths_all) == 4
    assert [p.nodes[1].id for p in paths_all] == [m1.id, m2.id, m3.id, m4.id]


def test_step_3_3_converging_alternative_paths_and_diamond_topology(session):
    """Scenario C: Verify converging alternative routes in a diamond topology are preserved without suppression."""
    user = User(email="diamond_test@nexora.ai")
    session.add(user)
    session.flush()

    ws = Workspace(user_id=user.id, name="Diamond Lab")
    session.add(ws)
    session.flush()

    col = Collection(workspace_id=ws.id, name="Diamond Collection")
    session.add(col)
    session.flush()

    ent_repo = GraphEntityRepository(session)
    src = ent_repo.create(collection_id=col.id, name="Source", entity_type="concept")
    tgt = ent_repo.create(collection_id=col.id, name="Target", entity_type="concept")
    node_a = ent_repo.create(collection_id=col.id, name="Branch A", entity_type="method")
    node_b = ent_repo.create(collection_id=col.id, name="Branch B", entity_type="method")

    rel_repo = GraphRelationshipRepository(session)
    # Diamond: Src -> A -> Tgt, Src -> B -> Tgt
    rel_repo.create(collection_id=col.id, relationship_type="proposes", source_type="entity", source_id=src.id, target_type="entity", target_id=node_a.id, confidence=0.9)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=node_a.id, target_type="entity", target_id=tgt.id, confidence=0.9)
    rel_repo.create(collection_id=col.id, relationship_type="proposes", source_type="entity", source_id=src.id, target_type="entity", target_id=node_b.id, confidence=0.85)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=node_b.id, target_type="entity", target_id=tgt.id, confidence=0.85)

    # Cross-link creating an additional 3-hop path: Src -> A -> B -> Tgt
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=node_a.id, target_type="entity", target_id=node_b.id, confidence=0.8)

    # Cycle between A and B: B -> A
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=node_b.id, target_type="entity", target_id=node_a.id, confidence=0.7)
    session.commit()

    service = KnowledgeGraphTraversalService()

    paths = service.find_paths(
        src.id, "entity",
        tgt.id, "entity",
        col.id,
        max_depth=3,
        max_paths=10,
        direction="outgoing",
        session=session,
    )

    # Must discover 4 valid paths:
    # 1. Src -> A -> Tgt (len 2)
    # 2. Src -> B -> Tgt (len 2)
    # 3. Src -> A -> B -> Tgt (len 3, via A -> B)
    # 4. Src -> B -> A -> Tgt (len 3, via B -> A)
    # Mutual link between A and B must not cause infinite loop or corrupt path continuity
    assert len(paths) == 4

    two_hop_paths = [p for p in paths if p.length == 2]
    assert len(two_hop_paths) == 2
    branch_intermediates = {p.nodes[1].id for p in two_hop_paths}
    assert branch_intermediates == {node_a.id, node_b.id}

    three_hop_paths = [p for p in paths if p.length == 3]
    assert len(three_hop_paths) == 2
    three_hop_routes = [[n.id for n in p.nodes] for p in three_hop_paths]
    assert [src.id, node_a.id, node_b.id, tgt.id] in three_hop_routes
    assert [src.id, node_b.id, node_a.id, tgt.id] in three_hop_routes

    # Verify path continuity across all returned paths
    for p in paths:
        assert p.start_node.id == src.id
        assert p.end_node.id == tgt.id
        assert len(p.nodes) == len(p.edges) + 1
        for i, edge in enumerate(p.edges):
            assert edge.source_key == p.nodes[i].node_key
            assert edge.target_key == p.nodes[i + 1].node_key


def test_step_3_3_exact_node_budget_boundaries_and_zero_overflow(session):
    """Scenario D: Verify exact node budget boundary semantics and strict prevention of budget overflow."""
    user = User(email="node_bound@nexora.ai")
    session.add(user)
    session.flush()

    ws = Workspace(user_id=user.id, name="Node Boundary Lab")
    session.add(ws)
    session.flush()

    col = Collection(workspace_id=ws.id, name="Node Boundary Collection")
    session.add(col)
    session.flush()

    ent_repo = GraphEntityRepository(session)
    seed = ent_repo.create(collection_id=col.id, name="Seed Node", entity_type="concept")
    n1 = ent_repo.create(collection_id=col.id, name="Neighbor 1", entity_type="concept")
    n2 = ent_repo.create(collection_id=col.id, name="Neighbor 2", entity_type="concept")
    n3 = ent_repo.create(collection_id=col.id, name="Neighbor 3", entity_type="concept")

    rel_repo = GraphRelationshipRepository(session)
    # Seed connects to exactly 3 distinct neighbors (total reachable nodes in 1-hop = 4: seed + 3 neighbors)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=seed.id, target_type="entity", target_id=n1.id, confidence=0.9)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=seed.id, target_type="entity", target_id=n2.id, confidence=0.8)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=seed.id, target_type="entity", target_id=n3.id, confidence=0.7)
    session.commit()

    service = KnowledgeGraphTraversalService()

    # 1. Exact capacity: max_nodes = 4 (seed + 3 neighbors)
    sg_exact = service.explore_multihop(seed.id, "entity", col.id, max_depth=1, max_nodes=4, direction="outgoing", session=session)
    assert len(sg_exact.nodes) == 4
    assert sg_exact.is_truncated is False
    assert sg_exact.truncation_reason is None

    # 2. One additional reachable node: max_nodes = 3
    # Traversal must strictly halt at exactly 3 nodes with ZERO overflow
    sg_overflow_guard = service.explore_multihop(seed.id, "entity", col.id, max_depth=1, max_nodes=3, direction="outgoing", session=session)
    assert len(sg_overflow_guard.nodes) == 3  # Zero overflow: strictly <= 3
    assert sg_overflow_guard.is_truncated is True
    assert sg_overflow_guard.truncation_reason == "max_nodes_reached"
    # Seed node is always present
    assert sg_overflow_guard.nodes[0].node_key == ("entity", seed.id)
    # All edges connect nodes strictly within the truncated subgraph
    for e in sg_overflow_guard.edges:
        assert e.source_key in sg_overflow_guard.node_keys
        assert e.target_key in sg_overflow_guard.node_keys

    # 3. max_nodes = 2
    sg_two = service.explore_multihop(seed.id, "entity", col.id, max_depth=1, max_nodes=2, direction="outgoing", session=session)
    assert len(sg_two.nodes) == 2
    assert sg_two.is_truncated is True
    assert sg_two.truncation_reason == "max_nodes_reached"

    # 4. max_nodes = 1 (seed only permitted)
    sg_one = service.explore_multihop(seed.id, "entity", col.id, max_depth=1, max_nodes=1, direction="outgoing", session=session)
    assert len(sg_one.nodes) == 1
    assert sg_one.nodes[0].node_key == ("entity", seed.id)
    assert len(sg_one.edges) == 0
    assert sg_one.is_truncated is True
    assert sg_one.truncation_reason == "max_nodes_reached"


def test_step_3_3_exact_edge_budget_boundaries_and_zero_overflow(session):
    """Scenario E: Verify exact edge budget boundary semantics and strict prevention of budget overflow."""
    user = User(email="edge_bound@nexora.ai")
    session.add(user)
    session.flush()

    ws = Workspace(user_id=user.id, name="Edge Boundary Lab")
    session.add(ws)
    session.flush()

    col = Collection(workspace_id=ws.id, name="Edge Boundary Collection")
    session.add(col)
    session.flush()

    ent_repo = GraphEntityRepository(session)
    seed = ent_repo.create(collection_id=col.id, name="Seed Edge Node", entity_type="concept")
    n1 = ent_repo.create(collection_id=col.id, name="Edge Neighbor 1", entity_type="concept")
    n2 = ent_repo.create(collection_id=col.id, name="Edge Neighbor 2", entity_type="concept")
    n3 = ent_repo.create(collection_id=col.id, name="Edge Neighbor 3", entity_type="concept")

    rel_repo = GraphRelationshipRepository(session)
    # Seed has exactly 3 outgoing edges
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=seed.id, target_type="entity", target_id=n1.id, confidence=0.95)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=seed.id, target_type="entity", target_id=n2.id, confidence=0.85)
    rel_repo.create(collection_id=col.id, relationship_type="relates_to", source_type="entity", source_id=seed.id, target_type="entity", target_id=n3.id, confidence=0.75)
    session.commit()

    service = KnowledgeGraphTraversalService()

    # 1. Exact capacity: max_edges = 3
    sg_exact = service.explore_multihop(seed.id, "entity", col.id, max_depth=1, max_edges=3, max_nodes=50, direction="outgoing", session=session)
    assert len(sg_exact.edges) == 3
    assert sg_exact.is_truncated is False
    assert sg_exact.truncation_reason is None

    # 2. One additional eligible edge: max_edges = 2
    # Must collect exactly 2 edges with ZERO overflow
    sg_overflow_guard = service.explore_multihop(seed.id, "entity", col.id, max_depth=1, max_edges=2, max_nodes=50, direction="outgoing", session=session)
    assert len(sg_overflow_guard.edges) == 2  # Zero overflow: strictly <= 2
    assert sg_overflow_guard.is_truncated is True
    assert sg_overflow_guard.truncation_reason == "max_edges_reached"

    # 3. max_edges = 1
    sg_one = service.explore_multihop(seed.id, "entity", col.id, max_depth=1, max_edges=1, max_nodes=50, direction="outgoing", session=session)
    assert len(sg_one.edges) == 1
    assert sg_one.is_truncated is True
    assert sg_one.truncation_reason == "max_edges_reached"


def test_step_3_3_determinism_independent_of_insertion_order(session):
    """Scenario F: Verify deterministic ordering of nodes, edges, and paths independent of database insertion order."""
    user = User(email="det_test@nexora.ai")
    session.add(user)
    session.flush()

    ws = Workspace(user_id=user.id, name="Determinism Lab")
    session.add(ws)
    session.flush()

    col1 = Collection(workspace_id=ws.id, name="Col Determinism 1")
    col2 = Collection(workspace_id=ws.id, name="Col Determinism 2")
    session.add_all([col1, col2])
    session.flush()

    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # In col1: Insert entities in alphabetical order
    s1 = ent_repo.create(collection_id=col1.id, name="Center", entity_type="concept")
    e1_a = ent_repo.create(collection_id=col1.id, name="Alpha", entity_type="concept")
    e1_b = ent_repo.create(collection_id=col1.id, name="Beta", entity_type="concept")
    e1_g = ent_repo.create(collection_id=col1.id, name="Gamma", entity_type="concept")

    # Relationships inserted with varying confidences
    rel_repo.create(collection_id=col1.id, relationship_type="relates_to", source_type="entity", source_id=s1.id, target_type="entity", target_id=e1_g.id, confidence=0.50)
    rel_repo.create(collection_id=col1.id, relationship_type="relates_to", source_type="entity", source_id=s1.id, target_type="entity", target_id=e1_a.id, confidence=0.90)
    rel_repo.create(collection_id=col1.id, relationship_type="relates_to", source_type="entity", source_id=s1.id, target_type="entity", target_id=e1_b.id, confidence=0.70)

    # In col2: Insert the logically identical graph in REVERSE insertion order
    e2_g = ent_repo.create(collection_id=col2.id, name="Gamma", entity_type="concept")
    e2_b = ent_repo.create(collection_id=col2.id, name="Beta", entity_type="concept")
    e2_a = ent_repo.create(collection_id=col2.id, name="Alpha", entity_type="concept")
    s2 = ent_repo.create(collection_id=col2.id, name="Center", entity_type="concept")

    rel_repo.create(collection_id=col2.id, relationship_type="relates_to", source_type="entity", source_id=s2.id, target_type="entity", target_id=e2_a.id, confidence=0.90)
    rel_repo.create(collection_id=col2.id, relationship_type="relates_to", source_type="entity", source_id=s2.id, target_type="entity", target_id=e2_b.id, confidence=0.70)
    rel_repo.create(collection_id=col2.id, relationship_type="relates_to", source_type="entity", source_id=s2.id, target_type="entity", target_id=e2_g.id, confidence=0.50)
    session.commit()

    service = KnowledgeGraphTraversalService()

    # Repeated queries on col1 across 5 iterations must yield identical sequences
    runs_nodes = []
    runs_edges = []
    for _ in range(5):
        sg = service.get_neighborhood(s1.id, "entity", col1.id, direction="outgoing", session=session)
        runs_nodes.append([n.node_key for n in sg.nodes])
        runs_edges.append([e.id for e in sg.edges])

    assert all(r == runs_nodes[0] for r in runs_nodes)
    assert all(r == runs_edges[0] for r in runs_edges)

    # Independent of insertion order: nodes in col1 and col2 must sort identically by (seed_first, name)
    sg1 = service.get_neighborhood(s1.id, "entity", col1.id, direction="outgoing", session=session)
    sg2 = service.get_neighborhood(s2.id, "entity", col2.id, direction="outgoing", session=session)

    # Seed is index 0 in both
    assert sg1.nodes[0].name == "Center"
    assert sg2.nodes[0].name == "Center"
    # Neighbors sort deterministically by name: Alpha, Beta, Gamma
    assert [n.name for n in sg1.nodes[1:]] == ["Alpha", "Beta", "Gamma"]
    assert [n.name for n in sg2.nodes[1:]] == ["Alpha", "Beta", "Gamma"]

    # Edges sort deterministically by confidence DESC: 0.90, 0.70, 0.50
    assert [e.confidence for e in sg1.edges] == [0.90, 0.70, 0.50]
    assert [e.confidence for e in sg2.edges] == [0.90, 0.70, 0.50]
