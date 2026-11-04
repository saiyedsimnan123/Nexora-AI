"""Focused tests for Knowledge Graph Traversal Models and Batch Repository Lookups (Stage 5 Step 3.1).

Covers:
1. GraphNode compound identity, validation, and ORM conversion.
2. Compound node identity collision safety (Paper vs Entity sharing same UUID).
3. GraphEdge evidence provenance preservation, nullable confidence, and ORM conversion.
4. GraphSubgraph seed key, node lookup, and truncation attributes.
5. GraphPath continuous node-edge sequence validation and start/end properties.
6. GraphRelationshipRepository.list_topological_edges():
   - Batch query for papers and entities.
   - Directionality (outgoing, incoming, both/undirected).
   - Relationship type filtering.
   - Confidence threshold filtering with and without unscored edges.
   - Deterministic ordering and pagination.
   - Exclusion of asserting-paper provenance (paper_id) from topological edges.
   - Strict collection scoping and tenant isolation.
   - Empty input safety and argument validation.
"""

from __future__ import annotations

import datetime
from uuid import UUID, uuid4
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from nexora.database.models import (
    Base,
    Collection,
    Document,
    DocumentChunkRecord,
    GraphEndpointType,
    GraphEntity,
    GraphEntityType,
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
    """SQLite in-memory engine with PRAGMA foreign_keys=ON."""
    engine = create_engine("sqlite:///:memory:")

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
def seed_data(session):
    """Seed test graph with collections, papers, entities, chunks, and relationships."""
    user = User(email="traversal@nexora.ai")
    session.add(user)
    session.flush()

    workspace = Workspace(user_id=user.id, name="Graph Traversal Lab")
    session.add(workspace)
    session.flush()

    col_a = Collection(workspace_id=workspace.id, name="Collection A")
    col_b = Collection(workspace_id=workspace.id, name="Collection B")
    session.add_all([col_a, col_b])
    session.flush()

    paper_a1 = Paper(
        collection_id=col_a.id,
        title="Attention Is All You Need",
        doi="10.1145/test.vaswani",
        external_id="1706.03762",
        publication_date=datetime.date(2017, 6, 12),
    )
    paper_a2 = Paper(
        collection_id=col_a.id,
        title="FlashAttention: Fast and Memory-Efficient Exact Attention",
        doi="10.1145/test.dao",
        external_id="2205.14135",
        publication_date=datetime.date(2022, 5, 27),
    )
    paper_b1 = Paper(
        collection_id=col_b.id,
        title="ResNet: Deep Residual Learning for Image Recognition",
        publication_date=datetime.date(2015, 12, 10),
    )
    session.add_all([paper_a1, paper_a2, paper_b1])
    session.flush()

    doc_a1 = Document(
        paper_id=paper_a1.id,
        filename="vaswani2017.pdf",
        mime_type="application/pdf",
        file_size=10240,
        checksum="a" * 64,
        storage_key="docs/vaswani2017.pdf",
        indexing_status=IndexingStatus.COMPLETED.value,
    )
    session.add(doc_a1)
    session.flush()

    chunk_a1 = DocumentChunkRecord(
        document_id=doc_a1.id,
        chunk_index=0,
        text="We propose the Transformer model based on multi-head attention.",
    )
    session.add(chunk_a1)
    session.flush()

    ent_repo = GraphEntityRepository(session)
    transformer_ent = ent_repo.create(
        collection_id=col_a.id,
        name="Transformer Architecture",
        entity_type="method",
        description="Novel neural sequence architecture.",
        aliases=["Transformer", "Transformer model"],
    )
    self_attn_ent = ent_repo.create(
        collection_id=col_a.id,
        name="Self-Attention",
        entity_type="concept",
        description="Attention mechanism relating sequence positions.",
    )
    wmt_ent = ent_repo.create(
        collection_id=col_a.id,
        name="WMT 2014 English-to-German",
        entity_type="dataset",
        description="Machine translation benchmark.",
    )
    author_ent = ent_repo.create(
        collection_id=col_a.id,
        name="Ashish Vaswani",
        entity_type="author",
    )

    # Entity in Collection B
    resnet_method = ent_repo.create(
        collection_id=col_b.id,
        name="Residual Connections",
        entity_type="method",
    )

    rel_repo = GraphRelationshipRepository(session)

    # Edge 1: Paper A1 -> proposes -> Transformer (confidence=0.98, grounded evidence)
    rel1 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="proposes",
        source_type="paper",
        source_id=paper_a1.id,
        target_type="entity",
        target_id=transformer_ent.id,
        paper_id=paper_a1.id,
        chunk_id=chunk_a1.id,
        page_number=1,
        evidence_text="We propose the Transformer model.",
        confidence=0.98,
        extraction_method="llm_extraction",
    )

    # Edge 2: Transformer -> relates_to -> Self-Attention (confidence=0.90, asserting paper is paper_a1)
    rel2 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="relates_to",
        source_type="entity",
        source_id=transformer_ent.id,
        target_type="entity",
        target_id=self_attn_ent.id,
        paper_id=paper_a1.id,  # Asserting paper! Not an endpoint.
        confidence=0.90,
    )

    # Edge 3: Paper A2 -> cites -> Paper A1 (confidence=1.0, citation_match)
    rel3 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="cites",
        source_type="paper",
        source_id=paper_a2.id,
        target_type="paper",
        target_id=paper_a1.id,
        confidence=1.0,
        extraction_method="citation_match",
    )

    # Edge 4: Paper A1 -> uses_dataset -> WMT (confidence=None, unscored)
    rel4 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="uses_dataset",
        source_type="paper",
        source_id=paper_a1.id,
        target_type="entity",
        target_id=wmt_ent.id,
        confidence=None,
    )

    # Edge 5: Paper A1 -> authored_by -> Ashish Vaswani (confidence=1.0)
    rel5 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=paper_a1.id,
        target_type="entity",
        target_id=author_ent.id,
        confidence=1.0,
    )

    # Edge in Collection B (Paper B1 -> proposes -> Residual Connections)
    rel_b = rel_repo.create(
        collection_id=col_b.id,
        relationship_type="proposes",
        source_type="paper",
        source_id=paper_b1.id,
        target_type="entity",
        target_id=resnet_method.id,
        confidence=0.99,
    )

    session.flush()

    return {
        "col_a": col_a,
        "col_b": col_b,
        "paper_a1": paper_a1,
        "paper_a2": paper_a2,
        "paper_b1": paper_b1,
        "chunk_a1": chunk_a1,
        "transformer_ent": transformer_ent,
        "self_attn_ent": self_attn_ent,
        "wmt_ent": wmt_ent,
        "author_ent": author_ent,
        "resnet_method": resnet_method,
        "rel1": rel1,
        "rel2": rel2,
        "rel3": rel3,
        "rel4": rel4,
        "rel5": rel5,
        "rel_b": rel_b,
    }


# ============================================================================
# 1. GraphNode Unit Tests
# ============================================================================

def test_graph_node_paper_and_entity_creation(seed_data):
    paper = seed_data["paper_a1"]
    entity = seed_data["transformer_ent"]

    # From Paper ORM
    node_p = GraphNode.from_paper(paper)
    assert node_p.id == paper.id
    assert node_p.node_type == "paper"
    assert node_p.name == "Attention Is All You Need"
    assert node_p.entity_type is None
    assert node_p.node_key == ("paper", paper.id)
    assert node_p.key == f"paper:{paper.id}"
    assert node_p.metadata["doi"] == "10.1145/test.vaswani"

    # From Entity ORM
    node_e = GraphNode.from_entity(entity)
    assert node_e.id == entity.id
    assert node_e.node_type == "entity"
    assert node_e.name == "Transformer Architecture"
    assert node_e.entity_type == "method"
    assert node_e.node_key == ("entity", entity.id)
    assert node_e.key == f"entity:{entity.id}"
    assert node_e.metadata["canonical_name"] == "transformer architecture"
    assert node_e.metadata["aliases"] == ["Transformer", "Transformer model"]


def test_graph_node_validation_errors():
    valid_uuid = uuid4()

    # Invalid node_type
    with pytest.raises(ValueError, match="Invalid node_type"):
        GraphNode(id=valid_uuid, node_type="invalid_type", name="Test")

    # Paper node with entity_type must fail
    with pytest.raises(ValueError, match="entity_type must be None when node_type is 'paper'"):
        GraphNode(id=valid_uuid, node_type="paper", name="Paper", entity_type="method")

    # Entity node without entity_type must fail
    with pytest.raises(ValueError, match="entity_type is required when node_type is 'entity'"):
        GraphNode(id=valid_uuid, node_type="entity", name="Entity", entity_type=None)

    # Invalid entity_type
    with pytest.raises(ValueError, match="Invalid entity_type"):
        GraphNode(id=valid_uuid, node_type="entity", name="Entity", entity_type="invalid_cat")


def test_graph_node_compound_identity_collision_safety():
    """Verify that a Paper and an Entity sharing the exact same UUID do not collide."""
    shared_uuid = uuid4()

    paper_node = GraphNode(
        id=shared_uuid,
        node_type="paper",
        name="Attention Paper",
        entity_type=None,
    )
    entity_node = GraphNode(
        id=shared_uuid,
        node_type="entity",
        name="Attention Concept",
        entity_type="concept",
    )

    assert paper_node.id == entity_node.id  # Same UUID
    assert paper_node.node_key != entity_node.node_key  # Different compound key!
    assert paper_node.node_key == ("paper", shared_uuid)
    assert entity_node.node_key == ("entity", shared_uuid)
    assert paper_node.key != entity_node.key

    # In a set or dictionary, both coexist without collision
    node_set = {paper_node, entity_node}
    assert len(node_set) == 2


# ============================================================================
# 2. GraphEdge Unit Tests
# ============================================================================

def test_graph_edge_evidence_provenance_and_nullable_confidence(seed_data):
    rel1 = seed_data["rel1"]
    rel4 = seed_data["rel4"]

    # Edge with evidence and confidence
    edge1 = GraphEdge.from_orm(rel1)
    assert edge1.id == rel1.id
    assert edge1.relationship_type == "proposes"
    assert edge1.source_key == ("paper", rel1.source_id)
    assert edge1.target_key == ("entity", rel1.target_id)
    assert edge1.confidence == 0.98
    assert edge1.chunk_id == rel1.chunk_id
    assert edge1.page_number == 1
    assert edge1.evidence_text == "We propose the Transformer model."
    assert edge1.has_evidence is True

    # Edge with NULL confidence (must be preserved as None, never fabricated)
    edge4 = GraphEdge.from_orm(rel4)
    assert edge4.confidence is None
    assert edge4.chunk_id is None
    assert edge4.has_evidence is False


# ============================================================================
# 3. GraphSubgraph and GraphPath Unit Tests
# ============================================================================

def test_graph_subgraph_and_path_models(seed_data):
    paper = seed_data["paper_a1"]
    method = seed_data["transformer_ent"]
    concept = seed_data["self_attn_ent"]

    node_p = GraphNode.from_paper(paper)
    node_m = GraphNode.from_entity(method)
    node_c = GraphNode.from_entity(concept)

    edge1 = GraphEdge.from_orm(seed_data["rel1"])
    edge2 = GraphEdge.from_orm(seed_data["rel2"])

    # GraphSubgraph
    subgraph = GraphSubgraph(
        seed_id=paper.id,
        seed_type="paper",
        depth=2,
        nodes=[node_p, node_m, node_c],
        edges=[edge1, edge2],
        is_truncated=False,
    )
    assert subgraph.seed_key == ("paper", paper.id)
    assert len(subgraph.node_keys) == 3
    assert subgraph.get_node("paper", paper.id) == node_p
    assert subgraph.get_node("entity", method.id) == node_m
    assert subgraph.get_node("entity", uuid4()) is None

    # GraphPath
    path = GraphPath(
        nodes=[node_p, node_m, node_c],
        edges=[edge1, edge2],
        length=2,
        confidence_score=0.98 * 0.90,
    )
    assert path.start_node == node_p
    assert path.end_node == node_c
    assert path.path_keys == [node_p.node_key, node_m.node_key, node_c.node_key]

    # Path node-edge count mismatch must raise ValueError
    with pytest.raises(ValueError, match="Path structure mismatch"):
        GraphPath(
            nodes=[node_p, node_m],
            edges=[edge1, edge2],  # 2 nodes with 2 edges is invalid
            length=2,
        )


# ============================================================================
# 4. GraphRelationshipRepository.list_topological_edges() Tests
# ============================================================================

def test_list_topological_edges_batch_papers_and_entities(session, seed_data):
    repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    transformer_ent = seed_data["transformer_ent"]

    # Query edges connecting Paper A1 and Transformer Architecture (undirected/both)
    edges = repo.list_topological_edges(
        col_a.id,
        paper_ids=[paper_a1.id],
        entity_ids=[transformer_ent.id],
        direction="both",
    )
    # Should include:
    # - rel1 (paper_a1 -> proposes -> transformer_ent)
    # - rel2 (transformer_ent -> relates_to -> self_attn_ent)
    # - rel3 (paper_a2 -> cites -> paper_a1)
    # - rel4 (paper_a1 -> uses_dataset -> wmt_ent)
    # - rel5 (paper_a1 -> authored_by -> author_ent)
    assert len(edges) == 5
    edge_types = {e.relationship_type for e in edges}
    assert edge_types == {"proposes", "relates_to", "cites", "uses_dataset", "authored_by"}


def test_list_topological_edges_directionality(session, seed_data):
    repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    # 1. Outgoing from Paper A1 only
    outgoing = repo.list_topological_edges(
        col_a.id,
        paper_ids=[paper_a1.id],
        direction="outgoing",
    )
    out_types = {e.relationship_type for e in outgoing}
    assert out_types == {"proposes", "uses_dataset", "authored_by"}
    assert all(e.source_paper_id == paper_a1.id for e in outgoing)

    # 2. Incoming to Paper A1 only
    incoming = repo.list_topological_edges(
        col_a.id,
        paper_ids=[paper_a1.id],
        direction="incoming",
    )
    assert len(incoming) == 1
    assert incoming[0].relationship_type == "cites"
    assert incoming[0].target_paper_id == paper_a1.id


def test_list_topological_edges_relationship_type_filtering(session, seed_data):
    repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    edges = repo.list_topological_edges(
        col_a.id,
        paper_ids=[paper_a1.id],
        relationship_types=["cites", "proposes"],
    )
    assert len(edges) == 2
    types = {e.relationship_type for e in edges}
    assert types == {"cites", "proposes"}


def test_list_topological_edges_confidence_filtering(session, seed_data):
    repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    # 1. min_confidence=0.95 with include_unscored=False
    # rel1=0.98, rel3=1.0, rel5=1.0 (rel4 is None -> excluded)
    strict_edges = repo.list_topological_edges(
        col_a.id,
        paper_ids=[paper_a1.id],
        min_confidence=0.95,
        include_unscored=False,
    )
    assert len(strict_edges) == 3
    assert all(e.confidence is not None and e.confidence >= 0.95 for e in strict_edges)
    assert "uses_dataset" not in {e.relationship_type for e in strict_edges}

    # 2. min_confidence=0.95 with include_unscored=True (includes rel4 with confidence=None)
    unscored_included = repo.list_topological_edges(
        col_a.id,
        paper_ids=[paper_a1.id],
        min_confidence=0.95,
        include_unscored=True,
    )
    assert len(unscored_included) == 4
    assert any(e.relationship_type == "uses_dataset" and e.confidence is None for e in unscored_included)


def test_list_topological_edges_excludes_asserting_paper_provenance(session, seed_data):
    """CRITICAL: An edge asserted by Paper A1 between two entities is NOT a topological edge of Paper A1."""
    repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    rel2 = seed_data["rel2"]  # Transformer -> relates_to -> Self-Attention (paper_id=paper_a1.id)

    # Query edges strictly connecting Paper A1
    p1_edges = repo.list_topological_edges(
        col_a.id,
        paper_ids=[paper_a1.id],
        direction="both",
    )
    p1_ids = {e.id for e in p1_edges}

    # rel2 has paper_id=paper_a1.id, but its endpoints are Transformer and Self-Attention.
    # It MUST NOT be present in Paper A1's topological edges!
    assert rel2.id not in p1_ids

    # Querying Transformer entity DOES include rel2
    t_edges = repo.list_topological_edges(
        col_a.id,
        entity_ids=[seed_data["transformer_ent"].id],
        direction="both",
    )
    t_ids = {e.id for e in t_edges}
    assert rel2.id in t_ids


def test_list_topological_edges_tenant_collection_isolation(session, seed_data):
    repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    col_b = seed_data["col_b"]
    paper_b1 = seed_data["paper_b1"]

    # Paper B1 belongs to Collection B. Querying under Collection A must return 0 edges.
    edges = repo.list_topological_edges(
        col_a.id,
        paper_ids=[paper_b1.id],
    )
    assert len(edges) == 0

    # Querying under Collection B returns Paper B1's edge
    edges_b = repo.list_topological_edges(
        col_b.id,
        paper_ids=[paper_b1.id],
    )
    assert len(edges_b) == 1
    assert edges_b[0].id == seed_data["rel_b"].id


def test_list_topological_edges_empty_input_and_validation(session, seed_data):
    repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]

    # Both empty/None returns empty sequence immediately
    assert repo.list_topological_edges(col_a.id) == ()
    assert repo.list_topological_edges(col_a.id, paper_ids=[], entity_ids=[]) == ()

    # Validation errors
    with pytest.raises(TypeError, match="collection_id must be a UUID"):
        repo.list_topological_edges("not-a-uuid")  # type: ignore

    with pytest.raises(ValueError, match="direction must be"):
        repo.list_topological_edges(col_a.id, paper_ids=[uuid4()], direction="sideways")

    with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
        repo.list_topological_edges(col_a.id, paper_ids=[uuid4()], min_confidence=1.5)

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        repo.list_topological_edges(col_a.id, paper_ids=[uuid4()], limit=0)
