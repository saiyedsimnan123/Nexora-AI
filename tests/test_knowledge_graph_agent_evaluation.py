"""Deterministic correctness, boundary, and security evaluation tests for Knowledge Graph Agent Tools.

Stage 5 Step 4.3 — Knowledge Graph Agent Evaluation.

Verifies:
1. Entity search exact matching, substring matching, entity-type filtering, and deterministic limits.
2. Neighborhood exploration across outgoing, incoming, both directions, and multi-hop BFS.
3. Node and edge budget truncation behavior and truncation reasons.
4. Complete evidence provenance preservation on nodes and edges.
5. Path discovery correctness for 1-hop and 2-hop connecting paths.
6. Multiplicative path confidence calculations.
7. Cycle prevention and graceful handling of nonexistent paths.
8. Analytics calculation correctness across summary, overview, author collaborations, and author connectivity.
9. Security fail-closed enforcement: anonymous access, cross-tenant isolation, and missing auth_service.
10. Input validation: malformed UUIDs, invalid parameter ranges, forbidden extra fields, and invalid query types.
11. Safe failure handling on unexpected underlying service exceptions.
"""

from __future__ import annotations

import datetime
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.agents.models import AgentContext, ToolResult
from nexora.agents.tools import (
    ExploreGraphNeighborhoodTool,
    FindGraphPathTool,
    QueryGraphAnalyticsTool,
    SearchGraphEntitiesTool,
)
from nexora.auth.service import AuthenticationService
from nexora.database.models.base import Base
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.graph_entity import GraphEntity
from nexora.database.models.graph_relationship import GraphRelationship
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.graph.analytics_service import KnowledgeGraphAnalyticsService
from nexora.graph.traversal import KnowledgeGraphTraversalService


# ---------------------------------------------------------------------------
# Fixed Deterministic Identifiers
# ---------------------------------------------------------------------------

USER_A_ID = UUID("11111111-1111-1111-1111-111111111111")
USER_B_ID = UUID("22222222-2222-2222-2222-222222222222")

WS_A_ID = UUID("aaaaaaaa-1111-1111-1111-111111111111")
WS_B_ID = UUID("bbbbbbbb-1111-1111-1111-111111111111")

COL_A_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
COL_B_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

PAPER_1_ID = UUID("10000000-0000-0000-0000-000000000001")
PAPER_2_ID = UUID("10000000-0000-0000-0000-000000000002")

DOC_1_ID = UUID("10000000-0000-0000-0000-000000000011")
CHUNK_1_ID = UUID("10000000-0000-0000-0000-000000000021")

ENTITY_TRANSFORMER_ID = UUID("20000000-0000-0000-0000-000000000001")
ENTITY_ATTENTION_ID = UUID("20000000-0000-0000-0000-000000000002")
ENTITY_VASWANI_ID = UUID("20000000-0000-0000-0000-000000000003")
ENTITY_SHAZEER_ID = UUID("20000000-0000-0000-0000-000000000004")
ENTITY_WMT_ID = UUID("20000000-0000-0000-0000-000000000005")

ENTITY_SECRET_B_ID = UUID("30000000-0000-0000-0000-000000000001")

REL_1_ID = UUID("40000000-0000-0000-0000-000000000001")
REL_2_ID = UUID("40000000-0000-0000-0000-000000000002")
REL_3_ID = UUID("40000000-0000-0000-0000-000000000003")
REL_4_ID = UUID("40000000-0000-0000-0000-000000000004")
REL_5_ID = UUID("40000000-0000-0000-0000-000000000005")


# ---------------------------------------------------------------------------
# Deterministic Benchmark Database Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def eval_db_env():
    """In-memory SQLite database populated with a deterministic benchmark graph corpus."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    session = session_factory()
    try:
        # 1. Tenants and Workspaces
        user_a = User(id=USER_A_ID, email="eval-tenant-a@example.com")
        user_b = User(id=USER_B_ID, email="eval-tenant-b@example.com")
        session.add_all([user_a, user_b])
        session.flush()

        ws_a = Workspace(id=WS_A_ID, name="Eval Workspace A", user_id=USER_A_ID)
        ws_b = Workspace(id=WS_B_ID, name="Eval Workspace B", user_id=USER_B_ID)
        session.add_all([ws_a, ws_b])
        session.flush()

        col_a = Collection(id=COL_A_ID, name="Transformers Research", workspace_id=WS_A_ID)
        col_b = Collection(id=COL_B_ID, name="Tenant B Private Data", workspace_id=WS_B_ID)
        session.add_all([col_a, col_b])
        session.flush()

        # 2. Papers and Chunk Evidence in Collection A
        p1 = Paper(
            id=PAPER_1_ID,
            collection_id=COL_A_ID,
            title="Attention Is All You Need",
            abstract="Seminal sequence transduction model using self-attention.",
            doi="10.48550/arXiv.1706.03762",
            publication_date=datetime.date(2017, 6, 12),
        )
        p2 = Paper(
            id=PAPER_2_ID,
            collection_id=COL_A_ID,
            title="Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
            abstract="Enables learning dependency beyond a fixed length without disrupting temporal coherence.",
            doi="10.48550/arXiv.1901.02860",
            publication_date=datetime.date(2019, 1, 9),
        )
        session.add_all([p1, p2])
        session.flush()

        doc1 = Document(
            id=DOC_1_ID,
            paper_id=PAPER_1_ID,
            filename="attention.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="a" * 64,
            storage_key="papers/attention.pdf",
            indexing_status="completed",
        )
        session.add(doc1)
        session.flush()

        chunk1 = DocumentChunkRecord(
            id=CHUNK_1_ID,
            document_id=DOC_1_ID,
            chunk_index=0,
            text="The Transformer relies entirely on self-attention to compute representations.",
        )
        session.add(chunk1)
        session.flush()

        # 3. Canonical Entities in Collection A (5 entities)
        e_transformer = GraphEntity(
            id=ENTITY_TRANSFORMER_ID,
            collection_id=COL_A_ID,
            name="Transformer",
            canonical_name="transformer",
            entity_type="method",
            description="Sequence-to-sequence neural model based entirely on attention mechanisms.",
            aliases=["Transformer model", "Attention architecture"],
        )
        e_attention = GraphEntity(
            id=ENTITY_ATTENTION_ID,
            collection_id=COL_A_ID,
            name="Self-Attention",
            canonical_name="self-attention",
            entity_type="method",
            description="Attention mechanism relating different positions of a single sequence.",
            aliases=["Intra-attention"],
        )
        e_vaswani = GraphEntity(
            id=ENTITY_VASWANI_ID,
            collection_id=COL_A_ID,
            name="Ashish Vaswani",
            canonical_name="ashish vaswani",
            entity_type="author",
            description="Primary researcher and author of the Transformer architecture.",
        )
        e_shazeer = GraphEntity(
            id=ENTITY_SHAZEER_ID,
            collection_id=COL_A_ID,
            name="Noam Shazeer",
            canonical_name="noam shazeer",
            entity_type="author",
            description="Co-author of the Transformer and mesh-tensorflow architecture.",
        )
        e_wmt = GraphEntity(
            id=ENTITY_WMT_ID,
            collection_id=COL_A_ID,
            name="WMT 2014 En-De",
            canonical_name="wmt 2014 en-de",
            entity_type="dataset",
            description="Standard English-to-German machine translation benchmark dataset.",
        )
        session.add_all([e_transformer, e_attention, e_vaswani, e_shazeer, e_wmt])

        # 4. Canonical Entity in Collection B (Tenant B Private Entity)
        e_secret_b = GraphEntity(
            id=ENTITY_SECRET_B_ID,
            collection_id=COL_B_ID,
            name="Secret Algorithm B",
            canonical_name="secret algorithm b",
            entity_type="method",
            description="Proprietary algorithm strictly confidential to Tenant B.",
        )
        session.add(e_secret_b)
        session.flush()

        # 5. Typed Ground-Truth Relationships in Collection A (5 edges)
        # R1: Paper 1 -> authored_by -> Vaswani (confidence 0.99)
        rel_1 = GraphRelationship(
            id=REL_1_ID,
            collection_id=COL_A_ID,
            source_type="paper",
            source_id=PAPER_1_ID,
            source_paper_id=PAPER_1_ID,
            target_type="entity",
            target_id=ENTITY_VASWANI_ID,
            target_entity_id=ENTITY_VASWANI_ID,
            relationship_type="authored_by",
            confidence=0.99,
            evidence_text="Attention Is All You Need was authored by Ashish Vaswani.",
            paper_id=PAPER_1_ID,
            chunk_id=CHUNK_1_ID,
            page_number=1,
            extraction_method="rule_based",
        )
        # R2: Paper 1 -> authored_by -> Shazeer (confidence 0.99)
        rel_2 = GraphRelationship(
            id=REL_2_ID,
            collection_id=COL_A_ID,
            source_type="paper",
            source_id=PAPER_1_ID,
            source_paper_id=PAPER_1_ID,
            target_type="entity",
            target_id=ENTITY_SHAZEER_ID,
            target_entity_id=ENTITY_SHAZEER_ID,
            relationship_type="authored_by",
            confidence=0.99,
            evidence_text="Noam Shazeer proposed multi-head attention components.",
            paper_id=PAPER_1_ID,
            chunk_id=CHUNK_1_ID,
            page_number=1,
            extraction_method="rule_based",
        )
        # R3: Paper 1 -> proposes -> Transformer (confidence 0.95)
        rel_3 = GraphRelationship(
            id=REL_3_ID,
            collection_id=COL_A_ID,
            source_type="paper",
            source_id=PAPER_1_ID,
            source_paper_id=PAPER_1_ID,
            target_type="entity",
            target_id=ENTITY_TRANSFORMER_ID,
            target_entity_id=ENTITY_TRANSFORMER_ID,
            relationship_type="proposes",
            confidence=0.95,
            evidence_text="In this paper we propose the Transformer, a model architecture.",
            paper_id=PAPER_1_ID,
            chunk_id=CHUNK_1_ID,
            page_number=1,
            extraction_method="rule_based",
        )
        # R4: Transformer -> uses_method -> Self-Attention (confidence 0.90)
        rel_4 = GraphRelationship(
            id=REL_4_ID,
            collection_id=COL_A_ID,
            source_type="entity",
            source_id=ENTITY_TRANSFORMER_ID,
            source_entity_id=ENTITY_TRANSFORMER_ID,
            target_type="entity",
            target_id=ENTITY_ATTENTION_ID,
            target_entity_id=ENTITY_ATTENTION_ID,
            relationship_type="uses_method",
            confidence=0.90,
            evidence_text="The Transformer relies entirely on self-attention.",
            paper_id=PAPER_1_ID,
            chunk_id=CHUNK_1_ID,
            page_number=2,
            extraction_method="rule_based",
        )
        # R5: Paper 1 -> evaluates_on -> WMT 2014 En-De (confidence 0.85)
        rel_5 = GraphRelationship(
            id=REL_5_ID,
            collection_id=COL_A_ID,
            source_type="paper",
            source_id=PAPER_1_ID,
            source_paper_id=PAPER_1_ID,
            target_type="entity",
            target_id=ENTITY_WMT_ID,
            target_entity_id=ENTITY_WMT_ID,
            relationship_type="evaluates_on",
            confidence=0.85,
            evidence_text="On the WMT 2014 English-to-German translation task, the Transformer achieved 28.4 BLEU.",
            paper_id=PAPER_1_ID,
            chunk_id=CHUNK_1_ID,
            page_number=7,
            extraction_method="rule_based",
        )
        session.add_all([rel_1, rel_2, rel_3, rel_4, rel_5])
        session.commit()

        auth_service = AuthenticationService(session_factory=session_factory)
        traversal_service = KnowledgeGraphTraversalService(session_factory=session_factory)
        analytics_service = KnowledgeGraphAnalyticsService(session_factory=session_factory)

        return {
            "session_factory": session_factory,
            "auth_service": auth_service,
            "traversal_service": traversal_service,
            "analytics_service": analytics_service,
            "user_a_context": AgentContext(
                user_id=USER_A_ID,
                auth_enforced=True,
                auth_service=auth_service,
                session_factory=session_factory,
            ),
            "user_b_context": AgentContext(
                user_id=USER_B_ID,
                auth_enforced=True,
                auth_service=auth_service,
                session_factory=session_factory,
            ),
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 1. Entity Search Relevance & Filtering
# ---------------------------------------------------------------------------

def test_eval_search_exact_match(eval_db_env):
    """Verify exact-match entity lookup returns 100% precision and recall on the target entity."""
    tool = SearchGraphEntitiesTool(session_factory=eval_db_env["session_factory"])
    res = tool.execute(
        {"collection_id": str(COL_A_ID), "query": "Transformer", "entity_type": "method"},
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    assert res.error is None
    entities = res.data
    assert len(entities) == 1
    target = entities[0]
    assert target["id"] == str(ENTITY_TRANSFORMER_ID)
    assert target["name"] == "Transformer"
    assert target["canonical_name"] == "transformer"
    assert target["entity_type"] == "method"
    assert target["collection_id"] == str(COL_A_ID)


def test_eval_search_substring_matching(eval_db_env):
    """Verify substring query matches relevant entities via case-insensitive canonical/name filters."""
    tool = SearchGraphEntitiesTool(session_factory=eval_db_env["session_factory"])
    res = tool.execute(
        {"collection_id": str(COL_A_ID), "query": "Attention"},
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    entities = res.data
    assert len(entities) == 1
    assert entities[0]["name"] == "Self-Attention"
    assert entities[0]["id"] == str(ENTITY_ATTENTION_ID)


def test_eval_search_type_filtering(eval_db_env):
    """Verify entity_type filter strictly scopes results to the requested entity category."""
    tool = SearchGraphEntitiesTool(session_factory=eval_db_env["session_factory"])

    # Query for authors only
    res_authors = tool.execute(
        {"collection_id": str(COL_A_ID), "entity_type": "author"},
        eval_db_env["user_a_context"],
    )
    assert res_authors.success is True
    author_names = {e["name"] for e in res_authors.data}
    assert author_names == {"Ashish Vaswani", "Noam Shazeer"}

    # Query for dataset only
    res_datasets = tool.execute(
        {"collection_id": str(COL_A_ID), "entity_type": "dataset"},
        eval_db_env["user_a_context"],
    )
    assert res_datasets.success is True
    assert len(res_datasets.data) == 1
    assert res_datasets.data[0]["name"] == "WMT 2014 En-De"


def test_eval_search_type_filtering_zero_match(eval_db_env):
    """Verify entity search returns empty list when query exists but category does not match."""
    tool = SearchGraphEntitiesTool(session_factory=eval_db_env["session_factory"])
    res = tool.execute(
        {"collection_id": str(COL_A_ID), "query": "Vaswani", "entity_type": "dataset"},
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    assert res.data == []
    assert "Found 0 entities" in res.summary


def test_eval_search_list_without_query_and_limit(eval_db_env):
    """Verify omitting query lists collection entities deterministically capped by limit."""
    tool = SearchGraphEntitiesTool(session_factory=eval_db_env["session_factory"])
    res = tool.execute(
        {"collection_id": str(COL_A_ID), "limit": 3},
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    assert len(res.data) == 3
    # Ordered by canonical_name: ashish vaswani, noam shazeer, self-attention
    assert res.data[0]["canonical_name"] == "ashish vaswani"
    assert res.data[1]["canonical_name"] == "noam shazeer"
    assert res.data[2]["canonical_name"] == "self-attention"


# ---------------------------------------------------------------------------
# 2. Neighborhood Exploration & Directional Semantics
# ---------------------------------------------------------------------------

def test_eval_neighborhood_outgoing_1_hop(eval_db_env):
    """Verify 1-hop outgoing traversal from Paper 1 returns all 4 outgoing entity relationships."""
    tool = ExploreGraphNeighborhoodTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "node_id": str(PAPER_1_ID),
            "node_type": "paper",
            "direction": "outgoing",
            "max_depth": 1,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    subgraph = res.data
    assert subgraph["seed_id"] == str(PAPER_1_ID)
    assert subgraph["seed_type"] == "paper"
    assert subgraph["depth"] == 1
    assert subgraph["is_truncated"] is False

    # 1 seed node + 4 neighbor entities = 5 nodes
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert str(PAPER_1_ID) in node_ids
    assert str(ENTITY_VASWANI_ID) in node_ids
    assert str(ENTITY_SHAZEER_ID) in node_ids
    assert str(ENTITY_TRANSFORMER_ID) in node_ids
    assert str(ENTITY_WMT_ID) in node_ids
    assert len(subgraph["nodes"]) == 5

    # 4 outgoing edges from paper
    edge_ids = {e["id"] for e in subgraph["edges"]}
    assert {str(REL_1_ID), str(REL_2_ID), str(REL_3_ID), str(REL_5_ID)} == edge_ids


def test_eval_neighborhood_incoming_1_hop(eval_db_env):
    """Verify incoming traversal from Self-Attention discovers its source Transformer model."""
    tool = ExploreGraphNeighborhoodTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "node_id": str(ENTITY_ATTENTION_ID),
            "node_type": "entity",
            "direction": "incoming",
            "max_depth": 1,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    subgraph = res.data
    assert len(subgraph["nodes"]) == 2
    assert {n["id"] for n in subgraph["nodes"]} == {str(ENTITY_ATTENTION_ID), str(ENTITY_TRANSFORMER_ID)}
    assert len(subgraph["edges"]) == 1
    assert subgraph["edges"][0]["id"] == str(REL_4_ID)
    assert subgraph["edges"][0]["relationship_type"] == "uses_method"


def test_eval_neighborhood_both_directions(eval_db_env):
    """Verify both-direction traversal discovers incoming paper and outgoing component."""
    tool = ExploreGraphNeighborhoodTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "node_id": str(ENTITY_TRANSFORMER_ID),
            "node_type": "entity",
            "direction": "both",
            "max_depth": 1,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    subgraph = res.data
    # Seed (Transformer) + Incoming source (Paper 1) + Outgoing target (Self-Attention) = 3 nodes
    assert len(subgraph["nodes"]) == 3
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert node_ids == {str(ENTITY_TRANSFORMER_ID), str(PAPER_1_ID), str(ENTITY_ATTENTION_ID)}
    assert len(subgraph["edges"]) == 2
    edge_types = {e["relationship_type"] for e in subgraph["edges"]}
    assert edge_types == {"proposes", "uses_method"}


def test_eval_neighborhood_multihop_bfs_depth_expansion(eval_db_env):
    """Verify 2-hop BFS traversal expands from Paper 1 through Transformer to Self-Attention."""
    tool = ExploreGraphNeighborhoodTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "node_id": str(PAPER_1_ID),
            "node_type": "paper",
            "direction": "outgoing",
            "max_depth": 2,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    subgraph = res.data
    assert subgraph["depth"] == 2
    # All 6 nodes in Collection A
    assert len(subgraph["nodes"]) == 6
    # All 5 relationships in Collection A
    assert len(subgraph["edges"]) == 5
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert str(ENTITY_ATTENTION_ID) in node_ids


# ---------------------------------------------------------------------------
# 3. Budget Truncation & Provenance Preservation
# ---------------------------------------------------------------------------

def test_eval_neighborhood_node_budget_truncation(eval_db_env):
    """Verify strict node budget truncation flags is_truncated and sets truncation_reason."""
    tool = ExploreGraphNeighborhoodTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "node_id": str(PAPER_1_ID),
            "node_type": "paper",
            "max_nodes": 3,
            "max_depth": 1,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    subgraph = res.data
    assert len(subgraph["nodes"]) == 3
    assert subgraph["is_truncated"] is True
    assert subgraph["truncation_reason"] == "max_nodes_reached"


def test_eval_neighborhood_edge_budget_truncation(eval_db_env):
    """Verify strict edge budget truncation flags is_truncated and caps returned edges."""
    tool = ExploreGraphNeighborhoodTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "node_id": str(PAPER_1_ID),
            "node_type": "paper",
            "max_edges": 2,
            "max_depth": 1,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    subgraph = res.data
    assert len(subgraph["edges"]) == 2
    assert subgraph["is_truncated"] is True
    assert subgraph["truncation_reason"] == "max_edges_reached"


def test_eval_evidence_provenance_preservation(eval_db_env):
    """Verify edges preserve grounding fields: chunk_id, paper_id, page_number, evidence_text."""
    tool = ExploreGraphNeighborhoodTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "node_id": str(PAPER_1_ID),
            "node_type": "paper",
            "max_depth": 1,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    edges = res.data["edges"]
    assert len(edges) >= 4
    for edge in edges:
        assert edge["paper_id"] == str(PAPER_1_ID)
        assert edge["chunk_id"] == str(CHUNK_1_ID)
        assert isinstance(edge["page_number"], int)
        assert len(edge["evidence_text"]) > 10
        assert edge["extraction_method"] == "rule_based"


# ---------------------------------------------------------------------------
# 4. Path Discovery & Multiplicative Confidence
# ---------------------------------------------------------------------------

def test_eval_path_discovery_1_hop(eval_db_env):
    """Verify 1-hop path discovery between Paper 1 and Vaswani."""
    tool = FindGraphPathTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "source_id": str(PAPER_1_ID),
            "source_type": "paper",
            "target_id": str(ENTITY_VASWANI_ID),
            "target_type": "entity",
            "direction": "outgoing",
            "max_depth": 2,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    paths = res.data
    assert len(paths) == 1
    p = paths[0]
    assert p["hops"] == 1
    assert p["length"] == 1
    assert p["confidence_score"] == 0.99
    assert p["cumulative_confidence"] == 0.99
    assert len(p["nodes"]) == 2
    assert p["nodes"][0]["id"] == str(PAPER_1_ID)
    assert p["nodes"][1]["id"] == str(ENTITY_VASWANI_ID)
    assert len(p["edges"]) == 1
    assert p["edges"][0]["id"] == str(REL_1_ID)


def test_eval_path_discovery_2_hop_multiplicative_confidence(eval_db_env):
    """Verify 2-hop path discovery and multiplicative confidence score (0.95 * 0.90 = 0.855)."""
    tool = FindGraphPathTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "source_id": str(PAPER_1_ID),
            "source_type": "paper",
            "target_id": str(ENTITY_ATTENTION_ID),
            "target_type": "entity",
            "direction": "outgoing",
            "max_depth": 2,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    paths = res.data
    assert len(paths) == 1
    p = paths[0]
    assert p["hops"] == 2
    assert p["length"] == 2

    # Confidence: 0.95 (proposes) * 0.90 (uses_method) = 0.855
    expected_confidence = round(0.95 * 0.90, 6)
    assert p["confidence_score"] == expected_confidence
    assert p["cumulative_confidence"] == expected_confidence

    # Node chain: Paper 1 -> Transformer -> Self-Attention
    node_chain = [n["id"] for n in p["nodes"]]
    assert node_chain == [str(PAPER_1_ID), str(ENTITY_TRANSFORMER_ID), str(ENTITY_ATTENTION_ID)]
    edge_types = [e["relationship_type"] for e in p["edges"]]
    assert edge_types == ["proposes", "uses_method"]


def test_eval_path_nonexistent_returns_empty_list(eval_db_env):
    """Verify nonexistent path returns empty list without failure."""
    tool = FindGraphPathTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "source_id": str(ENTITY_ATTENTION_ID),
            "source_type": "entity",
            "target_id": str(ENTITY_VASWANI_ID),
            "target_type": "entity",
            "direction": "outgoing",
            "max_depth": 2,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    assert res.data == []
    assert "Discovered 0 path(s)" in res.summary


def test_eval_path_cycle_prevention(eval_db_env):
    """Verify source == target path discovery returns trivial 0-hop path with confidence 1.0."""
    tool = FindGraphPathTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "source_id": str(ENTITY_TRANSFORMER_ID),
            "source_type": "entity",
            "target_id": str(ENTITY_TRANSFORMER_ID),
            "target_type": "entity",
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    paths = res.data
    assert len(paths) == 1
    assert paths[0]["hops"] == 0
    assert paths[0]["confidence_score"] == 1.0
    assert len(paths[0]["nodes"]) == 1
    assert paths[0]["nodes"][0]["id"] == str(ENTITY_TRANSFORMER_ID)


# ---------------------------------------------------------------------------
# 5. Knowledge Graph Analytics Accuracy
# ---------------------------------------------------------------------------

def test_eval_analytics_summary_accuracy(eval_db_env):
    """Verify summary analytics returns exact ground-truth entity and relationship counts."""
    tool = QueryGraphAnalyticsTool(eval_db_env["analytics_service"])
    res = tool.execute(
        {"collection_id": str(COL_A_ID), "query_type": "summary"},
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    data = res.data
    assert data["collection_id"] == str(COL_A_ID)
    assert data["total_entities"] == 5
    assert data["total_relationships"] == 5
    assert data["entity_counts_by_type"] == {"author": 2, "dataset": 1, "method": 2}
    assert data["relationship_counts_by_type"] == {
        "authored_by": 2,
        "evaluates_on": 1,
        "proposes": 1,
        "uses_method": 1,
    }


def test_eval_analytics_overview_completeness(eval_db_env):
    """Verify overview query returns complete distributions, collaborations, and connectivity."""
    tool = QueryGraphAnalyticsTool(eval_db_env["analytics_service"])
    res = tool.execute(
        {"collection_id": str(COL_A_ID), "query_type": "overview", "top_n": 5},
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    data = res.data
    assert data["summary"]["total_entities"] == 5
    assert data["summary"]["total_relationships"] == 5
    assert len(data["entity_distribution"]) == 3
    assert len(data["relationship_distribution"]) == 4
    assert len(data["top_collaborations"]) == 1
    assert len(data["author_connectivity"]) == 2


def test_eval_analytics_author_collaborations(eval_db_env):
    """Verify co-authorship collaboration pair is correctly derived from shared Paper 1."""
    tool = QueryGraphAnalyticsTool(eval_db_env["analytics_service"])
    res = tool.execute(
        {"collection_id": str(COL_A_ID), "query_type": "collaborations"},
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    collabs = res.data
    assert len(collabs) == 1
    c = collabs[0]
    author_ids = {c["author_id"], c["collaborator_id"]}
    assert author_ids == {str(ENTITY_VASWANI_ID), str(ENTITY_SHAZEER_ID)}
    assert c["collaboration_count"] == 1
    assert c["shared_paper_ids"] == [str(PAPER_1_ID)]


def test_eval_analytics_author_connectivity(eval_db_env):
    """Verify author connectivity computes paper_count, collaborator_count, and total_collaborations."""
    tool = QueryGraphAnalyticsTool(eval_db_env["analytics_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "query_type": "connectivity",
            "author_id": str(ENTITY_VASWANI_ID),
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is True
    conn = res.data
    assert len(conn) == 1
    m = conn[0]
    assert m["author_id"] == str(ENTITY_VASWANI_ID)
    assert m["author_name"] == "Ashish Vaswani"
    assert m["paper_count"] == 1
    assert m["collaborator_count"] == 1
    assert m["total_collaborations"] == 1


# ---------------------------------------------------------------------------
# 6. Security Enforcement & Tenant Isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tool_cls, input_args",
    [
        (SearchGraphEntitiesTool, {"collection_id": str(COL_A_ID), "query": "Transformer"}),
        (
            ExploreGraphNeighborhoodTool,
            {"collection_id": str(COL_A_ID), "node_id": str(PAPER_1_ID), "node_type": "paper"},
        ),
        (
            FindGraphPathTool,
            {
                "collection_id": str(COL_A_ID),
                "source_id": str(PAPER_1_ID),
                "source_type": "paper",
                "target_id": str(ENTITY_VASWANI_ID),
                "target_type": "entity",
            },
        ),
        (QueryGraphAnalyticsTool, {"collection_id": str(COL_A_ID), "query_type": "summary"}),
    ],
)
def test_eval_security_anonymous_access_denied_all_tools(eval_db_env, tool_cls, input_args):
    """Verify all 4 graph tools fail closed with unauthorized when called anonymously."""
    if tool_cls is SearchGraphEntitiesTool:
        tool = tool_cls(session_factory=eval_db_env["session_factory"])
    elif tool_cls in (ExploreGraphNeighborhoodTool, FindGraphPathTool):
        tool = tool_cls(eval_db_env["traversal_service"])
    else:
        tool = tool_cls(eval_db_env["analytics_service"])

    anon_context = AgentContext(
        user_id=None,
        auth_enforced=False,
        auth_service=eval_db_env["auth_service"],
        session_factory=eval_db_env["session_factory"],
    )
    res = tool.execute(input_args, anon_context)
    assert res.success is False
    assert res.error_type == "unauthorized"
    assert "Authentication required" in res.error


@pytest.mark.parametrize(
    "tool_cls, input_args",
    [
        (SearchGraphEntitiesTool, {"collection_id": str(COL_B_ID), "query": "Secret"}),
        (
            ExploreGraphNeighborhoodTool,
            {"collection_id": str(COL_B_ID), "node_id": str(ENTITY_SECRET_B_ID), "node_type": "entity"},
        ),
        (
            FindGraphPathTool,
            {
                "collection_id": str(COL_B_ID),
                "source_id": str(ENTITY_SECRET_B_ID),
                "source_type": "entity",
                "target_id": str(ENTITY_SECRET_B_ID),
                "target_type": "entity",
            },
        ),
        (QueryGraphAnalyticsTool, {"collection_id": str(COL_B_ID), "query_type": "summary"}),
    ],
)
def test_eval_security_cross_tenant_denial_all_tools(eval_db_env, tool_cls, input_args):
    """Verify authenticated User A is denied access when querying Tenant B's private collection."""
    if tool_cls is SearchGraphEntitiesTool:
        tool = tool_cls(session_factory=eval_db_env["session_factory"])
    elif tool_cls in (ExploreGraphNeighborhoodTool, FindGraphPathTool):
        tool = tool_cls(eval_db_env["traversal_service"])
    else:
        tool = tool_cls(eval_db_env["analytics_service"])

    res = tool.execute(input_args, eval_db_env["user_a_context"])
    assert res.success is False
    assert res.error_type == "forbidden"
    assert "forbidden" in res.error.lower()


@pytest.mark.parametrize(
    "tool_cls, input_args",
    [
        (SearchGraphEntitiesTool, {"collection_id": str(COL_A_ID), "query": "Transformer"}),
        (
            ExploreGraphNeighborhoodTool,
            {"collection_id": str(COL_A_ID), "node_id": str(PAPER_1_ID), "node_type": "paper"},
        ),
        (
            FindGraphPathTool,
            {
                "collection_id": str(COL_A_ID),
                "source_id": str(PAPER_1_ID),
                "source_type": "paper",
                "target_id": str(ENTITY_VASWANI_ID),
                "target_type": "entity",
            },
        ),
        (QueryGraphAnalyticsTool, {"collection_id": str(COL_A_ID), "query_type": "summary"}),
    ],
)
def test_eval_security_missing_auth_service_denied_all_tools(eval_db_env, tool_cls, input_args):
    """Verify all 4 graph tools fail closed when auth_service is missing."""
    if tool_cls is SearchGraphEntitiesTool:
        tool = tool_cls(session_factory=eval_db_env["session_factory"])
    elif tool_cls in (ExploreGraphNeighborhoodTool, FindGraphPathTool):
        tool = tool_cls(eval_db_env["traversal_service"])
    else:
        tool = tool_cls(eval_db_env["analytics_service"])

    no_auth_context = AgentContext(
        user_id=USER_A_ID,
        auth_enforced=True,
        auth_service=None,
        session_factory=eval_db_env["session_factory"],
    )
    res = tool.execute(input_args, no_auth_context)
    assert res.success is False
    assert res.error_type == "unauthorized"
    assert "Authorization service unavailable" in res.error


# ---------------------------------------------------------------------------
# 7. Input Validation & Adversarial Boundaries
# ---------------------------------------------------------------------------

def test_eval_malformed_uuid_rejected(eval_db_env):
    """Verify non-UUID strings are rejected with validation_error."""
    tool = SearchGraphEntitiesTool(session_factory=eval_db_env["session_factory"])
    res = tool.execute(
        {"collection_id": "not-a-valid-uuid", "query": "Transformer"},
        eval_db_env["user_a_context"],
    )
    assert res.success is False
    assert res.error_type == "validation_error"


def test_eval_forbidden_extra_fields_rejected(eval_db_env):
    """Verify extra input fields are strictly rejected under extra='forbid'."""
    tool = SearchGraphEntitiesTool(session_factory=eval_db_env["session_factory"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "query": "Transformer",
            "forbidden_extra_payload": True,
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is False
    assert res.error_type == "validation_error"
    assert "extra" in res.error.lower()


def test_eval_invalid_direction_parameter_rejected(eval_db_env):
    """Verify invalid direction parameter values are rejected."""
    tool = ExploreGraphNeighborhoodTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "node_id": str(PAPER_1_ID),
            "node_type": "paper",
            "direction": "sideways",
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is False
    assert res.error_type == "validation_error"


def test_eval_empty_relationship_types_rejected(eval_db_env):
    """Verify empty list relationship_types=[] is rejected."""
    tool = FindGraphPathTool(eval_db_env["traversal_service"])
    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "source_id": str(PAPER_1_ID),
            "source_type": "paper",
            "target_id": str(ENTITY_VASWANI_ID),
            "target_type": "entity",
            "relationship_types": [],
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is False
    assert res.error_type == "validation_error"


def test_eval_author_id_rejected_on_overview_and_summary(eval_db_env):
    """Verify author_id filter is rejected for overview and summary analytics queries."""
    tool = QueryGraphAnalyticsTool(eval_db_env["analytics_service"])
    res_overview = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "query_type": "overview",
            "author_id": str(ENTITY_VASWANI_ID),
        },
        eval_db_env["user_a_context"],
    )
    assert res_overview.success is False
    assert res_overview.error_type == "validation_error"
    assert "author_id filter is not supported" in res_overview.error


# ---------------------------------------------------------------------------
# 8. Unexpected Service Failures Handling
# ---------------------------------------------------------------------------

def test_eval_unexpected_service_failure_fails_safely(eval_db_env, monkeypatch):
    """Verify unexpected low-level service exceptions are caught as execution_error."""
    tool = ExploreGraphNeighborhoodTool(eval_db_env["traversal_service"])

    def _exploding_traversal(*args, **kwargs):
        raise OSError("Simulated disk I/O hardware failure")

    monkeypatch.setattr(eval_db_env["traversal_service"], "explore_multihop", _exploding_traversal)

    res = tool.execute(
        {
            "collection_id": str(COL_A_ID),
            "node_id": str(PAPER_1_ID),
            "node_type": "paper",
        },
        eval_db_env["user_a_context"],
    )
    assert res.success is False
    assert res.error_type == "execution_error"
    assert "Neighborhood exploration failed: OSError" in res.error
