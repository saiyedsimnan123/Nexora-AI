"""API integration and security tests for Stage 5 Step 4.1 Knowledge Graph Endpoints.

Endpoints tested:
1. GET  /api/v1/collections/{collection_id}/graph/summary
2. GET  /api/v1/collections/{collection_id}/graph/analytics
3. GET  /api/v1/collections/{collection_id}/graph/entities
4. GET  /api/v1/collections/{collection_id}/graph/entities/{entity_id}
5. POST /api/v1/collections/{collection_id}/graph/neighborhood
6. POST /api/v1/collections/{collection_id}/graph/paths
7. GET  /api/v1/collections/{collection_id}/graph/collaborations
"""

from __future__ import annotations

import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.api.app import create_app
from nexora.auth.service import AuthenticationService
from nexora.database.models.api_key import ApiKey
from nexora.database.models.base import Base
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.graph_entity import GraphEntity
from nexora.database.models.graph_relationship import GraphRelationship
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.database.repository.graph_entity import GraphEntityRepository
from nexora.database.repository.graph_relationship import GraphRelationshipRepository
from nexora.database.repository.paper import PaperRepository
from nexora.graph.analytics_service import KnowledgeGraphAnalyticsService
from nexora.graph.traversal import KnowledgeGraphTraversalService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_session_factory():
    """In-memory SQLite session factory with schema initialized."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def seeded_graph_data(sqlite_session_factory):
    """Seed SQLite database with two isolated tenants, research papers, entities, and relationships."""
    session = sqlite_session_factory()
    try:
        # 1. Tenants & Workspaces
        user_a = User(email="tenant-a@example.com")
        user_b = User(email="tenant-b@example.com")
        session.add_all([user_a, user_b])
        session.flush()

        ws_a = Workspace(name="Workspace A", user_id=user_a.id)
        ws_b = Workspace(name="Workspace B", user_id=user_b.id)
        session.add_all([ws_a, ws_b])
        session.flush()

        col_a = Collection(name="Collection A", workspace_id=ws_a.id)
        col_b = Collection(name="Collection B", workspace_id=ws_b.id)
        session.add_all([col_a, col_b])
        session.flush()

        # 2. Papers & Documents for Collection A
        paper_repo = PaperRepository(session)
        p1 = paper_repo.create(
            collection_id=col_a.id,
            title="Attention is All You Need",
            abstract="The seminal transformer paper.",
            publication_date=datetime.date(2017, 6, 12),
        )
        p2 = paper_repo.create(
            collection_id=col_a.id,
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            abstract="Pre-training deep bidirectional representations.",
            publication_date=datetime.date(2018, 10, 11),
        )

        doc1 = Document(
            paper_id=p1.id,
            filename="vaswani2017.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="a" * 64,
            storage_key="s3://vaswani2017.pdf",
            indexing_status="completed",
        )
        session.add(doc1)
        session.flush()

        chunk1 = DocumentChunkRecord(
            document_id=doc1.id,
            chunk_index=0,
            text="The Transformer is the first transduction model relying entirely on self-attention.",
        )
        chunk2 = DocumentChunkRecord(
            document_id=doc1.id,
            chunk_index=1,
            text="Ashish Vaswani and Jacob Devlin discussed multi-head attention mechanisms.",
        )
        session.add_all([chunk1, chunk2])
        session.flush()

        # 3. Graph Entities in Collection A
        entity_repo = GraphEntityRepository(session)
        e_vaswani = entity_repo.create(
            collection_id=col_a.id,
            name="Ashish Vaswani",
            entity_type="author",
            description="Primary author of Transformer paper.",
        )
        e_devlin = entity_repo.create(
            collection_id=col_a.id,
            name="Jacob Devlin",
            entity_type="author",
            description="Primary author of BERT paper.",
        )
        e_transformer = entity_repo.create(
            collection_id=col_a.id,
            name="Transformer",
            entity_type="method",
            description="Self-attention neural sequence architecture.",
        )
        e_bert = entity_repo.create(
            collection_id=col_a.id,
            name="BERT",
            entity_type="method",
            description="Bidirectional Encoder Representations from Transformers.",
        )
        e_attention = entity_repo.create(
            collection_id=col_a.id,
            name="Self-Attention",
            entity_type="concept",
            description="Attention mechanism relating positions within a sequence.",
        )
        e_glue = entity_repo.create(
            collection_id=col_a.id,
            name="GLUE Benchmark",
            entity_type="dataset",
            description="General Language Understanding Evaluation benchmark.",
        )

        # 4. Graph Relationships in Collection A
        rel_repo = GraphRelationshipRepository(session)
        # Author collaboration edge: Vaswani <-> Devlin
        r_collab = rel_repo.create(
            collection_id=col_a.id,
            relationship_type="co_authored_with",
            source_type="entity",
            source_id=e_vaswani.id,
            target_type="entity",
            target_id=e_devlin.id,
            paper_id=p1.id,
            chunk_id=chunk2.id,
            page_number=1,
            evidence_text="Ashish Vaswani and Jacob Devlin discussed multi-head attention mechanisms.",
            confidence=0.92,
            extraction_method="llm",
        )
        # Structural edge: Transformer -> Self-Attention
        r_trans_att = rel_repo.create(
            collection_id=col_a.id,
            relationship_type="uses_mechanism",
            source_type="entity",
            source_id=e_transformer.id,
            target_type="entity",
            target_id=e_attention.id,
            paper_id=p1.id,
            chunk_id=chunk1.id,
            page_number=2,
            evidence_text="The Transformer relies entirely on self-attention.",
            confidence=0.95,
            extraction_method="llm",
        )
        # Structural edge: BERT -> Transformer
        r_bert_trans = rel_repo.create(
            collection_id=col_a.id,
            relationship_type="based_on",
            source_type="entity",
            source_id=e_bert.id,
            target_type="entity",
            target_id=e_transformer.id,
            paper_id=p2.id,
            confidence=0.88,
            evidence_text="BERT builds directly upon the Transformer architecture.",
        )
        # Structural edge: BERT -> GLUE
        r_bert_glue = rel_repo.create(
            collection_id=col_a.id,
            relationship_type="evaluated_on",
            source_type="entity",
            source_id=e_bert.id,
            target_type="entity",
            target_id=e_glue.id,
            paper_id=p2.id,
            confidence=0.85,
            evidence_text="BERT obtains state-of-the-art results on GLUE.",
        )
        # Paper attribution edges: Paper 1 -> Vaswani & Paper 1 -> Devlin (authored_by)
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p1.id,
            target_type="entity",
            target_id=e_vaswani.id,
            paper_id=p1.id,
            confidence=0.95,
            evidence_text="Ashish Vaswani authored Attention is All You Need.",
        )
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p1.id,
            target_type="entity",
            target_id=e_devlin.id,
            paper_id=p1.id,
            confidence=0.92,
            evidence_text="Jacob Devlin co-authored Attention is All You Need.",
        )

        # 5. Graph Entities & Relationships in Tenant B (for isolation testing)
        e_b = entity_repo.create(
            collection_id=col_b.id,
            name="Isolated Entity B",
            entity_type="method",
            description="Entity belonging strictly to Tenant B.",
        )
        e_b2 = entity_repo.create(
            collection_id=col_b.id,
            name="Isolated Entity B2",
            entity_type="dataset",
            description="Second entity belonging strictly to Tenant B.",
        )
        rel_repo.create(
            collection_id=col_b.id,
            relationship_type="tested_on",
            source_type="entity",
            source_id=e_b.id,
            target_type="entity",
            target_id=e_b2.id,
            confidence=0.75,
        )

        # 6. API Keys
        key_repo = ApiKeyRepository(session)
        _, raw_token_a = key_repo.generate_key(user_id=user_a.id, name="Key Tenant A")
        _, raw_token_b = key_repo.generate_key(user_id=user_b.id, name="Key Tenant B")

        session.commit()

        return {
            "user_a_id": user_a.id,
            "user_b_id": user_b.id,
            "col_a_id": col_a.id,
            "col_b_id": col_b.id,
            "token_a": raw_token_a,
            "token_b": raw_token_b,
            "p1_id": p1.id,
            "p2_id": p2.id,
            "e_vaswani_id": e_vaswani.id,
            "e_devlin_id": e_devlin.id,
            "e_transformer_id": e_transformer.id,
            "e_bert_id": e_bert.id,
            "e_attention_id": e_attention.id,
            "e_glue_id": e_glue.id,
            "e_b_id": e_b.id,
            "chunk1_id": chunk1.id,
            "chunk2_id": chunk2.id,
        }
    finally:
        session.close()


def _build_test_client(sqlite_session_factory, enforce_auth: bool = True) -> TestClient:
    """Helper composing TestClient with wired session factory and auth service."""
    auth_service = AuthenticationService(session_factory=sqlite_session_factory)
    traversal_service = KnowledgeGraphTraversalService(session_factory=sqlite_session_factory)
    analytics_service = KnowledgeGraphAnalyticsService(session_factory=sqlite_session_factory)

    class DummyResearchService:
        def ask(self, req: Any) -> Any:
            pass

    app = create_app(
        DummyResearchService(),
        session_factory=sqlite_session_factory,
        auth_service=auth_service,
        enforce_auth=enforce_auth,
        graph_traversal_service=traversal_service,
        graph_analytics_service=analytics_service,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. GET /api/v1/collections/{collection_id}/graph/summary Tests
# ---------------------------------------------------------------------------

def test_get_summary_success(sqlite_session_factory, seeded_graph_data):
    """Verify 200 OK and accurate entity/relation counts for an authorized collection."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(f"/api/v1/collections/{col_id}/graph/summary", headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["collection_id"] == str(col_id)
    assert data["total_entities"] == 6
    assert data["total_relationships"] == 6
    assert data["entity_counts_by_type"] == {
        "author": 2,
        "method": 2,
        "concept": 1,
        "dataset": 1,
    }
    assert "co_authored_with" in data["relationship_counts_by_type"]
    assert "uses_mechanism" in data["relationship_counts_by_type"]


def test_get_summary_unauthenticated(sqlite_session_factory, seeded_graph_data):
    """Verify 401 Unauthorized when no credentials are provided."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]

    resp = client.get(f"/api/v1/collections/{col_id}/graph/summary")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_get_summary_unauthenticated_even_when_enforce_auth_false(sqlite_session_factory, seeded_graph_data):
    """Verify 401 Unauthorized for private collection graph access even if enforce_auth=False."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=False)
    col_id = seeded_graph_data["col_a_id"]

    resp = client.get(f"/api/v1/collections/{col_id}/graph/summary")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_get_summary_cross_tenant_forbidden(sqlite_session_factory, seeded_graph_data):
    """Verify 403 Forbidden when Tenant B attempts to access Tenant A's collection summary."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_a_id = seeded_graph_data["col_a_id"]
    headers_b = {"Authorization": f"Bearer {seeded_graph_data['token_b']}"}

    resp = client.get(f"/api/v1/collections/{col_a_id}/graph/summary", headers=headers_b)
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"


def test_get_summary_nonexistent_collection(sqlite_session_factory, seeded_graph_data):
    """Verify 404 Not Found for non-existent collection."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    missing_id = uuid4()
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(f"/api/v1/collections/{missing_id}/graph/summary", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_get_summary_invalid_uuid(sqlite_session_factory, seeded_graph_data):
    """Verify 400 Bad Request for malformed collection UUID."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get("/api/v1/collections/not-a-valid-uuid/graph/summary", headers=headers)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# 2. GET /api/v1/collections/{collection_id}/graph/analytics Tests
# ---------------------------------------------------------------------------

def test_get_analytics_success(sqlite_session_factory, seeded_graph_data):
    """Verify 200 OK and complete analytics payload with distributions and author metrics."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/analytics?top_collaborations_limit=10&author_connectivity_limit=10",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["collection_id"] == str(col_id)
    assert data["summary"]["total_entities"] == 6
    assert len(data["entity_distribution"]) > 0
    assert len(data["relationship_distribution"]) > 0

    # Top collaborations check
    assert len(data["top_collaborations"]) == 1
    collab = data["top_collaborations"][0]
    assert collab["author_name"] in ("Ashish Vaswani", "Jacob Devlin")
    assert collab["collaborator_name"] in ("Ashish Vaswani", "Jacob Devlin")
    assert collab["collaboration_count"] == 1
    assert str(seeded_graph_data["p1_id"]) in collab["shared_paper_ids"]

    # Author connectivity check
    assert len(data["author_connectivity"]) == 2
    author_names = {m["author_name"] for m in data["author_connectivity"]}
    assert "Ashish Vaswani" in author_names
    assert "Jacob Devlin" in author_names


def test_get_analytics_min_confidence_filter(sqlite_session_factory, seeded_graph_data):
    """Verify min_confidence parameter filters author collaboration records."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    # Edge has confidence 0.92. Filter with 0.95 should exclude it from collaborations
    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/analytics?min_confidence=0.95",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["top_collaborations"]) == 0


def test_get_analytics_invalid_min_confidence(sqlite_session_factory, seeded_graph_data):
    """Verify 422 Unprocessable Entity when min_confidence is outside [0.0, 1.0]."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/analytics?min_confidence=1.5",
        headers=headers,
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. GET /api/v1/collections/{collection_id}/graph/entities Tests
# ---------------------------------------------------------------------------

def test_list_entities_success(sqlite_session_factory, seeded_graph_data):
    """Verify listing entities with pagination and type filtering."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    # All entities
    resp = client.get(f"/api/v1/collections/{col_id}/graph/entities?limit=10", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] == 6
    assert len(data["entities"]) == 6
    assert data["offset"] == 0
    assert data["limit"] == 10

    # Filtered by entity_type="method"
    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/entities?entity_type=method",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] == 2
    assert {e["name"] for e in data["entities"]} == {"Transformer", "BERT"}


def test_search_entities_by_query(sqlite_session_factory, seeded_graph_data):
    """Verify substring entity search within an authorized collection."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/entities?query=trans&limit=10",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    names = [e["name"] for e in data["entities"]]
    assert "Transformer" in names
    assert len(data["entities"]) == 1


def test_search_entities_enforces_max_limit_20(sqlite_session_factory, seeded_graph_data):
    """Verify that search mode strictly rejects limit > 20 with 400 Bad Request."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/entities?query=trans&limit=25",
        headers=headers,
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# 4. GET /api/v1/collections/{collection_id}/graph/entities/{entity_id} Tests
# ---------------------------------------------------------------------------

def test_get_entity_detail_success(sqlite_session_factory, seeded_graph_data):
    """Verify 200 OK, entity metadata, and direct relationship edge serialization."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    bert_id = seeded_graph_data["e_bert_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/entities/{bert_id}",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"] == str(bert_id)
    assert data["name"] == "BERT"
    assert data["entity_type"] == "method"
    assert data["relationship_count"] == 2
    rel_types = {r["relationship_type"] for r in data["relationships"]}
    assert "based_on" in rel_types
    assert "evaluated_on" in rel_types


def test_get_entity_detail_not_found(sqlite_session_factory, seeded_graph_data):
    """Verify 404 Not Found when entity does not exist in the collection."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    missing_ent_id = uuid4()
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/entities/{missing_ent_id}",
        headers=headers,
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_get_entity_detail_cross_collection_isolation(sqlite_session_factory, seeded_graph_data):
    """Verify 404 Not Found when querying Tenant B's entity through Tenant A's collection."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_a_id = seeded_graph_data["col_a_id"]
    ent_b_id = seeded_graph_data["e_b_id"]
    headers_a = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(
        f"/api/v1/collections/{col_a_id}/graph/entities/{ent_b_id}",
        headers=headers_a,
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. POST /api/v1/collections/{collection_id}/graph/neighborhood Tests
# ---------------------------------------------------------------------------

def test_explore_neighborhood_success_with_evidence(sqlite_session_factory, seeded_graph_data):
    """Verify neighborhood exploration preserves chunk_id, page_number, confidence, and evidence_text."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    transformer_id = seeded_graph_data["e_transformer_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    payload = {
        "node_id": str(transformer_id),
        "node_type": "entity",
        "max_depth": 1,
        "max_nodes": 50,
        "max_edges": 100,
        "direction": "both",
        "include_unscored": True,
    }
    resp = client.post(
        f"/api/v1/collections/{col_id}/graph/neighborhood",
        json=payload,
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["seed_id"] == str(transformer_id)
    assert data["seed_type"] == "entity"
    assert data["depth"] == 1
    assert data["is_truncated"] is False

    node_names = {n["name"] for n in data["nodes"]}
    assert "Transformer" in node_names
    assert "Self-Attention" in node_names
    assert "BERT" in node_names

    # Check evidence provenance preservation
    uses_edges = [e for e in data["edges"] if e["relationship_type"] == "uses_mechanism"]
    assert len(uses_edges) == 1
    edge = uses_edges[0]
    assert edge["confidence"] == 0.95
    assert edge["page_number"] == 2
    assert edge["chunk_id"] == str(seeded_graph_data["chunk1_id"])
    assert edge["paper_id"] == str(seeded_graph_data["p1_id"])
    assert "relies entirely on self-attention" in edge["evidence_text"]


def test_explore_neighborhood_depth_hard_limit(sqlite_session_factory, seeded_graph_data):
    """Verify that max_depth > 3 is rejected by request validation with 422."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    payload = {
        "node_id": str(seeded_graph_data["e_transformer_id"]),
        "node_type": "entity",
        "max_depth": 4,  # Exceeds max 3
    }
    resp = client.post(
        f"/api/v1/collections/{col_id}/graph/neighborhood",
        json=payload,
        headers=headers,
    )
    assert resp.status_code == 422


def test_explore_neighborhood_forbids_extra_fields(sqlite_session_factory, seeded_graph_data):
    """Verify that extra unexpected fields are rejected with 422 (extra='forbid')."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    payload = {
        "node_id": str(seeded_graph_data["e_transformer_id"]),
        "node_type": "entity",
        "extra_field_not_allowed": "malicious_payload",
    }
    resp = client.post(
        f"/api/v1/collections/{col_id}/graph/neighborhood",
        json=payload,
        headers=headers,
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. POST /api/v1/collections/{collection_id}/graph/paths Tests
# ---------------------------------------------------------------------------

def test_find_paths_success_with_confidence(sqlite_session_factory, seeded_graph_data):
    """Verify pathfinding discovers multi-hop paths and computes multiplicative confidence score."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    bert_id = seeded_graph_data["e_bert_id"]
    attention_id = seeded_graph_data["e_attention_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    # BERT -> Transformer -> Self-Attention (2 hops, outgoing)
    payload = {
        "source_id": str(bert_id),
        "source_type": "entity",
        "target_id": str(attention_id),
        "target_type": "entity",
        "max_depth": 3,
        "max_paths": 5,
        "direction": "outgoing",
        "include_unscored": True,
    }
    resp = client.post(
        f"/api/v1/collections/{col_id}/graph/paths",
        json=payload,
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["path_count"] == 1
    path = data["paths"][0]
    assert path["length"] == 2
    assert len(path["nodes"]) == 3
    assert [n["name"] for n in path["nodes"]] == ["BERT", "Transformer", "Self-Attention"]
    assert len(path["edges"]) == 2

    # Confidence score: 0.88 * 0.95 = 0.836
    assert path["confidence_score"] == pytest.approx(0.88 * 0.95, rel=1e-3)
    assert path["is_partially_unscored"] is False

    # Check evidence provenance on second edge
    e2 = path["edges"][1]
    assert e2["relationship_type"] == "uses_mechanism"
    assert e2["page_number"] == 2
    assert e2["chunk_id"] == str(seeded_graph_data["chunk1_id"])


def test_find_paths_limits_enforced(sqlite_session_factory, seeded_graph_data):
    """Verify path depth > 3 and paths > 10 are rejected with 422."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    # Depth > 3
    payload = {
        "source_id": str(seeded_graph_data["e_bert_id"]),
        "source_type": "entity",
        "target_id": str(seeded_graph_data["e_attention_id"]),
        "target_type": "entity",
        "max_depth": 5,
    }
    resp = client.post(f"/api/v1/collections/{col_id}/graph/paths", json=payload, headers=headers)
    assert resp.status_code == 422

    # Paths > 10
    payload["max_depth"] = 2
    payload["max_paths"] = 15
    resp = client.post(f"/api/v1/collections/{col_id}/graph/paths", json=payload, headers=headers)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 7. GET /api/v1/collections/{collection_id}/graph/collaborations Tests
# ---------------------------------------------------------------------------

def test_get_collaborations_success(sqlite_session_factory, seeded_graph_data):
    """Verify 200 OK and co-authorship collaboration aggregations."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/collaborations?min_shared_papers=1",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["collection_id"] == str(col_id)
    assert data["total_count"] == 1
    c = data["collaborations"][0]
    assert c["collaboration_count"] == 1
    assert str(seeded_graph_data["p1_id"]) in c["shared_paper_ids"]


def test_get_collaborations_filter_by_author_id(sqlite_session_factory, seeded_graph_data):
    """Verify filtering collaborations by specific author_id."""
    client = _build_test_client(sqlite_session_factory, enforce_auth=True)
    col_id = seeded_graph_data["col_a_id"]
    vaswani_id = seeded_graph_data["e_vaswani_id"]
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.get(
        f"/api/v1/collections/{col_id}/graph/collaborations?author_id={vaswani_id}",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] == 1
    assert data["collaborations"][0]["author_name"] == "Ashish Vaswani"


# ---------------------------------------------------------------------------
# 8. Read-Only Invariance & Caller Session Tests
# ---------------------------------------------------------------------------

def test_read_only_invariance_on_caller_session(sqlite_session_factory, seeded_graph_data):
    """Verify that graph service executions preserve active caller transactions without commits/rollbacks."""
    session = sqlite_session_factory()
    try:
        col_id = seeded_graph_data["col_a_id"]
        analytics_service = KnowledgeGraphAnalyticsService(session_factory=sqlite_session_factory)
        traversal_service = KnowledgeGraphTraversalService(session_factory=sqlite_session_factory)

        assert session.is_active is True
        summary = analytics_service.get_summary(col_id, session=session)
        assert summary.total_entities == 6
        assert session.is_active is True

        neighborhood = traversal_service.get_neighborhood(
            seeded_graph_data["e_transformer_id"],
            "entity",
            col_id,
            session=session,
        )
        assert len(neighborhood.nodes) > 0
        assert session.is_active is True
    finally:
        session.close()
