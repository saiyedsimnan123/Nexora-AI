"""End-to-End Cross-Module Integration Tests for Knowledge Graph (Stage 5 Step 3.6).

Verifies the complete integration across all Stage 5 modules:
1. Extraction -> Repository (Structured & LLM extraction, evidence grounding, idempotence).
2. Repository -> Traversal (Neighborhood exploration, multi-hop BFS, pathfinding, endpoint invariants).
3. Repository -> Analytics (Distributions, author collaboration, connectivity, collection summary).
4. Cross-Module Consistency (Repository counts vs analytics summaries, traversal visibility vs persisted edges).
5. Strict Invariants (Collection isolation, compound identity, read-only session invariance).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.database.models import (
    Base,
    Collection,
    Document,
    DocumentChunkRecord,
    GraphEndpointType,
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
    PaperRepository,
)
from nexora.graph.analytics import (
    AuthorCollaborationRecord,
    AuthorConnectivityMetrics,
    AuthorPaperConnection,
    CollectionAnalyticsResult,
    CollectionGraphSummary,
    EntityTypeCount,
    RelationshipTypeCount,
)
from nexora.graph.analytics_service import KnowledgeGraphAnalyticsService
from nexora.graph.models import (
    ExtractedEntityCandidate,
    ExtractedGraphPayload,
    ExtractedRelationshipCandidate,
    GraphExtractionResult,
)
from nexora.graph.service import KnowledgeGraphExtractionService
from nexora.graph.traversal import KnowledgeGraphTraversalService
from nexora.graph.traversal_models import (
    GraphEdge,
    GraphNode,
    GraphPath,
    GraphSubgraph,
)


@pytest.fixture
def engine():
    """SQLite in-memory engine with PRAGMA foreign_keys=ON and StaticPool."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def session(session_factory):
    with session_factory() as sess:
        yield sess


from nexora.llm.models import LLMMessage, LLMResponse


class MockIntegrationLLMService:
    """Mock LLM service for cross-module integration tests."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.call_count += 1
        payload = {
            "entities": [
                {
                    "name": "Transformer Architecture",
                    "entity_type": "concept",
                    "description": "Novel sequence transduction architecture based entirely on attention.",
                    "aliases": ["Transformer"],
                },
                {
                    "name": "Scaled Dot-Product Attention",
                    "entity_type": "method",
                    "description": "An attention function computing dot products of queries with keys.",
                    "aliases": ["Self-Attention"],
                },
                {
                    "name": "WMT 2014 English-to-German",
                    "entity_type": "dataset",
                    "description": "Standard translation benchmark dataset.",
                    "aliases": ["WMT14 En-De"],
                },
            ],
            "relationships": [
                {
                    "source_name": "Attention Is All You Need",
                    "source_type": "paper",
                    "relationship_type": "introduces",
                    "target_name": "Transformer Architecture",
                    "target_type": "entity",
                    "confidence": 0.98,
                    "evidence_label": "[cite: 0]",
                },
                {
                    "source_name": "Transformer Architecture",
                    "source_type": "entity",
                    "relationship_type": "uses_method",
                    "target_name": "Scaled Dot-Product Attention",
                    "target_type": "entity",
                    "confidence": 0.95,
                    "evidence_label": "[cite: 0]",
                },
                {
                    "source_name": "Attention Is All You Need",
                    "source_type": "paper",
                    "relationship_type": "evaluates_on",
                    "target_name": "WMT 2014 English-to-German",
                    "target_type": "entity",
                    "confidence": 0.90,
                    "evidence_label": "[cite: 1]",
                },
            ],
        }
        return LLMResponse(text=json.dumps(payload), model="mock-integration-model")


@pytest.fixture
def mock_llm_service():
    return MockIntegrationLLMService()


@pytest.fixture
def seeded_environment(session):
    """Seed multi-collection research environment with papers, documents, and chunks."""
    user = User(email="lead_researcher@nexora.ai")
    session.add(user)
    session.flush()

    workspace = Workspace(user_id=user.id, name="Nexora Research Workspace")
    session.add(workspace)
    session.flush()

    col_nlp = Collection(workspace_id=workspace.id, name="NLP Foundations")
    col_vision = Collection(workspace_id=workspace.id, name="Vision Systems")
    col_empty = Collection(workspace_id=workspace.id, name="Empty Sandbox")
    session.add_all([col_nlp, col_vision, col_empty])
    session.flush()

    # Paper 1 in NLP: Vaswani et al.
    p1 = Paper(
        collection_id=col_nlp.id,
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
    )
    # Paper 2 in NLP: Devlin et al.
    p2 = Paper(
        collection_id=col_nlp.id,
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        abstract="We introduce a new language representation model called BERT...",
    )
    # Paper 3 in Vision: He et al.
    p3 = Paper(
        collection_id=col_vision.id,
        title="Deep Residual Learning for Image Recognition",
        abstract="Deeper neural networks are more difficult to train...",
    )
    session.add_all([p1, p2, p3])
    session.flush()

    # Documents and Chunks for Paper 1
    doc1 = Document(
        paper_id=p1.id,
        filename="vaswani2017.pdf",
        mime_type="application/pdf",
        file_size=10240,
        checksum="1" * 64,
        storage_key="docs/vaswani2017.pdf",
        indexing_status=IndexingStatus.COMPLETED.value,
    )
    session.add(doc1)
    session.flush()

    chunk1 = DocumentChunkRecord(
        document_id=doc1.id,
        chunk_index=0,
        text="The Transformer is the first transduction model relying entirely on self-attention to compute representations.",
    )
    chunk2 = DocumentChunkRecord(
        document_id=doc1.id,
        chunk_index=1,
        text="On the WMT 2014 English-to-German translation task, the big transformer model achieves state-of-the-art BLEU score of 28.4.",
    )
    session.add_all([chunk1, chunk2])
    session.commit()

    return {
        "user": user,
        "workspace": workspace,
        "col_nlp": col_nlp,
        "col_vision": col_vision,
        "col_empty": col_empty,
        "paper_vaswani": p1,
        "paper_bert": p2,
        "paper_resnet": p3,
        "chunk1": chunk1,
        "chunk2": chunk2,
    }


def test_01_end_to_end_extraction_to_repository_flow(session, session_factory, seeded_environment, mock_llm_service):
    """A. Extraction -> Repository: Extracts graph from paper, verifies persistence and grounded evidence."""
    col_id = seeded_environment["col_nlp"].id
    p1 = seeded_environment["paper_vaswani"]

    extraction_svc = KnowledgeGraphExtractionService(
        llm_service=mock_llm_service,
        session_factory=session_factory,
    )

    result = extraction_svc.extract_paper(
        paper_id=p1.id,
        collection_id=col_id,
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        session=session,
    )

    assert result.paper_id == p1.id
    assert result.collection_id == col_id
    assert result.entities_created > 0
    assert result.relationships_created > 0
    assert len(result.entities) >= 5  # 3 authors + 3 concepts/methods/datasets
    assert len(result.relationships) >= 5  # 3 authored_by + 3 extracted rels

    session.commit()

    # Verify repository persistence
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    entities = entity_repo.list_by_collection_id(col_id, limit=50)
    assert len(entities) == result.entities_created

    # Verify author entities exist
    author_names = {e.name for e in entities if e.entity_type == "author"}
    assert "Ashish Vaswani" in author_names
    assert "Noam Shazeer" in author_names
    assert "Niki Parmar" in author_names

    # Verify relationships and evidence grounding
    rels = rel_repo.list_by_collection_id(col_id, limit=50)
    assert len(rels) == result.relationships_created

    eval_rel = next(r for r in rels if r.relationship_type == "evaluates_on")
    assert eval_rel.chunk_id == seeded_environment["chunk2"].id
    assert eval_rel.evidence_text is not None
    assert "BLEU score" in eval_rel.evidence_text
    assert eval_rel.confidence == 0.90


def test_02_idempotent_repeated_extraction(session, session_factory, seeded_environment, mock_llm_service):
    """A. Extraction -> Repository: Repeated extraction against unchanged data is strictly idempotent."""
    col_id = seeded_environment["col_nlp"].id
    p1 = seeded_environment["paper_vaswani"]

    extraction_svc = KnowledgeGraphExtractionService(
        llm_service=mock_llm_service,
        session_factory=session_factory,
    )

    # First run
    res1 = extraction_svc.extract_paper(
        paper_id=p1.id,
        collection_id=col_id,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        session=session,
    )
    session.commit()

    created_entities_1 = res1.entities_created
    created_rels_1 = res1.relationships_created

    # Second run with exact same data
    res2 = extraction_svc.extract_paper(
        paper_id=p1.id,
        collection_id=col_id,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        session=session,
    )
    session.commit()

    # Reused counters incremented, zero new entities or relationships created
    assert res2.entities_created == 0
    assert res2.relationships_created == 0
    assert res2.entities_reused == created_entities_1
    assert res2.relationships_reused == created_rels_1


def test_03_repository_to_traversal_workflow(session, session_factory, seeded_environment, mock_llm_service):
    """B. Repository -> Traversal: Traverses extracted graph topology, verifies compound keys and invariants."""
    col_id = seeded_environment["col_nlp"].id
    p1 = seeded_environment["paper_vaswani"]

    extraction_svc = KnowledgeGraphExtractionService(
        llm_service=mock_llm_service,
        session_factory=session_factory,
    )
    _ = extraction_svc.extract_paper(
        paper_id=p1.id,
        collection_id=col_id,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        session=session,
    )
    session.commit()

    traversal_svc = KnowledgeGraphTraversalService(session_factory=session_factory)

    # 1-hop neighborhood from paper seed node
    neighborhood = traversal_svc.get_neighborhood(
        node_id=p1.id,
        node_type="paper",
        collection_id=col_id,
        session=session,
    )
    assert isinstance(neighborhood, GraphSubgraph)
    assert neighborhood.seed_id == p1.id
    assert neighborhood.seed_type == "paper"
    assert len(neighborhood.nodes) >= 4  # Seed paper + 2 authors + Transformer Architecture + WMT14
    assert len(neighborhood.edges) >= 3  # authored_by + introduces + evaluates_on

    # Multi-hop BFS from paper through concept to method
    multihop = traversal_svc.explore_multihop(
        start_id=p1.id,
        start_type="paper",
        collection_id=col_id,
        max_depth=2,
        session=session,
    )
    assert multihop.depth == 2
    # Reachable nodes includes the method "Scaled Dot-Product Attention" at hop 2
    node_names = {n.name for n in multihop.nodes}
    assert "Scaled Dot-Product Attention" in node_names

    # Path finding: Paper -> Method
    method_node = next(n for n in multihop.nodes if n.name == "Scaled Dot-Product Attention")
    paths = traversal_svc.find_paths(
        source_id=p1.id,
        source_type="paper",
        target_id=method_node.id,
        target_type="entity",
        collection_id=col_id,
        max_depth=3,
        session=session,
    )
    assert len(paths) == 1
    path = paths[0]
    assert path.length == 2
    assert path.start_node.name == "Attention Is All You Need"
    assert path.end_node.name == "Scaled Dot-Product Attention"
    # Preserved path confidence: 0.98 * 0.95 = 0.931
    assert path.confidence_score is not None
    assert pytest.approx(path.confidence_score, 0.001) == 0.98 * 0.95
    assert path.is_partially_unscored is False


def test_04_repository_to_analytics_workflow(session, session_factory, seeded_environment, mock_llm_service):
    """C. Repository -> Analytics: Calculates summary, distributions, co-authorship, and connectivity."""
    col_id = seeded_environment["col_nlp"].id
    p1 = seeded_environment["paper_vaswani"]
    p2 = seeded_environment["paper_bert"]

    extraction_svc = KnowledgeGraphExtractionService(
        llm_service=mock_llm_service,
        session_factory=session_factory,
    )

    # Extract P1: authored by Vaswani and Shazeer
    _ = extraction_svc.extract_paper(
        paper_id=p1.id,
        collection_id=col_id,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        session=session,
    )

    # Extract P2: authored by Shazeer and Jacob Devlin
    _ = extraction_svc.extract_paper(
        paper_id=p2.id,
        collection_id=col_id,
        authors=["Noam Shazeer", "Jacob Devlin"],
        session=session,
    )
    session.commit()

    analytics_svc = KnowledgeGraphAnalyticsService(session_factory=session_factory)

    # Summary
    summary = analytics_svc.get_summary(col_id, session=session)
    assert isinstance(summary, CollectionGraphSummary)
    assert summary.total_entities >= 4  # Vaswani, Shazeer, Devlin + extracted concepts
    assert summary.entity_counts_by_type["author"] == 3

    # Author collaborations: Shazeer collaborated with Vaswani (on P1) and Devlin (on P2)
    collabs = analytics_svc.get_author_collaborations(col_id, session=session)
    assert len(collabs) == 2

    # Author connectivity
    connectivity = analytics_svc.get_author_connectivity(col_id, session=session)
    assert len(connectivity) == 3

    m_shazeer = next(m for m in connectivity if m.author_name == "Noam Shazeer")
    assert m_shazeer.paper_count == 2
    assert m_shazeer.collaborator_count == 2  # Vaswani and Devlin
    assert m_shazeer.total_collaborations == 2

    m_vaswani = next(m for m in connectivity if m.author_name == "Ashish Vaswani")
    assert m_vaswani.paper_count == 1
    assert m_vaswani.collaborator_count == 1  # Shazeer
    assert m_vaswani.total_collaborations == 1


def test_05_cross_module_consistency_and_collection_isolation(session, session_factory, seeded_environment, mock_llm_service):
    """D. Cross-Module Consistency: Cross-validates repository counts, traversal edges, and analytics summaries."""
    col_nlp_id = seeded_environment["col_nlp"].id
    col_vision_id = seeded_environment["col_vision"].id
    col_empty_id = seeded_environment["col_empty"].id
    p1 = seeded_environment["paper_vaswani"]

    extraction_svc = KnowledgeGraphExtractionService(
        llm_service=mock_llm_service,
        session_factory=session_factory,
    )
    analytics_svc = KnowledgeGraphAnalyticsService(session_factory=session_factory)
    traversal_svc = KnowledgeGraphTraversalService(session_factory=session_factory)

    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Extract NLP graph
    _ = extraction_svc.extract_paper(
        paper_id=p1.id,
        collection_id=col_nlp_id,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        session=session,
    )
    session.commit()

    # 1. Compare Repository counts with Analytics summaries
    repo_entity_count = entity_repo.count_by_collection(col_nlp_id)
    repo_rel_count = rel_repo.count_by_collection(col_nlp_id)
    analytics_summary = analytics_svc.get_summary(col_nlp_id, session=session)
    composed_analytics = analytics_svc.get_collection_analytics(col_nlp_id, session=session)

    assert analytics_summary.total_entities == repo_entity_count
    assert analytics_summary.total_relationships == repo_rel_count
    assert composed_analytics.summary.total_entities == repo_entity_count
    assert composed_analytics.summary.total_relationships == repo_rel_count

    # 2. Compare Extracted relationships with Traversal-visible edges
    all_persisted_edges = rel_repo.list_by_collection_id(col_nlp_id, limit=100)
    neighborhood = traversal_svc.get_neighborhood(
        node_id=p1.id,
        node_type="paper",
        collection_id=col_nlp_id,
        session=session,
    )
    # Every 1-hop traversal edge must correspond to a persisted relationship
    persisted_edge_ids = {r.id for r in all_persisted_edges}
    for edge in neighborhood.edges:
        assert edge.id in persisted_edge_ids
        assert edge.paper_id == p1.id or edge.source_id == p1.id or edge.target_id == p1.id

    # 3. Compare Author-Paper connections with Author Connectivity metrics
    author_connections = analytics_svc.get_author_paper_connections(col_nlp_id, session=session)
    author_connectivity = analytics_svc.get_author_connectivity(col_nlp_id, session=session)

    # Both Vaswani and Shazeer have 1 connection to P1
    for author_metric in author_connectivity:
        conns_for_author = [c for c in author_connections if c.author_id == author_metric.author_id]
        assert author_metric.paper_count == len(conns_for_author)

    # 4. Collection Isolation across All 3 Services
    # Vision collection remains pristine
    assert entity_repo.count_by_collection(col_vision_id) == 0
    assert rel_repo.count_by_collection(col_vision_id) == 0
    assert analytics_svc.get_summary(col_vision_id, session=session).is_empty is True
    assert analytics_svc.get_author_collaborations(col_vision_id, session=session) == []

    # Empty collection remains pristine
    assert entity_repo.count_by_collection(col_empty_id) == 0
    assert rel_repo.count_by_collection(col_empty_id) == 0
    assert analytics_svc.get_summary(col_empty_id, session=session).is_empty is True

    # Traversal across wrong collection raises RecordNotFoundError
    with pytest.raises(Exception):
        traversal_svc.get_neighborhood(
            node_id=p1.id,
            node_type="paper",
            collection_id=col_vision_id,  # P1 is not in vision collection
            session=session,
        )


def test_06_invariants_caller_session_and_provenance_isolation(session, session_factory, seeded_environment):
    """E. Invariants: Caller session invariance and non-endpoint assertion protection."""
    col_id = seeded_environment["col_nlp"].id
    p1 = seeded_environment["paper_vaswani"]

    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    c1 = entity_repo.create(collection_id=col_id, name="Attention Method", entity_type="method")
    d1 = entity_repo.create(collection_id=col_id, name="Test Benchmark", entity_type="dataset")
    session.commit()

    # Create relationship between C1 and D1 where paper_id=p1.id is asserting provenance
    rel_repo.create(
        collection_id=col_id,
        relationship_type="evaluates_on",
        source_type="entity",
        source_id=c1.id,
        target_type="entity",
        target_id=d1.id,
        paper_id=p1.id,  # Asserting provenance, NOT an endpoint
    )
    session.commit()

    analytics_svc = KnowledgeGraphAnalyticsService(session_factory=session_factory)
    traversal_svc = KnowledgeGraphTraversalService(session_factory=session_factory)

    # 1. paper_id must never appear as a traversal endpoint
    neighborhood_p1 = traversal_svc.get_neighborhood(
        node_id=p1.id,
        node_type="paper",
        collection_id=col_id,
        session=session,
    )
    # The C1 -> D1 edge must NOT connect to P1
    assert len(neighborhood_p1.edges) == 0

    # 2. paper_id must never be counted as an author connection
    conns = analytics_svc.get_author_paper_connections(col_id, session=session)
    assert conns == []

    # 3. Caller-owned session invariance
    probe = Paper(collection_id=col_id, title="Probe Paper Uncommitted")
    session.add(probe)
    session.flush()

    assert probe in session.identity_map.values()
    assert session.is_active is True

    with patch.object(session, "commit", wraps=session.commit) as mock_commit, \
         patch.object(session, "rollback", wraps=session.rollback) as mock_rollback, \
         patch.object(session, "close", wraps=session.close) as mock_close:

        _ = analytics_svc.get_collection_analytics(col_id, session=session)
        _ = traversal_svc.explore_multihop(
            start_id=c1.id,
            start_type="entity",
            collection_id=col_id,
            session=session,
        )

        mock_commit.assert_not_called()
        mock_rollback.assert_not_called()
        mock_close.assert_not_called()

    assert session.is_active is True
    session.rollback()
    assert session.get(Paper, probe.id) is None
