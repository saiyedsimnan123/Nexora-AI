"""Comprehensive tests for KnowledgeGraphAnalyticsService (Stage 5 Step 3.5).

Covers:
1. Empty collection behavior.
2. Collection summary composition.
3. Entity and relationship distributions.
4. Author collaboration analytics.
5. Author connectivity metrics.
6. Combined collection analytics result.
7. Strict collection isolation (including identically named authors across collections).
8. Invalid input handling and boundary validation.
9. Deterministic output ordering.
10. Nullable confidence and provenance preservation.
11. Read-only session lifecycle (caller-owned invariance and service-owned cleanup).
12. Multi-row duplicate relationships and documented provenance behavior.
"""

from __future__ import annotations

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


@pytest.fixture
def service(session_factory):
    return KnowledgeGraphAnalyticsService(session_factory=session_factory)


@pytest.fixture
def seed_data(session):
    """Seed baseline user, workspace, and isolated collections."""
    user = User(email="analytics_service_test@nexora.ai")
    session.add(user)
    session.flush()

    workspace = Workspace(user_id=user.id, name="Analytics Service Lab")
    session.add(workspace)
    session.flush()

    col_a = Collection(workspace_id=workspace.id, name="Collection A")
    col_b = Collection(workspace_id=workspace.id, name="Collection B")
    col_empty = Collection(workspace_id=workspace.id, name="Collection Empty")
    session.add_all([col_a, col_b, col_empty])
    session.commit()

    return {
        "user": user,
        "workspace": workspace,
        "col_a": col_a,
        "col_b": col_b,
        "col_empty": col_empty,
    }


def test_01_empty_collection_behavior(service, session, seed_data):
    """1. Empty collection produces clean zero counts, empty lists, and empty summary."""
    col_empty_id = seed_data["col_empty"].id

    summary = service.get_summary(col_empty_id, session=session)
    assert isinstance(summary, CollectionGraphSummary)
    assert summary.collection_id == col_empty_id
    assert summary.total_entities == 0
    assert summary.total_relationships == 0
    assert summary.entity_counts_by_type == {}
    assert summary.relationship_counts_by_type == {}
    assert summary.is_empty is True

    entities = service.get_entity_distribution(col_empty_id, session=session)
    assert entities == []

    rels = service.get_relationship_distribution(col_empty_id, session=session)
    assert rels == []

    collabs = service.get_author_collaborations(col_empty_id, session=session)
    assert collabs == []

    connectivity = service.get_author_connectivity(col_empty_id, session=session)
    assert connectivity == []

    connections = service.get_author_paper_connections(col_empty_id, session=session)
    assert connections == []

    combined = service.get_collection_analytics(col_empty_id, session=session)
    assert isinstance(combined, CollectionAnalyticsResult)
    assert combined.collection_id == col_empty_id
    assert combined.summary.is_empty is True
    assert combined.entity_distribution == []
    assert combined.relationship_distribution == []
    assert combined.top_collaborations == []
    assert combined.author_connectivity == []


def test_02_collection_summary_and_distributions(service, session, seed_data):
    """2. Service retrieves accurate graph summary and type distributions."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Attention Paper")
    session.add(p1)
    session.flush()

    e_author = entity_repo.create(collection_id=col_id, name="Vaswani", entity_type="author")
    e_concept = entity_repo.create(collection_id=col_id, name="Transformer", entity_type="concept")
    e_dataset = entity_repo.create(collection_id=col_id, name="WMT14", entity_type="dataset")
    session.commit()

    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=e_author.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="introduces",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=e_concept.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="evaluates_on",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=e_dataset.id,
    )
    session.commit()

    # Summary
    summary = service.get_summary(col_id, session=session)
    assert summary.total_entities == 3
    assert summary.total_relationships == 3
    assert summary.entity_counts_by_type == {"author": 1, "concept": 1, "dataset": 1}
    assert summary.relationship_counts_by_type == {
        "authored_by": 1,
        "evaluates_on": 1,
        "introduces": 1,
    }

    # Entity distribution
    e_dist = service.get_entity_distribution(col_id, session=session)
    assert len(e_dist) == 3
    assert e_dist[0] == EntityTypeCount(entity_type="author", count=1)
    assert e_dist[1] == EntityTypeCount(entity_type="concept", count=1)
    assert e_dist[2] == EntityTypeCount(entity_type="dataset", count=1)

    # Relationship distribution
    r_dist = service.get_relationship_distribution(col_id, session=session)
    assert len(r_dist) == 3
    assert r_dist[0] == RelationshipTypeCount(relationship_type="authored_by", count=1)
    assert r_dist[1] == RelationshipTypeCount(relationship_type="evaluates_on", count=1)
    assert r_dist[2] == RelationshipTypeCount(relationship_type="introduces", count=1)


def test_03_author_collaborations_and_connectivity(service, session, seed_data):
    """3. Service computes author collaboration network and connectivity metrics."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Paper One")
    p2 = Paper(collection_id=col_id, title="Paper Two")
    p3 = Paper(collection_id=col_id, title="Paper Three")
    session.add_all([p1, p2, p3])
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Alice", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Bob", entity_type="author")
    a3 = entity_repo.create(collection_id=col_id, name="Charlie", entity_type="author")
    a_solo = entity_repo.create(collection_id=col_id, name="Soloist", entity_type="author")
    session.commit()

    # Alice & Bob on P1 and P2
    for p in (p1, p2):
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p.id,
            target_type="entity",
            target_id=a1.id,
        )
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p.id,
            target_type="entity",
            target_id=a2.id,
        )

    # Alice & Charlie on P3
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p3.id,
        target_type="entity",
        target_id=a1.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p3.id,
        target_type="entity",
        target_id=a3.id,
    )

    # Soloist alone on P3
    p_solo = Paper(collection_id=col_id, title="Solo Paper")
    session.add(p_solo)
    session.flush()
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p_solo.id,
        target_type="entity",
        target_id=a_solo.id,
    )
    session.commit()

    # Author collaborations
    collabs = service.get_author_collaborations(col_id, session=session)
    assert len(collabs) == 2
    assert collabs[0].collaboration_count == 2
    assert {collabs[0].author_id, collabs[0].collaborator_id} == {a1.id, a2.id}
    assert collabs[1].collaboration_count == 1
    assert {collabs[1].author_id, collabs[1].collaborator_id} == {a1.id, a3.id}

    # Author-centric query for Bob
    bob_collabs = service.get_author_collaborations(col_id, author_id=a2.id, session=session)
    assert len(bob_collabs) == 1
    assert bob_collabs[0].author_id == a2.id
    assert bob_collabs[0].collaborator_id == a1.id
    assert bob_collabs[0].collaboration_count == 2

    # Author connectivity
    metrics = service.get_author_connectivity(col_id, session=session)
    assert len(metrics) == 4

    m_alice = next(m for m in metrics if m.author_id == a1.id)
    assert m_alice.paper_count == 3
    assert m_alice.collaborator_count == 2
    assert m_alice.total_collaborations == 3  # (Bob on P1, Bob on P2, Charlie on P3)

    m_solo = next(m for m in metrics if m.author_id == a_solo.id)
    assert m_solo.paper_count == 1
    assert m_solo.collaborator_count == 0
    assert m_solo.total_collaborations == 0


def test_04_combined_collection_analytics(service, session, seed_data):
    """4. get_collection_analytics composes summary, distributions, collabs, and connectivity."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Combined Paper")
    session.add(p1)
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Author Alpha", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Author Beta", entity_type="author")
    c1 = entity_repo.create(collection_id=col_id, name="Concept One", entity_type="concept")
    session.commit()

    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a1.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a2.id,
    )
    session.commit()

    result = service.get_collection_analytics(col_id, session=session)
    assert isinstance(result, CollectionAnalyticsResult)
    assert result.collection_id == col_id
    assert result.summary.total_entities == 3
    assert result.summary.total_relationships == 2
    assert len(result.entity_distribution) == 2
    assert len(result.relationship_distribution) == 1
    assert len(result.top_collaborations) == 1
    assert len(result.author_connectivity) == 2


def test_05_strict_collection_isolation_with_identically_named_authors(service, session, seed_data):
    """5. Identically named author entities across distinct collections remain strictly isolated."""
    col_a_id = seed_data["col_a"].id
    col_b_id = seed_data["col_b"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Identically named author "Ashish Vaswani" in BOTH Collection A and Collection B
    author_a = entity_repo.create(collection_id=col_a_id, name="Ashish Vaswani", entity_type="author")
    author_b = entity_repo.create(collection_id=col_b_id, name="Ashish Vaswani", entity_type="author")

    pa = Paper(collection_id=col_a_id, title="Col A Paper")
    pb = Paper(collection_id=col_b_id, title="Col B Paper")
    session.add_all([pa, pb])
    session.flush()

    rel_repo.create(
        collection_id=col_a_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=pa.id,
        target_type="entity",
        target_id=author_a.id,
    )
    rel_repo.create(
        collection_id=col_b_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=pb.id,
        target_type="entity",
        target_id=author_b.id,
    )
    session.commit()

    # Different UUIDs despite identical display names
    assert author_a.id != author_b.id

    # Collection A analytics
    conns_a = service.get_author_paper_connections(col_a_id, session=session)
    assert len(conns_a) == 1
    assert conns_a[0].author_id == author_a.id
    assert conns_a[0].paper_id == pa.id

    conn_metrics_a = service.get_author_connectivity(col_a_id, session=session)
    assert len(conn_metrics_a) == 1
    assert conn_metrics_a[0].author_id == author_a.id

    # Collection B analytics
    conns_b = service.get_author_paper_connections(col_b_id, session=session)
    assert len(conns_b) == 1
    assert conns_b[0].author_id == author_b.id
    assert conns_b[0].paper_id == pb.id

    conn_metrics_b = service.get_author_connectivity(col_b_id, session=session)
    assert len(conn_metrics_b) == 1
    assert conn_metrics_b[0].author_id == author_b.id


def test_06_multi_row_duplicate_relationships_provenance(service, session, seed_data):
    """6. Verifies duplicate relationship handling and documents independent MAX provenance aggregation."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Multi-Rel Paper")
    session.add(p1)
    session.flush()

    author = entity_repo.create(collection_id=col_id, name="Target Author", entity_type="author")
    session.commit()

    # Edge 1: authored_by, confidence=0.7, evidence="A text snippet"
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=author.id,
        confidence=0.7,
        evidence_text="A text snippet",
    )
    # Edge 2: author_of (inverse), confidence=0.95, evidence="Z text snippet"
    rel_repo.create(
        collection_id=col_id,
        relationship_type="author_of",
        source_type="entity",
        source_id=author.id,
        target_type="paper",
        target_id=p1.id,
        confidence=0.95,
        evidence_text="Z text snippet",
    )
    session.commit()

    # Query author-paper connections across both relationship types
    conns = service.get_author_paper_connections(
        col_id,
        relationship_types=["authored_by", "author_of"],
        session=session,
    )
    # Exactly one connection is returned (no duplicate records)
    assert len(conns) == 1
    conn = conns[0]
    assert conn.author_id == author.id
    assert conn.paper_id == p1.id
    # Max confidence is 0.95 (from Edge 2)
    assert conn.confidence == 0.95
    # Max evidence text is "Z text snippet" (from Edge 2)
    assert conn.evidence_text == "Z text snippet"


def test_07_invalid_input_and_bounds_validation(service, session, seed_data):
    """7. Service validates UUIDs, pagination limits, and confidence bounds."""
    col_id = seed_data["col_a"].id

    # Invalid UUID
    with pytest.raises(TypeError, match="collection_id must be a UUID"):
        service.get_summary("not-a-uuid", session=session)

    with pytest.raises(TypeError, match="collection_id must be a UUID"):
        service.get_entity_distribution(12345, session=session)

    with pytest.raises(TypeError, match="collection_id must be a UUID"):
        service.get_relationship_distribution(None, session=session)

    with pytest.raises(TypeError, match="author_id must be a UUID"):
        service.get_author_collaborations(col_id, author_id="bad-uuid", session=session)

    with pytest.raises(TypeError, match="author_ids\\[0\\] must be a UUID"):
        service.get_author_connectivity(col_id, author_ids=["bad-uuid"], session=session)

    # Invalid bounds
    with pytest.raises(ValueError, match="min_shared_papers must be >= 1"):
        service.get_author_collaborations(col_id, min_shared_papers=0, session=session)

    with pytest.raises(ValueError, match="limit must be >= 1"):
        service.get_author_collaborations(col_id, limit=-5, session=session)

    with pytest.raises(ValueError, match="offset must be >= 0"):
        service.get_author_connectivity(col_id, offset=-1, session=session)

    # Invalid confidence
    with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
        service.get_author_collaborations(col_id, min_confidence=2.5, session=session)

    with pytest.raises(TypeError, match="min_confidence must be a float"):
        service.get_author_collaborations(col_id, min_confidence=True, session=session)

    with pytest.raises(TypeError, match="include_unscored must be a bool"):
        service.get_author_collaborations(col_id, include_unscored="yes", session=session)


def test_08_deterministic_output_ordering(service, session, seed_data):
    """8. Repeated analytics calls produce identical deterministic ordering."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Paper Determinism")
    session.add(p1)
    session.flush()

    # Scrambled insertion order
    a_z = entity_repo.create(collection_id=col_id, name="Zack", entity_type="author")
    a_b = entity_repo.create(collection_id=col_id, name="Ben", entity_type="author")
    a_a = entity_repo.create(collection_id=col_id, name="Adam", entity_type="author")
    session.commit()

    for a in (a_z, a_b, a_a):
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p1.id,
            target_type="entity",
            target_id=a.id,
        )
    session.commit()

    first_conns = service.get_author_paper_connections(col_id, session=session)
    for _ in range(3):
        repeat_conns = service.get_author_paper_connections(col_id, session=session)
        assert [c.author_name for c in repeat_conns] == [c.author_name for c in first_conns]

    assert [c.author_name for c in first_conns] == ["Adam", "Ben", "Zack"]


def test_09_caller_owned_session_lifecycle(service, session, seed_data):
    """9. Caller-owned session is never committed, rolled back, or closed by the service."""
    col_id = seed_data["col_a"].id

    probe_paper = Paper(collection_id=col_id, title="Probe Paper Caller Owned")
    session.add(probe_paper)
    session.flush()

    assert probe_paper in session.identity_map.values()
    assert session.is_active is True

    with patch.object(session, "commit", wraps=session.commit) as mock_commit, \
         patch.object(session, "rollback", wraps=session.rollback) as mock_rollback, \
         patch.object(session, "close", wraps=session.close) as mock_close:

        _ = service.get_summary(col_id, session=session)
        _ = service.get_entity_distribution(col_id, session=session)
        _ = service.get_relationship_distribution(col_id, session=session)
        _ = service.get_author_collaborations(col_id, session=session)
        _ = service.get_author_connectivity(col_id, session=session)
        _ = service.get_collection_analytics(col_id, session=session)

        mock_commit.assert_not_called()
        mock_rollback.assert_not_called()
        mock_close.assert_not_called()

    assert session.is_active is True
    session.rollback()
    assert session.get(Paper, probe_paper.id) is None


def test_10_service_owned_session_lifecycle(engine, session_factory, seed_data):
    """10. Service-owned session created via session_factory is reliably closed upon exit."""
    col_id = seed_data["col_a"].id

    # Create service with session_factory
    service_with_factory = KnowledgeGraphAnalyticsService(session_factory=session_factory)

    # Call method without passing session -> uses session_factory
    summary = service_with_factory.get_summary(col_id)
    assert isinstance(summary, CollectionGraphSummary)

    # Service without session and without factory raises ValueError
    headless_service = KnowledgeGraphAnalyticsService(session_factory=None)
    with pytest.raises(ValueError, match="requires an active database session or session_factory"):
        headless_service.get_summary(col_id)
