"""Comprehensive tests for Knowledge Graph Analytics Models and Repository Aggregations (Stage 5 Step 3.4).

Covers all 12 required scenarios:
1. Empty collection aggregation.
2. Entity counts by type.
3. Relationship counts by type.
4. Collection isolation across two collections.
5. Author-to-paper aggregation.
6. Author collaboration with shared papers.
7. Authors with no collaboration.
8. Duplicate relationship handling.
9. Nullable confidence handling.
10. Deterministic ordering.
11. Read-only session behavior.
12. Correct use of endpoint columns rather than GraphRelationship.paper_id.
Plus model validation and input argument boundaries.
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import UUID, uuid4
import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session
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
    CollectionGraphSummary,
    EntityTypeCount,
    RelationshipTypeCount,
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
def session(engine):
    with Session(engine) as sess:
        yield sess


@pytest.fixture
def seed_data(session):
    """Seed baseline user, workspace, and two isolated collections."""
    user = User(email="analytics_tester@nexora.ai")
    session.add(user)
    session.flush()

    workspace = Workspace(user_id=user.id, name="Analytics Research Lab")
    session.add(workspace)
    session.flush()

    col_a = Collection(workspace_id=workspace.id, name="Collection A - AI Architecture")
    col_b = Collection(workspace_id=workspace.id, name="Collection B - Vision Systems")
    col_empty = Collection(workspace_id=workspace.id, name="Collection C - Empty")
    session.add_all([col_a, col_b, col_empty])
    session.commit()

    return {
        "user": user,
        "workspace": workspace,
        "col_a": col_a,
        "col_b": col_b,
        "col_empty": col_empty,
    }


def test_01_empty_collection_aggregation(session, seed_data):
    """1. Empty collection aggregation returns exact zero counts and empty structures."""
    col_empty_id = seed_data["col_empty"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Entity counts
    assert entity_repo.count_by_collection(col_empty_id) == 0
    assert entity_repo.count_by_type(col_empty_id) == {}
    assert entity_repo.get_entity_type_counts(col_empty_id) == []

    # Relationship counts
    assert rel_repo.count_by_collection(col_empty_id) == 0
    assert rel_repo.count_by_type(col_empty_id) == {}
    assert rel_repo.get_relationship_type_counts(col_empty_id) == []

    # Graph summary
    summary = rel_repo.get_graph_summary(col_empty_id, entity_repo=entity_repo)
    assert isinstance(summary, CollectionGraphSummary)
    assert summary.collection_id == col_empty_id
    assert summary.total_entities == 0
    assert summary.total_relationships == 0
    assert summary.entity_counts_by_type == {}
    assert summary.relationship_counts_by_type == {}
    assert summary.is_empty is True

    # Author connections and collaborations
    conns = rel_repo.list_author_paper_connections(col_empty_id)
    assert conns == []

    collabs = rel_repo.aggregate_author_collaborations(col_empty_id)
    assert collabs == []

    metrics = rel_repo.get_author_connectivity_metrics(col_empty_id)
    assert metrics == []


def test_02_entity_counts_by_type(session, seed_data):
    """2. Entity counts grouped by type produce accurate counts and typed models."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)

    # Create various entity types
    entity_repo.create(collection_id=col_id, name="Transformer", entity_type="concept")
    entity_repo.create(collection_id=col_id, name="Self-Attention", entity_type="concept")
    entity_repo.create(collection_id=col_id, name="BERT", entity_type="method")
    entity_repo.create(collection_id=col_id, name="SQuAD", entity_type="dataset")
    entity_repo.create(collection_id=col_id, name="Ashish Vaswani", entity_type="author")
    session.commit()

    assert entity_repo.count_by_collection(col_id) == 5

    counts = entity_repo.count_by_type(col_id)
    assert counts == {
        "author": 1,
        "concept": 2,
        "dataset": 1,
        "method": 1,
    }

    typed_counts = entity_repo.get_entity_type_counts(col_id)
    assert len(typed_counts) == 4
    assert typed_counts[0] == EntityTypeCount(entity_type="author", count=1)
    assert typed_counts[1] == EntityTypeCount(entity_type="concept", count=2)
    assert typed_counts[2] == EntityTypeCount(entity_type="dataset", count=1)
    assert typed_counts[3] == EntityTypeCount(entity_type="method", count=1)


def test_03_relationship_counts_by_type(session, seed_data):
    """3. Relationship counts grouped by type produce accurate counts and typed models."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Create paper and entities
    p1 = Paper(collection_id=col_id, title="Attention Is All You Need")
    session.add(p1)
    session.flush()

    author = entity_repo.create(collection_id=col_id, name="Ashish Vaswani", entity_type="author")
    concept = entity_repo.create(collection_id=col_id, name="Transformer", entity_type="concept")
    method = entity_repo.create(collection_id=col_id, name="Self-Attention", entity_type="method")
    dataset = entity_repo.create(collection_id=col_id, name="WMT14", entity_type="dataset")
    session.commit()

    # Create relationships of different types
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=author.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="introduces",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=concept.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="uses_method",
        source_type="entity",
        source_id=concept.id,
        target_type="entity",
        target_id=method.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="evaluates_on",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=dataset.id,
    )
    session.commit()

    assert rel_repo.count_by_collection(col_id) == 4

    type_counts = rel_repo.count_by_type(col_id)
    assert type_counts == {
        "authored_by": 1,
        "evaluates_on": 1,
        "introduces": 1,
        "uses_method": 1,
    }

    typed_counts = rel_repo.get_relationship_type_counts(col_id)
    assert len(typed_counts) == 4
    assert typed_counts[0] == RelationshipTypeCount(relationship_type="authored_by", count=1)
    assert typed_counts[1] == RelationshipTypeCount(relationship_type="evaluates_on", count=1)
    assert typed_counts[2] == RelationshipTypeCount(relationship_type="introduces", count=1)
    assert typed_counts[3] == RelationshipTypeCount(relationship_type="uses_method", count=1)

    summary = rel_repo.get_graph_summary(col_id)
    assert summary.total_entities == 4
    assert summary.total_relationships == 4
    assert summary.is_empty is False


def test_04_collection_isolation_across_two_collections(session, seed_data):
    """4. Collection isolation guarantees zero data leakage across separate collections."""
    col_a_id = seed_data["col_a"].id
    col_b_id = seed_data["col_b"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Collection A data
    pa = Paper(collection_id=col_a_id, title="NLP Foundations")
    session.add(pa)
    session.flush()
    ea1 = entity_repo.create(collection_id=col_a_id, name="NLP Author", entity_type="author")
    ea2 = entity_repo.create(collection_id=col_a_id, name="Tokenization", entity_type="concept")
    rel_repo.create(
        collection_id=col_a_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=pa.id,
        target_type="entity",
        target_id=ea1.id,
    )

    # Collection B data
    pb = Paper(collection_id=col_b_id, title="Vision Foundations")
    session.add(pb)
    session.flush()
    eb1 = entity_repo.create(collection_id=col_b_id, name="Vision Author", entity_type="author")
    eb2 = entity_repo.create(collection_id=col_b_id, name="ResNet", entity_type="concept")
    eb3 = entity_repo.create(collection_id=col_b_id, name="ImageNet", entity_type="dataset")
    rel_repo.create(
        collection_id=col_b_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=pb.id,
        target_type="entity",
        target_id=eb1.id,
    )
    rel_repo.create(
        collection_id=col_b_id,
        relationship_type="introduces",
        source_type="paper",
        source_id=pb.id,
        target_type="entity",
        target_id=eb2.id,
    )
    session.commit()

    # Check Collection A
    assert entity_repo.count_by_collection(col_a_id) == 2
    assert entity_repo.count_by_type(col_a_id) == {"author": 1, "concept": 1}
    assert rel_repo.count_by_collection(col_a_id) == 1
    assert rel_repo.count_by_type(col_a_id) == {"authored_by": 1}

    conns_a = rel_repo.list_author_paper_connections(col_a_id)
    assert len(conns_a) == 1
    assert conns_a[0].author_name == "NLP Author"
    assert conns_a[0].paper_title == "NLP Foundations"

    # Check Collection B
    assert entity_repo.count_by_collection(col_b_id) == 3
    assert entity_repo.count_by_type(col_b_id) == {"author": 1, "concept": 1, "dataset": 1}
    assert rel_repo.count_by_collection(col_b_id) == 2
    assert rel_repo.count_by_type(col_b_id) == {"authored_by": 1, "introduces": 1}

    conns_b = rel_repo.list_author_paper_connections(col_b_id)
    assert len(conns_b) == 1
    assert conns_b[0].author_name == "Vision Author"
    assert conns_b[0].paper_title == "Vision Foundations"


def test_05_author_to_paper_aggregation(session, seed_data):
    """5. Author-to-paper aggregation retrieves verified connections with provenance and filters."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Attention Is All You Need")
    p2 = Paper(collection_id=col_id, title="BERT: Pre-training")
    session.add_all([p1, p2])
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Ashish Vaswani", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Noam Shazeer", entity_type="author")
    a3 = entity_repo.create(collection_id=col_id, name="Jacob Devlin", entity_type="author")
    session.commit()

    # P1 authored by a1 and a2
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a1.id,
        confidence=0.95,
        evidence_text="Ashish Vaswani was the first author of Attention Is All You Need.",
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a2.id,
        confidence=0.90,
        evidence_text="Noam Shazeer co-authored the Transformer architecture paper.",
    )
    # P2 authored by a3
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p2.id,
        target_type="entity",
        target_id=a3.id,
        confidence=0.88,
        evidence_text="Jacob Devlin introduced BERT.",
    )
    session.commit()

    all_conns = rel_repo.list_author_paper_connections(col_id)
    assert len(all_conns) == 3

    # Check connection details
    v_conn = next(c for c in all_conns if c.author_name == "Ashish Vaswani")
    assert v_conn.author_id == a1.id
    assert v_conn.paper_id == p1.id
    assert v_conn.paper_title == "Attention Is All You Need"
    assert v_conn.confidence == 0.95
    assert "first author" in v_conn.evidence_text

    # Filter by author_ids
    conns_a1 = rel_repo.list_author_paper_connections(col_id, author_ids=[a1.id])
    assert len(conns_a1) == 1
    assert conns_a1[0].author_id == a1.id

    # Filter by paper_ids
    conns_p2 = rel_repo.list_author_paper_connections(col_id, paper_ids=[p2.id])
    assert len(conns_p2) == 1
    assert conns_p2[0].author_id == a3.id


def test_06_author_collaboration_with_shared_papers(session, seed_data):
    """6. Author collaboration aggregation accurately pairs co-authors and counts shared papers."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Paper Alpha")
    p2 = Paper(collection_id=col_id, title="Paper Beta")
    p3 = Paper(collection_id=col_id, title="Paper Gamma")
    session.add_all([p1, p2, p3])
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Alice Researcher", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Bob Scientist", entity_type="author")
    a3 = entity_repo.create(collection_id=col_id, name="Charlie Engineer", entity_type="author")
    session.commit()

    # Alice and Bob co-authored Alpha and Beta
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
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p2.id,
        target_type="entity",
        target_id=a1.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p2.id,
        target_type="entity",
        target_id=a2.id,
    )

    # Alice and Charlie co-authored Gamma
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
    session.commit()

    collabs = rel_repo.aggregate_author_collaborations(col_id)
    # Expect 2 collaboration pairs: (Alice, Bob) with 2 shared papers, (Alice, Charlie) with 1 shared paper
    assert len(collabs) == 2

    # Ordered by collaboration_count DESC
    rec1 = collabs[0]
    assert rec1.collaboration_count == 2
    assert {rec1.author_id, rec1.collaborator_id} == {a1.id, a2.id}
    assert set(rec1.shared_paper_ids) == {p1.id, p2.id}

    rec2 = collabs[1]
    assert rec2.collaboration_count == 1
    assert {rec2.author_id, rec2.collaborator_id} == {a1.id, a3.id}
    assert set(rec2.shared_paper_ids) == {p3.id}

    # Filter with min_shared_papers=2
    collabs_min2 = rel_repo.aggregate_author_collaborations(col_id, min_shared_papers=2)
    assert len(collabs_min2) == 1
    assert collabs_min2[0].collaboration_count == 2

    # Filter by specific author_id
    collabs_bob = rel_repo.aggregate_author_collaborations(col_id, author_id=a2.id)
    assert len(collabs_bob) == 1
    assert collabs_bob[0].author_id == a2.id
    assert collabs_bob[0].collaborator_id == a1.id


def test_07_authors_with_no_collaboration(session, seed_data):
    """7. Authors with solo papers or no papers have zero collaboration records and accurate metrics."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p_solo = Paper(collection_id=col_id, title="Solo Research Investigation")
    session.add(p_solo)
    session.flush()

    a_solo = entity_repo.create(collection_id=col_id, name="Solo Researcher", entity_type="author")
    a_idle = entity_repo.create(collection_id=col_id, name="Idle Author", entity_type="author")
    session.commit()

    # a_solo authored p_solo alone
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p_solo.id,
        target_type="entity",
        target_id=a_solo.id,
    )
    session.commit()

    # No collaboration pairs should exist
    collabs_all = rel_repo.aggregate_author_collaborations(col_id)
    assert collabs_all == []

    collabs_solo = rel_repo.aggregate_author_collaborations(col_id, author_id=a_solo.id)
    assert collabs_solo == []

    collabs_idle = rel_repo.aggregate_author_collaborations(col_id, author_id=a_idle.id)
    assert collabs_idle == []

    # Connectivity metrics
    metrics = rel_repo.get_author_connectivity_metrics(
        col_id, author_ids=[a_solo.id, a_idle.id]
    )
    assert len(metrics) == 2

    m_solo = next(m for m in metrics if m.author_id == a_solo.id)
    assert m_solo.paper_count == 1
    assert m_solo.collaborator_count == 0
    assert m_solo.total_collaborations == 0

    m_idle = next(m for m in metrics if m.author_id == a_idle.id)
    assert m_idle.paper_count == 0
    assert m_idle.collaborator_count == 0
    assert m_idle.total_collaborations == 0


def test_08_duplicate_relationship_handling(session, seed_data):
    """8. Duplicate or multi-extracted relationships between same author and paper do not inflate metrics."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Duplicate Handling Paper")
    session.add(p1)
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Author One", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Author Two", entity_type="author")
    session.commit()

    # Create standard relationship
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a1.id,
        confidence=0.85,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a2.id,
        confidence=0.90,
    )
    # Also add an inverse relationship or alternative relationship type between same author and paper
    rel_repo.create(
        collection_id=col_id,
        relationship_type="author_of",
        source_type="entity",
        source_id=a1.id,
        target_type="paper",
        target_id=p1.id,
        confidence=0.95,
    )
    session.commit()

    # If querying all relationship types: (authored_by, author_of)
    conns = rel_repo.list_author_paper_connections(
        col_id,
        relationship_types=["authored_by", "author_of"],
    )
    # Author One should be returned only once for Paper One (with max confidence 0.95)
    a1_conns = [c for c in conns if c.author_id == a1.id]
    assert len(a1_conns) == 1
    assert a1_conns[0].confidence == 0.95

    # Check collaboration aggregation
    collabs = rel_repo.aggregate_author_collaborations(col_id)
    assert len(collabs) == 1
    # Exactly 1 shared paper, not duplicated
    assert collabs[0].collaboration_count == 1
    assert collabs[0].shared_paper_ids == [p1.id]


def test_09_nullable_confidence_handling(session, seed_data):
    """9. Aggregations preserve nullable confidence honestly without fabricating defaults."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Paper Scored High")
    p2 = Paper(collection_id=col_id, title="Paper Unscored")
    p3 = Paper(collection_id=col_id, title="Paper Scored Low")
    session.add_all([p1, p2, p3])
    session.flush()

    author = entity_repo.create(collection_id=col_id, name="Test Author", entity_type="author")
    session.commit()

    # R1: confidence = 0.9
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=author.id,
        confidence=0.9,
    )
    # R2: confidence = None (unscored)
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p2.id,
        target_type="entity",
        target_id=author.id,
        confidence=None,
    )
    # R3: confidence = 0.3
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p3.id,
        target_type="entity",
        target_id=author.id,
        confidence=0.3,
    )
    session.commit()

    # Query 1: min_confidence=0.5, include_unscored=True -> expect P1 (0.9) and P2 (None)
    conns_unscored = rel_repo.list_author_paper_connections(
        col_id, min_confidence=0.5, include_unscored=True
    )
    assert len(conns_unscored) == 2
    paper_ids_unscored = {c.paper_id for c in conns_unscored}
    assert paper_ids_unscored == {p1.id, p2.id}
    p2_conn = next(c for c in conns_unscored if c.paper_id == p2.id)
    assert p2_conn.confidence is None  # honest None, not fabricated 0.0 or 1.0

    # Query 2: min_confidence=0.5, include_unscored=False -> expect only P1 (0.9)
    conns_scored_only = rel_repo.list_author_paper_connections(
        col_id, min_confidence=0.5, include_unscored=False
    )
    assert len(conns_scored_only) == 1
    assert conns_scored_only[0].paper_id == p1.id
    assert conns_scored_only[0].confidence == 0.9

    # Query 3: min_confidence=None -> expect all 3 (P1, P2, P3)
    conns_all = rel_repo.list_author_paper_connections(col_id, min_confidence=None)
    assert len(conns_all) == 3


def test_10_deterministic_ordering(session, seed_data):
    """10. Repeated aggregations against identical data produce deterministic ordering."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Insert papers with intentional title ordering
    p_b = Paper(collection_id=col_id, title="Bravo Paper")
    p_a = Paper(collection_id=col_id, title="Alpha Paper")
    session.add_all([p_b, p_a])
    session.flush()

    # Insert authors in scrambled order
    a_z = entity_repo.create(collection_id=col_id, name="Zara Author", entity_type="author")
    a_m = entity_repo.create(collection_id=col_id, name="Mary Author", entity_type="author")
    a_a = entity_repo.create(collection_id=col_id, name="Alan Author", entity_type="author")
    session.commit()

    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p_a.id,
        target_type="entity",
        target_id=a_z.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p_b.id,
        target_type="entity",
        target_id=a_m.id,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p_a.id,
        target_type="entity",
        target_id=a_a.id,
    )
    session.commit()

    # Run connections query 5 times, assert identical result ordering every time
    conns_run1 = rel_repo.list_author_paper_connections(col_id)
    for _ in range(4):
        conns_subsequent = rel_repo.list_author_paper_connections(col_id)
        assert [c.author_name for c in conns_subsequent] == [c.author_name for c in conns_run1]
        assert [c.paper_title for c in conns_subsequent] == [c.paper_title for c in conns_run1]

    # Deterministic alphabetical ordering by author_name
    author_names = [c.author_name for c in conns_run1]
    assert author_names == ["Alan Author", "Mary Author", "Zara Author"]


def test_11_read_only_session_behavior(session, seed_data):
    """11. All repository aggregation operations are read-only and preserve caller transaction state."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Caller starts an active transaction with pending uncommitted changes
    probe_paper = Paper(
        collection_id=col_id,
        title="Uncommitted Probe Paper That Caller Controls",
    )
    session.add(probe_paper)
    session.flush()

    assert probe_paper in session.identity_map.values()
    assert session.is_active is True

    # Spy on session methods to guarantee repository does NOT touch caller transaction lifecycle
    with patch.object(session, "commit", wraps=session.commit) as spy_commit, \
         patch.object(session, "rollback", wraps=session.rollback) as spy_rollback, \
         patch.object(session, "close", wraps=session.close) as spy_close:

        # Execute all aggregation methods
        _ = entity_repo.count_by_collection(col_id)
        _ = entity_repo.count_by_type(col_id)
        _ = entity_repo.get_entity_type_counts(col_id)
        _ = rel_repo.count_by_collection(col_id)
        _ = rel_repo.count_by_type(col_id)
        _ = rel_repo.get_relationship_type_counts(col_id)
        _ = rel_repo.get_graph_summary(col_id, entity_repo=entity_repo)
        _ = rel_repo.list_author_paper_connections(col_id)
        _ = rel_repo.aggregate_author_collaborations(col_id)
        _ = rel_repo.get_author_connectivity_metrics(col_id)

        # None of commit, rollback, or close must have been invoked
        spy_commit.assert_not_called()
        spy_rollback.assert_not_called()
        spy_close.assert_not_called()

    # Caller session remains active and unchanged
    assert session.is_active is True
    assert probe_paper in session.identity_map.values()

    # Caller safely rolls back their transaction; probe is cleanly discarded
    session.rollback()
    reloaded_paper = session.get(Paper, probe_paper.id)
    assert reloaded_paper is None


def test_12_correct_use_of_endpoint_columns_rather_than_paper_id(session, seed_data):
    """12. GraphRelationship.paper_id is asserting provenance and must NEVER be treated as a graph endpoint."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Create paper P1 and entities
    p1 = Paper(collection_id=col_id, title="Provenance Asserting Paper")
    session.add(p1)
    session.flush()

    c1 = entity_repo.create(collection_id=col_id, name="Self-Attention Mechanism", entity_type="concept")
    m1 = entity_repo.create(collection_id=col_id, name="Transformer Layer", entity_type="method")
    a1 = entity_repo.create(collection_id=col_id, name="Valid Author Entity", entity_type="author")
    session.commit()

    # Create a relationship BETWEEN C1 AND M1, where paper_id=p1.id is asserting provenance
    rel_repo.create(
        collection_id=col_id,
        relationship_type="uses_method",
        source_type="entity",
        source_id=c1.id,
        target_type="entity",
        target_id=m1.id,
        paper_id=p1.id,  # Asserting provenance field! NOT an endpoint!
    )
    session.commit()

    # Confirm that paper_id is populated on the relationship
    all_rels = rel_repo.list_by_collection_id(col_id)
    assert len(all_rels) == 1
    assert all_rels[0].paper_id == p1.id
    assert all_rels[0].source_paper_id is None
    assert all_rels[0].target_paper_id is None

    # list_author_paper_connections MUST NOT use paper_id to fabricate an authorship connection
    conns = rel_repo.list_author_paper_connections(col_id)
    assert conns == []

    # aggregate_author_collaborations MUST NOT treat P1 as authored by anyone
    collabs = rel_repo.aggregate_author_collaborations(col_id)
    assert collabs == []

    # get_author_connectivity_metrics MUST NOT report P1 as authored by any author
    metrics = rel_repo.get_author_connectivity_metrics(col_id)
    assert len(metrics) == 1
    assert metrics[0].author_id == a1.id
    assert metrics[0].paper_count == 0

    # Now add genuine endpoint connection: P1 is authored by A1 (source_paper_id=p1.id, target_entity_id=a1.id)
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a1.id,
    )
    session.commit()

    # Now the connection is properly discovered
    conns_after = rel_repo.list_author_paper_connections(col_id)
    assert len(conns_after) == 1
    assert conns_after[0].author_id == a1.id
    assert conns_after[0].paper_id == p1.id


def test_13_model_validation_and_integrity():
    """13. Validate analytics Pydantic models for consistency, strict typing, and validation rules."""
    a1_id = uuid4()
    a2_id = uuid4()
    p1_id = uuid4()
    p2_id = uuid4()

    # AuthorCollaborationRecord consistency
    rec = AuthorCollaborationRecord(
        author_id=a1_id,
        author_name="Alice",
        collaborator_id=a2_id,
        collaborator_name="Bob",
        shared_paper_ids=[p1_id, p2_id],
        collaboration_count=2,
    )
    assert rec.collaboration_count == 2
    assert len(rec.canonical_pair_keys) == 2

    # Self-collaboration should fail
    with pytest.raises(ValueError, match="Self-collaboration prohibited"):
        AuthorCollaborationRecord(
            author_id=a1_id,
            author_name="Alice",
            collaborator_id=a1_id,
            collaborator_name="Alice",
            shared_paper_ids=[p1_id],
            collaboration_count=1,
        )

    # Count mismatch should fail
    with pytest.raises(ValueError, match="collaboration_count .* must match"):
        AuthorCollaborationRecord(
            author_id=a1_id,
            author_name="Alice",
            collaborator_id=a2_id,
            collaborator_name="Bob",
            shared_paper_ids=[p1_id],
            collaboration_count=5,
        )

    # Deduplication of duplicate shared_paper_ids
    rec_dedup = AuthorCollaborationRecord(
        author_id=a1_id,
        author_name="Alice",
        collaborator_id=a2_id,
        collaborator_name="Bob",
        shared_paper_ids=[p1_id, p1_id],  # Duplicate paper ID in input
        collaboration_count=1,
    )
    assert len(rec_dedup.shared_paper_ids) == 1

    # AuthorConnectivityMetrics
    conn_metrics = AuthorConnectivityMetrics(
        author_id=a1_id,
        author_name="Alice",
        paper_count=5,
        collaborator_count=3,
        total_collaborations=8,
    )
    assert conn_metrics.paper_count == 5
    assert conn_metrics.total_collaborations == 8


def test_14_repository_argument_validation(session, seed_data):
    """14. Repository methods validate UUID types, pagination limits, and confidence bounds."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Non-UUID collection_id
    with pytest.raises(TypeError, match="collection_id must be a UUID"):
        entity_repo.count_by_collection("invalid-uuid")

    with pytest.raises(TypeError, match="collection_id must be a UUID"):
        entity_repo.count_by_type("invalid-uuid")

    with pytest.raises(TypeError, match="collection_id must be a UUID"):
        rel_repo.count_by_collection("invalid-uuid")

    with pytest.raises(TypeError, match="collection_id must be a UUID"):
        rel_repo.count_by_type("invalid-uuid")

    with pytest.raises(TypeError, match="collection_id must be a UUID"):
        rel_repo.list_author_paper_connections("invalid-uuid")

    # Invalid pagination
    with pytest.raises(ValueError, match="offset must be a non-negative integer"):
        rel_repo.list_author_paper_connections(col_id, offset=-1)

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        rel_repo.list_author_paper_connections(col_id, limit=0)

    # Invalid confidence range
    with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
        rel_repo.list_author_paper_connections(col_id, min_confidence=1.5)

    with pytest.raises(TypeError, match="min_confidence must be a float"):
        rel_repo.list_author_paper_connections(col_id, min_confidence=True)
