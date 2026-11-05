"""Tests and benchmarks for Knowledge Graph Analytics Optimization (Stage 5 Step 4.5).

Verifies the optimizations for:
- Task 1: Database-side aggregation for author collaborations (GraphRelationshipRepository.aggregate_author_collaborations)
- Task 2: Database-level author-specific filtering for connectivity metrics (GraphRelationshipRepository.get_author_connectivity_metrics)

Covers all 10 required regression scenarios:
1. Two authors sharing 1 paper
2. Three authors sharing 1 paper
3. Authors with multiple shared papers
4. Different collections with unrelated authors (collection isolation)
5. Confidence thresholds (min_confidence)
6. Nullable confidence handling (include_unscored True vs False)
7. Specific author filtering (author_id and author_ids)
8. Empty / non-existent author IDs
9. Deterministic ordering and tie-breaking
10. Large synthetic author groups (author consortia / paper with 50 authors)

Plus empirical benchmark measurements reporting fixture size, authors/papers, execution time, and correctness.
"""

from __future__ import annotations

import time
from uuid import UUID, uuid4
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from nexora.database.models import (
    Base,
    Collection,
    GraphEntity,
    GraphRelationship,
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
    """Seed baseline user, workspace, and isolated collections."""
    user = User(email="analytics_opt_test@nexora.ai")
    session.add(user)
    session.flush()

    workspace = Workspace(user_id=user.id, name="Analytics Optimization Lab")
    session.add(workspace)
    session.flush()

    col_a = Collection(workspace_id=workspace.id, name="Optimization Collection A")
    col_b = Collection(workspace_id=workspace.id, name="Optimization Collection B")
    session.add_all([col_a, col_b])
    session.commit()

    return {
        "user": user,
        "workspace": workspace,
        "col_a": col_a,
        "col_b": col_b,
    }


def test_01_two_authors_sharing_one_paper(session, seed_data):
    """Scenario 1: Two authors sharing exactly 1 paper."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Quantum Computing Fundamentals")
    session.add(p1)
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Alice Martin", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Bob Chen", entity_type="author")
    session.flush()

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

    # 1. Author collaborations
    collabs = rel_repo.aggregate_author_collaborations(col_id)
    assert len(collabs) == 1
    rec = collabs[0]
    assert {rec.author_id, rec.collaborator_id} == {a1.id, a2.id}
    assert rec.shared_paper_ids == [p1.id]
    assert rec.collaboration_count == 1

    # 2. Author connectivity
    metrics = rel_repo.get_author_connectivity_metrics(col_id)
    assert len(metrics) == 2
    m_dict = {m.author_id: m for m in metrics}
    assert m_dict[a1.id].paper_count == 1
    assert m_dict[a1.id].collaborator_count == 1
    assert m_dict[a1.id].total_collaborations == 1
    assert m_dict[a2.id].paper_count == 1
    assert m_dict[a2.id].collaborator_count == 1
    assert m_dict[a2.id].total_collaborations == 1


def test_02_three_authors_sharing_one_paper(session, seed_data):
    """Scenario 2: Three authors sharing 1 paper produces exactly 3 pairwise collaborations."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Distributed Consensus Protocols")
    session.add(p1)
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Alice", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Bob", entity_type="author")
    a3 = entity_repo.create(collection_id=col_id, name="Charlie", entity_type="author")
    session.flush()

    for author in [a1, a2, a3]:
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p1.id,
            target_type="entity",
            target_id=author.id,
        )
    session.commit()

    # Combinations: (A, B), (A, C), (B, C) -> exactly 3 pairs
    collabs = rel_repo.aggregate_author_collaborations(col_id)
    assert len(collabs) == 3
    for c in collabs:
        assert c.collaboration_count == 1
        assert c.shared_paper_ids == [p1.id]

    pairs = {frozenset([c.author_id, c.collaborator_id]) for c in collabs}
    assert pairs == {
        frozenset([a1.id, a2.id]),
        frozenset([a1.id, a3.id]),
        frozenset([a2.id, a3.id]),
    }

    # Connectivity: Each author co-authored with 2 collaborators on 1 paper
    metrics = rel_repo.get_author_connectivity_metrics(col_id)
    assert len(metrics) == 3
    for m in metrics:
        assert m.paper_count == 1
        assert m.collaborator_count == 2
        assert m.total_collaborations == 2


def test_03_authors_with_multiple_shared_papers(session, seed_data):
    """Scenario 3: Authors sharing multiple papers aggregate correctly and filter by min_shared_papers."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Paper 1")
    p2 = Paper(collection_id=col_id, title="Paper 2")
    p3 = Paper(collection_id=col_id, title="Paper 3")
    session.add_all([p1, p2, p3])
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Author A", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Author B", entity_type="author")
    a3 = entity_repo.create(collection_id=col_id, name="Author C", entity_type="author")
    session.flush()

    # A and B co-author P1, P2, P3
    # B and C co-author P3
    for p in [p1, p2, p3]:
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

    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p3.id,
        target_type="entity",
        target_id=a3.id,
    )
    session.commit()

    # All collaborations
    collabs_all = rel_repo.aggregate_author_collaborations(col_id, min_shared_papers=1)
    assert len(collabs_all) == 3  # (A, B) with 3 papers, (A, C) with 1 paper (on P3), (B, C) with 1 paper (on P3)

    # Top collaboration is (A, B) with 3 shared papers
    top = collabs_all[0]
    assert {top.author_id, top.collaborator_id} == {a1.id, a2.id}
    assert top.collaboration_count == 3
    assert set(top.shared_paper_ids) == {p1.id, p2.id, p3.id}

    # Filtering by min_shared_papers=2 should return ONLY (A, B)
    collabs_filtered = rel_repo.aggregate_author_collaborations(col_id, min_shared_papers=2)
    assert len(collabs_filtered) == 1
    assert {collabs_filtered[0].author_id, collabs_filtered[0].collaborator_id} == {a1.id, a2.id}
    assert collabs_filtered[0].collaboration_count == 3

    # Connectivity metrics
    metrics = rel_repo.get_author_connectivity_metrics(col_id)
    m_dict = {m.author_id: m for m in metrics}

    # Author B has 3 papers: on P1 (with A), on P2 (with A), on P3 (with A and C)
    # Total collaborations = 1 + 1 + 2 = 4; Collaborator count = 2 ({A, C})
    assert m_dict[a2.id].paper_count == 3
    assert m_dict[a2.id].collaborator_count == 2
    assert m_dict[a2.id].total_collaborations == 4

    # Author A has 3 papers: on P1 (with B), on P2 (with B), on P3 (with B and C)
    # Total collaborations = 1 + 1 + 2 = 4; Collaborator count = 2 ({B, C})
    assert m_dict[a1.id].paper_count == 3
    assert m_dict[a1.id].collaborator_count == 2
    assert m_dict[a1.id].total_collaborations == 4

    # Author C has 1 paper (P3 with A and B)
    # Total collaborations = 2 ({A, B}); Collaborator count = 2
    assert m_dict[a3.id].paper_count == 1
    assert m_dict[a3.id].collaborator_count == 2
    assert m_dict[a3.id].total_collaborations == 2


def test_04_different_collections_unrelated_authors(session, seed_data):
    """Scenario 4: Strict collection isolation prevents cross-tenant author or collaboration leakage."""
    col_a_id = seed_data["col_a"].id
    col_b_id = seed_data["col_b"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Collection A: Paper PA, Authors AA1, AA2
    pa = Paper(collection_id=col_a_id, title="Collection A Paper")
    session.add(pa)
    session.flush()
    aa1 = entity_repo.create(collection_id=col_a_id, name="Author Alpha", entity_type="author")
    aa2 = entity_repo.create(collection_id=col_a_id, name="Author Beta", entity_type="author")
    rel_repo.create(
        collection_id=col_a_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=pa.id,
        target_type="entity",
        target_id=aa1.id,
    )
    rel_repo.create(
        collection_id=col_a_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=pa.id,
        target_type="entity",
        target_id=aa2.id,
    )

    # Collection B: Paper PB, Authors BB1, BB2
    pb = Paper(collection_id=col_b_id, title="Collection B Paper")
    session.add(pb)
    session.flush()
    bb1 = entity_repo.create(collection_id=col_b_id, name="Author Gamma", entity_type="author")
    bb2 = entity_repo.create(collection_id=col_b_id, name="Author Delta", entity_type="author")
    rel_repo.create(
        collection_id=col_b_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=pb.id,
        target_type="entity",
        target_id=bb1.id,
    )
    rel_repo.create(
        collection_id=col_b_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=pb.id,
        target_type="entity",
        target_id=bb2.id,
    )
    session.commit()

    # Verify Col A
    collabs_a = rel_repo.aggregate_author_collaborations(col_a_id)
    assert len(collabs_a) == 1
    assert {collabs_a[0].author_id, collabs_a[0].collaborator_id} == {aa1.id, aa2.id}

    metrics_a = rel_repo.get_author_connectivity_metrics(col_a_id)
    assert len(metrics_a) == 2
    assert {m.author_id for m in metrics_a} == {aa1.id, aa2.id}

    # Verify Col B
    collabs_b = rel_repo.aggregate_author_collaborations(col_b_id)
    assert len(collabs_b) == 1
    assert {collabs_b[0].author_id, collabs_b[0].collaborator_id} == {bb1.id, bb2.id}

    metrics_b = rel_repo.get_author_connectivity_metrics(col_b_id)
    assert len(metrics_b) == 2
    assert {m.author_id for m in metrics_b} == {bb1.id, bb2.id}

    # Querying Col A for an author that belongs to Col B returns empty
    cross_metrics = rel_repo.get_author_connectivity_metrics(col_a_id, author_ids=[bb1.id])
    assert cross_metrics == []

    cross_collabs = rel_repo.aggregate_author_collaborations(col_a_id, author_id=bb1.id)
    assert cross_collabs == []


def test_05_confidence_threshold_filtering(session, seed_data):
    """Scenario 5: min_confidence filters out relationships below the specified threshold."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="High Confidence Paper")
    p2 = Paper(collection_id=col_id, title="Low Confidence Paper")
    session.add_all([p1, p2])
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Researcher A", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Researcher B", entity_type="author")
    session.flush()

    # P1 relationships have high confidence
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a1.id,
        confidence=0.90,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a2.id,
        confidence=0.85,
    )

    # P2 relationships have lower confidence
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p2.id,
        target_type="entity",
        target_id=a1.id,
        confidence=0.40,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p2.id,
        target_type="entity",
        target_id=a2.id,
        confidence=0.45,
    )
    session.commit()

    # Threshold at 0.75: only P1 qualifies
    collabs_high = rel_repo.aggregate_author_collaborations(col_id, min_confidence=0.75)
    assert len(collabs_high) == 1
    assert collabs_high[0].collaboration_count == 1
    assert collabs_high[0].shared_paper_ids == [p1.id]

    metrics_high = rel_repo.get_author_connectivity_metrics(col_id, min_confidence=0.75)
    for m in metrics_high:
        assert m.paper_count == 1
        assert m.collaborator_count == 1
        assert m.total_collaborations == 1

    # Threshold at 0.30: both P1 and P2 qualify
    collabs_low = rel_repo.aggregate_author_collaborations(col_id, min_confidence=0.30)
    assert len(collabs_low) == 1
    assert collabs_low[0].collaboration_count == 2
    assert set(collabs_low[0].shared_paper_ids) == {p1.id, p2.id}

    metrics_low = rel_repo.get_author_connectivity_metrics(col_id, min_confidence=0.30)
    for m in metrics_low:
        assert m.paper_count == 2
        assert m.collaborator_count == 1
        assert m.total_collaborations == 2


def test_06_include_unscored_behavior(session, seed_data):
    """Scenario 6: include_unscored determines whether relationships with NULL confidence pass."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Unscored Collaboration Paper")
    session.add(p1)
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Scored Author", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Unscored Author", entity_type="author")
    session.flush()

    # a1 has confidence 0.95; a2 has NULL confidence
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a1.id,
        confidence=0.95,
    )
    rel_repo.create(
        collection_id=col_id,
        relationship_type="authored_by",
        source_type="paper",
        source_id=p1.id,
        target_type="entity",
        target_id=a2.id,
        confidence=None,
    )
    session.commit()

    # Case A: min_confidence=0.8, include_unscored=True -> unscored relationship is included
    collabs_included = rel_repo.aggregate_author_collaborations(
        col_id, min_confidence=0.8, include_unscored=True
    )
    assert len(collabs_included) == 1
    assert collabs_included[0].collaboration_count == 1

    metrics_included = rel_repo.get_author_connectivity_metrics(
        col_id, min_confidence=0.8, include_unscored=True
    )
    assert len(metrics_included) == 2
    for m in metrics_included:
        assert m.paper_count == 1
        assert m.collaborator_count == 1

    # Case B: min_confidence=0.8, include_unscored=False -> unscored relationship is excluded
    collabs_excluded = rel_repo.aggregate_author_collaborations(
        col_id, min_confidence=0.8, include_unscored=False
    )
    assert len(collabs_excluded) == 0

    metrics_excluded = rel_repo.get_author_connectivity_metrics(
        col_id, min_confidence=0.8, include_unscored=False
    )
    # Scored author has 1 paper, but 0 collaborators (unscored co-author was excluded)
    m_dict = {m.author_id: m for m in metrics_excluded}
    assert m_dict[a1.id].paper_count == 1
    assert m_dict[a1.id].collaborator_count == 0
    assert m_dict[a1.id].total_collaborations == 0
    # Unscored author has 0 papers and 0 collaborators
    assert m_dict[a2.id].paper_count == 0
    assert m_dict[a2.id].collaborator_count == 0


def test_07_specific_author_filtering(session, seed_data):
    """Scenario 7: Filtering by author_id restricts output to targeted author collaborations and metrics."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    p1 = Paper(collection_id=col_id, title="Target Author Paper")
    p2 = Paper(collection_id=col_id, title="Unrelated Paper")
    session.add_all([p1, p2])
    session.flush()

    target_author = entity_repo.create(collection_id=col_id, name="Target Author", entity_type="author")
    collab_1 = entity_repo.create(collection_id=col_id, name="Collaborator 1", entity_type="author")
    collab_2 = entity_repo.create(collection_id=col_id, name="Collaborator 2", entity_type="author")
    other_author_1 = entity_repo.create(collection_id=col_id, name="Other Author 1", entity_type="author")
    other_author_2 = entity_repo.create(collection_id=col_id, name="Other Author 2", entity_type="author")
    session.flush()

    # Target author + Collaborators on P1
    for a in [target_author, collab_1, collab_2]:
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p1.id,
            target_type="entity",
            target_id=a.id,
        )

    # Other authors on P2 (no target author)
    for a in [other_author_1, other_author_2]:
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p2.id,
            target_type="entity",
            target_id=a.id,
        )
    session.commit()

    # 1. Author collaborations filtered by target_author.id
    target_collabs = rel_repo.aggregate_author_collaborations(col_id, author_id=target_author.id)
    assert len(target_collabs) == 2
    for rec in target_collabs:
        assert rec.author_id == target_author.id
        assert rec.collaborator_id in {collab_1.id, collab_2.id}
        assert rec.collaboration_count == 1
        assert rec.shared_paper_ids == [p1.id]

    # 2. Connectivity metrics filtered by [target_author.id]
    target_metrics = rel_repo.get_author_connectivity_metrics(col_id, author_ids=[target_author.id])
    assert len(target_metrics) == 1
    m = target_metrics[0]
    assert m.author_id == target_author.id
    assert m.paper_count == 1
    assert m.collaborator_count == 2
    assert m.total_collaborations == 2


def test_08_empty_and_nonexistent_author_ids(session, seed_data):
    """Scenario 8: Empty author sequences and non-existent IDs return clean empty results."""
    col_id = seed_data["col_a"].id
    rel_repo = GraphRelationshipRepository(session)

    non_existent = uuid4()

    # Empty author_ids list
    assert rel_repo.get_author_connectivity_metrics(col_id, author_ids=[]) == []

    # Non-existent UUID
    assert rel_repo.get_author_connectivity_metrics(col_id, author_ids=[non_existent]) == []
    assert rel_repo.aggregate_author_collaborations(col_id, author_id=non_existent) == []


def test_09_deterministic_ordering_and_tie_breaking(session, seed_data):
    """Scenario 9: Sorting tie-breakers strictly adhere to deterministic specification."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Create 3 authors with distinct names
    # Author Alpha, Author Beta, Author Gamma
    p1 = Paper(collection_id=col_id, title="Paper Alpha-Beta")
    p2 = Paper(collection_id=col_id, title="Paper Alpha-Gamma")
    session.add_all([p1, p2])
    session.flush()

    a1 = entity_repo.create(collection_id=col_id, name="Alpha", entity_type="author")
    a2 = entity_repo.create(collection_id=col_id, name="Beta", entity_type="author")
    a3 = entity_repo.create(collection_id=col_id, name="Gamma", entity_type="author")
    session.flush()

    # Alpha & Beta share P1 (count 1)
    # Alpha & Gamma share P2 (count 1)
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
        target_id=a3.id,
    )
    session.commit()

    # Collaborations:
    # Both pairs have collaboration_count=1.
    # Tie-breaking:
    # 1. collaboration_count DESC (tied at 1)
    # 2. author_name ASC
    # 3. collaborator_name ASC: Beta comes before Gamma
    collabs = rel_repo.aggregate_author_collaborations(col_id)
    assert len(collabs) == 2
    # Verify that the pairs are deterministically ordered
    names = [(c.author_name.lower(), c.collaborator_name.lower()) for c in collabs]
    assert names == sorted(names)

    # Connectivity metrics:
    # Alpha: collaborator_count=2, total_collabs=2, paper_count=2
    # Beta: collaborator_count=1, total_collabs=1, paper_count=1
    # Gamma: collaborator_count=1, total_collabs=1, paper_count=1
    metrics = rel_repo.get_author_connectivity_metrics(col_id)
    assert len(metrics) == 3
    assert metrics[0].author_name == "Alpha"
    # Beta and Gamma are tied on counts, so Beta comes before Gamma by name ASC
    assert metrics[1].author_name == "Beta"
    assert metrics[2].author_name == "Gamma"


def test_10_large_synthetic_author_consortium(session, seed_data):
    """Scenario 10: Large author consortium (50 authors on 1 paper) evaluates efficiently without combinatorial explosion."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Single paper with 50 authors
    paper = Paper(collection_id=col_id, title="Human Genome Consortium 2026")
    session.add(paper)
    session.flush()

    num_authors = 50
    authors = []
    for i in range(num_authors):
        author = entity_repo.create(
            collection_id=col_id,
            name=f"Consortium Author {i:03d}",
            entity_type="author",
        )
        authors.append(author)

    for author in authors:
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=paper.id,
            target_type="entity",
            target_id=author.id,
        )
    session.commit()

    # Total expected pairs = 50 * 49 / 2 = 1225
    expected_pairs = (num_authors * (num_authors - 1)) // 2
    assert expected_pairs == 1225

    # 1. Fetch with limit=10 (database aggregation should only return 10 rows, avoiding memory explosion)
    t0 = time.perf_counter()
    collabs_limited = rel_repo.aggregate_author_collaborations(col_id, limit=10)
    dur_limited = time.perf_counter() - t0
    assert len(collabs_limited) == 10
    for c in collabs_limited:
        assert c.collaboration_count == 1
        assert c.shared_paper_ids == [paper.id]

    # 2. Fetch all pairs with limit=2000
    t1 = time.perf_counter()
    collabs_all = rel_repo.aggregate_author_collaborations(col_id, limit=2000)
    dur_all = time.perf_counter() - t1
    assert len(collabs_all) == 1225

    # 3. Connectivity metrics for a single author (database-level author scoping)
    target = authors[0]
    t2 = time.perf_counter()
    target_metric = rel_repo.get_author_connectivity_metrics(col_id, author_ids=[target.id])
    dur_metric = time.perf_counter() - t2
    assert len(target_metric) == 1
    tm = target_metric[0]
    assert tm.author_id == target.id
    assert tm.paper_count == 1
    assert tm.collaborator_count == 49
    assert tm.total_collaborations == 49

    # Assert reasonable execution durations (under 1.5 seconds for SQLite in-memory)
    assert dur_limited < 1.5
    assert dur_all < 2.0
    assert dur_metric < 0.5


def test_11_benchmarks_and_performance_report(session, seed_data):
    """Task 4 Benchmark: Reproducible latency and throughput measurements across varying graph scales."""
    col_id = seed_data["col_b"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Scale 1: Small consortium (10 authors on 1 paper)
    # Scale 2: Medium consortium (30 authors on 2 papers)
    # Scale 3: Large consortium (50 authors on 3 papers)

    results = []
    scale_configs = [
        ("Small", 10, 1),
        ("Medium", 30, 2),
        ("Large", 50, 3),
    ]

    for label, author_count, paper_count in scale_configs:
        # Create papers
        papers = []
        for p_idx in range(paper_count):
            p = Paper(collection_id=col_id, title=f"Bench Paper {label} {p_idx}")
            session.add(p)
            papers.append(p)
        session.flush()

        # Create authors
        authors = []
        for a_idx in range(author_count):
            a = entity_repo.create(
                collection_id=col_id,
                name=f"Bench Author {label} {a_idx:03d}",
                entity_type="author",
            )
            authors.append(a)

        # Connect all authors to all papers in this scale
        for p in papers:
            for a in authors:
                rel_repo.create(
                    collection_id=col_id,
                    relationship_type="authored_by",
                    source_type="paper",
                    source_id=p.id,
                    target_type="entity",
                    target_id=a.id,
                )
        session.commit()

        # Warm-up run
        rel_repo.aggregate_author_collaborations(col_id, limit=20)
        rel_repo.get_author_connectivity_metrics(col_id, author_ids=[authors[0].id])

        # Benchmark aggregate_author_collaborations (limit=50)
        collab_times = []
        for _ in range(5):
            t_start = time.perf_counter()
            collabs = rel_repo.aggregate_author_collaborations(col_id, limit=50)
            collab_times.append((time.perf_counter() - t_start) * 1000.0)
        collab_p50 = sorted(collab_times)[len(collab_times) // 2]

        # Benchmark get_author_connectivity_metrics for single author
        conn_times = []
        for _ in range(5):
            t_start = time.perf_counter()
            metrics = rel_repo.get_author_connectivity_metrics(col_id, author_ids=[authors[0].id])
            conn_times.append((time.perf_counter() - t_start) * 1000.0)
        conn_p50 = sorted(conn_times)[len(conn_times) // 2]

        expected_pairs_for_group = (author_count * (author_count - 1)) // 2
        results.append({
            "scale": label,
            "authors": author_count,
            "papers": paper_count,
            "total_pairs_in_graph": expected_pairs_for_group,
            "collab_latency_p50_ms": round(collab_p50, 3),
            "conn_single_author_p50_ms": round(conn_p50, 3),
            "correct_collab_count": collabs[0].collaboration_count == paper_count,
            "correct_collaborator_count": metrics[0].collaborator_count >= author_count - 1,
        })

    # Assert correctness for all scales
    for r in results:
        assert r["correct_collab_count"] is True
        assert r["correct_collaborator_count"] is True
        assert r["collab_latency_p50_ms"] < 150.0  # Well within performance bounds
        assert r["conn_single_author_p50_ms"] < 50.0


def test_12_unicode_author_sorting_tie_and_comparison(session, seed_data):
    """Task 1 & 2: Regression test for Unicode author names with equal collaboration counts on tie boundary."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Anchor author who co-authors with all candidate authors
    anchor = entity_repo.create(collection_id=col_id, name="Anchor Researcher", entity_type="author")

    # Candidate authors with Unicode names and equal collaboration counts:
    # "ástrid" (starts with lowercase á, code point U+00E1 / 225)
    # "Élise" (starts with uppercase É, code point U+00C9 / 201; in UTF-8: 0xC3 0x89)
    # "Ömer" (starts with uppercase Ö, code point U+00D6 / 214; in UTF-8: 0xC3 0x96)
    c1 = entity_repo.create(collection_id=col_id, name="ástrid", entity_type="author")
    c2 = entity_repo.create(collection_id=col_id, name="Élise", entity_type="author")
    c3 = entity_repo.create(collection_id=col_id, name="Ömer", entity_type="author")

    p1 = Paper(collection_id=col_id, title="Paper 1")
    p2 = Paper(collection_id=col_id, title="Paper 2")
    p3 = Paper(collection_id=col_id, title="Paper 3")
    session.add_all([p1, p2, p3])
    session.flush()

    for p, author in [(p1, c1), (p2, c2), (p3, c3)]:
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p.id,
            target_type="entity",
            target_id=anchor.id,
        )
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p.id,
            target_type="entity",
            target_id=author.id,
        )
    session.commit()

    # Query with limit=1, where the limit falls directly on a sorting tie
    # (all 3 pairs have collaboration_count=1)
    res_limit_1 = rel_repo.aggregate_author_collaborations(col_id, author_id=anchor.id, limit=1)
    assert len(res_limit_1) == 1

    # Query all 3 to observe full ordering
    res_all = rel_repo.aggregate_author_collaborations(col_id, author_id=anchor.id, limit=10)
    assert len(res_all) == 3

    # Compare with previous Python in-memory sorting logic from Git HEAD
    # In previous Python implementation:
    #   records.sort(key=lambda r: (-r.collaboration_count, r.author_name.lower(), r.collaborator_name.lower(), str(r.author_id), str(r.collaborator_id)))
    # With author_id=anchor.id:
    #   r.author_name is "Anchor Researcher"
    #   r.collaborator_name is c.name
    # In Python:
    #   "ástrid".lower() -> "ástrid" (U+00E1, 225)
    #   "Élise".lower() -> "élise" (U+00E9, 233)
    #   "Ömer".lower() -> "ömer" (U+00F6, 246)
    # Python Unicode order: "ástrid" (225) < "élise" (233) < "ömer" (246)
    # In SQLite SQL:
    #   SQLite's default lower() only folds ASCII [A-Z], leaving "Élise" as 0xC3 0x89 and "ástrid" as 0xC3 0xA1.
    #   In binary UTF-8 collation: 0x89 < 0xA1, so SQLite ranks "Élise" before "ástrid".
    sql_top_1 = res_limit_1[0].collaborator_name
    python_sorted = sorted(
        res_all,
        key=lambda r: (
            -r.collaboration_count,
            r.author_name.lower(),
            r.collaborator_name.lower(),
            str(r.author_id),
            str(r.collaborator_id),
        ),
    )
    python_top_1 = python_sorted[0].collaborator_name

    # With the boundary-tie strategy, SQL retrieves all candidates at the boundary count (count >= 1)
    # and Python's Unicode-aware sorting strictly determines the top-1.
    # Therefore, sql_top_1 matches python_top_1 exactly: "ástrid".
    assert sql_top_1 == "ástrid"
    assert python_top_1 == "ástrid"
    assert [r.collaborator_name for r in res_all] == ["ástrid", "Élise", "Ömer"]


def test_14_boundary_tie_groups_and_different_counts(session, seed_data):
    """Scenario 14: Comprehensive test for boundary tie groups, different collaboration counts, and ASCII/Unicode ties."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # 1. Test different collaboration counts:
    # Author Zack has 3 shared papers with Anchor (name starts with Z, but count=3)
    # Author Aaron has 1 shared paper with Anchor (name starts with A, but count=1)
    anchor = entity_repo.create(collection_id=col_id, name="Lead Principal", entity_type="author")
    zack = entity_repo.create(collection_id=col_id, name="Zack HighCount", entity_type="author")
    aaron = entity_repo.create(collection_id=col_id, name="Aaron LowCount", entity_type="author")

    pz1 = Paper(collection_id=col_id, title="Zack Paper 1")
    pz2 = Paper(collection_id=col_id, title="Zack Paper 2")
    pz3 = Paper(collection_id=col_id, title="Zack Paper 3")
    pa1 = Paper(collection_id=col_id, title="Aaron Paper 1")
    session.add_all([pz1, pz2, pz3, pa1])
    session.flush()

    for p in [pz1, pz2, pz3]:
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=anchor.id)
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=zack.id)

    rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=pa1.id, target_type="entity", target_id=anchor.id)
    rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=pa1.id, target_type="entity", target_id=aaron.id)
    session.commit()

    # Query top-1: Zack must be chosen first despite alphabetical name order because count=3 > count=1
    top_diff = rel_repo.aggregate_author_collaborations(col_id, author_id=anchor.id, limit=1)
    assert len(top_diff) == 1
    assert top_diff[0].collaborator_name == "Zack HighCount"
    assert top_diff[0].collaboration_count == 3

    # 2. Test tie group larger than requested limit:
    # Create 20 authors in a tie group sharing 1 paper with anchor, and request limit=5
    col_b_id = seed_data["col_b"].id
    tie_anchor = entity_repo.create(collection_id=col_b_id, name="Tie Anchor", entity_type="author")
    tie_paper = Paper(collection_id=col_b_id, title="Large Tie Group Paper")
    session.add(tie_paper)
    session.flush()

    rel_repo.create(collection_id=col_b_id, relationship_type="authored_by", source_type="paper", source_id=tie_paper.id, target_type="entity", target_id=tie_anchor.id)

    # 20 authors named Author_00 to Author_19
    tie_authors = []
    for i in range(20):
        ta = entity_repo.create(collection_id=col_b_id, name=f"TieAuthor_{i:02d}", entity_type="author")
        tie_authors.append(ta)
        rel_repo.create(collection_id=col_b_id, relationship_type="authored_by", source_type="paper", source_id=tie_paper.id, target_type="entity", target_id=ta.id)
    session.commit()

    # Request limit=5 from a 20-member tie group (boundary count is 1, candidate set is 20)
    top_5 = rel_repo.aggregate_author_collaborations(col_b_id, author_id=tie_anchor.id, limit=5)
    assert len(top_5) == 5
    # The top 5 must strictly be TieAuthor_00 through TieAuthor_04 by deterministic ordering
    expected_top_5_names = [f"TieAuthor_{i:02d}" for i in range(5)]
    assert [r.collaborator_name for r in top_5] == expected_top_5_names
    for r in top_5:
        assert r.collaboration_count == 1
        assert r.shared_paper_ids == [tie_paper.id]

    # 3. Unicode and ASCII mixed tie group with limit=3:
    # Authors: "Charlie", "ástrid", "Bob", "Élise", "Alice"
    col_c = Collection(workspace_id=seed_data["workspace"].id, name="Mixed Tie Collection")
    session.add(col_c)
    session.flush()

    mix_anchor = entity_repo.create(collection_id=col_c.id, name="Mix Anchor", entity_type="author")
    mix_paper = Paper(collection_id=col_c.id, title="Mixed Paper")
    session.add(mix_paper)
    session.flush()

    rel_repo.create(collection_id=col_c.id, relationship_type="authored_by", source_type="paper", source_id=mix_paper.id, target_type="entity", target_id=mix_anchor.id)

    mixed_names = ["Charlie", "ástrid", "Bob", "Élise", "Alice"]
    for mname in mixed_names:
        ma = entity_repo.create(collection_id=col_c.id, name=mname, entity_type="author")
        rel_repo.create(collection_id=col_c.id, relationship_type="authored_by", source_type="paper", source_id=mix_paper.id, target_type="entity", target_id=ma.id)
    session.commit()

    # Python Unicode order:
    # "alice".lower() -> "alice" (ASCII 'a' / 0x61)
    # "bob".lower() -> "bob" (ASCII 'b' / 0x62)
    # "charlie".lower() -> "charlie" (ASCII 'c' / 0x63)
    # "ástrid".lower() -> "ástrid" (Unicode 'á' / 0xE1 / 225)
    # "élise".lower() -> "élise" (Unicode 'é' / 0xE9 / 233)
    # When limit=3: must return ["Alice", "Bob", "Charlie"]
    mix_top_3 = rel_repo.aggregate_author_collaborations(col_c.id, author_id=mix_anchor.id, limit=3)
    assert len(mix_top_3) == 3
    assert [r.collaborator_name for r in mix_top_3] == ["Alice", "Bob", "Charlie"]

    # When limit=5: must return all 5 in exact Python Unicode order:
    mix_all = rel_repo.aggregate_author_collaborations(col_c.id, author_id=mix_anchor.id, limit=5)
    assert len(mix_all) == 5
    assert [r.collaborator_name for r in mix_all] == ["Alice", "Bob", "Charlie", "ástrid", "Élise"]


def test_13_benchmark_head_vs_optimized_comparison(session, seed_data):
    """Task 4 & 5: Benchmark previous implementation from Git HEAD against optimized implementation."""
    col_id = seed_data["col_b"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Build a dedicated consortium fixture: 50 authors on 1 paper (1,225 pairs)
    paper = Paper(collection_id=col_id, title="Consortium Scale Benchmark 50")
    session.add(paper)
    session.flush()

    num_authors = 50
    authors = []
    for i in range(num_authors):
        a = entity_repo.create(
            collection_id=col_id,
            name=f"Bench Consortium Author {i:03d}",
            entity_type="author",
        )
        authors.append(a)

    for a in authors:
        rel_repo.create(
            collection_id=col_id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=paper.id,
            target_type="entity",
            target_id=a.id,
        )
    session.commit()

    from sqlalchemy import select, and_, func

    # 1. Legacy (Git HEAD commit 3c68b74a5e) algorithm for aggregate_author_collaborations
    def legacy_aggregate_author_collaborations(repo, col_id, *, limit=100):
        conns = repo.list_author_paper_connections(
            collection_id=col_id,
            offset=0,
            limit=50000,
        )
        if not conns:
            return []
        paper_authors = {}
        author_names = {}
        for conn in conns:
            author_names[conn.author_id] = conn.author_name
            paper_authors.setdefault(conn.paper_id, []).append((conn.author_id, conn.author_name))

        pair_shared_papers = {}
        for pid, auths in paper_authors.items():
            if len(auths) < 2:
                continue
            unique_authors = list({a_id: a_name for a_id, a_name in auths}.items())
            num_a = len(unique_authors)
            for i in range(num_a):
                for j in range(i + 1, num_a):
                    a1_id, a1_name = unique_authors[i]
                    a2_id, a2_name = unique_authors[j]
                    pair_key = (a1_id, a2_id) if a1_id < a2_id else (a2_id, a1_id)
                    pair_shared_papers.setdefault(pair_key, []).append(pid)

        records = []
        for (a1_id, a2_id), pids in pair_shared_papers.items():
            unique_pids = sorted(set(pids))
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

    # 2. Initial SQL Optimization (pushed LIMIT directly to SQL ORDER BY, fast top-N heap, but Unicode collation risk)
    def initial_sql_aggregate_author_collaborations(repo, col_id, *, limit=100):
        stmt1 = (
            select(
                GraphEntity.id.label("author_id"),
                GraphEntity.name.label("author_name"),
                Paper.id.label("paper_id"),
            )
            .select_from(GraphRelationship)
            .join(Paper, and_(Paper.id == GraphRelationship.source_paper_id, Paper.collection_id == col_id))
            .join(GraphEntity, and_(GraphEntity.id == GraphRelationship.target_entity_id, GraphEntity.collection_id == col_id, GraphEntity.entity_type == "author"))
            .where(GraphRelationship.collection_id == col_id, GraphRelationship.relationship_type == "authored_by")
        )
        stmt2 = (
            select(
                GraphEntity.id.label("author_id"),
                GraphEntity.name.label("author_name"),
                Paper.id.label("paper_id"),
            )
            .select_from(GraphRelationship)
            .join(Paper, and_(Paper.id == GraphRelationship.target_paper_id, Paper.collection_id == col_id))
            .join(GraphEntity, and_(GraphEntity.id == GraphRelationship.source_entity_id, GraphEntity.collection_id == col_id, GraphEntity.entity_type == "author"))
            .where(GraphRelationship.collection_id == col_id, GraphRelationship.relationship_type == "authored_by")
        )
        ap = stmt1.union(stmt2).subquery("ap_init")
        ap1 = ap.alias("ap1_init")
        ap2 = ap.alias("ap2_init")

        collab_stmt = (
            select(
                ap1.c.author_id.label("author_id"),
                ap1.c.author_name.label("author_name"),
                ap2.c.author_id.label("collaborator_id"),
                ap2.c.author_name.label("collaborator_name"),
                func.count(func.distinct(ap1.c.paper_id)).label("collaboration_count"),
            )
            .select_from(ap1)
            .join(ap2, ap1.c.paper_id == ap2.c.paper_id)
            .where(ap1.c.author_id < ap2.c.author_id)
            .group_by(ap1.c.author_id, ap1.c.author_name, ap2.c.author_id, ap2.c.author_name)
            .order_by(
                func.count(func.distinct(ap1.c.paper_id)).desc(),
                func.lower(ap1.c.author_name).asc(),
                func.lower(ap2.c.author_name).asc(),
                ap1.c.author_id.asc(),
                ap2.c.author_id.asc(),
            )
            .limit(limit)
        )
        rows = repo._session.execute(collab_stmt).all()
        if not rows:
            return []
        relevant_author_ids = list({r[0] for r in rows} | {r[2] for r in rows})
        author_papers = {}
        papers_query = select(ap.c.author_id, ap.c.paper_id).select_from(ap).where(ap.c.author_id.in_(relevant_author_ids))
        for aid, pid in repo._session.execute(papers_query).all():
            author_papers.setdefault(aid, set()).add(pid)
        records = [
            AuthorCollaborationRecord(
                author_id=r[0], author_name=r[1], collaborator_id=r[2], collaborator_name=r[3],
                shared_paper_ids=sorted(author_papers.get(r[0], set()) & author_papers.get(r[2], set())),
                collaboration_count=r[4],
            )
            for r in rows
        ]
        return records

    # 3. Legacy (Git HEAD commit 3c68b74a5e) algorithm for get_author_connectivity_metrics
    def legacy_get_author_connectivity_metrics(repo, col_id, *, author_ids=None, offset=0, limit=100):
        author_stmt = select(GraphEntity.id, GraphEntity.name).where(
            GraphEntity.collection_id == col_id,
            GraphEntity.entity_type == "author",
        )
        if author_ids:
            author_stmt = author_stmt.where(GraphEntity.id.in_(author_ids))
        author_stmt = author_stmt.order_by(GraphEntity.name.asc(), GraphEntity.id.asc())
        author_rows = repo._session.execute(author_stmt).all()
        if not author_rows:
            return []

        connections = repo.list_author_paper_connections(
            collection_id=col_id,
            offset=0,
            limit=50000,
        )
        author_papers = {}
        paper_authors = {}
        for conn in connections:
            author_papers.setdefault(conn.author_id, set()).add(conn.paper_id)
            paper_authors.setdefault(conn.paper_id, set()).add(conn.author_id)

        metrics = []
        for aid, aname in author_rows:
            pids = author_papers.get(aid, set())
            paper_count = len(pids)
            collaborators = set()
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

    # --- Warm-up runs ---
    legacy_aggregate_author_collaborations(rel_repo, col_id, limit=20)
    initial_sql_aggregate_author_collaborations(rel_repo, col_id, limit=20)
    rel_repo.aggregate_author_collaborations(col_id, limit=20)
    legacy_get_author_connectivity_metrics(rel_repo, col_id, author_ids=[authors[0].id])
    rel_repo.get_author_connectivity_metrics(col_id, author_ids=[authors[0].id])

    # --- Benchmark Fixture A: 50 authors on 1 paper (1,225 pairs in single tie-group) ---
    leg_collab_a = []
    for _ in range(5):
        t0 = time.perf_counter()
        res_leg = legacy_aggregate_author_collaborations(rel_repo, col_id, limit=20)
        leg_collab_a.append((time.perf_counter() - t0) * 1000.0)

    init_collab_a = []
    for _ in range(5):
        t0 = time.perf_counter()
        res_init = initial_sql_aggregate_author_collaborations(rel_repo, col_id, limit=20)
        init_collab_a.append((time.perf_counter() - t0) * 1000.0)

    opt_collab_a = []
    for _ in range(5):
        t0 = time.perf_counter()
        res_opt = rel_repo.aggregate_author_collaborations(col_id, limit=20)
        opt_collab_a.append((time.perf_counter() - t0) * 1000.0)

    assert len(res_leg) == 20
    assert len(res_init) == 20
    assert len(res_opt) == 20

    # Single author connectivity on Fixture A
    target_id = authors[0].id
    leg_conn_a = []
    for _ in range(5):
        t0 = time.perf_counter()
        m_leg = legacy_get_author_connectivity_metrics(rel_repo, col_id, author_ids=[target_id])
        leg_conn_a.append((time.perf_counter() - t0) * 1000.0)

    opt_conn_a = []
    for _ in range(5):
        t0 = time.perf_counter()
        m_opt = rel_repo.get_author_connectivity_metrics(col_id, author_ids=[target_id])
        opt_conn_a.append((time.perf_counter() - t0) * 1000.0)

    assert m_leg[0].collaborator_count == m_opt[0].collaborator_count == 49

    # --- Setup Fixture B: 100 low-count pairs (count=1) + 2 high-count collaboration pairs (count=5 and count=3) ---
    col_c = Collection(workspace_id=seed_data["workspace"].id, name="Low Count Benchmark Collection")
    session.add(col_c)
    session.flush()

    # Pair 1: Alpha & Beta on 5 papers
    a_alpha = entity_repo.create(collection_id=col_c.id, name="Alpha Leader", entity_type="author")
    a_beta = entity_repo.create(collection_id=col_c.id, name="Beta Co-Leader", entity_type="author")
    for p_i in range(5):
        p_hi = Paper(collection_id=col_c.id, title=f"High Count AlphaBeta {p_i}")
        session.add(p_hi)
        session.flush()
        rel_repo.create(collection_id=col_c.id, relationship_type="authored_by", source_type="paper", source_id=p_hi.id, target_type="entity", target_id=a_alpha.id)
        rel_repo.create(collection_id=col_c.id, relationship_type="authored_by", source_type="paper", source_id=p_hi.id, target_type="entity", target_id=a_beta.id)

    # Pair 2: Gamma & Delta on 3 papers
    a_gamma = entity_repo.create(collection_id=col_c.id, name="Gamma Specialist", entity_type="author")
    a_delta = entity_repo.create(collection_id=col_c.id, name="Delta Specialist", entity_type="author")
    for p_i in range(3):
        p_med = Paper(collection_id=col_c.id, title=f"Med Count GammaDelta {p_i}")
        session.add(p_med)
        session.flush()
        rel_repo.create(collection_id=col_c.id, relationship_type="authored_by", source_type="paper", source_id=p_med.id, target_type="entity", target_id=a_gamma.id)
        rel_repo.create(collection_id=col_c.id, relationship_type="authored_by", source_type="paper", source_id=p_med.id, target_type="entity", target_id=a_delta.id)

    # 100 low-count pairs on 100 papers
    for i in range(100):
        p_low = Paper(collection_id=col_c.id, title=f"Disjoint Low Paper {i}")
        session.add(p_low)
        session.flush()
        ax = entity_repo.create(collection_id=col_c.id, name=f"LowAuth_{i:03d}_A", entity_type="author")
        ay = entity_repo.create(collection_id=col_c.id, name=f"LowAuth_{i:03d}_B", entity_type="author")
        rel_repo.create(collection_id=col_c.id, relationship_type="authored_by", source_type="paper", source_id=p_low.id, target_type="entity", target_id=ax.id)
        rel_repo.create(collection_id=col_c.id, relationship_type="authored_by", source_type="paper", source_id=p_low.id, target_type="entity", target_id=ay.id)
    session.commit()

    # Warm-up Fixture B
    legacy_aggregate_author_collaborations(rel_repo, col_c.id, limit=2)
    initial_sql_aggregate_author_collaborations(rel_repo, col_c.id, limit=2)
    rel_repo.aggregate_author_collaborations(col_c.id, limit=2)
    legacy_get_author_connectivity_metrics(rel_repo, col_c.id, author_ids=[a_alpha.id])
    rel_repo.get_author_connectivity_metrics(col_c.id, author_ids=[a_alpha.id])

    # Benchmark Collaboration Aggregation on Fixture B (limit=2)
    leg_collab_b = []
    for _ in range(5):
        t0 = time.perf_counter()
        r_leg_b = legacy_aggregate_author_collaborations(rel_repo, col_c.id, limit=2)
        leg_collab_b.append((time.perf_counter() - t0) * 1000.0)

    init_collab_b = []
    for _ in range(5):
        t0 = time.perf_counter()
        r_init_b = initial_sql_aggregate_author_collaborations(rel_repo, col_c.id, limit=2)
        init_collab_b.append((time.perf_counter() - t0) * 1000.0)

    opt_collab_b = []
    for _ in range(5):
        t0 = time.perf_counter()
        r_opt_b = rel_repo.aggregate_author_collaborations(col_c.id, limit=2)
        opt_collab_b.append((time.perf_counter() - t0) * 1000.0)

    assert len(r_leg_b) == 2
    assert len(r_init_b) == 2
    assert len(r_opt_b) == 2
    # Verify top collaboration is Alpha & Beta with count 5, and second is Gamma & Delta with count 3
    assert r_opt_b[0].collaboration_count == 5
    assert r_opt_b[1].collaboration_count == 3
    assert {r_opt_b[0].author_name, r_opt_b[0].collaborator_name} == {"Alpha Leader", "Beta Co-Leader"}

    # Benchmark Single-Author Connectivity on Fixture B (author in 5 papers out of 108 papers in collection)
    leg_conn_b = []
    for _ in range(5):
        t0 = time.perf_counter()
        m_leg_b = legacy_get_author_connectivity_metrics(rel_repo, col_c.id, author_ids=[a_alpha.id])
        leg_conn_b.append((time.perf_counter() - t0) * 1000.0)

    opt_conn_b = []
    for _ in range(5):
        t0 = time.perf_counter()
        m_opt_b = rel_repo.get_author_connectivity_metrics(col_c.id, author_ids=[a_alpha.id])
        opt_conn_b.append((time.perf_counter() - t0) * 1000.0)

    assert m_leg_b[0].paper_count == m_opt_b[0].paper_count == 5
    assert m_leg_b[0].collaborator_count == m_opt_b[0].collaborator_count == 1
    assert m_leg_b[0].total_collaborations == m_opt_b[0].total_collaborations == 5

    # Print comprehensive comparative measurements
    print("\n=== THREE-WAY COMPARATIVE BENCHMARK MEASUREMENTS ===")
    print("\nFixture A: Large Tie-Group (50 authors on 1 paper = 1,225 pairs in single tie-group, limit=20)")
    print(f"  1. Legacy Git HEAD p50:         {sorted(leg_collab_a)[2]:.3f} ms (min: {min(leg_collab_a):.3f}, max: {max(leg_collab_a):.3f})")
    print(f"  2. Initial SQL (heap limit) p50: {sorted(init_collab_a)[2]:.3f} ms (min: {min(init_collab_a):.3f}, max: {max(init_collab_a):.3f})")
    print(f"  3. Optimized Boundary-Tie p50:   {sorted(opt_collab_a)[2]:.3f} ms (min: {min(opt_collab_a):.3f}, max: {max(opt_collab_a):.3f})")
    print(f"  Connectivity Single Author p50: Legacy: {sorted(leg_conn_a)[2]:.3f} ms vs Optimized: {sorted(opt_conn_a)[2]:.3f} ms")

    print("\nFixture B: Many Low-Count Pairs (100 low-count pairs + 2 high-count collaborations = 102 pairs, limit=2)")
    print(f"  1. Legacy Git HEAD p50:         {sorted(leg_collab_b)[2]:.3f} ms (min: {min(leg_collab_b):.3f}, max: {max(leg_collab_b):.3f})")
    print(f"  2. Initial SQL (heap limit) p50: {sorted(init_collab_b)[2]:.3f} ms (min: {min(init_collab_b):.3f}, max: {max(init_collab_b):.3f})")
    print(f"  3. Optimized Boundary-Tie p50:   {sorted(opt_collab_b)[2]:.3f} ms (min: {min(opt_collab_b):.3f}, max: {max(opt_collab_b):.3f})")
    print(f"  Connectivity Single Author p50: Legacy: {sorted(leg_conn_b)[2]:.3f} ms vs Optimized: {sorted(opt_conn_b)[2]:.3f} ms")


def test_15_exhaustive_exact_result_parity_against_legacy(session, seed_data):
    """Scenario 15: Exhaustive field-by-field and order-by-order parity verification against legacy implementation."""
    col_id = seed_data["col_a"].id
    entity_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    # Legacy function from Git HEAD
    def legacy_collabs(repo, cid, **kwargs):
        limit = kwargs.get("limit", 100)
        author_id = kwargs.get("author_id", None)
        min_shared = kwargs.get("min_shared_papers", 1)
        min_conf = kwargs.get("min_confidence", None)
        inc_unscored = kwargs.get("include_unscored", True)

        conns = repo.list_author_paper_connections(
            collection_id=cid,
            min_confidence=min_conf,
            include_unscored=inc_unscored,
            offset=0,
            limit=50000,
        )
        if not conns:
            return []
        paper_authors = {}
        author_names = {}
        for conn in conns:
            author_names[conn.author_id] = conn.author_name
            paper_authors.setdefault(conn.paper_id, []).append((conn.author_id, conn.author_name))

        pair_shared = {}
        for pid, auths in paper_authors.items():
            if len(auths) < 2:
                continue
            u = list({a_id: a_name for a_id, a_name in auths}.items())
            num_a = len(u)
            for i in range(num_a):
                for j in range(i + 1, num_a):
                    a1_id, a1_name = u[i]
                    a2_id, a2_name = u[j]
                    if author_id is not None:
                        if a1_id == author_id:
                            pair_key = (a1_id, a2_id)
                        elif a2_id == author_id:
                            pair_key = (a2_id, a1_id)
                        else:
                            continue
                    else:
                        pair_key = (a1_id, a2_id) if a1_id < a2_id else (a2_id, a1_id)
                    pair_shared.setdefault(pair_key, []).append(pid)

        records = []
        for (a1_id, a2_id), pids in pair_shared.items():
            u_pids = sorted(set(pids))
            if len(u_pids) < min_shared:
                continue
            records.append(
                AuthorCollaborationRecord(
                    author_id=a1_id,
                    author_name=author_names[a1_id],
                    collaborator_id=a2_id,
                    collaborator_name=author_names[a2_id],
                    shared_paper_ids=u_pids,
                    collaboration_count=len(u_pids),
                )
            )
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

    # Create test authors: ASCII, Unicode, and two identical names with different UUIDs
    a_alice = entity_repo.create(collection_id=col_id, name="Alice Researcher", entity_type="author")
    a_bob = entity_repo.create(collection_id=col_id, name="Bob Scientist", entity_type="author")
    a_astrid = entity_repo.create(collection_id=col_id, name="ástrid", entity_type="author")
    a_elise = entity_repo.create(collection_id=col_id, name="Élise", entity_type="author")
    a_omer = entity_repo.create(collection_id=col_id, name="Ömer", entity_type="author")
    # Duplicate names with different IDs
    a_john_1 = GraphEntity(collection_id=col_id, name="John Smith", canonical_name="john smith 1", entity_type="author")
    a_john_2 = GraphEntity(collection_id=col_id, name="John Smith", canonical_name="john smith 2", entity_type="author")
    session.add_all([a_john_1, a_john_2])
    session.flush()
    assert a_john_1.id != a_john_2.id

    # Create papers across different collaboration count groups:
    # Alice & Bob share 4 papers (count = 4)
    # Alice & ástrid share 2 papers (count = 2)
    # Alice & Élise share 2 papers (count = 2)
    # Alice & Ömer share 2 papers (count = 2)
    # John 1 & John 2 share 1 paper (count = 1)
    # Bob & John 1 share 1 paper with low confidence (0.4)
    # Bob & John 2 share 1 paper with NULL confidence
    p_ab = [Paper(collection_id=col_id, title=f"AliceBob Paper {i}") for i in range(4)]
    p_astrid = [Paper(collection_id=col_id, title=f"AliceAstrid Paper {i}") for i in range(2)]
    p_elise = [Paper(collection_id=col_id, title=f"AliceElise Paper {i}") for i in range(2)]
    p_omer = [Paper(collection_id=col_id, title=f"AliceOmer Paper {i}") for i in range(2)]
    p_johns = Paper(collection_id=col_id, title="Johns Co-Authored Paper")
    p_conf_low = Paper(collection_id=col_id, title="Low Conf Paper")
    p_conf_null = Paper(collection_id=col_id, title="Null Conf Paper")

    all_papers = p_ab + p_astrid + p_elise + p_omer + [p_johns, p_conf_low, p_conf_null]
    session.add_all(all_papers)
    session.flush()

    for p in p_ab:
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=a_alice.id, confidence=0.95)
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=a_bob.id, confidence=0.90)

    for p in p_astrid:
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=a_alice.id, confidence=0.95)
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=a_astrid.id, confidence=0.90)

    for p in p_elise:
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=a_alice.id, confidence=0.95)
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=a_elise.id, confidence=0.90)

    for p in p_omer:
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=a_alice.id, confidence=0.95)
        rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p.id, target_type="entity", target_id=a_omer.id, confidence=0.90)

    rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p_johns.id, target_type="entity", target_id=a_john_1.id, confidence=0.95)
    rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p_johns.id, target_type="entity", target_id=a_john_2.id, confidence=0.95)

    rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p_conf_low.id, target_type="entity", target_id=a_bob.id, confidence=0.40)
    rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p_conf_low.id, target_type="entity", target_id=a_john_1.id, confidence=0.40)

    rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p_conf_null.id, target_type="entity", target_id=a_bob.id, confidence=0.90)
    rel_repo.create(collection_id=col_id, relationship_type="authored_by", source_type="paper", source_id=p_conf_null.id, target_type="entity", target_id=a_john_2.id, confidence=None)
    session.commit()

    # Define helper to verify 100% exact parity across every field of every record
    def assert_exact_parity(**kwargs):
        leg = legacy_collabs(rel_repo, col_id, **kwargs)
        opt = rel_repo.aggregate_author_collaborations(col_id, **kwargs)
        assert len(opt) == len(leg), f"Count mismatch for kwargs={kwargs}: {len(opt)} vs {len(leg)}"
        for i in range(len(leg)):
            l_rec = leg[i]
            o_rec = opt[i]
            assert o_rec.author_id == l_rec.author_id, f"author_id mismatch at {i}: {o_rec} vs {l_rec}"
            assert o_rec.author_name == l_rec.author_name, f"author_name mismatch at {i}: {o_rec} vs {l_rec}"
            assert o_rec.collaborator_id == l_rec.collaborator_id, f"collaborator_id mismatch at {i}: {o_rec} vs {l_rec}"
            assert o_rec.collaborator_name == l_rec.collaborator_name, f"collaborator_name mismatch at {i}: {o_rec} vs {l_rec}"
            assert o_rec.collaboration_count == l_rec.collaboration_count, f"collaboration_count mismatch at {i}: {o_rec} vs {l_rec}"
            assert o_rec.shared_paper_ids == l_rec.shared_paper_ids, f"shared_paper_ids mismatch at {i}: {o_rec} vs {l_rec}"
            assert o_rec == l_rec, f"Record equality mismatch at {i}"

    # 1. limit = 1 (boundary is top-1)
    assert_exact_parity(limit=1)

    # 2. limit smaller than total pairs (limit=2)
    assert_exact_parity(limit=2)

    # 3. limit equal to total pairs
    all_recs = legacy_collabs(rel_repo, col_id, limit=100)
    total_pairs = len(all_recs)
    assert_exact_parity(limit=total_pairs)

    # 4. limit greater than total pairs (limit=50)
    assert_exact_parity(limit=50)

    # 5. Multiple collaboration count groups (min_shared_papers = 1, limit=10)
    assert_exact_parity(min_shared_papers=1, limit=10)

    # 6. Large boundary tie groups:
    # Slicing directly into the tie group of count=2 (Alice & astrid, Alice & Elise, Alice & Omer)
    assert_exact_parity(limit=3)

    # 7. Unicode and ASCII tie ordering (Alice's collaborators: astrid vs Elise vs Omer)
    assert_exact_parity(author_id=a_alice.id, limit=2)
    assert_exact_parity(author_id=a_alice.id, limit=3)

    # 8. Equal names with different IDs (John Smith 1 vs John Smith 2)
    assert_exact_parity(min_shared_papers=1, limit=10)

    # 9. min_shared_papers filtering (min=2, min=3, min=4)
    assert_exact_parity(min_shared_papers=2, limit=10)
    assert_exact_parity(min_shared_papers=3, limit=10)
    assert_exact_parity(min_shared_papers=4, limit=10)

    # 10. Confidence threshold filtering (min_confidence=0.75 excludes p_conf_low)
    assert_exact_parity(min_confidence=0.75, limit=10)

    # 11. include_unscored behavior (True vs False)
    assert_exact_parity(min_confidence=0.75, include_unscored=True, limit=10)
    assert_exact_parity(min_confidence=0.75, include_unscored=False, limit=10)

    # 12. author_id filtering across various authors
    assert_exact_parity(author_id=a_bob.id, limit=5)
    assert_exact_parity(author_id=a_john_1.id, limit=5)
    assert_exact_parity(author_id=a_john_2.id, limit=5)
