"""Comprehensive tests for Stage 5 Step 1: Knowledge Graph Foundation.

Covers:
1. Entity creation and retrieval.
2. Canonical-name normalization.
3. Deduplication within one collection.
4. Identical names in separate collections.
5. Relationship creation and deduplication.
6. Valid and invalid relationship endpoints.
7. Evidence provenance and invalid provenance.
8. Confidence validation.
9. Cross-collection access and relationship isolation.
10. Foreign-key cascade and set-null behavior.
11. Repeated ingestion operations (idempotence).
12. Existing database model regressions.
"""

from __future__ import annotations

import datetime
from uuid import UUID, uuid4
import pytest
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import Session

from nexora.database.models import (
    Author,
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
    PaperAuthor,
    PaperReference,
    User,
    Workspace,
    normalize_entity_name,
)
from nexora.database.repository import (
    AuthorRepository,
    CollectionRepository,
    ConstraintError,
    DocumentChunkRepository,
    DocumentRepository,
    DuplicateRecordError,
    GraphEntityRepository,
    GraphRelationshipRepository,
    PaperAuthorRepository,
    PaperReferenceRepository,
    PaperRepository,
    RecordNotFoundError,
    UserRepository,
    WorkspaceRepository,
)


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
    """Seed baseline User, Workspace, Collections, Papers, Documents, and Chunks."""
    user = User(email="researcher@nexora.ai")
    session.add(user)
    session.flush()

    workspace = Workspace(user_id=user.id, name="AI Research Lab")
    session.add(workspace)
    session.flush()

    col_a = Collection(workspace_id=workspace.id, name="Collection A - LLM Architectures")
    col_b = Collection(workspace_id=workspace.id, name="Collection B - Computer Vision")
    session.add_all([col_a, col_b])
    session.flush()

    paper_a1 = Paper(
        collection_id=col_a.id,
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models...",
    )
    paper_a2 = Paper(
        collection_id=col_a.id,
        title="FlashAttention: Fast and Memory-Efficient Exact Attention",
        abstract="Transformers are slow and memory-hungry on long sequences...",
    )
    paper_b1 = Paper(
        collection_id=col_b.id,
        title="Deep Residual Learning for Image Recognition",
        abstract="Deeper neural networks are more difficult to train...",
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
        text="On the WMT 2014 English-to-German translation task, the big transformer model...",
    )
    chunk_a2 = DocumentChunkRecord(
        document_id=doc_a1.id,
        chunk_index=1,
        text="We call our particular attention Scaled Dot-Product Attention...",
    )
    session.add_all([chunk_a1, chunk_a2])
    session.flush()

    return {
        "user": user,
        "workspace": workspace,
        "col_a": col_a,
        "col_b": col_b,
        "paper_a1": paper_a1,
        "paper_a2": paper_a2,
        "paper_b1": paper_b1,
        "doc_a1": doc_a1,
        "chunk_a1": chunk_a1,
        "chunk_a2": chunk_a2,
    }


# ============================================================================
# 1. Entity creation and retrieval
# ============================================================================

def test_entity_creation_and_retrieval(session, seed_data):
    repo = GraphEntityRepository(session)
    col_a = seed_data["col_a"]

    entity = repo.create(
        collection_id=col_a.id,
        name="Transformer Architecture",
        entity_type="concept",
        description="Self-attention based neural network architecture.",
        aliases=["Transformers", "Transformer model"],
        entity_metadata={"invented": 2017},
    )

    assert entity.id is not None
    assert entity.canonical_name == "transformer architecture"
    assert entity.entity_type == "concept"
    assert entity.aliases == ["Transformers", "Transformer model"]

    # Scoped retrieval by ID
    retrieved = repo.get_by_id_scoped(entity.id, collection_id=col_a.id)
    assert retrieved is not None
    assert retrieved.name == "Transformer Architecture"

    # Canonical name lookup
    by_canonical = repo.get_by_canonical_name(
        collection_id=col_a.id,
        entity_type="concept",
        canonical_name="transformer architecture",
    )
    assert by_canonical is not None
    assert by_canonical.id == entity.id

    # List by collection
    all_entities = repo.list_by_collection_id(col_a.id)
    assert len(all_entities) == 1
    assert all_entities[0].id == entity.id

    # Search entities
    search_res = repo.search_entities(collection_id=col_a.id, query="transformer")
    assert len(search_res) == 1
    assert search_res[0].id == entity.id


def test_entity_creation_all_valid_types(session, seed_data):
    repo = GraphEntityRepository(session)
    col_a = seed_data["col_a"]

    valid_types = ["concept", "method", "dataset", "metric", "author", "other"]
    created_entities = []

    for t in valid_types:
        ent = repo.create(
            collection_id=col_a.id,
            name=f"Entity of type {t}",
            entity_type=t,
        )
        created_entities.append(ent)
        assert ent.entity_type == t

    assert len(repo.list_by_collection_id(col_a.id)) == len(valid_types)


# ============================================================================
# 2. Canonical-name normalization
# ============================================================================

def test_canonical_name_normalization():
    assert normalize_entity_name("  FlashAttention-2  ") == "flashattention-2"
    assert normalize_entity_name("FLASHATTENTION-2") == "flashattention-2"
    assert normalize_entity_name("In-Context   Learning") == "in-context learning"
    assert normalize_entity_name("\tChain\nof\tThought\n") == "chain of thought"

    with pytest.raises(ValueError):
        normalize_entity_name("   ")

    with pytest.raises(TypeError):
        normalize_entity_name(123)  # type: ignore


# ============================================================================
# 3. Deduplication within one collection
# ============================================================================

def test_entity_deduplication_within_one_collection(session, seed_data):
    repo = GraphEntityRepository(session)
    col_a = seed_data["col_a"]

    repo.create(
        collection_id=col_a.id,
        name="LoRA",
        entity_type="method",
    )

    # Attempting to create duplicate canonical entity in same collection fails
    with pytest.raises(DuplicateRecordError):
        repo.create(
            collection_id=col_a.id,
            name="  lora  ",  # Normalizes to 'lora'
            entity_type="method",
        )

    # get_or_create safely returns existing entity
    existing, created = repo.get_or_create(
        collection_id=col_a.id,
        name="LoRA",
        entity_type="method",
        description="Low-Rank Adaptation",
    )
    assert created is False
    assert existing.canonical_name == "lora"
    assert existing.description == "Low-Rank Adaptation"


# ============================================================================
# 4. Identical names in separate collections
# ============================================================================

def test_identical_canonical_names_in_separate_collections(session, seed_data):
    repo = GraphEntityRepository(session)
    col_a = seed_data["col_a"]
    col_b = seed_data["col_b"]

    ent_a = repo.create(
        collection_id=col_a.id,
        name="FlashAttention",
        entity_type="method",
    )
    ent_b = repo.create(
        collection_id=col_b.id,
        name="FlashAttention",
        entity_type="method",
    )

    assert ent_a.id != ent_b.id
    assert ent_a.canonical_name == ent_b.canonical_name == "flashattention"
    assert ent_a.collection_id == col_a.id
    assert ent_b.collection_id == col_b.id

    # Verify each collection lists only its own entity
    list_a = repo.list_by_collection_id(col_a.id)
    list_b = repo.list_by_collection_id(col_b.id)
    assert len(list_a) == 1 and list_a[0].id == ent_a.id
    assert len(list_b) == 1 and list_b[0].id == ent_b.id


# ============================================================================
# 5. Relationship creation and deduplication
# ============================================================================

def test_relationship_creation_and_deduplication(session, seed_data):
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    dataset_ent = ent_repo.create(
        collection_id=col_a.id,
        name="WMT 2014 English-to-German",
        entity_type="dataset",
    )

    rel = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="evaluates_on",
        source_type="paper",
        source_id=paper_a1.id,
        target_type="entity",
        target_id=dataset_ent.id,
        confidence=0.95,
        evidence_text="On the WMT 2014 English-to-German translation task, the big transformer model...",
    )

    assert rel.id is not None
    assert rel.relationship_type == "evaluates_on"
    assert rel.source_paper_id == paper_a1.id
    assert rel.target_entity_id == dataset_ent.id
    assert rel.has_evidence is True

    # Duplicate creation fails
    with pytest.raises(DuplicateRecordError):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="evaluates_on",
            source_type="paper",
            source_id=paper_a1.id,
            target_type="entity",
            target_id=dataset_ent.id,
        )

    # get_or_create succeeds idempotently
    existing, created = rel_repo.get_or_create(
        collection_id=col_a.id,
        relationship_type="evaluates_on",
        source_type="paper",
        source_id=paper_a1.id,
        target_type="entity",
        target_id=dataset_ent.id,
        confidence=0.98,  # Higher confidence enriches existing
    )
    assert created is False
    assert existing.id == rel.id
    assert existing.confidence == 0.98


# ============================================================================
# 6. Valid and invalid relationship endpoints
# ============================================================================

def test_relationship_invalid_endpoints(session, seed_data):
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]

    method_ent = ent_repo.create(
        collection_id=col_a.id,
        name="Self-Attention",
        entity_type="method",
    )

    # 1. Self-loop prohibited
    with pytest.raises(ConstraintError, match="Self-referencing relationship prohibited"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="relates_to",
            source_type="entity",
            source_id=method_ent.id,
            target_type="entity",
            target_id=method_ent.id,
        )

    # 2. Non-existent source
    with pytest.raises(RecordNotFoundError, match="Source entity with id"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="relates_to",
            source_type="entity",
            source_id=uuid4(),
            target_type="entity",
            target_id=method_ent.id,
        )

    # 3. Invalid endpoint type
    with pytest.raises(ValueError, match="Invalid source_type 'cluster'"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="relates_to",
            source_type="cluster",  # type: ignore
            source_id=method_ent.id,
            target_type="entity",
            target_id=method_ent.id,
        )


# ============================================================================
# 7. Evidence provenance and invalid provenance
# ============================================================================

def test_evidence_provenance_and_validation(session, seed_data):
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    paper_a2 = seed_data["paper_a2"]
    chunk_a1 = seed_data["chunk_a1"]

    method = ent_repo.create(collection_id=col_a.id, name="Scaled Dot-Product Attention", entity_type="method")

    # Valid grounded relationship
    rel = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="proposes",
        source_type="paper",
        source_id=paper_a1.id,
        target_type="entity",
        target_id=method.id,
        paper_id=paper_a1.id,
        chunk_id=chunk_a1.id,
        page_number=3,
        evidence_text="We call our particular attention Scaled Dot-Product Attention.",
        confidence=0.99,
        extraction_method="llm_extraction",
    )
    assert rel.has_evidence is True
    assert rel.page_number == 3
    assert rel.chunk_id == chunk_a1.id

    # Ungrounded relationship
    rel_unsupported = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="discusses",
        source_type="paper",
        source_id=paper_a2.id,
        target_type="entity",
        target_id=method.id,
    )
    assert rel_unsupported.has_evidence is False

    # Invalid page number (<= 0)
    with pytest.raises(ConstraintError, match="page_number must be an integer >= 1"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="discusses",
            source_type="paper",
            source_id=paper_a1.id,
            target_type="entity",
            target_id=method.id,
            page_number=0,
        )

    # Blank evidence text
    with pytest.raises(ConstraintError, match="evidence_text cannot be blank"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="discusses",
            source_type="paper",
            source_id=paper_a1.id,
            target_type="entity",
            target_id=method.id,
            evidence_text="   ",
        )

    # Chunk belonging to paper_a1, but claiming paper_id=paper_a2
    with pytest.raises(ConstraintError, match="Evidence chunk belongs to paper"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="discusses",
            source_type="paper",
            source_id=paper_a2.id,
            target_type="entity",
            target_id=method.id,
            paper_id=paper_a2.id,
            chunk_id=chunk_a1.id,  # chunk belongs to paper_a1!
        )


# ============================================================================
# 8. Confidence validation
# ============================================================================

def test_confidence_validation(session, seed_data):
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    metric = ent_repo.create(collection_id=col_a.id, name="BLEU-4", entity_type="metric")

    # Valid boundary confidence scores
    rel_0 = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="reports_metric",
        source_type="paper",
        source_id=paper_a1.id,
        target_type="entity",
        target_id=metric.id,
        confidence=0.0,
    )
    assert rel_0.confidence == 0.0

    # Negative confidence
    with pytest.raises(ConstraintError, match="confidence must be between 0.0 and 1.0"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="measures",
            source_type="paper",
            source_id=paper_a1.id,
            target_type="entity",
            target_id=metric.id,
            confidence=-0.1,
        )

    # Over 1.0 confidence
    with pytest.raises(ConstraintError, match="confidence must be between 0.0 and 1.0"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="measures",
            source_type="paper",
            source_id=paper_a1.id,
            target_type="entity",
            target_id=metric.id,
            confidence=1.05,
        )


# ============================================================================
# 9. Cross-collection access and relationship isolation
# ============================================================================

def test_cross_collection_relationship_isolation(session, seed_data):
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    col_b = seed_data["col_b"]
    paper_a1 = seed_data["paper_a1"]
    paper_b1 = seed_data["paper_b1"]

    ent_b = ent_repo.create(
        collection_id=col_b.id,
        name="Residual Connections",
        entity_type="concept",
    )

    # 1. Prohibit relationship connecting paper in col_a to entity in col_b
    with pytest.raises(ConstraintError, match="Cross-collection relationship prohibited: Target entity"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="uses",
            source_type="paper",
            source_id=paper_a1.id,
            target_type="entity",
            target_id=ent_b.id,
        )

    # 2. Prohibit relationship between paper_a1 and paper_b1 within collection A
    with pytest.raises(ConstraintError, match="Cross-collection relationship prohibited: Target paper"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="extends",
            source_type="paper",
            source_id=paper_a1.id,
            target_type="paper",
            target_id=paper_b1.id,
        )

    # 3. Prohibit cross-collection evidence provenance
    ent_a = ent_repo.create(collection_id=col_a.id, name="Self-Attention", entity_type="method")
    with pytest.raises(ConstraintError, match="Cross-collection evidence provenance prohibited"):
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="uses",
            source_type="paper",
            source_id=paper_a1.id,
            target_type="entity",
            target_id=ent_a.id,
            paper_id=paper_b1.id,  # paper belongs to col_b!
        )

    # 4. Scoped query isolation: collection A cannot see collection B's entities
    assert ent_repo.get_by_id_scoped(ent_b.id, collection_id=col_a.id) is None
    with pytest.raises(RecordNotFoundError):
        ent_repo.get_by_id_scoped_or_raise(ent_b.id, collection_id=col_a.id)


# ============================================================================
# 10. Foreign-key cascade and set-null behavior
# ============================================================================

def test_foreign_key_cascades_and_set_null(session, seed_data):
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    chunk_a1 = seed_data["chunk_a1"]

    entity = ent_repo.create(collection_id=col_a.id, name="Multi-Head Attention", entity_type="method")

    rel = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="proposes",
        source_type="paper",
        source_id=paper_a1.id,
        target_type="entity",
        target_id=entity.id,
        paper_id=paper_a1.id,
        chunk_id=chunk_a1.id,
        evidence_text="Multi-head attention allows the model to jointly attend...",
    )
    session.flush()

    # 1. Deleting chunk sets chunk_id to NULL on relationship without deleting relationship
    session.delete(chunk_a1)
    session.flush()
    session.expire_all()

    refreshed_rel = session.get(GraphRelationship, rel.id)
    assert refreshed_rel is not None
    assert refreshed_rel.chunk_id is None
    assert refreshed_rel.evidence_text == "Multi-head attention allows the model to jointly attend..."

    # 2. Deleting entity cascades and removes the relationship
    rel_id = rel.id
    session.delete(entity)
    session.flush()
    session.expire_all()
    assert session.get(GraphRelationship, rel_id) is None


def test_cascade_delete_collection_cleans_graph(session, seed_data):
    ent_repo = GraphEntityRepository(session)
    col_b = seed_data["col_b"]

    ent = ent_repo.create(collection_id=col_b.id, name="ResNet-50", entity_type="method")
    ent_id = ent.id
    col_b_id = col_b.id

    # Deleting collection_b cascades to graph_entities
    session.delete(col_b)
    session.flush()
    session.expire_all()

    assert session.get(GraphEntity, ent_id) is None
    assert session.get(Collection, col_b_id) is None


# ============================================================================
# 11. Repeated ingestion operations (idempotence)
# ============================================================================

def test_repeated_ingestion_idempotence(session, seed_data):
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    # Simulating 3 consecutive extraction passes on the same paper
    for i in range(3):
        # 1. get_or_create entity
        entity, ent_created = ent_repo.get_or_create(
            collection_id=col_a.id,
            name="Positional Encoding",
            entity_type="concept",
            description="Adds order awareness to sequence models",
        )
        if i == 0:
            assert ent_created is True
        else:
            assert ent_created is False

        # 2. get_or_create relationship
        rel, rel_created = rel_repo.get_or_create(
            collection_id=col_a.id,
            relationship_type="uses",
            source_type="paper",
            source_id=paper_a1.id,
            target_type="entity",
            target_id=entity.id,
            confidence=0.85 + (i * 0.05),
            evidence_text="Since our model contains no recurrence, we add positional encodings...",
        )
        if i == 0:
            assert rel_created is True
        else:
            assert rel_created is False

    # Verify no duplicate records exist in the database
    entities = ent_repo.list_by_collection_id(col_a.id)
    assert len(entities) == 1
    rels = rel_repo.list_by_collection_id(col_a.id)
    assert len(rels) == 1
    assert rels[0].confidence == pytest.approx(0.95)


# ============================================================================
# 12. Existing database models and repository regression check
# ============================================================================

def test_existing_author_and_paper_reference_models_intact(session, seed_data):
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    paper_a2 = seed_data["paper_a2"]

    # Test Author and PaperAuthor
    author_repo = AuthorRepository(session)
    paper_author_repo = PaperAuthorRepository(session)

    author = author_repo.create(name="Ashish Vaswani")
    authorship = paper_author_repo.create(
        paper_id=paper_a1.id,
        author_id=author.id,
        author_order=1,
    )
    assert authorship.id is not None
    assert author_repo.get_by_name("Ashish Vaswani") is not None

    # Test PaperReference (citation)
    ref_repo = PaperReferenceRepository(session)
    citation = ref_repo.create(
        source_paper_id=paper_a2.id,
        target_paper_id=paper_a1.id,
    )
    assert citation.id is not None
    assert ref_repo.get_by_source_and_target(paper_a2.id, paper_a1.id) is not None

    # Document Chunk repository check
    chunk_repo = DocumentChunkRepository(session)
    doc_chunks = chunk_repo.list_by_document_id(seed_data["doc_a1"].id)
    assert len(doc_chunks) == 2


# ============================================================================
# 13. Database-level endpoint consistency constraint checks
# ============================================================================

def test_database_level_endpoint_consistency_constraints(session, seed_data):
    """Verify that raw database inserts violating endpoint consistency fail with IntegrityError."""
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    paper_a2 = seed_data["paper_a2"]

    # 1. source_type is paper, but source_paper_id is NULL -> rejected at DB level
    with pytest.raises(Exception):
        session.execute(
            text(
                "INSERT INTO graph_relationships "
                "(id, collection_id, relationship_type, source_type, source_id, target_type, target_id, source_paper_id, target_paper_id) "
                "VALUES (:id, :col_id, 'cites', 'paper', :src_id, 'paper', :tgt_id, NULL, :tgt_paper_id)"
            ),
            {
                "id": str(uuid4()).replace("-", ""),
                "col_id": str(col_a.id).replace("-", ""),
                "src_id": str(paper_a1.id).replace("-", ""),
                "tgt_id": str(paper_a2.id).replace("-", ""),
                "tgt_paper_id": str(paper_a2.id).replace("-", ""),
            },
        )
        session.flush()
    session.rollback()

    # 2. source_type is paper, but source_paper_id != source_id -> rejected at DB level
    with pytest.raises(Exception):
        session.execute(
            text(
                "INSERT INTO graph_relationships "
                "(id, collection_id, relationship_type, source_type, source_id, target_type, target_id, source_paper_id, target_paper_id) "
                "VALUES (:id, :col_id, 'cites', 'paper', :src_id, 'paper', :tgt_id, :mismatched_src, :tgt_paper_id)"
            ),
            {
                "id": str(uuid4()).replace("-", ""),
                "col_id": str(col_a.id).replace("-", ""),
                "src_id": str(paper_a1.id).replace("-", ""),
                "mismatched_src": str(paper_a2.id).replace("-", ""),
                "tgt_id": str(paper_a2.id).replace("-", ""),
                "tgt_paper_id": str(paper_a2.id).replace("-", ""),
            },
        )
        session.flush()
    session.rollback()


# ============================================================================
# 14. Relationship update validation and tenant isolation
# ============================================================================

def test_relationship_update_scoped_validation(session, seed_data):
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    col_a = seed_data["col_a"]
    col_b = seed_data["col_b"]
    paper_a1 = seed_data["paper_a1"]
    paper_b1 = seed_data["paper_b1"]

    concept = ent_repo.create(collection_id=col_a.id, name="Attention Mechanism", entity_type="concept")
    rel = rel_repo.create(
        collection_id=col_a.id,
        relationship_type="discusses",
        source_type="paper",
        source_id=paper_a1.id,
        target_type="entity",
        target_id=concept.id,
        confidence=0.7,
        evidence_text="Initial evidence text",
    )

    # 1. Valid update
    updated = rel_repo.update_scoped(
        rel.id,
        collection_id=col_a.id,
        confidence=0.95,
        evidence_text="Updated evidence excerpt from section 3",
        page_number=5,
    )
    assert updated.confidence == 0.95
    assert updated.page_number == 5
    assert updated.evidence_text == "Updated evidence excerpt from section 3"

    # 2. Cannot update with out-of-range confidence
    with pytest.raises(ConstraintError, match="confidence must be between 0.0 and 1.0"):
        rel_repo.update_scoped(rel.id, collection_id=col_a.id, confidence=1.5)

    # 3. Cannot update with invalid page number (< 1)
    with pytest.raises(ConstraintError, match="page_number must be an integer >= 1"):
        rel_repo.update_scoped(rel.id, collection_id=col_a.id, page_number=-1)

    # 4. Cannot update with cross-collection evidence paper
    with pytest.raises(ConstraintError, match="Cross-collection evidence provenance prohibited"):
        rel_repo.update_scoped(rel.id, collection_id=col_a.id, paper_id=paper_b1.id)

    # 5. Accessing with different collection_id raises RecordNotFoundError
    with pytest.raises(RecordNotFoundError):
        rel_repo.update_scoped(rel.id, collection_id=col_b.id, confidence=0.8)
