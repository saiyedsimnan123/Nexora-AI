"""Focused tests for Knowledge Graph Extraction Service (Stage 5 Step 2).

Covers:
1. Deterministic author extraction and relational/graph linking.
2. Citation extraction and matching within collection.
3. Structured paper analysis mapping into graph entities/edges.
4. LLM-assisted extraction from document chunks.
5. Strict validation of malformed LLM responses.
6. Evidence provenance grounding and missing evidence handling.
7. Collection boundary and tenant isolation.
8. Entity and relationship deduplication across repeated runs.
9. Extraction failure without document/chunk data loss (transaction savepoint rollback).
10. Idempotent re-processing without confidence degradation.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4
import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session

from nexora.database.models import (
    Author,
    Base,
    Collection,
    Document,
    DocumentChunkRecord,
    GraphEntity,
    GraphEntityType,
    GraphRelationship,
    IndexingStatus,
    Paper,
    PaperAuthor,
    PaperReference,
    User,
    Workspace,
)
from nexora.database.repository import (
    AuthorRepository,
    ConstraintError,
    GraphEntityRepository,
    GraphRelationshipRepository,
    PaperAuthorRepository,
    PaperReferenceRepository,
    PaperRepository,
    RecordNotFoundError,
)
from nexora.graph.models import (
    ExtractedEntityCandidate,
    ExtractedGraphPayload,
    ExtractedRelationshipCandidate,
    GraphExtractionResult,
    InvalidExtractionPayloadError,
)
from nexora.graph.prompt import GraphExtractionPromptBuilder
from nexora.graph.service import (
    KnowledgeGraphExtractionService,
    _resolve_evidence_item,
)
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem
from nexora.research.analysis import AnalysisField, StructuredPaperAnalysis


class MockLLMService:
    """Deterministic mock LLM service returning structured graph JSON."""

    def __init__(self, response_text: str | None = None) -> None:
        self.call_count = 0
        self.last_messages: list[LLMMessage] = []
        self._response_text = response_text

    def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages

        if self._response_text is not None:
            return LLMResponse(text=self._response_text, model="mock-model")

        # Default valid structured JSON
        payload = {
            "entities": [
                {
                    "name": "Transformer Architecture",
                    "entity_type": "method",
                    "description": "Novel neural network architecture based on self-attention mechanisms.",
                    "aliases": ["Transformers", "Transformer model"],
                },
                {
                    "name": "Self-Attention",
                    "entity_type": "concept",
                    "description": "An attention mechanism relating different positions of a single sequence.",
                },
                {
                    "name": "WMT 2014 English-to-German",
                    "entity_type": "dataset",
                    "description": "Standard translation benchmark dataset.",
                },
                {
                    "name": "BLEU-4",
                    "entity_type": "metric",
                    "description": "Bilingual Evaluation Understudy benchmark score.",
                },
            ],
            "relationships": [
                {
                    "source_name": "Attention Is All You Need",
                    "source_type": "paper",
                    "target_name": "Transformer Architecture",
                    "target_type": "entity",
                    "relationship_type": "proposes",
                    "evidence_label": "[cite: 0]",
                    "evidence_quote": "We propose the Transformer, a model architecture eschewing recurrence...",
                    "confidence": 0.98,
                },
                {
                    "source_name": "Transformer Architecture",
                    "source_type": "entity",
                    "target_name": "Self-Attention",
                    "target_type": "entity",
                    "relationship_type": "relates_to",
                    "evidence_label": "[cite: 0]",
                    "evidence_quote": "The Transformer relies entirely on an attention mechanism to draw global dependencies...",
                    "confidence": 0.95,
                },
                {
                    "source_name": "Attention Is All You Need",
                    "source_type": "paper",
                    "target_name": "WMT 2014 English-to-German",
                    "target_type": "entity",
                    "relationship_type": "uses_dataset",
                    "evidence_label": "[cite: 1]",
                    "evidence_quote": "On the WMT 2014 English-to-German translation task, the big transformer model...",
                    "confidence": 0.94,
                },
            ],
        }
        return LLMResponse(text=json.dumps(payload), model="mock-model")


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
    """Seed user, workspace, collections, papers, documents, and chunks."""
    user = User(email="scientist@nexora.ai")
    session.add(user)
    session.flush()

    workspace = Workspace(user_id=user.id, name="Scientific Discovery Lab")
    session.add(workspace)
    session.flush()

    col_a = Collection(workspace_id=workspace.id, name="NLP Collection")
    col_b = Collection(workspace_id=workspace.id, name="Robotics Collection")
    session.add_all([col_a, col_b])
    session.flush()

    paper_a1 = Paper(
        collection_id=col_a.id,
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional networks...",
        doi="10.48550/arXiv.1706.03762",
        external_id="1706.03762",
    )
    paper_a2 = Paper(
        collection_id=col_a.id,
        title="FlashAttention: Fast and Memory-Efficient Exact Attention",
        abstract="Transformers are slow and memory-hungry on long sequences...",
        doi="10.48550/arXiv.2205.14135",
        external_id="2205.14135",
    )
    paper_b1 = Paper(
        collection_id=col_b.id,
        title="End-to-End Robotic Manipulation",
        abstract="Deep reinforcement learning for robotic grasping...",
    )
    session.add_all([paper_a1, paper_a2, paper_b1])
    session.flush()

    doc_a1 = Document(
        paper_id=paper_a1.id,
        filename="vaswani2017.pdf",
        mime_type="application/pdf",
        file_size=20480,
        checksum="f" * 64,
        storage_key="docs/vaswani2017.pdf",
        indexing_status=IndexingStatus.COMPLETED.value,
    )
    session.add(doc_a1)
    session.flush()

    chunk_a1 = DocumentChunkRecord(
        document_id=doc_a1.id,
        chunk_index=0,
        text="We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism to draw global dependencies between input and output.",
    )
    chunk_a2 = DocumentChunkRecord(
        document_id=doc_a1.id,
        chunk_index=1,
        text="On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU.",
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
# 1. Deterministic author extraction and relational/graph linking
# ============================================================================

def test_deterministic_author_extraction(session, seed_data):
    service = KnowledgeGraphExtractionService()
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    authors = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        authors=authors,
        run_llm=False,
    )

    assert result.status == "success"
    assert result.authors_linked == 3
    assert result.entities_created == 3
    assert result.relationships_created == 3

    # Verify relational tables populated
    author_repo = AuthorRepository(session)
    pa_repo = PaperAuthorRepository(session)
    for idx, name in enumerate(authors):
        author = author_repo.get_by_name(name)
        assert author is not None
        pa = pa_repo.get_by_paper_and_author(paper_a1.id, author.id)
        assert pa is not None
        assert pa.author_order == idx + 1

    # Verify graph entities & relationships
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)

    ent = ent_repo.get_by_canonical_name(
        collection_id=col_a.id,
        entity_type=GraphEntityType.AUTHOR.value,
        canonical_name="ashish vaswani",
    )
    assert ent is not None

    rels = rel_repo.list_by_paper_id(paper_a1.id, collection_id=col_a.id)
    assert len(rels) == 3
    assert all(r.relationship_type == "authored_by" for r in rels)
    assert all(r.confidence == 1.0 for r in rels)


# ============================================================================
# 2. Citation extraction and matching within collection
# ============================================================================

def test_citation_extraction_matches_in_collection(session, seed_data):
    service = KnowledgeGraphExtractionService()
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    paper_a2 = seed_data["paper_a2"]

    # FlashAttention (paper_a2) cites Attention Is All You Need (paper_a1)
    citations = [
        str(paper_a1.id),
        "1706.03762",  # external_id match
        "Non-Existent Target 999",  # unresolvable
    ]

    result = service.extract_paper(
        paper_a2.id,
        collection_id=col_a.id,
        session=session,
        citations=citations,
        run_llm=False,
    )

    assert result.status == "success"
    assert result.citations_linked == 1  # Exactly one unique citation link established

    # Verify PaperReference created
    ref_repo = PaperReferenceRepository(session)
    ref = ref_repo.get_by_source_and_target(paper_a2.id, paper_a1.id)
    assert ref is not None

    # Verify graph relationship created
    rel_repo = GraphRelationshipRepository(session)
    rels = rel_repo.list_by_paper_id(paper_a2.id, collection_id=col_a.id)
    assert len(rels) == 1
    assert rels[0].relationship_type == "cites"
    assert rels[0].source_paper_id == paper_a2.id
    assert rels[0].target_paper_id == paper_a1.id


def test_cross_collection_citations_are_skipped(session, seed_data):
    """Citation targets belonging to another collection must not bridge collections."""
    service = KnowledgeGraphExtractionService()
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    paper_b1 = seed_data["paper_b1"]  # belongs to col_b!

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        citations=[str(paper_b1.id)],
        run_llm=False,
    )

    assert result.status == "success"
    assert result.citations_linked == 0  # Ignored safely


# ============================================================================
# 3. Structured paper analysis mapping into graph entities/edges
# ============================================================================

def test_extract_from_structured_analysis(session, seed_data):
    service = KnowledgeGraphExtractionService()
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    chunk_a1 = seed_data["chunk_a1"]

    analysis = StructuredPaperAnalysis(
        paper_id=str(paper_a1.id),
        title="Attention Is All You Need",
        problem_and_objectives=AnalysisField(
            summary="Sequence transduction without recurrent connections.",
            status="reported",
            sources=[{"chunk_id": str(chunk_a1.id), "page_number": 1, "text": chunk_a1.text}],
        ),
        methodology_and_algorithms=AnalysisField(
            summary="Multi-Head Self-Attention network.",
            status="reported",
            sources=[{"chunk_id": str(chunk_a1.id), "page_number": 2, "text": chunk_a1.text}],
        ),
        models_and_datasets=AnalysisField(
            summary="WMT 2014 English-to-German translation benchmark.",
            status="reported",
            sources=[{"chunk_id": str(chunk_a1.id), "page_number": 3, "text": chunk_a1.text}],
        ),
        experiments_and_metrics=AnalysisField(
            summary="BLEU score evaluation.",
            status="reported",
            sources=[{"chunk_id": str(chunk_a1.id), "page_number": 4, "text": chunk_a1.text}],
        ),
        results_and_findings=AnalysisField(summary="Achieved SOTA.", status="reported"),
        limitations=AnalysisField(summary="Quadratic memory complexity.", status="reported"),
        future_work=AnalysisField(summary="Apply to other modalities.", status="reported"),
    )

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        analysis=analysis,
        run_llm=False,
    )

    assert result.status == "success"
    assert result.entities_created == 4
    assert result.relationships_created == 4

    rel_repo = GraphRelationshipRepository(session)
    rels = rel_repo.list_by_paper_id(paper_a1.id, collection_id=col_a.id)
    types = {r.relationship_type for r in rels}
    assert types == {"proposes", "uses_dataset", "reports_metric", "discusses"}


# ============================================================================
# 4. LLM-assisted extraction from document chunks
# ============================================================================

def test_llm_assisted_extraction_from_chunks(session, seed_data):
    mock_llm = MockLLMService()
    service = KnowledgeGraphExtractionService(llm_service=mock_llm)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        run_llm=True,
    )

    assert result.status == "success"
    assert mock_llm.call_count == 1
    assert result.entities_created == 4
    assert result.relationships_created == 3

    ent_repo = GraphEntityRepository(session)
    transformer_ent = ent_repo.get_by_canonical_name(
        collection_id=col_a.id,
        entity_type="method",
        canonical_name="transformer architecture",
    )
    assert transformer_ent is not None
    assert transformer_ent.aliases == ["Transformers", "Transformer model"]

    rel_repo = GraphRelationshipRepository(session)
    rels = rel_repo.list_by_paper_id(paper_a1.id, collection_id=col_a.id)
    assert len(rels) >= 2


# ============================================================================
# 5. Strict validation of malformed LLM responses
# ============================================================================

def test_malformed_llm_json_handled_safely(session, seed_data):
    broken_llm = MockLLMService(response_text="This is not valid JSON at all!")
    service = KnowledgeGraphExtractionService(llm_service=broken_llm)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        run_llm=True,
    )

    assert result.status == "failed"
    assert len(result.errors) > 0
    assert any("invalid json" in err.lower() for err in result.errors)


def test_invalid_schema_payload_handled_safely(session, seed_data):
    # Missing required 'name' or invalid entity type
    invalid_payload = {
        "entities": [{"name": "Missing Type"}],
        "relationships": [],
    }
    broken_llm = MockLLMService(response_text=json.dumps(invalid_payload))
    service = KnowledgeGraphExtractionService(llm_service=broken_llm)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        run_llm=True,
    )

    assert result.status == "failed"
    assert len(result.errors) > 0


# ============================================================================
# 6. Evidence provenance grounding and missing evidence handling
# ============================================================================

def test_evidence_provenance_grounding(session, seed_data):
    mock_llm = MockLLMService()
    service = KnowledgeGraphExtractionService(llm_service=mock_llm)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    chunk_a1 = seed_data["chunk_a1"]

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        run_llm=True,
    )

    assert result.status == "success"
    rel_repo = GraphRelationshipRepository(session)
    rels = rel_repo.list_by_paper_id(paper_a1.id, collection_id=col_a.id)

    proposes_rel = next((r for r in rels if r.relationship_type == "proposes"), None)
    assert proposes_rel is not None
    assert proposes_rel.chunk_id == chunk_a1.id
    assert proposes_rel.has_evidence is True
    assert "eschewing recurrence" in proposes_rel.evidence_text


# ============================================================================
# 7. Collection boundary and tenant isolation
# ============================================================================

def test_cross_collection_paper_extraction_rejected(session, seed_data):
    service = KnowledgeGraphExtractionService()
    col_a = seed_data["col_a"]
    paper_b1 = seed_data["paper_b1"]  # belongs to col_b!

    # Attempting to extract paper_b1 within collection A must fail closed
    result = service.extract_paper(
        paper_b1.id,
        collection_id=col_a.id,
        session=session,
        run_llm=False,
    )

    assert result.status == "failed"
    assert any("Cross-collection extraction prohibited" in err for err in result.errors)


# ============================================================================
# 8. Repeated processing of same paper (idempotence)
# ============================================================================

def test_repeated_extraction_is_idempotent(session, seed_data):
    mock_llm = MockLLMService()
    service = KnowledgeGraphExtractionService(llm_service=mock_llm)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    authors = ["Ashish Vaswani", "Noam Shazeer"]

    # First pass
    res1 = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        authors=authors,
        run_llm=True,
    )
    assert res1.status == "success"
    total_entities_1 = res1.total_entities
    total_rels_1 = res1.total_relationships

    # Second pass
    res2 = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        authors=authors,
        run_llm=True,
    )
    assert res2.status == "success"
    assert res2.entities_created == 0  # Reused all
    assert res2.relationships_created == 0  # Reused all
    assert res2.entities_reused == total_entities_1
    assert res2.relationships_reused == total_rels_1

    # Verify DB table row counts didn't double
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    assert len(ent_repo.list_by_collection_id(col_a.id)) == total_entities_1
    assert len(rel_repo.list_by_collection_id(col_a.id)) == total_rels_1


# ============================================================================
# 9. Extraction failure without document/chunk data loss (savepoint rollback)
# ============================================================================

def test_extraction_failure_preserves_documents_and_chunks(session, seed_data):
    broken_llm = MockLLMService(response_text="INVALID JSON THAT CRASHES")
    service = KnowledgeGraphExtractionService(llm_service=broken_llm)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]
    doc_a1 = seed_data["doc_a1"]
    chunk_a1 = seed_data["chunk_a1"]

    # Verify paper and doc exist before extraction
    assert session.get(Paper, paper_a1.id) is not None
    assert session.get(Document, doc_a1.id) is not None
    assert session.get(DocumentChunkRecord, chunk_a1.id) is not None

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        authors=["Ashish Vaswani", "Noam Shazeer"],
        run_llm=True,
    )
    assert result.status == "failed"
    assert result.entities_created == 0
    assert result.entities_reused == 0
    assert result.relationships_created == 0
    assert result.relationships_reused == 0
    assert result.authors_linked == 0
    assert len(result.entities) == 0
    assert len(result.relationships) == 0
    assert len(result.errors) > 0

    # Pre-existing data remains 100% intact after failed extraction
    assert session.get(Paper, paper_a1.id) is not None
    assert session.get(Document, doc_a1.id) is not None
    assert session.get(DocumentChunkRecord, chunk_a1.id) is not None

    # No partial graph records survived in DB
    ent_repo = GraphEntityRepository(session)
    rel_repo = GraphRelationshipRepository(session)
    assert len(ent_repo.list_by_collection_id(col_a.id)) == 0
    assert len(rel_repo.list_by_collection_id(col_a.id)) == 0


# ============================================================================
# 10. Audit regression tests: evidence validation, parsing, rollback, incoming
# ============================================================================

def test_unverified_chunk_uuid_rejected_safely(session, seed_data):
    """An unverified UUID not in extraction context must not crash extraction or fabricate chunk_id."""
    fake_chunk_id = uuid4()
    payload = {
        "entities": [
            {
                "name": "Attention Mechanism",
                "entity_type": "concept",
                "description": "General attention mechanism.",
            }
        ],
        "relationships": [
            {
                "source_name": "Attention Is All You Need",
                "source_type": "paper",
                "target_name": "Attention Mechanism",
                "target_type": "entity",
                "relationship_type": "discusses",
                "evidence_label": str(fake_chunk_id),
                "evidence_quote": "Attention is all you need.",
                "confidence": 0.85,
            }
        ],
    }
    llm = MockLLMService(response_text=json.dumps(payload))
    service = KnowledgeGraphExtractionService(llm_service=llm)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        run_llm=True,
    )

    assert result.status == "success"
    rel_repo = GraphRelationshipRepository(session)
    rels = rel_repo.list_by_paper_id(paper_a1.id, collection_id=col_a.id)
    assert len(rels) == 1
    # Chunk ID must be None because fake UUID was not in context
    assert rels[0].chunk_id is None
    assert rels[0].relationship_type == "discusses"


def test_evidence_label_strict_parsing_documented_formats(seed_data):
    """Documented citation labels must resolve accurately to context items."""
    chunk_a1 = seed_data["chunk_a1"]
    doc_a1 = seed_data["doc_a1"]
    paper_a1 = seed_data["paper_a1"]

    item0 = EvidenceItem(
        document_id=str(doc_a1.id),
        chunk_id=str(chunk_a1.id),
        text="The Transformer model architecture.",
        score=1.0,
        metadata={"paper_id": str(paper_a1.id), "page_number": 2},
        position=0,
    )
    context = EvidenceContext(query="test", items=[item0])

    # 1. [cite: 0]
    cid, page, text = _resolve_evidence_item("[cite: 0]", context)
    assert cid == chunk_a1.id
    assert page == 2

    # 2. cite: 0
    cid, page, text = _resolve_evidence_item("cite: 0", context)
    assert cid == chunk_a1.id

    # 3. [0]
    cid, page, text = _resolve_evidence_item("[0]", context)
    assert cid == chunk_a1.id

    # 4. [cite: 0] (Page 2)
    cid, page, text = _resolve_evidence_item("[cite: 0] (Page 2)", context)
    assert cid == chunk_a1.id

    # 5. [E1]
    cid, page, text = _resolve_evidence_item("[E1]", context)
    assert cid == chunk_a1.id

    # 6. E1
    cid, page, text = _resolve_evidence_item("E1", context)
    assert cid == chunk_a1.id

    # 7. Direct verified chunk UUID
    cid, page, text = _resolve_evidence_item(str(chunk_a1.id), context)
    assert cid == chunk_a1.id


def test_evidence_label_rejects_unrelated_text_and_out_of_range(seed_data):
    """Unrelated numeric strings and out-of-range indices must not resolve to unrelated chunks."""
    chunk_a1 = seed_data["chunk_a1"]
    doc_a1 = seed_data["doc_a1"]
    paper_a1 = seed_data["paper_a1"]

    item0 = EvidenceItem(
        document_id=str(doc_a1.id),
        chunk_id=str(chunk_a1.id),
        text="Item 0 text",
        score=1.0,
        metadata={"paper_id": str(paper_a1.id)},
        position=0,
    )
    context = EvidenceContext(query="test", items=[item0])

    # Unrelated numeric strings must not resolve to chunk 0 or any other chunk
    assert _resolve_evidence_item("Section 2", context) == (None, None, None)
    assert _resolve_evidence_item("Page 3", context) == (None, None, None)
    assert _resolve_evidence_item("Table 1", context) == (None, None, None)
    assert _resolve_evidence_item("Equation 0", context) == (None, None, None)

    # Out of range indices
    assert _resolve_evidence_item("[cite: 999]", context) == (None, None, None)
    assert _resolve_evidence_item("[E999]", context) == (None, None, None)
    assert _resolve_evidence_item("cite: 5", context) == (None, None, None)

    # Unverified UUID
    assert _resolve_evidence_item(str(uuid4()), context) == (None, None, None)

    # Malformed strings
    assert _resolve_evidence_item("not a citation", context) == (None, None, None)
    assert _resolve_evidence_item("", context) == (None, None, None)
    assert _resolve_evidence_item(None, context) == (None, None, None)


def test_incoming_entity_to_current_paper_relationship_supported(session, seed_data):
    """Incoming entity-to-paper relationships are supported while genuine self-loops are blocked."""
    payload = {
        "entities": [
            {
                "name": "Transformer Architecture",
                "entity_type": "method",
                "description": "Novel neural network architecture.",
            }
        ],
        "relationships": [
            # Valid incoming relationship: entity -> current paper
            {
                "source_name": "Transformer Architecture",
                "source_type": "entity",
                "target_name": "Attention Is All You Need",
                "target_type": "paper",
                "relationship_type": "introduced_in",
                "evidence_label": "[cite: 0]",
                "evidence_quote": "We propose the Transformer.",
                "confidence": 0.95,
            },
            # Genuine self-loop: current paper -> current paper
            {
                "source_name": "Attention Is All You Need",
                "source_type": "paper",
                "target_name": "Attention Is All You Need",
                "target_type": "paper",
                "relationship_type": "cites",
                "evidence_label": "[cite: 0]",
                "evidence_quote": "Self citation loop.",
                "confidence": 0.95,
            },
        ],
    }
    llm = MockLLMService(response_text=json.dumps(payload))
    service = KnowledgeGraphExtractionService(llm_service=llm)
    col_a = seed_data["col_a"]
    paper_a1 = seed_data["paper_a1"]

    result = service.extract_paper(
        paper_a1.id,
        collection_id=col_a.id,
        session=session,
        run_llm=True,
    )

    assert result.status == "success"
    rel_repo = GraphRelationshipRepository(session)
    rels = rel_repo.list_by_paper_id(paper_a1.id, collection_id=col_a.id)

    # Genuine self-loop was blocked
    cites_self = [r for r in rels if r.relationship_type == "cites" and r.source_id == r.target_id]
    assert len(cites_self) == 0

    # Incoming entity-to-paper relationship was created
    introduced_in_rel = next((r for r in rels if r.relationship_type == "introduced_in"), None)
    assert introduced_in_rel is not None
    assert introduced_in_rel.source_type == "entity"
    assert introduced_in_rel.target_type == "paper"
    assert introduced_in_rel.target_paper_id == paper_a1.id
