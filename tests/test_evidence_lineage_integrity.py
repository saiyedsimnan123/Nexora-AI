"""End-to-End Evidence Lineage Integrity Tests (Milestone 12C-29).

Validates that evidence identity (document_id, chunk_id, paper_id, filename, position, label)
and citations remain completely accurate and traceable across the complete research workflow:
Ingestion -> Storage -> Retrieval -> Structured Analysis -> Claim Verification -> Gap Discovery -> API Serialization.

Covers all Milestone 12C-29 specifications:
1. End-to-end lineage tracing proving valid evidence retains original identity and citations across full stack.
2. De-duplication and reference integrity under repeated and duplicate citation labels.
3. Fail-closed rejection of missing, blank, or malformed metadata (document_id, chunk_id, paper_id).
4. Paper scoping isolation across multi-paper corpora (mismatched paper IDs rejected).
5. Unsupported and contradicted claims remain uncertain/unresolved and never silently upgraded to verified findings.
6. API wire responses strictly preserve separation of documented findings (confidence=1.0) and inferred gaps (confidence<1.0).
7. Zero-evidence and empty corpus handling produce explicit, safe outcomes with zero hallucinated evidence.
8. Full integration with FastAPI application and endpoints.
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.api.app import create_app
from nexora.api.models import (
    PaperAnalysisResponse,
    PaperComparisonResponse,
    ResearchGapsResponse,
    ResearchVerifyResponse,
)
from nexora.api.service import ResearchApplicationService
from nexora.database.models import (
    Base,
    Collection,
    Document,
    DocumentChunkRecord,
    Paper,
    User,
    Workspace,
)
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.llm.base import LLMProvider
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.service import LLMService
from nexora.research.analysis import AnalysisField, StructuredPaperAnalysis
from nexora.research.analysis_service import PaperAnalysisService, PaperComparisonService
from nexora.research.gap import GapCategory, ResearchGap, ResearchGapReport
from nexora.research.gap_service import ResearchGapService
from nexora.research.verification import (
    AnswerVerificationResult,
    ClaimToEvidenceVerifier,
    ClaimVerification,
    ClaimVerificationStatus,
    RawClaim,
)
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem


# ---------------------------------------------------------------------------
# Test Doubles
# ---------------------------------------------------------------------------
class ControllableScriptedLLMProvider(LLMProvider):
    """Controllable LLM Provider recording calls and returning scripted responses."""

    def __init__(
        self,
        responses: list[str] | None = None,
        default_response: str = "",
    ):
        self.responses = list(responses) if responses else []
        self.default_response = default_response
        self.calls: list[list[LLMMessage]] = []

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        if self.responses:
            text = self.responses.pop(0)
        else:
            text = self.default_response
        return LLMResponse(
            text=text,
            model=model or "mock-lineage-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            raw={"provider": "mock", "finish_reason": "stop"},
        )


class FakeResearchServiceForLineage:
    """Mock research application service supporting ask and verify with full evidence context."""

    def __init__(self, context: EvidenceContext | None = None, verifier: ClaimToEvidenceVerifier | None = None):
        self.context = context
        self.verifier = verifier or ClaimToEvidenceVerifier()

    def ask(self, request: Any) -> Any:
        from nexora.api.models import ResearchResponse

        return ResearchResponse(
            text="Transformer self-attention exhibits quadratic complexity O(N^2) [E1].",
            query=request.query,
            model="mock",
            usage={"total_tokens": 15},
            evidence_count=len(self.context.items) if self.context else 0,
        )

    def verify(self, request: Any) -> Any:
        ctx = self.context or EvidenceContext(query=request.query, items=[])
        answer_text = request.text or "Self-attention has quadratic complexity [E1]."
        v_res = self.verifier.verify_answer(answer_text, ctx, query=request.query)
        return ResearchVerifyResponse(
            text=answer_text,
            query=request.query,
            model="mock",
            usage={"total_tokens": 20},
            evidence_count=len(ctx.items),
            verification=v_res.to_dict(),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def lineage_database():
    """In-memory SQLite database populated with two papers, documents, and traceable chunks."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = session_factory()
    try:
        user = User(email=f"lineage_tester-{uuid.uuid4()}@example.com")
        session.add(user)
        session.flush()

        workspace = Workspace(name="Lineage Workspace", user_id=user.id)
        session.add(workspace)
        session.flush()

        collection = Collection(
            name="Lineage Benchmark Collection",
            workspace_id=workspace.id,
        )
        session.add(collection)
        session.flush()

        # Paper 1: Transformer
        paper1 = PaperRepository(session).create(
            collection_id=collection.id,
            title="Attention Is All You Need",
            abstract="Transformer sequence transduction.",
        )
        doc1 = DocumentRepository(session).create(
            paper_id=paper1.id,
            filename="transformer_2017.pdf",
            file_size=2048,
            checksum="1" * 64,
            storage_key="s3://transformer_2017.pdf",
            indexing_status="completed",
        )
        chunk_repo = DocumentChunkRepository(session)
        chunk1_0 = chunk_repo.create(
            document_id=doc1.id,
            chunk_index=0,
            text="Self-attention connects all positions with constant number of operations.",
        )
        chunk1_1 = chunk_repo.create(
            document_id=doc1.id,
            chunk_index=1,
            text="A key limitation is the quadratic computational and memory complexity O(N^2) in sequence length.",
        )

        # Paper 2: FlashAttention
        paper2 = PaperRepository(session).create(
            collection_id=collection.id,
            title="FlashAttention: Fast and Memory-Efficient Exact Attention",
            abstract="IO-aware exact attention.",
        )
        doc2 = DocumentRepository(session).create(
            paper_id=paper2.id,
            filename="flashattention_2022.pdf",
            file_size=4096,
            checksum="2" * 64,
            storage_key="s3://flashattention_2022.pdf",
            indexing_status="completed",
        )
        chunk2_0 = chunk_repo.create(
            document_id=doc2.id,
            chunk_index=0,
            text="FlashAttention uses tiling to compute exact attention in SRAM without materializing full N x N intermediate matrices in HBM.",
        )
        chunk2_1 = chunk_repo.create(
            document_id=doc2.id,
            chunk_index=1,
            text="A current limitation is that backward pass recomputes attention matrices and requires specific GPU memory architectures.",
        )

        session.commit()

        yield {
            "engine": engine,
            "session_factory": session_factory,
            "paper1_id": str(paper1.id),
            "paper2_id": str(paper2.id),
            "collection_id": str(collection.id),
            "doc1_id": str(doc1.id),
            "doc2_id": str(doc2.id),
            "chunk1_0_id": str(chunk1_0.id),
            "chunk1_1_id": str(chunk1_1.id),
            "chunk2_0_id": str(chunk2_0.id),
            "chunk2_1_id": str(chunk2_1.id),
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 1. Full Pipeline Lineage Tracing Tests
# ---------------------------------------------------------------------------
def test_end_to_end_lineage_trace_from_ingestion_to_api(lineage_database):
    """Scenario 1: Trace document, chunk, and paper identity from database through

    structured analysis, claim verification, research gap discovery, and API serialization.
    """
    p1_id = lineage_database["paper1_id"]
    p2_id = lineage_database["paper2_id"]
    doc1_id = lineage_database["doc1_id"]
    doc2_id = lineage_database["doc2_id"]
    chunk1_1_id = lineage_database["chunk1_1_id"]
    chunk2_1_id = lineage_database["chunk2_1_id"]

    # Scripted LLM outputs
    analysis_p1_json = json.dumps({
        "problem_and_objectives": {"summary": "Replace recurrence with self-attention.", "status": "reported", "evidence_labels": ["[E1]"]},
        "methodology_and_algorithms": {"summary": "Multi-head attention architecture.", "status": "reported", "evidence_labels": ["[E1]"]},
        "models_and_datasets": {"summary": "WMT 2014 English-to-German.", "status": "reported", "evidence_labels": ["[E1]"]},
        "experiments_and_metrics": {"summary": "BLEU score benchmarks.", "status": "reported", "evidence_labels": ["[E1]"]},
        "results_and_findings": {"summary": "State-of-the-art translation performance.", "status": "reported", "evidence_labels": ["[E1]"]},
        "limitations": {"summary": "Quadratic computational and memory complexity O(N^2).", "status": "reported", "evidence_labels": ["[E2]"]},
        "future_work": {"summary": "Not reported in the provided evidence.", "status": "not_reported", "evidence_labels": []},
    })
    analysis_p2_json = json.dumps({
        "problem_and_objectives": {"summary": "Accelerate attention via IO awareness.", "status": "reported", "evidence_labels": ["[E1]"]},
        "methodology_and_algorithms": {"summary": "Tiling and kernel fusion.", "status": "reported", "evidence_labels": ["[E1]"]},
        "models_and_datasets": {"summary": "BERT, GPT-2, Long-range Arena.", "status": "reported", "evidence_labels": ["[E1]"]},
        "experiments_and_metrics": {"summary": "Wall-clock speedup and peak memory.", "status": "reported", "evidence_labels": ["[E1]"]},
        "results_and_findings": {"summary": "2-4x wall clock speedup.", "status": "reported", "evidence_labels": ["[E1]"]},
        "limitations": {"summary": "Backward pass recomputes attention matrices.", "status": "reported", "evidence_labels": ["[E2]"]},
        "future_work": {"summary": "Not reported in the provided evidence.", "status": "not_reported", "evidence_labels": []},
    })
    gap_synth_json = json.dumps({
        "inferred_gaps": [{
            "title": "Hardware Generalization Gap",
            "description": "Exact attention tiling requires specific GPU SRAM architectures.",
            "reasoning": "Synthesized across Transformer complexity and FlashAttention hardware bounds.",
            "source_paper_ids": [p1_id, p2_id],
            "evidence_labels": ["[E2]", "[E4]"],  # E2 from P1, E4 from P2 in combined context
            "confidence": 0.82,
            "limitations": ["Hypothesis requires testing across non-NVIDIA accelerators."],
        }],
        "summary": "Identified documented limitations and cross-paper hardware dependency gap.",
        "limitations": ["Evaluated strictly across the 2 provided papers."],
    })

    llm_provider = ControllableScriptedLLMProvider(
        responses=[analysis_p1_json, analysis_p1_json, analysis_p2_json, gap_synth_json]
    )
    llm_service = LLMService(llm_provider)

    analysis_service = PaperAnalysisService(
        llm_service,
        session_factory=lineage_database["session_factory"],
    )
    comparison_service = PaperComparisonService(
        analysis_service,
        llm_service,
    )
    gap_service = ResearchGapService(
        llm_service,
        paper_analysis_service=analysis_service,
        session_factory=lineage_database["session_factory"],
    )

    app = create_app(
        FakeResearchServiceForLineage(),
        analysis_service=analysis_service,
        comparison_service=comparison_service,
        gap_service=gap_service,
        session_factory=lineage_database["session_factory"],
    )
    client = TestClient(app)

    # 1. Test Paper Analysis Lineage via API
    resp_analysis = client.post(f"/api/v1/papers/{p1_id}/analyze")
    assert resp_analysis.status_code == 200, resp_analysis.text
    data_analysis = resp_analysis.json()
    assert data_analysis["paper_id"] == p1_id
    lim_field = data_analysis["limitations"]
    assert lim_field["status"] == "reported"
    assert lim_field["evidence_labels"] == ["[E2]"]
    assert len(lim_field["sources"]) == 1
    src = lim_field["sources"][0]
    # Verify EXACT provenance trace back to database entities
    assert src["document_id"] == doc1_id
    assert src["chunk_id"] == chunk1_1_id
    assert src["paper_id"] == p1_id
    assert src["filename"] == "transformer_2017.pdf"
    assert "quadratic computational and memory complexity" in src["text"]

    # 2. Test Research Gap Discovery Lineage via API
    resp_gaps = client.post("/api/v1/research/gaps", json={"paper_ids": [p1_id, p2_id]})
    assert resp_gaps.status_code == 200, resp_gaps.text
    data_gaps = resp_gaps.json()

    # Documented gaps: exactly 2 author-documented limitations
    assert len(data_gaps["documented_gaps"]) == 2
    doc_gaps = {g["source_paper_ids"][0]: g for g in data_gaps["documented_gaps"]}

    # Verify Paper 1 documented gap
    p1_gap = doc_gaps[p1_id]
    assert p1_gap["category"] == "documented"
    assert p1_gap["confidence"] == 1.0
    assert len(p1_gap["supporting_sources"]) == 1
    p1_src = p1_gap["supporting_sources"][0]
    assert p1_src["document_id"] == doc1_id
    assert p1_src["chunk_id"] == chunk1_1_id
    assert p1_src["paper_id"] == p1_id
    assert p1_src["filename"] == "transformer_2017.pdf"

    # Verify Paper 2 documented gap
    p2_gap = doc_gaps[p2_id]
    assert p2_gap["category"] == "documented"
    assert p2_gap["confidence"] == 1.0
    assert len(p2_gap["supporting_sources"]) == 1
    p2_src = p2_gap["supporting_sources"][0]
    assert p2_src["document_id"] == doc2_id
    assert p2_src["chunk_id"] == chunk2_1_id
    assert p2_src["paper_id"] == p2_id
    assert p2_src["filename"] == "flashattention_2022.pdf"

    # Inferred gaps: cross-paper candidate hypothesis
    assert len(data_gaps["inferred_gaps"]) == 1
    inf_gap = data_gaps["inferred_gaps"][0]
    assert inf_gap["category"] == "inferred"
    assert inf_gap["confidence"] == 0.82
    assert inf_gap["confidence"] < 1.0
    assert len(inf_gap["supporting_sources"]) == 2
    # Verify both source chunks trace back to respective distinct papers and documents
    src_pids = [s["paper_id"] for s in inf_gap["supporting_sources"]]
    src_doc_ids = [s["document_id"] for s in inf_gap["supporting_sources"]]
    src_chunk_ids = [s["chunk_id"] for s in inf_gap["supporting_sources"]]
    assert p1_id in src_pids and p2_id in src_pids
    assert doc1_id in src_doc_ids and doc2_id in src_doc_ids
    assert chunk1_1_id in src_chunk_ids and chunk2_1_id in src_chunk_ids


# ---------------------------------------------------------------------------
# 2. De-duplication and Reference Integrity Tests
# ---------------------------------------------------------------------------
def test_lineage_preservation_under_duplicate_and_repeated_citations(lineage_database):
    """Scenario 2: Repeated citation references de-duplicate cleanly while preserving identity."""
    from nexora.research.analysis_service import _parse_field_dict

    p1_id = lineage_database["paper1_id"]
    doc1_id = lineage_database["doc1_id"]
    chunk1_0_id = lineage_database["chunk1_0_id"]

    item = EvidenceItem(
        document_id=doc1_id,
        chunk_id=chunk1_0_id,
        text="Self-attention connects all positions.",
        score=1.0,
        metadata={"paper_id": p1_id, "filename": "transformer.pdf"},
        position=0,
    )
    context = EvidenceContext(query="query", items=[item])

    # LLM outputs repeated identical citations
    raw_dict = {
        "summary": "Attention connects all positions.",
        "status": "reported",
        "evidence_labels": ["[E1]", "[E1]", "E1", "[e1]"],
    }
    field = _parse_field_dict(raw_dict, context, filter_citations=True, expected_paper_id=p1_id)
    # De-duplicated cleanly
    assert field.evidence_labels == ["[E1]"]
    assert len(field.sources) == 1
    assert field.sources[0]["document_id"] == doc1_id
    assert field.sources[0]["chunk_id"] == chunk1_0_id
    assert field.sources[0]["paper_id"] == p1_id


# ---------------------------------------------------------------------------
# 3. Fail-Closed Metadata Validation Tests
# ---------------------------------------------------------------------------
def test_lineage_rejection_missing_chunk_metadata_end_to_end():
    """Scenario 3: Evidence chunks missing document_id or chunk_id or paper_id fail closed."""
    # 1. Missing document_id in EvidenceItem raises ValueError/TypeError
    with pytest.raises((ValueError, TypeError)):
        EvidenceItem(
            document_id="",  # Empty document_id
            chunk_id="chk-1",
            text="Evidence",
            score=1.0,
            metadata={"paper_id": "p1"},
            position=0,
        )

    # 2. Missing chunk_id in EvidenceItem raises ValueError/TypeError
    with pytest.raises((ValueError, TypeError)):
        EvidenceItem(
            document_id="doc-1",
            chunk_id="   ",  # Blank chunk_id
            text="Evidence",
            score=1.0,
            metadata={"paper_id": "p1"},
            position=0,
        )

    # 3. Missing paper_id in EvidenceItem fails closed in paper-scoped analysis
    from nexora.research.analysis_service import _parse_field_dict

    unscoped_item = EvidenceItem(
        document_id="doc-1",
        chunk_id="chk-1",
        text="Evidence without paper identity.",
        score=1.0,
        metadata={},  # Missing paper_id!
        position=0,
    )
    context = EvidenceContext(query="test", items=[unscoped_item])
    field = _parse_field_dict(
        {"summary": "Summary", "status": "reported", "evidence_labels": ["[E1]"]},
        context,
        filter_citations=True,
        expected_paper_id="expected-paper-123",
    )
    # Rejected and downgraded to uncertain
    assert field.evidence_labels == []
    assert field.sources == []
    assert field.status == "uncertain"


# ---------------------------------------------------------------------------
# 4. Multi-Paper Scoping Isolation Tests
# ---------------------------------------------------------------------------
def test_lineage_rejection_mismatched_paper_evidence_across_multiple_papers():
    """Scenario 4: Evidence from Paper B cannot satisfy Paper A's scoped claims."""
    from nexora.research.analysis_service import _parse_field_dict

    item_b = EvidenceItem(
        document_id="doc-paper-b",
        chunk_id="chk-paper-b",
        text="FlashAttention SRAM tiling.",
        score=1.0,
        metadata={"paper_id": "paper-B-uuid"},
        position=0,
    )
    context = EvidenceContext(query="test", items=[item_b])

    # Paper A claims [E1], but [E1] belongs to Paper B
    field = _parse_field_dict(
        {"summary": "Paper A claims FlashAttention technique.", "status": "reported", "evidence_labels": ["[E1]"]},
        context,
        filter_citations=True,
        expected_paper_id="paper-A-uuid",
    )
    assert field.evidence_labels == []
    assert field.sources == []
    assert field.status == "uncertain"

    # ClaimToEvidenceVerifier also flags mismatch
    verifier = ClaimToEvidenceVerifier(expected_paper_id="paper-A-uuid")
    claim = RawClaim(
        claim_text="Paper A utilizes SRAM tiling [E1].",
        cited_labels=["[E1]"],
    )
    result = verifier.verify_claim(claim, context)
    assert result.status == ClaimVerificationStatus.UNRESOLVED.value
    assert result.is_unresolved is True
    assert "different paper" in result.reasoning


# ---------------------------------------------------------------------------
# 5. Unsupported Claims Remain Uncertain or Unresolved Tests
# ---------------------------------------------------------------------------
def test_unsupported_claim_never_silently_upgraded_to_verified(lineage_database):
    """Scenario 5: An unsupported or contradicted claim is never upgraded to a verified documented gap."""
    p1_id = lineage_database["paper1_id"]
    p2_id = lineage_database["paper2_id"]

    # Analysis claims 'reported' for future work but cites non-existent label [E99]
    analysis_p1 = json.dumps({
        "problem_and_objectives": {"summary": "Objective", "status": "reported", "evidence_labels": ["[E1]"]},
        "methodology_and_algorithms": {"summary": "Algorithm", "status": "reported", "evidence_labels": ["[E1]"]},
        "models_and_datasets": {"summary": "Dataset", "status": "reported", "evidence_labels": ["[E1]"]},
        "experiments_and_metrics": {"summary": "Metrics", "status": "reported", "evidence_labels": ["[E1]"]},
        "results_and_findings": {"summary": "Findings", "status": "reported", "evidence_labels": ["[E1]"]},
        "limitations": {"summary": "Unsubstantiated limitation [E99].", "status": "reported", "evidence_labels": ["[E99]"]},
        "future_work": {"summary": "Uncited future direction.", "status": "reported", "evidence_labels": []},
    })
    analysis_p2 = json.dumps({
        "problem_and_objectives": {"summary": "Objective", "status": "reported", "evidence_labels": ["[E1]"]},
        "methodology_and_algorithms": {"summary": "Algorithm", "status": "reported", "evidence_labels": ["[E1]"]},
        "models_and_datasets": {"summary": "Dataset", "status": "reported", "evidence_labels": ["[E1]"]},
        "experiments_and_metrics": {"summary": "Metrics", "status": "reported", "evidence_labels": ["[E1]"]},
        "results_and_findings": {"summary": "Findings", "status": "reported", "evidence_labels": ["[E1]"]},
        "limitations": {"summary": "Not reported in the provided evidence.", "status": "not_reported", "evidence_labels": []},
        "future_work": {"summary": "Not reported in the provided evidence.", "status": "not_reported", "evidence_labels": []},
    })
    gap_synth = json.dumps({"inferred_gaps": [], "summary": "Summary", "limitations": []})

    llm = LLMService(ControllableScriptedLLMProvider(responses=[analysis_p1, analysis_p2, gap_synth]))
    gap_service = ResearchGapService(
        llm,
        session_factory=lineage_database["session_factory"],
    )

    report = gap_service.discover_gaps_for_papers([p1_id, p2_id])
    # Both unverified claims must be rejected from documented_gaps
    assert len(report.documented_gaps) == 0
    # Both must be recorded in limitations
    assert any("uncertain/unverified" in lim or "did not resolve" in lim for lim in report.limitations)


# ---------------------------------------------------------------------------
# 6. API Response Distinction of Documented vs Inferred Gaps Tests
# ---------------------------------------------------------------------------
def test_api_response_preserves_strict_separation_of_documented_and_inferred_gaps(lineage_database):
    """Scenario 6: API wire representation preserves documented findings vs inferred hypotheses."""
    p1_id = lineage_database["paper1_id"]

    doc_source = {
        "label": "[E1]",
        "document_id": lineage_database["doc1_id"],
        "chunk_id": lineage_database["chunk1_1_id"],
        "paper_id": p1_id,
        "filename": "transformer.pdf",
        "score": 1.0,
        "text": "Quadratic memory complexity.",
        "position": 0,
    }
    doc_gap = ResearchGap(
        gap_id="gap-doc-1",
        category=GapCategory.DOCUMENTED.value,
        title="Documented Limitation: Quadratic Scaling",
        description="Authors reported quadratic scaling.",
        reasoning="Explicit author statement.",
        source_paper_ids=[p1_id],
        evidence_labels=["[E1]"],
        supporting_sources=[doc_source],
        contradicting_sources=[],
        confidence=1.0,
        limitations=["Author-reported finding."],
    )

    inf_gap = ResearchGap(
        gap_id="gap-inf-1",
        category=GapCategory.INFERRED.value,
        title="Candidate Gap: Cross-Layer Reuse",
        description="Inferred potential for cross-layer KV caching.",
        reasoning="Synthesized across literature.",
        source_paper_ids=[p1_id],
        evidence_labels=["[E1]"],
        supporting_sources=[doc_source],
        contradicting_sources=[],
        confidence=0.75,
        limitations=["Requires experimental verification."],
    )

    report = ResearchGapReport(
        paper_ids=[p1_id],
        collection_id=None,
        documented_gaps=[doc_gap],
        inferred_gaps=[inf_gap],
        total_gaps_count=2,
        summary="Separation test report.",
        limitations=[],
        model="mock-model",
    )

    class StubGapService:
        def discover_gaps_for_papers(self, paper_ids, **kwargs):
            return report

    app = create_app(
        FakeResearchServiceForLineage(),
        gap_service=StubGapService(),
        session_factory=lineage_database["session_factory"],
    )
    client = TestClient(app)

    resp = client.post("/api/v1/research/gaps", json={"paper_ids": [p1_id]})
    assert resp.status_code == 200
    data = resp.json()

    # Documented gaps
    assert len(data["documented_gaps"]) == 1
    d_gap = data["documented_gaps"][0]
    assert d_gap["category"] == "documented"
    assert d_gap["confidence"] == 1.0
    assert d_gap["is_documented"] is True
    assert d_gap["is_inferred"] is False

    # Inferred gaps
    assert len(data["inferred_gaps"]) == 1
    i_gap = data["inferred_gaps"][0]
    assert i_gap["category"] == "inferred"
    assert i_gap["confidence"] == 0.75
    assert i_gap["confidence"] < 1.0
    assert i_gap["is_documented"] is False
    assert i_gap["is_inferred"] is True


# ---------------------------------------------------------------------------
# 7. Empty Evidence Handling Tests
# ---------------------------------------------------------------------------
def test_empty_evidence_and_zero_chunk_handling_end_to_end():
    """Scenario 7: Zero evidence produces safe, non-hallucinated empty outcomes without fabricated citations."""
    p_empty = str(uuid.uuid4())

    class EmptyAnalysisService:
        def analyze_paper(self, paper_id: str, **kwargs):
            return StructuredPaperAnalysis.empty(paper_id=paper_id)

        def _load_evidence_from_database(self, paper_uuid):
            return None, EvidenceContext(query="test", items=[])

    gap_service = ResearchGapService(
        LLMService(ControllableScriptedLLMProvider(default_response="{}")),
        paper_analysis_service=EmptyAnalysisService(),
    )

    app = create_app(
        FakeResearchServiceForLineage(),
        gap_service=gap_service,
    )
    client = TestClient(app)

    resp = client.post(f"/api/v1/papers/{p_empty}/gaps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_gaps_count"] == 0
    assert data["documented_gaps"] == []
    assert data["inferred_gaps"] == []
    assert "No evidence chunks were found" in data["summary"]


# ---------------------------------------------------------------------------
# 8. Malformed Metadata Type Safety Tests
# ---------------------------------------------------------------------------
def test_malformed_evidence_metadata_types_fail_closed():
    """Scenario 8: Evidence chunks with non-string, boolean, or corrupt metadata fail closed."""
    # 1. Non-string paper_id in source record raises TypeError
    with pytest.raises(TypeError, match="must be a string or None"):
        ResearchGap(
            gap_id="gap-1",
            category=GapCategory.DOCUMENTED.value,
            title="Title",
            description="Desc",
            reasoning="Reasoning",
            source_paper_ids=["p1"],
            evidence_labels=["[E1]"],
            supporting_sources=[{
                "label": "[E1]",
                "document_id": "doc-1",
                "chunk_id": "chk-1",
                "text": "text",
                "paper_id": 12345,  # Invalid type!
            }],
            contradicting_sources=[],
            confidence=1.0,
            limitations=[],
        )

    # 2. Boolean score in EvidenceItem raises TypeError
    with pytest.raises(TypeError, match="score must be a number"):
        EvidenceItem(
            document_id="doc-1",
            chunk_id="chk-1",
            text="text",
            score=True,  # Invalid type!
            metadata={"paper_id": "p1"},
            position=0,
        )

    # 3. Non-string metadata keys in EvidenceItem raise TypeError
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        EvidenceItem(
            document_id="doc-1",
            chunk_id="chk-1",
            text="text",
            score=1.0,
            metadata={123: "value"},  # Non-string key!
            position=0,
        )
