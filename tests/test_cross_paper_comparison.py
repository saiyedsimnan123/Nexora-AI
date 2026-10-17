"""Tests for Cross-Paper Comparison and Evidence Synthesis.

Nexora core principle:
"Nexora does not tell users what to believe. It shows what the available evidence justifies believing."

Verifies:
1. Demonstrated improvement on shared, comparable benchmarks.
2. Conflicting results and performance trade-offs classified as 'Conflicting evidence'.
3. Incomparable experiments and disparate benchmarks classified without spurious superiority claims.
4. Missing evidence handled with explicit 'not_reported' and 'Insufficient evidence' without hallucination.
5. Citation provenance preserving document IDs, chunk IDs, page numbers, and exact excerpts.
6. Cross-paper isolation ensuring zero chunk or citation leakage across papers.
7. API endpoint POST /api/v1/research/compare with request validation and error propagation.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.api.app import create_app
from nexora.api.errors import (
    ComparisonServiceError,
    DocumentNotFoundError,
    InvalidRequestError,
)
from nexora.api.models import (
    ResearchCompareRequest,
    ResearchCompareResponse,
)
from nexora.api.service import ResearchApplicationService
from nexora.bootstrap.application import (
    build_analysis_service,
    build_comparison_service,
    build_http_app,
)
from nexora.config.settings import AppSettings
from nexora.database.models import Collection, Document, DocumentChunkRecord, Paper, User, Workspace
from nexora.database.models.base import Base
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.llm.base import LLMProvider
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.service import LLMService
from nexora.research.comparison import (
    COMPARISON_MATRIX_DIMENSIONS,
    ConclusionClassification,
    CrossPaperComparisonResult,
    MatrixDimensionComparison,
    MatrixEvidenceSource,
    MatrixPaperFinding,
    VALID_CONCLUSIONS,
)
from nexora.research.cross_paper_prompt import CrossPaperComparisonPromptBuilder
from nexora.research.cross_paper_service import (
    CrossPaperComparisonService,
    evaluate_cross_paper_conclusion,
)
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import VectorRecord


# ---------------------------------------------------------------------------
# Test Doubles
# ---------------------------------------------------------------------------
class ControllableComparisonLLMProvider(LLMProvider):
    def __init__(self, response_text: str = "", error: Exception | None = None):
        self.response_text = response_text
        self.error = error
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
        if self.error:
            raise self.error
        return LLMResponse(
            text=self.response_text,
            model=model or "test-comparison-model",
            usage={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
            raw={"request_id": "test_comp_req_123"},
        )


def _make_cross_paper_json(
    paper_ids: list[str],
    *,
    conclusion: str = "Demonstrated improvement",
    conclusion_reasoning: str = "Rigorous comparative evaluation across shared metrics.",
    comparability_status: str = "comparable",
    conflict: bool = False,
    disparate: bool = False,
    missing_p2: bool = False,
) -> str:
    p1 = paper_ids[0]
    p2 = paper_ids[1]

    d1 = "SQuAD v2.0" if not disparate else "ImageNet-1k"
    d2 = "SQuAD v2.0" if not disparate else "LibriSpeech"

    res1 = "Achieves 91.2% F1 on SQuAD." if not conflict else "Achieves 93.5% F1 on SQuAD, but 3.5x slower latency."
    res2 = "Achieves 88.4% F1 on SQuAD." if not conflict else "Achieves 89.0% F1 on SQuAD, with 1.0x baseline latency."

    cite_p1 = ["[E1]"]
    cite_p2 = ["[E1]"] if not missing_p2 else []

    status_p2 = "reported" if not missing_p2 else "not_reported"
    summary_p2 = f"Evaluated on {d2}" if not missing_p2 else "Not reported in the provided evidence."

    matrix = {
        "methods": {
            "comparison_summary": "Comparison of model architectures and fine-tuning mechanisms.",
            "key_differences": ["Paper 1 uses low-rank adapter injection", "Paper 2 uses prefix tuning"],
            "key_similarities": ["Both preserve frozen base model weights"],
            "per_paper": {
                p1: {
                    "summary": "Low-rank adaptation with rank r=8.",
                    "status": "reported",
                    "evidence_labels": cite_p1,
                    "details": {"method": "LoRA", "rank": 8},
                },
                p2: {
                    "summary": "Prefix tuning with continuous virtual tokens." if not missing_p2 else "Not reported in the provided evidence.",
                    "status": status_p2,
                    "evidence_labels": cite_p2,
                    "details": {"method": "PrefixTuning"} if not missing_p2 else {},
                },
            },
        },
        "datasets": {
            "comparison_summary": "Datasets evaluated across papers.",
            "key_differences": ["Different domains" if disparate else "Same benchmark suite"],
            "key_similarities": ["Public benchmarks"],
            "per_paper": {
                p1: {
                    "summary": f"Trained and evaluated on {d1}.",
                    "status": "reported",
                    "evidence_labels": cite_p1,
                    "details": {"datasets": [d1]},
                },
                p2: {
                    "summary": summary_p2,
                    "status": status_p2,
                    "evidence_labels": cite_p2,
                    "details": {"datasets": [d2]} if not missing_p2 else {},
                },
            },
        },
        "baselines": {
            "comparison_summary": "Comparison baselines.",
            "key_differences": [],
            "key_similarities": ["Full fine-tuning baseline"],
            "per_paper": {
                p1: {
                    "summary": "Compared against full fine-tuning.",
                    "status": "reported",
                    "evidence_labels": cite_p1,
                    "details": {"baselines": ["Full Fine-Tuning"]},
                },
                p2: {
                    "summary": "Compared against full fine-tuning." if not missing_p2 else "Not reported in the provided evidence.",
                    "status": status_p2,
                    "evidence_labels": cite_p2,
                    "details": {"baselines": ["Full Fine-Tuning"]} if not missing_p2 else {},
                },
            },
        },
        "metrics": {
            "comparison_summary": "Evaluation metrics.",
            "key_differences": ["Latency trade-offs" if conflict else "None"],
            "key_similarities": ["F1 score", "Accuracy"],
            "per_paper": {
                p1: {
                    "summary": "Evaluated on F1 score and trainable parameter percentage.",
                    "status": "reported",
                    "evidence_labels": cite_p1,
                    "details": {"metrics": ["F1", "params"]},
                },
                p2: {
                    "summary": "Evaluated on F1 score and inference latency." if not missing_p2 else "Not reported in the provided evidence.",
                    "status": status_p2,
                    "evidence_labels": cite_p2,
                    "details": {"metrics": ["F1", "latency"]} if not missing_p2 else {},
                },
            },
        },
        "results": {
            "comparison_summary": "Empirical results across evaluated tasks.",
            "key_differences": ["Conflicting speed-vs-accuracy trade-offs" if conflict else "Paper 1 outperforms Paper 2"],
            "key_similarities": [],
            "per_paper": {
                p1: {
                    "summary": res1,
                    "status": "reported",
                    "evidence_labels": cite_p1,
                    "details": {"result_score": 91.2},
                },
                p2: {
                    "summary": res2 if not missing_p2 else "Not reported in the provided evidence.",
                    "status": status_p2,
                    "evidence_labels": cite_p2,
                    "details": {"result_score": 88.4} if not missing_p2 else {},
                },
            },
        },
        "limitations": {
            "comparison_summary": "Documented limitations.",
            "key_differences": [],
            "key_similarities": ["Requires hyperparameter tuning"],
            "per_paper": {
                p1: {
                    "summary": "Adapter rank selection requires empirical validation.",
                    "status": "reported",
                    "evidence_labels": cite_p1,
                    "details": {},
                },
                p2: {
                    "summary": "Optimization stability issues with deep prefixes." if not missing_p2 else "Not reported in the provided evidence.",
                    "status": status_p2,
                    "evidence_labels": cite_p2,
                    "details": {},
                },
            },
        },
        "comparability": {
            "comparison_summary": "Comparability of evaluation setups." if not disparate else "The compared papers evaluate on disparate benchmarks with non-overlapping datasets and evaluation targets.",
            "key_differences": ["Disparate evaluation domains and benchmarks" if disparate else "Minor batch size variation"],
            "key_similarities": ["Evaluated on common benchmark" if not disparate else "None"],
            "per_paper": {
                p1: {
                    "summary": f"Setup on {d1}.",
                    "status": "reported",
                    "evidence_labels": cite_p1,
                    "details": {},
                },
                p2: {
                    "summary": f"Setup on {d2}." if not missing_p2 else "Not reported in the provided evidence.",
                    "status": status_p2,
                    "evidence_labels": cite_p2,
                    "details": {},
                },
            },
        },
    }

    return json.dumps({
        "matrix": matrix,
        "comparability_status": comparability_status,
        "conclusion": conclusion,
        "conclusion_reasoning": conclusion_reasoning,
        "overall_summary": "Comparative synthesis across papers.",
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sqlite_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory


def _seed_paper_with_chunk(
    session_factory,
    paper_id: uuid.UUID,
    title: str,
    chunk_text: str,
    page_number: int = 1,
) -> None:
    session = session_factory()
    try:
        user = User(email=f"user_{uuid4().hex[:8]}@example.com")
        session.add(user)
        session.flush()

        workspace = Workspace(name="Test WS", user_id=user.id)
        session.add(workspace)
        session.flush()

        collection = Collection(
            name="Test Col",
            workspace_id=workspace.id,
        )
        session.add(collection)
        session.flush()

        paper = Paper(
            id=paper_id,
            title=title,
            collection_id=collection.id,
        )
        session.add(paper)
        session.flush()

        doc = Document(
            paper_id=paper.id,
            filename=f"{title.lower().replace(' ', '_')}.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="a" * 64,
            storage_key=f"papers/{paper_id}.pdf#page={page_number}",
            indexing_status="completed",
        )
        session.add(doc)
        session.flush()

        chunk = DocumentChunkRecord(
            document_id=doc.id,
            chunk_index=0,
            text=chunk_text,
        )
        chunk.page_number = page_number
        session.add(chunk)
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_cross_paper_comparison_demonstrated_improvement(sqlite_session_factory):
    """Verifies that comparable benchmarks with substantiated metrics yield Demonstrated improvement."""
    p1 = uuid4()
    p2 = uuid4()
    p1_str = str(p1)
    p2_str = str(p2)

    _seed_paper_with_chunk(
        sqlite_session_factory,
        p1,
        "LoRA: Low-Rank Adaptation",
        "LoRA achieves 91.2% F1 on SQuAD v2.0 with only 0.1% trainable parameters.",
        page_number=3,
    )
    _seed_paper_with_chunk(
        sqlite_session_factory,
        p2,
        "Prefix-Tuning: Optimizing Continuous Prompts",
        "Prefix-Tuning achieves 88.4% F1 on SQuAD v2.0 with 0.1% trainable parameters.",
        page_number=5,
    )

    response_json = _make_cross_paper_json(
        [p1_str, p2_str],
        conclusion="Demonstrated improvement",
        conclusion_reasoning="LoRA outperforms Prefix-Tuning by +2.8% F1 on the shared SQuAD v2.0 benchmark.",
        comparability_status="comparable",
    )
    llm = ControllableComparisonLLMProvider(response_text=response_json)
    service = CrossPaperComparisonService(llm, session_factory=sqlite_session_factory)

    result = service.compare([p1_str, p2_str])

    assert isinstance(result, CrossPaperComparisonResult)
    assert result.conclusion == "Demonstrated improvement"
    assert result.comparability_status == "comparable"
    assert "SQuAD" in result.conclusion_reasoning

    # Verify all 7 matrix dimensions are present
    assert set(result.matrix.keys()) == set(COMPARISON_MATRIX_DIMENSIONS)
    for dim in COMPARISON_MATRIX_DIMENSIONS:
        assert isinstance(result.matrix[dim], MatrixDimensionComparison)
        assert p1_str in result.matrix[dim].per_paper
        assert p2_str in result.matrix[dim].per_paper

    # Verify citation provenance
    methods_p1 = result.matrix["methods"].per_paper[p1_str]
    assert methods_p1.status == "reported"
    assert len(methods_p1.sources) > 0
    src = methods_p1.sources[0]
    assert isinstance(src, MatrixEvidenceSource)
    assert src.paper_id == p1_str
    assert src.page_number == 3
    assert "91.2% F1" in src.text_excerpt


def test_cross_paper_comparison_conflicting_results(sqlite_session_factory):
    """Verifies that performance trade-offs or contradictory findings classify as Conflicting evidence."""
    p1 = uuid4()
    p2 = uuid4()
    p1_str = str(p1)
    p2_str = str(p2)

    _seed_paper_with_chunk(
        sqlite_session_factory,
        p1,
        "Model Alpha",
        "Model Alpha achieves 93.5% F1 on SQuAD, but suffers a 3.5x slower latency regression.",
        page_number=4,
    )
    _seed_paper_with_chunk(
        sqlite_session_factory,
        p2,
        "Model Beta",
        "Model Beta achieves 89.0% F1 on SQuAD, maintaining baseline 1.0x latency.",
        page_number=6,
    )

    response_json = _make_cross_paper_json(
        [p1_str, p2_str],
        conclusion="Conflicting evidence",
        conclusion_reasoning="Trade-off between higher accuracy and severe latency degradation.",
        conflict=True,
    )
    llm = ControllableComparisonLLMProvider(response_text=response_json)
    service = CrossPaperComparisonService(llm, session_factory=sqlite_session_factory)

    result = service.compare([p1_str, p2_str])

    assert result.conclusion == ConclusionClassification.CONFLICTING_EVIDENCE.value
    assert "trade-off" in result.conclusion_reasoning.lower() or "latency" in result.conclusion_reasoning.lower()


def test_cross_paper_comparison_incomparable_experiments_guardrail(sqlite_session_factory):
    """Verifies that non-overlapping benchmarks are flagged as incomparable and never assert superiority."""
    p1 = uuid4()
    p2 = uuid4()
    p1_str = str(p1)
    p2_str = str(p2)

    _seed_paper_with_chunk(
        sqlite_session_factory,
        p1,
        "Vision Transformer",
        "Evaluated strictly on ImageNet-1k top-1 classification accuracy.",
        page_number=2,
    )
    _seed_paper_with_chunk(
        sqlite_session_factory,
        p2,
        "Speech Conformer",
        "Evaluated strictly on LibriSpeech clean and other word error rates.",
        page_number=7,
    )

    # LLM spuriously claims "Demonstrated improvement" across incomparable setups
    spurious_json = _make_cross_paper_json(
        [p1_str, p2_str],
        conclusion="Demonstrated improvement",
        conclusion_reasoning="Vision Transformer is superior.",
        comparability_status="incomparable",
        disparate=True,
    )
    llm = ControllableComparisonLLMProvider(response_text=spurious_json)
    service = CrossPaperComparisonService(llm, session_factory=sqlite_session_factory)

    result = service.compare([p1_str, p2_str])

    # Guardrail must override false claim to Limited evidence
    assert result.conclusion == ConclusionClassification.LIMITED_EVIDENCE.value
    assert result.comparability_status == "incomparable"
    assert "cannot be established" in result.conclusion_reasoning or "disparate" in result.conclusion_reasoning


def test_cross_paper_comparison_missing_evidence_all_papers():
    """Verifies that zero evidence across all papers short-circuits to Insufficient evidence without hallucinations."""
    p1_str = str(uuid4())
    p2_str = str(uuid4())

    vector_store = InMemoryVectorStore()
    llm = ControllableComparisonLLMProvider(response_text="{}")
    service = CrossPaperComparisonService(llm, vector_store=vector_store)

    result = service.compare([p1_str, p2_str])

    assert result.conclusion == ConclusionClassification.INSUFFICIENT_EVIDENCE.value
    assert result.comparability_status == "insufficient_evidence"
    assert "No evidence" in result.conclusion_reasoning
    # Verify no LLM call was wasted
    assert len(llm.calls) == 0

    for dim in COMPARISON_MATRIX_DIMENSIONS:
        finding1 = result.matrix[dim].per_paper[p1_str]
        finding2 = result.matrix[dim].per_paper[p2_str]
        assert finding1.status == "not_reported"
        assert finding2.status == "not_reported"
        assert len(finding1.sources) == 0
        assert len(finding2.sources) == 0


def test_cross_paper_comparison_missing_evidence_one_paper_never_infers_superiority(sqlite_session_factory):
    """Verifies that when one paper has evidence and the other has none, superiority is never inferred."""
    p1 = uuid4()
    p2 = uuid4()
    p1_str = str(p1)
    p2_str = str(p2)

    # Only p1 has evidence; p2 has none in vector store
    vector_store = InMemoryVectorStore()
    vector_store._records[str(uuid4())] = VectorRecord(
        id="chunk_p1",
        vector=[1.0, 0.0],
        payload={
            "paper_id": p1_str,
            "title": "Solo Paper",
            "text": "Reported impressive 95.0% accuracy on GLUE benchmark.",
            "chunk_index": 0,
            "page_number": 1,
        },
    )

    # LLM claims p1 is superior to p2 despite p2 having zero evidence
    llm_output = _make_cross_paper_json(
        [p1_str, p2_str],
        conclusion="Demonstrated improvement",
        conclusion_reasoning="Paper 1 is superior because Paper 2 has no data.",
        missing_p2=True,
    )
    llm = ControllableComparisonLLMProvider(response_text=llm_output)
    service = CrossPaperComparisonService(llm, vector_store=vector_store)

    result = service.compare([p1_str, p2_str])

    # Guardrail rule: Never infer superiority from one paper alone
    assert result.conclusion == ConclusionClassification.INSUFFICIENT_EVIDENCE.value
    assert "cannot be inferred without evidence from all papers" in result.conclusion_reasoning


def test_cross_paper_isolation_zero_leakage():
    """Verifies that an attempt to cite Paper A's chunks under Paper B is strictly blocked."""
    p1_str = str(uuid4())
    p2_str = str(uuid4())

    vector_store = InMemoryVectorStore()
    # Add chunk for p1
    vector_store._records["c1"] = VectorRecord(
        id="c1",
        vector=[1.0, 0.0],
        payload={
            "paper_id": p1_str,
            "title": "Paper One",
            "text": "Unique text belonging exclusively to Paper One.",
            "chunk_index": 0,
            "page_number": 2,
        },
    )
    # Add chunk for p2
    vector_store._records["c2"] = VectorRecord(
        id="c2",
        vector=[0.0, 1.0],
        payload={
            "paper_id": p2_str,
            "title": "Paper Two",
            "text": "Unique text belonging exclusively to Paper Two.",
            "chunk_index": 0,
            "page_number": 4,
        },
    )

    # LLM tries cross-paper leakage: citing [E1] (which belongs to P1) under P2's finding
    malicious_output = _make_cross_paper_json([p1_str, p2_str])
    parsed = json.loads(malicious_output)
    # Under methods for p2, attempt to cite [E1] of p1
    parsed["matrix"]["methods"]["per_paper"][p2_str]["evidence_labels"] = ["[E1]"]
    llm = ControllableComparisonLLMProvider(response_text=json.dumps(parsed))
    service = CrossPaperComparisonService(llm, vector_store=vector_store)

    result = service.compare([p1_str, p2_str])

    methods_p2 = result.matrix["methods"].per_paper[p2_str]
    # Any source under p2 MUST strictly have paper_id == p2_str
    for src in methods_p2.sources:
        assert src.paper_id == p2_str
        assert src.chunk_id != "c1"

    # Methods for p1 must also have only p1 sources
    methods_p1 = result.matrix["methods"].per_paper[p1_str]
    for src in methods_p1.sources:
        assert src.paper_id == p1_str


def test_cross_paper_api_endpoint_success(sqlite_session_factory):
    """Verifies POST /api/v1/research/compare returns 200 with complete comparison matrix."""
    p1 = uuid4()
    p2 = uuid4()
    p1_str = str(p1)
    p2_str = str(p2)

    _seed_paper_with_chunk(
        sqlite_session_factory,
        p1,
        "Paper Alpha",
        "Methodology Alpha with 92% benchmark accuracy.",
        page_number=1,
    )
    _seed_paper_with_chunk(
        sqlite_session_factory,
        p2,
        "Paper Beta",
        "Methodology Beta with 89% benchmark accuracy.",
        page_number=3,
    )

    response_json = _make_cross_paper_json([p1_str, p2_str])
    llm = ControllableComparisonLLMProvider(response_text=response_json)
    service = CrossPaperComparisonService(
        llm,
        session_factory=sqlite_session_factory,
    )

    # Dummy research service to satisfy create_app
    class DummyResearchService:
        def ask(self, req: Any) -> Any:
            pass

    app = create_app(
        DummyResearchService(),
        comparison_service=service,
        session_factory=sqlite_session_factory,
    )

    client = TestClient(app)
    resp = client.post(
        "/api/v1/research/compare",
        json={"paper_ids": [p1_str, p2_str], "query": "accuracy comparison"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["paper_ids"] == [p1_str, p2_str]
    assert data["conclusion"] in VALID_CONCLUSIONS
    assert data["comparability_status"] in ("comparable", "partially_comparable", "incomparable", "insufficient_evidence")
    assert "methods" in data["matrix"]
    assert "results" in data["matrix"]
    assert "comparability" in data["matrix"]


def test_cross_paper_api_endpoint_validation_errors(sqlite_session_factory):
    """Verifies request validation for invalid input, single paper, duplicates, and missing papers."""
    llm = ControllableComparisonLLMProvider(response_text="{}")
    service = CrossPaperComparisonService(
        llm,
        session_factory=sqlite_session_factory,
    )

    class DummyResearchService:
        def ask(self, req: Any) -> Any:
            pass

    app = create_app(
        DummyResearchService(),
        comparison_service=service,
        session_factory=sqlite_session_factory,
    )
    client = TestClient(app)

    # 1. Single paper ID (< 2)
    resp = client.post(
        "/api/v1/research/compare",
        json={"paper_ids": [str(uuid4())]},
    )
    assert resp.status_code == 422 or resp.status_code == 400

    # 2. Duplicate paper IDs
    valid_id = str(uuid4())
    resp = client.post(
        "/api/v1/research/compare",
        json={"paper_ids": [valid_id, valid_id]},
    )
    assert resp.status_code == 400

    # 3. Invalid UUID string
    resp = client.post(
        "/api/v1/research/compare",
        json={"paper_ids": ["not-a-uuid", str(uuid4())]},
    )
    assert resp.status_code == 400

    # 4. Non-existent paper ID in DB
    resp = client.post(
        "/api/v1/research/compare",
        json={"paper_ids": [str(uuid4()), str(uuid4())]},
    )
    assert resp.status_code == 404
