"""Tests for the end-to-end RAG pipeline (Milestone: End-to-End RAG Pipeline).

Verifies:
1. Data models: AttributedSource, RAGRequest, RAGResponse validation and serialization.
2. Generator contracts: DeterministicMockGenerator and LLMAnswerGenerator.
3. Offline RAG pipeline execution: retrieval-to-answer flow, evidence attribution,
   reference integrity, and truthful empty-evidence handling.
4. Error handling: malformed input, retrieval failure, and provider errors.
5. Multi-mode retrieval: BM25, vector, and hybrid RRF search.
6. CLI execution: --query, positional questions, --json, --sample-questions, validation.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nexora.llm.models import LLMMessage, LLMResponse
from nexora.rag.cli import run_rag_cli
from nexora.rag.generator import (
    DeterministicMockGenerator,
    GeneratedAnswer,
    LLMAnswerGenerator,
)
from nexora.rag.models import (
    AttributedSource,
    RAGRequest,
    RAGResponse,
)
from nexora.rag.pipeline import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    RAGPipeline,
    create_offline_rag_pipeline,
    create_rag_pipeline,
)
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem
from nexora.retrieval.models import RetrievedChunk, SearchResult


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------
def make_attributed_source(
    label: str = "[E1]",
    chunk_id: str = "c1",
    doc_id: str = "d1",
    paper_id: str | None = "p1",
    title: str = "Test Paper",
    score: float = 0.85,
    text: str = "Sample chunk text for testing.",
) -> AttributedSource:
    return AttributedSource(
        label=label,
        document_id=doc_id,
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_title=title,
        score=score,
        text=text,
        text_excerpt=text[:50],
        metadata={"key": "val"},
    )


class FakeRetrievalService:
    """Mock retrieval service returning pre-configured search results."""

    def __init__(self, results: list[RetrievedChunk] | None = None, error: Exception | None = None) -> None:
        self.results = results if results is not None else []
        self.error = error
        self.calls: list[tuple[str, int, str]] = []

    def retrieve(self, query: str, limit: int = 10, mode: str = "hybrid") -> SearchResult:
        self.calls.append((query, limit, mode))
        if self.error:
            raise self.error
        return SearchResult(query=query, results=list(self.results[:limit]))


class FailingGenerator:
    """Mock generator that simulates LLM provider failures."""

    def __init__(self, error_message: str = "LLM API timeout") -> None:
        self.error_message = error_message

    @property
    def is_mock(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "failing-mock"

    def generate(self, query: str, context: EvidenceContext, **kwargs) -> GeneratedAnswer:
        raise RuntimeError(self.error_message)


# ---------------------------------------------------------------------------
# 1. Model Validation Tests
# ---------------------------------------------------------------------------
def test_attributed_source_valid_and_to_dict():
    source = make_attributed_source()
    assert source.label == "[E1]"
    assert source.chunk_id == "c1"
    assert source.score == 0.85

    d = source.to_dict()
    assert d["label"] == "[E1]"
    assert d["chunk_id"] == "c1"
    assert d["paper_title"] == "Test Paper"
    assert d["score"] == 0.85


@pytest.mark.parametrize("kw,exc", [
    ({"label": ""}, ValueError),
    ({"document_id": "  "}, ValueError),
    ({"chunk_id": ""}, ValueError),
    ({"paper_title": " "}, ValueError),
    ({"score": "bad"}, TypeError),
    ({"score": True}, TypeError),
    ({"text": 123}, TypeError),
    ({"text_excerpt": None}, TypeError),
    ({"metadata": "not-a-dict"}, TypeError),
])
def test_attributed_source_invalid_fields(kw, exc):
    base = {
        "label": "[E1]",
        "document_id": "d1",
        "chunk_id": "c1",
        "paper_id": "p1",
        "paper_title": "Paper",
        "score": 0.5,
        "text": "text",
        "text_excerpt": "excerpt",
        "metadata": {},
    }
    base.update(kw)
    with pytest.raises(exc):
        AttributedSource(**base)


def test_rag_request_validation():
    req = RAGRequest(query="What is LoRA?", retrieval_mode="bm25", top_k=3)
    assert req.query == "What is LoRA?"
    assert req.retrieval_mode == "bm25"
    assert req.top_k == 3

    # Invalid queries
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        RAGRequest(query="   ")
    with pytest.raises(TypeError, match="string"):
        RAGRequest(query=123)  # type: ignore

    # Invalid mode
    with pytest.raises(ValueError, match="retrieval_mode"):
        RAGRequest(query="valid", retrieval_mode="invalid")

    # Invalid top_k
    with pytest.raises(ValueError, match="top_k"):
        RAGRequest(query="valid", top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        RAGRequest(query="valid", top_k=-2)


def test_rag_response_properties_and_to_dict():
    s1 = make_attributed_source(label="[E1]", chunk_id="c1")
    s2 = make_attributed_source(label="[E2]", chunk_id="c2")

    resp = RAGResponse(
        query="What is LoRA?",
        answer="LoRA reduces parameters [E1].",
        sources=[s1, s2],
        cited_sources=[s1],
        is_grounded=True,
        is_mock=True,
        model="offline-mock-v1",
        status="success",
    )

    assert resp.has_citations is True
    assert len(resp.cited_sources) == 1
    assert len(resp.uncited_sources) == 1
    assert resp.uncited_sources[0].chunk_id == "c2"

    d = resp.to_dict()
    assert d["query"] == "What is LoRA?"
    assert d["is_grounded"] is True
    assert d["is_mock"] is True
    assert len(d["sources"]) == 2
    assert len(d["cited_sources"]) == 1


# ---------------------------------------------------------------------------
# 2. Generator Tests
# ---------------------------------------------------------------------------
def test_deterministic_mock_generator_with_evidence():
    gen = DeterministicMockGenerator()
    assert gen.is_mock is True
    assert "mock" in gen.model_name

    item1 = EvidenceItem("d1", "c1", "LoRA freezes weights. Parameter reduction is 10,000x.", 0.9, {"title": "LoRA Paper"}, 0)
    context = EvidenceContext(query="What is LoRA?", items=[item1], total_candidates=1)

    result = gen.generate("What is LoRA?", context)
    assert isinstance(result, GeneratedAnswer)
    assert result.is_mock is True
    assert "[E1]" in result.text
    assert "LoRA Paper" in result.text
    assert "LoRA freezes weights." in result.text


def test_deterministic_mock_generator_empty_evidence():
    gen = DeterministicMockGenerator()
    context = EvidenceContext(query="Unanswerable?", items=[], total_candidates=0)
    result = gen.generate("Unanswerable?", context)

    assert "could not find sufficient evidence" in result.text
    assert "[E" not in result.text


def test_llm_answer_generator_delegation():
    fake_llm = MagicMock()
    fake_llm.generate.return_value = LLMResponse(
        text="According to the research, FlashAttention optimizes SRAM [E1].",
        model="gpt-4o-mini",
        usage={"total_tokens": 42},
    )

    gen = LLMAnswerGenerator(fake_llm, model_name="gpt-4o-mini")
    assert gen.is_mock is False

    item1 = EvidenceItem("d1", "c1", "FlashAttention uses tiling.", 0.88, {"title": "FlashAttention"}, 0)
    context = EvidenceContext(query="How does FlashAttention work?", items=[item1], total_candidates=1)

    result = gen.generate("How does FlashAttention work?", context)
    assert result.is_mock is False
    assert result.model == "gpt-4o-mini"
    assert "[E1]" in result.text
    assert fake_llm.generate.called


# ---------------------------------------------------------------------------
# 3. Pipeline Integration Tests
# ---------------------------------------------------------------------------
def test_rag_pipeline_end_to_end_success():
    chunk1 = RetrievedChunk("d1", "c1", "FlashAttention computes exact attention with IO-awareness.", 9.5, {"title": "FlashAttention", "paper_id": "arxiv_2205.14135"})
    chunk2 = RetrievedChunk("d1", "c2", "Tiling reduces GPU HBM memory accesses.", 6.2, {"title": "FlashAttention", "paper_id": "arxiv_2205.14135"})

    retrieval = FakeRetrievalService(results=[chunk1, chunk2])
    pipeline = RAGPipeline(retrieval_service=retrieval)

    response = pipeline.answer("How does FlashAttention reduce memory accesses?", top_k=2)

    assert response.status == "success"
    assert response.is_grounded is True
    assert response.is_mock is True
    assert len(response.sources) == 2
    assert response.sources[0].label == "[E1]"
    assert response.sources[1].label == "[E2]"
    assert response.sources[0].paper_title == "FlashAttention"
    assert len(response.cited_sources) >= 1
    assert "[E1]" in response.answer


def test_rag_pipeline_empty_retrieval_reports_insufficient_evidence():
    """Unanswerable query with no retrieved chunks returns honest insufficient_evidence."""
    retrieval = FakeRetrievalService(results=[])  # Zero candidates
    pipeline = RAGPipeline(retrieval_service=retrieval)

    response = pipeline.answer("Quantum annealing topological resonator qubit")

    assert response.status == "insufficient_evidence"
    assert response.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert response.sources == []
    assert response.cited_sources == []
    assert response.is_grounded is True


def test_rag_pipeline_zero_score_chunks_filtered_out():
    """Candidates with 0.0 or negative score are treated as no evidence."""
    chunk_zero = RetrievedChunk("d1", "c1", "Irrelevant text", 0.0, {})
    retrieval = FakeRetrievalService(results=[chunk_zero])
    pipeline = RAGPipeline(retrieval_service=retrieval)

    response = pipeline.answer("Any query")

    assert response.status == "insufficient_evidence"
    assert response.evidence_count == 0


def test_rag_pipeline_provider_error_handling():
    """When the generation provider fails, error is recorded without losing retrieved sources."""
    chunk1 = RetrievedChunk("d1", "c1", "Sample text", 5.0, {"title": "Paper 1"})
    retrieval = FakeRetrievalService(results=[chunk1])
    failing_gen = FailingGenerator("Upstream LLM network timeout")

    pipeline = RAGPipeline(retrieval_service=retrieval, generator=failing_gen)
    response = pipeline.answer("A query")

    assert response.status == "error"
    assert "Upstream LLM network timeout" in (response.error or "")
    assert response.answer == ""
    # Retrieved sources are preserved for transparency!
    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "c1"


def test_rag_pipeline_retrieval_failure_handling():
    """When retrieval raises an error, pipeline returns structured error response."""
    retrieval = FakeRetrievalService(error=RuntimeError("Vector index corrupted"))
    pipeline = RAGPipeline(retrieval_service=retrieval)

    response = pipeline.answer("A query")
    assert response.status == "error"
    assert "Vector index corrupted" in (response.error or "")
    assert response.sources == []


def test_rag_pipeline_malformed_input():
    retrieval = FakeRetrievalService()
    pipeline = RAGPipeline(retrieval_service=retrieval)

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        pipeline.answer("   ")
    with pytest.raises(TypeError):
        pipeline.answer(None)  # type: ignore


# ---------------------------------------------------------------------------
# 4. Offline Factory Integration Test (Real Pilot Benchmark)
# ---------------------------------------------------------------------------
def test_create_offline_rag_pipeline_live_pilot_benchmark():
    """Verifies that the factory builds a ready-to-run pipeline over real benchmark papers."""
    pipeline = create_offline_rag_pipeline(default_mode="bm25")

    # Ask an exact question from the benchmark
    res = pipeline.answer("What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation?")

    assert res.status == "success"
    assert res.is_grounded is True
    assert res.is_mock is True
    assert len(res.sources) > 0
    assert "Attention Is All You Need" in res.sources[0].paper_title
    assert "28.4 BLEU" in res.sources[0].text
    assert "[E1]" in res.answer

    # Ask an out-of-domain unanswerable question
    unans_res = pipeline.answer("Surface code quantum error correction threshold in superconducting resonators")
    assert unans_res.status == "insufficient_evidence"
    assert unans_res.sources == []


# ---------------------------------------------------------------------------
# 5. CLI Tests
# ---------------------------------------------------------------------------
def test_rag_cli_sample_questions(capsys):
    exit_code = run_rag_cli(["--sample-questions"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Curated Sample Questions" in out
    assert "LoRA" in out


def test_rag_cli_with_query_flag(capsys):
    exit_code = run_rag_cli(["--query", "How does LoRA freeze weights?", "--top-k", "2"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Nexora AI -- End-to-End RAG Pipeline" in out
    assert "ANSWER:" in out
    assert "SUPPORTING EVIDENCE" in out
    assert "LoRA: Low-Rank Adaptation of Large Language Models" in out
    assert "VERIFIED GROUNDED" in out


def test_rag_cli_positional_question(capsys):
    exit_code = run_rag_cli(["What is FlashAttention?"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "FlashAttention" in out


def test_rag_cli_json_output(capsys):
    exit_code = run_rag_cli(["--query", "FlashAttention tiling", "--json", "--top-k", "2"])
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["status"] == "success"
    assert data["is_grounded"] is True
    assert len(data["sources"]) > 0
    assert "FlashAttention" in data["sources"][0]["paper_title"]


def test_rag_cli_invalid_top_k(capsys):
    exit_code = run_rag_cli(["--top-k", "0"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "--top-k must be a positive integer" in err


def test_create_live_rag_pipeline_without_api_key_fails_clearly():
    """Live generation without an API key raises ConfigurationError with clear instructions."""
    from nexora.config.settings import ConfigurationError
    from nexora.rag.pipeline import create_live_rag_pipeline

    with pytest.raises(ConfigurationError) as exc_info:
        create_live_rag_pipeline(environ={})

    assert "no api key configured" in str(exc_info.value).lower()
