"""Unit and integration tests for Direct Paper Comparison and Structured Research Insights."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nexora.api.app import create_app
from nexora.bootstrap.application import (
    build_analysis_service,
    build_comparison_service,
    build_http_app,
)
from nexora.config.settings import AppSettings
from nexora.llm.base import LLMProvider
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.rag.models import AttributedSource, RAGResponse
from nexora.rag.pipeline import RAGPipeline, _build_structured_research_insights
from nexora.research.comparison_table import (
    BENCHMARK_PAPERS,
    build_11_dimension_comparison_table,
    normalize_paper_identifier,
)


class MockLLMProvider(LLMProvider):
    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text='{"overall_summary": "Comparative analysis across benchmark papers.", "dimensions": {}}',
            model=model or "mock-model",
            usage={"total_tokens": 100},
        )


def test_normalize_paper_identifier():
    # Benchmark string key
    u1, k1 = normalize_paper_identifier("arxiv_1706.03762")
    assert u1 == BENCHMARK_PAPERS["arxiv_1706.03762"]["uuid"]
    assert k1 == "arxiv_1706.03762"

    # Benchmark UUID
    u2, k2 = normalize_paper_identifier(u1)
    assert u2 == u1
    assert k2 == "arxiv_1706.03762"


def test_build_11_dimension_comparison_table_completeness():
    p1 = BENCHMARK_PAPERS["arxiv_1706.03762"]["uuid"]
    p2 = BENCHMARK_PAPERS["arxiv_2106.09685"]["uuid"]

    table = build_11_dimension_comparison_table(
        [p1, p2],
        {p1: "Attention Is All You Need", p2: "LoRA"},
        per_paper_evidence={p1: [], p2: []},
    )

    rows = table["rows"]
    assert len(rows) == 11

    dimension_keys = [r["key"] for r in rows]
    assert dimension_keys == [
        "title",
        "publication_year",
        "research_problem",
        "main_contribution",
        "methodology",
        "model_architecture",
        "dataset",
        "experimental_results",
        "computational_efficiency",
        "limitations",
        "future_work",
    ]

    # Verify honest reporting of unreported fields
    future_work_row = next(r for r in rows if r["key"] == "future_work")
    assert future_work_row["per_paper"][p1]["status"] == "not_reported"
    assert future_work_row["per_paper"][p1]["value"] == "Not reported in retrieved evidence."
    assert future_work_row["per_paper"][p2]["status"] == "not_reported"
    assert future_work_row["per_paper"][p2]["value"] == "Not reported in retrieved evidence."

    # Verify below-table fields
    assert "key_similarities" in table and len(table["key_similarities"]) > 0
    assert "key_differences" in table and len(table["key_differences"]) > 0
    assert "direct_evidence_findings" in table and len(table["direct_evidence_findings"]) > 0
    assert "evidence_gaps" in table and len(table["evidence_gaps"]) > 0
    assert table["conclusion"] in (
        "Demonstrated improvement",
        "Limited evidence",
        "Conflicting evidence",
        "Insufficient evidence",
    )
    assert len(table["conclusion_reasoning"]) > 0


def test_structured_research_insights_paper_isolation():
    s1 = AttributedSource(
        label="[E1]",
        document_id="doc_attn",
        chunk_id="chunk_attn_01",
        paper_id="arxiv_1706.03762",
        paper_title="Attention Is All You Need",
        score=0.95,
        text="The Transformer achieves 28.4 BLEU on English-to-German translation task.",
        text_excerpt="The Transformer achieves 28.4 BLEU.",
    )
    s2 = AttributedSource(
        label="[E2]",
        document_id="doc_lora",
        chunk_id="chunk_lora_01",
        paper_id="arxiv_2106.09685",
        paper_title="LoRA: Low-Rank Adaptation of Large Language Models",
        score=0.92,
        text="LoRA reduces trainable parameters by 10,000 times and GPU memory requirement by 3 times on GPT-3 175B.",
        text_excerpt="LoRA reduces trainable parameters by 10,000 times.",
    )

    evidence_by_paper, comp, lim, conc, reasoning = _build_structured_research_insights(
        "How do Transformer and LoRA improve efficiency?",
        [s1, s2],
        is_insufficient=False,
    )

    assert len(evidence_by_paper) == 2

    # Check isolation: LoRA excerpts only under LoRA, Attention only under Attention
    attn_group = next(p for p in evidence_by_paper if p["paper_id"] == "arxiv_1706.03762")
    lora_group = next(p for p in evidence_by_paper if p["paper_id"] == "arxiv_2106.09685")

    assert "[E1]" in attn_group["citations"]
    assert "[E2]" not in attn_group["citations"]
    assert any("28.4 BLEU" in exc for exc in attn_group["excerpts"])
    assert not any("10,000 times" in exc for exc in attn_group["excerpts"])

    assert "[E2]" in lora_group["citations"]
    assert "[E1]" not in lora_group["citations"]
    assert any("10,000 times" in exc for exc in lora_group["excerpts"])
    assert not any("28.4 BLEU" in exc for exc in lora_group["excerpts"])

    # Check comparison of findings
    assert comp is not None
    assert len(comp["agreements"]) > 0
    assert len(comp["differences"]) > 0
    assert len(comp["direct_evidence"]) == 2

    # Check conclusion
    assert conc == "Demonstrated improvement"


class MockResearchService:
    def ask(self, request):
        return {"text": "mock"}


def test_api_list_papers_endpoint():
    from nexora.api.app import create_app
    from nexora.rag.pipeline import create_offline_rag_pipeline

    app = create_app(MockResearchService(), rag_pipeline=create_offline_rag_pipeline())
    client = TestClient(app)

    # 1. List all papers
    resp = client.get("/api/v1/papers")
    assert resp.status_code == 200
    data = resp.json()
    assert "papers" in data
    assert data["total"] >= 3
    titles = [p["title"] for p in data["papers"]]
    assert "Attention Is All You Need" in titles
    assert "LoRA: Low-Rank Adaptation of Large Language Models" in titles
    assert "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" in titles

    # 2. Search by query
    resp_search = client.get("/api/v1/papers?query=LoRA")
    assert resp_search.status_code == 200
    search_data = resp_search.json()
    assert search_data["total"] >= 1
    assert any("LoRA" in p["title"] for p in search_data["papers"])

    # 3. Search by DOI
    resp_doi = client.get("/api/v1/papers?query=10.48550/arXiv.1706.03762")
    assert resp_doi.status_code == 200
    doi_data = resp_doi.json()
    assert doi_data["total"] >= 1
    assert doi_data["papers"][0]["title"] == "Attention Is All You Need"


def test_api_rag_query_structured_insights():
    from nexora.api.app import create_app
    from nexora.rag.pipeline import create_offline_rag_pipeline

    app = create_app(MockResearchService(), rag_pipeline=create_offline_rag_pipeline())
    client = TestClient(app)

    resp = client.post(
        "/api/v1/research/query",
        json={
            "query": "How does LoRA reduce trainable parameters on GPT-3?",
            "retrieval_mode": "hybrid",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "evidence_by_paper" in data
    assert len(data["evidence_by_paper"]) >= 1
    assert "comparison_of_findings" in data
    assert "limitations_and_conflicts" in data
    assert data["conclusion"] in (
        "Demonstrated improvement",
        "Limited evidence",
        "Conflicting evidence",
        "Insufficient evidence",
    )
    assert data["conclusion_reasoning"] is not None


def test_api_research_compare_attention_and_lora_canonical_and_alias_uuids():
    """Verify POST /api/v1/research/compare succeeds for Attention vs LoRA with both canonical and alias UUIDs."""
    from nexora.research.analysis_service import PaperAnalysisService, PaperComparisonService
    from nexora.llm.service import LLMService

    llm_service = LLMService(MockLLMProvider())
    analysis_service = PaperAnalysisService(llm_service)
    comparison_service = PaperComparisonService(analysis_service, llm_service)

    app = create_app(
        MockResearchService(),
        analysis_service=analysis_service,
        comparison_service=comparison_service,
        enforce_auth=False,
    )
    client = TestClient(app)

    # 1. Canonical benchmark UUIDs
    p1 = BENCHMARK_PAPERS["arxiv_1706.03762"]["uuid"]
    p2 = BENCHMARK_PAPERS["arxiv_2106.09685"]["uuid"]

    res_canonical = client.post("/api/v1/research/compare", json={"paper_ids": [p1, p2]})
    assert res_canonical.status_code == 200
    data_canonical = res_canonical.json()
    assert "comparison_table" in data_canonical
    assert len(data_canonical["comparison_table"]["rows"]) == 11
    assert data_canonical["paper_titles"][p1] == "Attention Is All You Need"
    assert data_canonical["paper_titles"][p2] == "LoRA: Low-Rank Adaptation of Large Language Models"

    # 2. Legacy / preview alias UUIDs
    alias1 = "541c4d4f-4d6b-592f-981f-7e9b04f3ce37"
    alias2 = "553e414c-112f-59ea-96b0-7d7d3d756b19"

    res_alias = client.post("/api/v1/research/compare", json={"paper_ids": [alias1, alias2]})
    assert res_alias.status_code == 200
    data_alias = res_alias.json()
    assert "comparison_table" in data_alias
    assert len(data_alias["comparison_table"]["rows"]) == 11
    assert data_alias["paper_titles"][p1] == "Attention Is All You Need"
    assert data_alias["paper_titles"][p2] == "LoRA: Low-Rank Adaptation of Large Language Models"


def test_api_research_compare_error_propagation_and_detail():
    """Verify that when comparison fails, HTTP 500 contains descriptive error details and is not masked."""
    from nexora.research.analysis_service import PaperAnalysisService, PaperComparisonService
    from nexora.llm.service import LLMService

    class FailingLLMProvider(LLMProvider):
        def generate(self, messages, **kwargs):
            raise RuntimeError("Upstream model rate limit exceeded: try again in 20s")

    llm_service = LLMService(FailingLLMProvider())
    analysis_service = PaperAnalysisService(llm_service)
    comparison_service = PaperComparisonService(analysis_service, llm_service)

    app = create_app(
        MockResearchService(),
        analysis_service=analysis_service,
        comparison_service=comparison_service,
        enforce_auth=False,
    )
    client = TestClient(app)

    p1 = BENCHMARK_PAPERS["arxiv_1706.03762"]["uuid"]
    p2 = BENCHMARK_PAPERS["arxiv_2106.09685"]["uuid"]

    res = client.post("/api/v1/research/compare", json={"paper_ids": [p1, p2]})
    assert res.status_code == 500
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "comparison_failed"
    # Ensure error message is useful and includes the upstream failure cause
    assert "The paper comparison could not be completed" in data["error"]["message"]
    assert "Upstream model rate limit exceeded" in data["error"]["message"]
