"""Regression tests for canonical benchmark corpus seeding and UUID retrieval in RAG pipeline.

Verifies:
1. build_http_app seeds the canonical benchmark corpus into both BM25Index and vector_store on startup.
2. Hybrid retrieval answers benchmark queries with positive evidence chunks and citation attribution.
3. Querying by paper UUID (as exposed by GET /api/v1/papers) matches benchmark chunks in both BM25 and vector store.
4. RAGPipeline.answer() succeeds and produces grounded answers for general and paper-scoped queries.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexora.bootstrap.application import build_http_app, build_research_application
from nexora.config.settings import AppSettings
from nexora.embedding.config import EmbeddingConfig
from nexora.embedding.factory import create_embedding_provider
from nexora.embedding.service import EmbeddingService
from nexora.llm.config import LLMConfig
from nexora.rag.generator import DeterministicMockGenerator
from nexora.rag.pipeline import create_rag_pipeline
from nexora.research.comparison_table import BENCHMARK_PAPERS
from nexora.retrieval.bm25 import BM25Index
from nexora.retrieval.dataset import get_default_research_dataset
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.service import RetrievalService


@pytest.fixture
def test_embedding_service() -> EmbeddingService:
    provider = create_embedding_provider(EmbeddingConfig(provider="hash", dimension=64))
    return EmbeddingService(provider)


@pytest.fixture
def app_settings() -> AppSettings:
    return AppSettings(
        llm=LLMConfig(model="offline-mock", provider="gemini"),
        embedding=SimpleNamespace(dimension=64),
        vector_store="memory",
    )


def test_build_http_app_seeds_benchmark_corpus(test_embedding_service: EmbeddingService, app_settings: AppSettings) -> None:
    """build_http_app must automatically populate BM25 and vector store with benchmark chunks."""
    vec_store = InMemoryVectorStore(dimension=64)
    app = build_http_app(
        app_settings,
        embedding_service=test_embedding_service,
        vector_store=vec_store,
    )

    rag = app.state.rag_pipeline
    assert rag is not None
    assert getattr(rag._retrieval, "_bm25_index", None) is not None
    assert rag._retrieval._bm25_index.total_documents >= 10
    assert vec_store.count() >= 10


def test_rag_pipeline_hybrid_answers_with_evidence(test_embedding_service: EmbeddingService, app_settings: AppSettings) -> None:
    """Benchmark queries must retrieve grounded evidence chunks and citations."""
    vec_store = InMemoryVectorStore(dimension=64)
    app = build_http_app(
        app_settings,
        embedding_service=test_embedding_service,
        vector_store=vec_store,
    )
    rag = app.state.rag_pipeline

    queries = [
        "How do FlashAttention IO optimizations achieve 4.3x speedup on A100 GPUs?",
        "What is low-rank adaptation in LoRA?",
        "What tasks was BERT evaluated on?",
        "How does RoBERTa improve over BERT?",
        "What is the attention mechanism in Transformer?",
    ]

    for q in queries:
        resp = rag.answer(q)
        assert resp.status == "success", f"Query {q!r} failed with status {resp.status}"
        assert resp.evidence_count >= 2, f"Query {q!r} returned {resp.evidence_count} evidence chunks"
        assert len(resp.sources) >= 2
        assert "[E1]" in resp.answer


def test_paper_uuid_scoped_retrieval(test_embedding_service: EmbeddingService, app_settings: AppSettings) -> None:
    """Queries scoped by benchmark paper UUID must resolve and retrieve chunks for that paper."""
    vec_store = InMemoryVectorStore(dimension=64)
    app = build_http_app(
        app_settings,
        embedding_service=test_embedding_service,
        vector_store=vec_store,
    )
    rag = app.state.rag_pipeline

    # Test FlashAttention UUID scoping
    fa_uuid = BENCHMARK_PAPERS["arxiv_2205.14135"]["uuid"]
    resp_fa = rag.answer("What speedup is achieved?", paper_id=fa_uuid)
    assert resp_fa.status == "success"
    assert resp_fa.evidence_count >= 1
    assert any("FlashAttention" in s.paper_title for s in resp_fa.sources)

    # Test LoRA UUID scoping
    lora_uuid = BENCHMARK_PAPERS["arxiv_2106.09685"]["uuid"]
    resp_lora = rag.answer("What does LoRA freeze and inject?", paper_id=lora_uuid)
    assert resp_lora.status == "success"
    assert resp_lora.evidence_count >= 1
    assert any("LoRA" in s.paper_title for s in resp_lora.sources)


def test_bm25_and_retrieval_service_uuid_resolution(test_embedding_service: EmbeddingService) -> None:
    """RetrievalService and BM25Index must resolve both UUID and arXiv ID."""
    ds = get_default_research_dataset()
    seeded = []
    for c in ds.chunks:
        cd = dict(c)
        bm_info = BENCHMARK_PAPERS.get(cd.get("paper_id"))
        if bm_info:
            cd["paper_uuid"] = bm_info["uuid"]
        cd["is_benchmark"] = True
        seeded.append(cd)

    bm25 = BM25Index()
    bm25.index_chunks(seeded)

    vec_store = InMemoryVectorStore(dimension=64)
    service = RetrievalService(test_embedding_service, vec_store, bm25_index=bm25)

    fa_uuid = BENCHMARK_PAPERS["arxiv_2205.14135"]["uuid"]

    # BM25 search by UUID
    res_bm25_uuid = bm25.search("speedup", paper_id=fa_uuid)
    assert len(res_bm25_uuid.results) >= 1

    # BM25 search by arXiv ID
    res_bm25_arxiv = bm25.search("speedup", paper_id="arxiv_2205.14135")
    assert len(res_bm25_arxiv.results) >= 1

    # RetrievalService retrieve in bm25 mode by UUID
    res_ret_uuid = service.retrieve("speedup", mode="bm25", paper_id=fa_uuid)
    assert len(res_ret_uuid.results) >= 1
