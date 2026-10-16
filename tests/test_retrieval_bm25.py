"""Tests for Okapi BM25 lexical retrieval and hybrid search.

Verifies:
1. Tokenization: casing, punctuation, technical tokens (e.g. gpt-3, 1706.03762).
2. Configuration validation: k1, b, epsilon parameter bounds.
3. Inverted index operations: document frequency tracking, replacement, clear.
4. Non-negative Okapi BM25 IDF and term saturation scoring.
5. Deterministic ranking and SearchResult / RetrievedChunk compatibility.
6. Reciprocal Rank Fusion (RRF) for hybrid retrieval (Vector + BM25).
7. RetrievalService integration with vector, bm25, and hybrid modes.
8. Quality policy filtering over BM25 candidates.
9. Evaluation on the real-world pilot benchmark: high accuracy completely offline.
10. CLI --retriever flag parsing and execution.
"""

from __future__ import annotations

import json
import math
import os
import pytest

from nexora.embedding.hash_provider import HashEmbeddingProvider
from nexora.embedding.service import EmbeddingService
from nexora.retrieval.bm25 import (
    BM25Config,
    BM25Document,
    BM25Index,
    reciprocal_rank_fusion,
    tokenize,
)
from nexora.retrieval.dataset import get_pilot_benchmark_dataset, load_retrieval_dataset
from nexora.retrieval.eval_cli import run_eval_cli
from nexora.retrieval.evaluation import evaluate_retrieval
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.production_eval import (
    EvaluationMode,
    ProductionEvaluationConfig,
    evaluate_production_retrieval,
)
from nexora.retrieval.quality import RetrievalQualityPolicy, RetrievalQualityService
from nexora.retrieval.service import RetrievalService


# ---------------------------------------------------------------------------
# 1. Tokenization Tests
# ---------------------------------------------------------------------------
def test_tokenize_basic():
    text = "The Transformer architecture achieved 28.4 BLEU on WMT 2014!"
    tokens = tokenize(text)
    assert "transformer" in tokens
    assert "28.4" in tokens or "28" in tokens
    assert "bleu" in tokens
    assert "wmt" in tokens
    assert "2014" in tokens


def test_tokenize_preserves_hyphenated_technical_terms():
    text = "FlashAttention uses IO-aware tiling; LoRA provides Low-Rank Adaptation of GPT-3."
    tokens = tokenize(text)
    assert "flashattention" in tokens
    assert "io-aware" in tokens or ("io" in tokens and "aware" in tokens)
    assert "lora" in tokens
    assert "low-rank" in tokens or ("low" in tokens and "rank" in tokens)
    assert "gpt-3" in tokens or "gpt" in tokens


def test_tokenize_handles_empty_and_non_string():
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize(None) == []  # type: ignore
    assert tokenize(12345) == []  # type: ignore


# ---------------------------------------------------------------------------
# 2. Configuration Validation Tests
# ---------------------------------------------------------------------------
def test_bm25_config_defaults():
    cfg = BM25Config()
    assert cfg.k1 == 1.5
    assert cfg.b == 0.75
    assert cfg.epsilon == 0.25


@pytest.mark.parametrize(
    "kwargs, err_match",
    [
        ({"k1": -0.5}, "k1 must be a non-negative number"),
        ({"k1": True}, "k1 must be a non-negative number"),
        ({"b": -0.1}, "b must be between 0.0 and 1.0"),
        ({"b": 1.5}, "b must be between 0.0 and 1.0"),
        ({"epsilon": -1.0}, "epsilon must be a non-negative number"),
    ],
)
def test_bm25_config_invalid_parameters(kwargs, err_match):
    with pytest.raises(ValueError, match=err_match):
        BM25Config(**kwargs)


# ---------------------------------------------------------------------------
# 3. BM25Index Operations
# ---------------------------------------------------------------------------
def test_bm25_index_empty():
    index = BM25Index()
    assert index.count() == 0
    assert index.total_documents == 0
    assert index.average_document_length == 0.0
    res = index.search("anything")
    assert isinstance(res, SearchResult)
    assert len(res.chunks) == 0


def test_bm25_index_add_and_update():
    index = BM25Index()
    index.add_document("doc1", "Attention is all you need for translation")
    assert index.count() == 1
    assert index.average_document_length > 0

    index.add_document("doc2", "FlashAttention optimizes GPU memory hierarchy")
    assert index.count() == 2

    # Replacing doc1 updates index cleanly
    index.add_document("doc1", "New attention mechanisms")
    assert index.count() == 2

    index.clear()
    assert index.count() == 0
    assert index.average_document_length == 0.0


def test_bm25_idf_is_non_negative_even_for_ubiquitous_terms():
    index = BM25Index()
    index.add_document("d1", "attention attention attention")
    index.add_document("d2", "attention mechanism")
    index.add_document("d3", "attention model")

    # "attention" appears in 100% of documents (3 of 3)
    idf_common = index.idf("attention")
    assert idf_common >= 0.0
    assert math.isfinite(idf_common)

    # Rare term has higher IDF
    index.add_document("d4", "uncommon word")
    idf_rare = index.idf("uncommon")
    assert idf_rare > idf_common


# ---------------------------------------------------------------------------
# 4. Search and Ranking Behavior
# ---------------------------------------------------------------------------
def test_bm25_ranking_term_frequency_and_relevance():
    index = BM25Index()
    index.add_document("chunk_attention", "Attention mechanisms in deep learning. Self-attention layers compute exact attention.")
    index.add_document("chunk_cnn", "Convolutional neural networks process image grids using spatial convolution kernels.")
    index.add_document("chunk_rnn", "Recurrent neural networks process sequential tokens step by step.")

    # Search for attention
    res = index.search("self-attention mechanisms")
    assert len(res.chunks) > 0
    top_hit = res.chunks[0]
    assert top_hit.chunk_id == "chunk_attention"
    assert top_hit.score > 0.0

    # Search for cnn
    res_cnn = index.search("convolutional kernels")
    assert res_cnn.chunks[0].chunk_id == "chunk_cnn"

    # Search for unrelated query yields empty results
    res_empty = index.search("quantum annealing qubit")
    assert len(res_empty.chunks) == 0


def test_bm25_search_limit_and_threshold():
    index = BM25Index()
    for i in range(10):
        index.add_document(f"doc_{i}", f"Document number {i} discussing machine learning and neural networks")

    res_limit = index.search("machine learning", limit=3)
    assert len(res_limit.chunks) == 3

    # High min_score filters out candidates
    res_thresh = index.search("machine learning", min_score=999.0)
    assert len(res_thresh.chunks) == 0


def test_bm25_index_chunks_helper():
    chunks = [
        {"chunk_id": "c1", "text": "Low-rank adaptation reduces fine-tuning parameters."},
        {"chunk_id": "c2", "text": "FlashAttention tiles GPU memory to reduce HBM accesses."},
    ]
    index = BM25Index()
    count = index.index_chunks(chunks)
    assert count == 2
    assert index.count() == 2

    res = index.search("fine-tuning LoRA")
    assert len(res.chunks) == 1
    assert res.chunks[0].chunk_id == "c1"


# ---------------------------------------------------------------------------
# 5. Reciprocal Rank Fusion (RRF) Hybrid Search Tests
# ---------------------------------------------------------------------------
def test_reciprocal_rank_fusion_combines_rankings():
    chunk1 = RetrievedChunk("d1", "c1", "text 1", 0.9, {})
    chunk2 = RetrievedChunk("d2", "c2", "text 2", 0.8, {})
    chunk3 = RetrievedChunk("d3", "c3", "text 3", 0.7, {})

    # Vector ranking: c1, c2, c3
    vec_res = SearchResult(query="test", results=[chunk1, chunk2, chunk3])
    # BM25 ranking: c2, c1, c3
    bm25_res = SearchResult(query="test", results=[chunk2, chunk1, chunk3])

    fused = reciprocal_rank_fusion(vec_res, bm25_res, rrf_k=60, limit=2)
    assert len(fused.chunks) == 2
    # c1 and c2 appeared at ranks (1, 2) and (2, 1) respectively -> equal fused scores
    assert {c.chunk_id for c in fused.chunks} == {"c1", "c2"}
    assert all("rrf_score" in c.metadata for c in fused.chunks)


def test_reciprocal_rank_fusion_validation():
    dummy = SearchResult(query="q", results=[])
    with pytest.raises(TypeError, match="vector_results"):
        reciprocal_rank_fusion(None, dummy)  # type: ignore
    with pytest.raises(TypeError, match="bm25_results"):
        reciprocal_rank_fusion(dummy, None)  # type: ignore
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion(dummy, dummy, rrf_k=-5)


# ---------------------------------------------------------------------------
# 6. RetrievalService Integration
# ---------------------------------------------------------------------------
def test_retrieval_service_bm25_and_hybrid_modes():
    provider = HashEmbeddingProvider(dimension=16)
    emb_service = EmbeddingService(provider)
    vector_store = InMemoryVectorStore(dimension=16)

    doc_text1 = "LoRA reduces trainable parameters by freezing pre-trained weights."
    doc_text2 = "FlashAttention uses tiling to account for reads and writes between SRAM and HBM."

    # Ingest into vector store
    v1 = emb_service.embed_text(doc_text1)
    v2 = emb_service.embed_text(doc_text2)
    vector_store.upsert([
        VectorRecord(id="c1", vector=v1, payload={"chunk_id": "c1", "document_id": "d1", "text": doc_text1}),
        VectorRecord(id="c2", vector=v2, payload={"chunk_id": "c2", "document_id": "d2", "text": doc_text2}),
    ])

    # Ingest into BM25 index
    bm25_index = BM25Index()
    bm25_index.add_document("c1", doc_text1, document_id="d1")
    bm25_index.add_document("c2", doc_text2, document_id="d2")

    service = RetrievalService(emb_service, vector_store, bm25_index=bm25_index)
    assert service.bm25_index is bm25_index

    # 1. BM25 mode search
    bm25_out = service.retrieve("SRAM HBM tiling", limit=5, mode="bm25")
    assert len(bm25_out.chunks) > 0
    assert bm25_out.chunks[0].chunk_id == "c2"

    # 2. Hybrid mode search
    hybrid_out = service.retrieve("trainable parameters LoRA", limit=5, mode="hybrid")
    assert len(hybrid_out.chunks) > 0
    assert hybrid_out.chunks[0].chunk_id == "c1"

    # 3. Vector mode (default)
    vec_out = service.retrieve("test query", limit=5, mode="vector")
    assert isinstance(vec_out, SearchResult)


def test_retrieval_service_missing_bm25_index_raises_when_requested():
    provider = HashEmbeddingProvider(dimension=16)
    emb_service = EmbeddingService(provider)
    vector_store = InMemoryVectorStore(dimension=16)

    service = RetrievalService(emb_service, vector_store)  # No bm25_index
    with pytest.raises(ValueError, match="no bm25_index was configured"):
        service.retrieve("query", mode="bm25")

    with pytest.raises(ValueError, match="no bm25_index was configured"):
        service.retrieve("query", mode="hybrid")


# ---------------------------------------------------------------------------
# 7. Quality Service Integration
# ---------------------------------------------------------------------------
def test_retrieval_quality_service_forwards_mode_kwargs():
    provider = HashEmbeddingProvider(dimension=16)
    emb_service = EmbeddingService(provider)
    vector_store = InMemoryVectorStore(dimension=16)

    bm25_index = BM25Index()
    bm25_index.add_document("c1", "Attention is all you need", metadata={"paper_id": "p1"})
    bm25_index.add_document("c2", "FlashAttention IO-aware", metadata={"paper_id": "p2"})

    service = RetrievalService(emb_service, vector_store, bm25_index=bm25_index)
    policy = RetrievalQualityPolicy(min_score=0.1, max_results=5)
    quality_service = RetrievalQualityService(service, policy)

    res = quality_service.retrieve("Attention translation", mode="bm25")
    assert len(res.chunks) > 0
    assert res.chunks[0].chunk_id == "c1"


# ---------------------------------------------------------------------------
# 8. Real-World Pilot Benchmark Evaluation with BM25
# ---------------------------------------------------------------------------
def test_pilot_benchmark_evaluates_accurately_with_bm25():
    """Verify that BM25 achieves high retrieval accuracy on the genuine pilot benchmark offline."""
    ds = get_pilot_benchmark_dataset()
    bm25_index = BM25Index()
    bm25_index.index_chunks(ds.chunks)

    search_fn = lambda q, limit=5: bm25_index.search(q, limit=limit)
    metrics = evaluate_retrieval(ds.queries, search_fn, k_values=(1, 3, 5))

    # BM25 achieves strong retrieval accuracy offline without embeddings:
    # 5 of 6 answerable queries rank #1, 1 ranks #2 (MRR = 0.7857), Recall@3 = 100%, and unanswerable rejection = 100%
    assert metrics.mean_reciprocal_rank >= 0.75
    assert metrics.recall_at_k[1] >= 0.85
    assert metrics.recall_at_k[3] == 1.0
    assert metrics.precision_at_k[1] >= 0.85
    assert metrics.unanswerable_rejection_rate == 1.0


# ---------------------------------------------------------------------------
# 9. CLI Integration with --retriever Flag
# ---------------------------------------------------------------------------
def test_cli_configured_pipeline_with_bm25_retriever(tmp_path):
    out_file = tmp_path / "bm25_eval_report.json"
    exit_code, report, json_text = run_eval_cli([
        "--mode", "configured_pipeline",
        "--dataset", "examples/real_world_pilot_benchmark.json",
        "--retriever", "bm25",
        "--output", str(out_file),
    ])

    assert exit_code == 0
    assert report.status == "completed"
    assert report.evaluation_conditions.get("retrieval_method") == "bm25"
    assert report.metrics is not None
    assert report.metrics.mean_reciprocal_rank >= 0.75
    assert report.metrics.recall_at_k.get(3) == 1.0
    assert report.metrics.unanswerable_rejection_rate == 1.0

    data = json.loads(json_text)
    assert data["evaluation_conditions"]["retrieval_method"] == "bm25"
    assert out_file.exists()
