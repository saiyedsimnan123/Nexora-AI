"""Test suite for retrieval quality evaluation and benchmark measurements (Milestone 12C-30).

Verifies:
1. Deterministic benchmark dataset design and corpus integrity.
2. Reproducible baseline metrics on raw vector retrieval.
3. Measurable retrieval quality improvements with RetrievalQualityPolicy.
4. Exact-term, semantic, and multi-paper query ranking accuracy.
5. Discrimination of misleading near-matches.
6. Clean rejection of unanswerable and empty ground-truth queries.
7. Prevention of cross-paper contamination via paper-scoping policy.
8. Preservation of evidence metadata throughout the retrieval pipeline.
9. Strict separation of retrieval evaluation from LLM answer generation (zero LLM judge dependency).
"""

from __future__ import annotations

import pytest

from nexora.retrieval.benchmark import (
    DOC_DISTRACTOR,
    DOC_FLASHATTN,
    DOC_LORA,
    DOC_TRANSFORMER,
    PAPER_DISTRACTOR,
    PAPER_FLASHATTN,
    PAPER_LORA,
    PAPER_TRANSFORMER,
    BenchmarkSemanticEmbeddingService,
    build_benchmark_corpus,
    get_benchmark_cases,
    run_retrieval_benchmark,
)
from nexora.retrieval.evaluation import (
    RetrievalEvaluationMetrics,
    compute_precision_recall_at_k,
    compute_reciprocal_rank,
    evaluate_retrieval,
)
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk, SearchResult
from nexora.retrieval.quality import RetrievalQualityPolicy, RetrievalQualityService
from nexora.retrieval.service import RetrievalService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def embedding_service() -> BenchmarkSemanticEmbeddingService:
    return BenchmarkSemanticEmbeddingService()


@pytest.fixture
def populated_vector_store(embedding_service: BenchmarkSemanticEmbeddingService) -> InMemoryVectorStore:
    store = InMemoryVectorStore(dimension=embedding_service.dimension)
    corpus_records = build_benchmark_corpus(embedding_service)
    store.upsert(corpus_records)
    return store


@pytest.fixture
def raw_retrieval_service(
    embedding_service: BenchmarkSemanticEmbeddingService,
    populated_vector_store: InMemoryVectorStore,
) -> RetrievalService:
    return RetrievalService(embedding_service, populated_vector_store, cache_queries=True)


# ---------------------------------------------------------------------------
# 1. Benchmark Corpus and Cases Design Integrity
# ---------------------------------------------------------------------------
def test_benchmark_corpus_and_cases_integrity(
    embedding_service: BenchmarkSemanticEmbeddingService,
    populated_vector_store: InMemoryVectorStore,
):
    """Verify that the benchmark dataset contains 4 papers, 15 chunks, valid metadata, and 13 cases."""
    assert populated_vector_store.count() == 15

    cases = get_benchmark_cases()
    assert len(cases) == 13

    categories = {c["category"] for c in cases}
    assert categories == {
        "exact_term",
        "semantic",
        "multi_paper",
        "misleading_near_match",
        "unanswerable_empty",
    }

    # Verify all expected chunk IDs exist in corpus records
    corpus_records = build_benchmark_corpus(embedding_service)
    corpus_ids = {r.id for r in corpus_records}
    for case in cases:
        for cid in case["expected_chunk_ids"]:
            assert cid in corpus_ids, f"Expected chunk {cid} not found in benchmark corpus"

    # Verify embeddings are unit-normalized
    for r in corpus_records:
        norm = sum(x * x for x in r.vector)
        assert pytest.approx(norm, rel=1e-5) == 1.0


# ---------------------------------------------------------------------------
# 2. Reproducible Baseline Retrieval Metrics
# ---------------------------------------------------------------------------
def test_baseline_retrieval_metrics(raw_retrieval_service: RetrievalService):
    """Establish and record the reproducible baseline on raw vector retrieval.

    Baseline demonstrates:
    - High ranking on answerable queries (MRR ~0.96)
    - Defect: Unanswerable queries return low-scoring irrelevant chunks instead of clean rejection
      (rejection rate = 0.0 because raw store returns top-k unconditionally).
    """
    cases = get_benchmark_cases()

    # Retrieve directly from raw RetrievalService (limit=5)
    def _raw_retrieve(q: str, limit: int = 5) -> SearchResult:
        return raw_retrieval_service.retrieve(q, limit=limit)

    baseline_metrics = evaluate_retrieval(cases, _raw_retrieve, k_values=(1, 3, 5))

    assert baseline_metrics.total_queries == 13
    assert baseline_metrics.unanswerable_queries == 2
    # Baseline weakness: 0% clean rejection of unanswerable queries (2 false alarms)
    assert baseline_metrics.unanswerable_rejections == 0
    assert baseline_metrics.unanswerable_rejection_rate == 0.0
    assert baseline_metrics.empty_retrievals == 0

    # Overall MRR across all 13 queries (unanswerable queries have RR = 0.0)
    assert baseline_metrics.mean_reciprocal_rank >= 0.75
    # Answerable categories achieve perfect rank-1 retrieval
    assert baseline_metrics.category_metrics["exact_term"]["mean_reciprocal_rank"] == 1.0
    assert baseline_metrics.category_metrics["semantic"]["mean_reciprocal_rank"] == 1.0
    assert baseline_metrics.category_metrics["misleading_near_match"]["mean_reciprocal_rank"] == 1.0


# ---------------------------------------------------------------------------
# 3. Quality Policy Demonstrates Measurable Improvement
# ---------------------------------------------------------------------------
def test_quality_policy_demonstrates_measurable_improvement(
    raw_retrieval_service: RetrievalService,
):
    """Prove that RetrievalQualityPolicy with min_score and per-paper limits improves quality.

    Measurements prove:
    - Unanswerable clean rejection rate improves from 0.0 (0%) in baseline to 1.0 (100%) in quality service!
    - Empty retrievals correctly reflect the 2 cleanly rejected unanswerable queries.
    - MRR and Recall remain high while eliminating false alarms.
    """
    cases = get_benchmark_cases()

    # Raw baseline runner
    baseline_metrics = evaluate_retrieval(
        cases,
        lambda q, limit=5: raw_retrieval_service.retrieve(q, limit=limit),
        k_values=(1, 3, 5),
    )

    # Tuned Quality Service with min_score threshold
    quality_policy = RetrievalQualityPolicy(
        min_score=0.35,
        max_results=5,
        max_results_per_paper=2,
        max_results_per_document=2,
    )
    quality_service = RetrievalQualityService(raw_retrieval_service, quality_policy)

    quality_metrics = evaluate_retrieval(
        cases,
        lambda q, limit=5: quality_service.retrieve(q),
        k_values=(1, 3, 5),
    )

    # 1. Measurable improvement: Unanswerable clean rejection rate
    assert baseline_metrics.unanswerable_rejection_rate == 0.0
    assert quality_metrics.unanswerable_rejection_rate == 1.0  # 100% improvement!
    assert quality_metrics.unanswerable_rejections == 2

    # 2. Empty retrievals accurately capture the clean rejections
    assert quality_metrics.empty_retrievals == 2

    # 3. High precision and ranking preserved
    assert quality_metrics.mean_reciprocal_rank >= 0.75
    assert quality_metrics.precision_at_k[1] >= 0.80


# ---------------------------------------------------------------------------
# 4. Exact-Term Retrieval Evaluation
# ---------------------------------------------------------------------------
def test_exact_term_retrieval_relevance_and_mrr(raw_retrieval_service: RetrievalService):
    """Verify that exact-term scientific queries achieve rank-1 retrieval."""
    cases = [c for c in get_benchmark_cases() if c["category"] == "exact_term"]
    assert len(cases) == 3

    metrics = evaluate_retrieval(
        cases,
        lambda q, limit=3: raw_retrieval_service.retrieve(q, limit=limit),
        k_values=(1, 3),
    )

    # All exact-term queries rank expected chunk at position 1
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.recall_at_k[1] == 1.0
    assert metrics.precision_at_k[1] == 1.0


# ---------------------------------------------------------------------------
# 5. Semantic Concept Retrieval Evaluation
# ---------------------------------------------------------------------------
def test_semantic_concept_retrieval_relevance(raw_retrieval_service: RetrievalService):
    """Verify that semantic queries retrieve target chunks despite conceptual paraphrase."""
    cases = [c for c in get_benchmark_cases() if c["category"] == "semantic"]
    assert len(cases) == 3

    metrics = evaluate_retrieval(
        cases,
        lambda q, limit=3: raw_retrieval_service.retrieve(q, limit=limit),
        k_values=(1, 3),
    )

    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.recall_at_k[1] == 1.0


# ---------------------------------------------------------------------------
# 6. Multi-Paper Retrieval and Diversity
# ---------------------------------------------------------------------------
def test_multi_paper_retrieval_and_diversity(raw_retrieval_service: RetrievalService):
    """Verify multi-paper queries retrieve relevant chunks from both distinct papers."""
    cases = [c for c in get_benchmark_cases() if c["category"] == "multi_paper"]
    assert len(cases) == 2

    policy = RetrievalQualityPolicy(max_results=5, max_results_per_paper=2)
    service = RetrievalQualityService(raw_retrieval_service, policy)

    for case in cases:
        res = service.retrieve(case["query"])
        retrieved_ids = {c.chunk_id for c in res.results}
        retrieved_papers = {c.metadata.get("paper_id") for c in res.results}

        # Both expected chunks retrieved
        for expected_id in case["expected_chunk_ids"]:
            assert expected_id in retrieved_ids

        # Both target papers represented in the retrieved candidates
        for target_paper in case["target_paper_ids"]:
            assert target_paper in retrieved_papers


# ---------------------------------------------------------------------------
# 7. Misleading Near-Match Discrimination
# ---------------------------------------------------------------------------
def test_misleading_near_match_discrimination(raw_retrieval_service: RetrievalService):
    """Verify that queries with distractors rank the true domain match above misleading near-matches."""
    cases = [c for c in get_benchmark_cases() if c["category"] == "misleading_near_match"]
    assert len(cases) == 3

    metrics = evaluate_retrieval(
        cases,
        lambda q, limit=3: raw_retrieval_service.retrieve(q, limit=limit),
        k_values=(1, 3),
    )

    # True specific matches are ranked above generic distractors
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.recall_at_k[1] == 1.0


# ---------------------------------------------------------------------------
# 8. Cross-Paper Contamination Prevention
# ---------------------------------------------------------------------------
def test_cross_paper_contamination_prevention(raw_retrieval_service: RetrievalService):
    """Verify that allowed_paper_ids strictly prevents chunks from other papers from leaking into results."""
    # Query mentions attention and IO/memory, which exists across P1, P2, and P4
    query = "Attention mechanisms and computational complexity"

    # 1. Unscoped retrieval contains chunks from multiple papers
    unscoped_res = raw_retrieval_service.retrieve(query, limit=5)
    unscoped_papers = {c.metadata.get("paper_id") for c in unscoped_res.results}
    assert len(unscoped_papers) > 1

    # 2. Paper-scoped policy for Transformer (P1)
    p1_policy = RetrievalQualityPolicy(allowed_paper_ids=(PAPER_TRANSFORMER,))
    p1_service = RetrievalQualityService(raw_retrieval_service, p1_policy)
    p1_res = p1_service.retrieve(query)

    assert len(p1_res.results) > 0
    assert all(c.metadata.get("paper_id") == PAPER_TRANSFORMER for c in p1_res.results)
    assert not any(c.metadata.get("paper_id") == PAPER_FLASHATTN for c in p1_res.results)
    assert not any(c.metadata.get("paper_id") == PAPER_DISTRACTOR for c in p1_res.results)

    # 3. Paper-scoped policy for FlashAttention (P2)
    p2_policy = RetrievalQualityPolicy(allowed_paper_ids=(PAPER_FLASHATTN,))
    p2_service = RetrievalQualityService(raw_retrieval_service, p2_policy)
    p2_res = p2_service.retrieve(query)

    assert len(p2_res.results) > 0
    assert all(c.metadata.get("paper_id") == PAPER_FLASHATTN for c in p2_res.results)
    assert not any(c.metadata.get("paper_id") == PAPER_TRANSFORMER for c in p2_res.results)


# ---------------------------------------------------------------------------
# 9. Evidence Metadata Preservation Across Pipeline
# ---------------------------------------------------------------------------
def test_evidence_metadata_preservation_across_pipeline(
    raw_retrieval_service: RetrievalService,
):
    """Verify that document_id, chunk_id, paper_id, title, filename, and score are preserved."""
    policy = RetrievalQualityPolicy(min_score=0.30, max_results=5)
    service = RetrievalQualityService(raw_retrieval_service, policy)

    res = service.retrieve("What is the scaling factor in Scaled Dot-Product Attention?")
    assert len(res.results) > 0

    top_chunk = res.results[0]
    assert top_chunk.chunk_id == "chunk_p1_c1"
    assert top_chunk.document_id == DOC_TRANSFORMER
    assert top_chunk.metadata.get("paper_id") == PAPER_TRANSFORMER
    assert top_chunk.metadata.get("title") == "Attention Is All You Need"
    assert top_chunk.metadata.get("filename") == "vaswani_2017.pdf"
    assert top_chunk.metadata.get("chunk_index") == 0
    assert top_chunk.score > 0.5
    assert "Scaled Dot-Product Attention" in top_chunk.text


# ---------------------------------------------------------------------------
# 10. Separation of Retrieval Evaluation from LLM Answer Generation
# ---------------------------------------------------------------------------
def test_retrieval_evaluation_separate_from_llm_judge():
    """Verify that retrieval metrics are computed deterministically without any LLM judge."""
    cases = [
        {"query": "q1", "expected_chunk_ids": ["c1", "c2"]},
        {"query": "q2", "expected_chunk_ids": []},
    ]

    # Deterministic mock search
    def _mock_search(q: str, limit: int = 3) -> SearchResult:
        if q == "q1":
            return SearchResult("q1", [
                RetrievedChunk("d1", "c1", "text 1", 0.9, {}),
                RetrievedChunk("d1", "c3", "text 3", 0.5, {}),
            ])
        # Clean rejection for q2
        return SearchResult("q2", [])

    metrics = evaluate_retrieval(cases, _mock_search, k_values=(1, 2))

    assert metrics.total_queries == 2
    assert metrics.unanswerable_queries == 1
    assert metrics.unanswerable_rejections == 1
    assert metrics.unanswerable_rejection_rate == 1.0

    # q1 has first relevant at rank 1 -> RR = 1.0; q2 has RR = 0.0 -> average = 0.5
    assert metrics.mean_reciprocal_rank == 0.5


# ---------------------------------------------------------------------------
# 11. Unanswerable and Empty Ground-Truth Edge Cases
# ---------------------------------------------------------------------------
def test_unanswerable_and_empty_ground_truth_definitions():
    """Verify formal definitions of Precision@k, Recall@k, and MRR for empty cases."""
    # 1. Empty ground truth, clean rejection (retrieved is empty) -> (1.0, 1.0)
    p_clean, r_clean = compute_precision_recall_at_k([], set(), 3)
    assert (p_clean, r_clean) == (1.0, 1.0)

    # 2. Empty ground truth, false alarms retrieved -> (0.0, 0.0)
    p_false_alarm, r_false_alarm = compute_precision_recall_at_k(["c1", "c2"], set(), 3)
    assert (p_false_alarm, r_false_alarm) == (0.0, 0.0)

    # 3. Non-empty ground truth, empty retrieval -> (0.0, 0.0)
    p_miss, r_miss = compute_precision_recall_at_k([], {"c1"}, 3)
    assert (p_miss, r_miss) == (0.0, 0.0)

    # 4. compute_reciprocal_rank on empty target or empty retrieval -> 0.0
    assert compute_reciprocal_rank([], {"c1"}) == 0.0
    assert compute_reciprocal_rank(["c1"], set()) == 0.0
    assert compute_reciprocal_rank([], set()) == 0.0


# ---------------------------------------------------------------------------
# 12. Helper Runner Validation
# ---------------------------------------------------------------------------
def test_run_retrieval_benchmark_helper(raw_retrieval_service: RetrievalService):
    """Verify run_retrieval_benchmark helper runs smoothly and returns valid metrics."""
    metrics = run_retrieval_benchmark(raw_retrieval_service.retrieve, k_values=(1, 3))
    assert isinstance(metrics, RetrievalEvaluationMetrics)
    assert metrics.total_queries == 13
    assert len(metrics.recall_at_k) == 2
    assert len(metrics.precision_at_k) == 2
