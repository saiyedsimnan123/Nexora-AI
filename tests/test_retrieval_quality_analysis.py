"""Tests for Milestone 12C-34: Dataset-Level Retrieval Quality Analysis.

Verifies:
1. Per-query-category retrieval metrics aggregation, handling missing categories
   (bucketed to 'uncategorized') and empty categories (0.0 metrics without divide-by-zero).
2. Baseline retrieval vs quality-filtered retrieval comparison with honest delta reporting
   and regression tracking (net_effect: improved, degraded, mixed, unchanged).
3. Preservation of illustrative, benchmark, and unknown dataset provenance.
4. Ground-truth relevance isolation (labels used strictly for scoring, never passed to retrieval).
5. Integration with evaluate_production_retrieval and eval_cli.
"""

from __future__ import annotations

import json
from typing import Any
import pytest

from nexora.retrieval.dataset import ResearchRetrievalDataset
from nexora.retrieval.eval_cli import run_eval_cli
from nexora.retrieval.evaluation import (
    RetrievalEvaluationMetrics,
    compare_retrieval_metrics,
    evaluate_retrieval,
)
from nexora.retrieval.models import RetrievedChunk, SearchResult
from nexora.retrieval.production_eval import (
    EvaluationMode,
    ProductionEvaluationConfig,
    evaluate_production_retrieval,
)
from nexora.retrieval.quality import RetrievalQualityPolicy


def _make_dummy_search_result(chunk_ids: list[str]) -> SearchResult:
    """Helper to create dummy SearchResult with given chunk IDs."""
    chunks = [
        RetrievedChunk(
            document_id=f"doc_{cid}",
            chunk_id=cid,
            text=f"Sample text for {cid}",
            score=0.9 - (idx * 0.1),
            metadata={"paper_id": f"paper_{cid}"},
        )
        for idx, cid in enumerate(chunk_ids)
    ]
    return SearchResult(query="test", results=chunks)


# ===========================================================================
# 1. Category Aggregation & Empty Group Handling
# ===========================================================================

def test_category_aggregation_with_explicit_categories():
    """Verify metrics are grouped by category when category labels are present."""
    cases = [
        {"query": "q1", "expected_chunk_ids": ["c1"], "category": "exact_match"},
        {"query": "q2", "expected_chunk_ids": ["c2"], "category": "exact_match"},
        {"query": "q3", "expected_chunk_ids": ["c3"], "category": "semantic"},
    ]

    def mock_retrieve(query: str, limit: int = 5) -> SearchResult:
        if query == "q1":
            return _make_dummy_search_result(["c1"])  # Rank 1 -> RR = 1.0
        elif query == "q2":
            return _make_dummy_search_result(["other", "c2"])  # Rank 2 -> RR = 0.5
        else:
            return _make_dummy_search_result(["c3"])  # Rank 1 -> RR = 1.0

    metrics = evaluate_retrieval(cases, mock_retrieve, k_values=(1, 3))

    assert metrics.category_metrics is not None
    assert set(metrics.category_metrics.keys()) == {"exact_match", "semantic"}

    exact = metrics.category_metrics["exact_match"]
    assert exact["count"] == 2
    assert exact["mean_reciprocal_rank"] == 0.75  # (1.0 + 0.5) / 2
    assert exact["precision_at_k"][1] == 0.5  # (1/1 + 0/1) / 2
    assert exact["recall_at_k"][1] == 0.5  # (1/1 + 0/1) / 2
    assert exact["precision_at_k"][3] == 0.3333  # (1/3 + 1/3) / 2
    assert exact["recall_at_k"][3] == 1.0  # both found within 3

    sem = metrics.category_metrics["semantic"]
    assert sem["count"] == 1
    assert sem["mean_reciprocal_rank"] == 1.0


def test_category_aggregation_missing_category_bucketed_as_uncategorized():
    """Queries missing category label are grouped into 'uncategorized' when categories exist."""
    cases = [
        {"query": "q1", "expected_chunk_ids": ["c1"], "category": "theory"},
        {"query": "q2", "expected_chunk_ids": ["c2"]},  # Missing category key
        {"query": "q3", "expected_chunk_ids": ["c3"], "category": None},  # None category
        {"query": "q4", "expected_chunk_ids": ["c4"], "category": "   "},  # Whitespace category
    ]

    def mock_retrieve(query: str, limit: int = 5) -> SearchResult:
        c_map = {"q1": "c1", "q2": "c2", "q3": "c3", "q4": "c4"}
        return _make_dummy_search_result([c_map[query]])

    metrics = evaluate_retrieval(cases, mock_retrieve, k_values=(1, 3))

    assert metrics.category_metrics is not None
    assert set(metrics.category_metrics.keys()) == {"theory", "uncategorized"}
    assert metrics.category_metrics["theory"]["count"] == 1
    assert metrics.category_metrics["uncategorized"]["count"] == 3
    assert metrics.category_metrics["uncategorized"]["mean_reciprocal_rank"] == 1.0


def test_empty_category_groups_handled_without_divide_by_zero():
    """Supplied category list with zero matches returns 0.0 metrics with count 0."""
    cases = [
        {"query": "q1", "expected_chunk_ids": ["c1"], "category": "active_cat"},
    ]

    def mock_retrieve(query: str, limit: int = 5) -> SearchResult:
        return _make_dummy_search_result(["c1"])

    metrics = evaluate_retrieval(
        cases,
        mock_retrieve,
        k_values=(1, 5),
        categories=["active_cat", "empty_cat_1", "empty_cat_2"],
    )

    assert metrics.category_metrics is not None
    assert set(metrics.category_metrics.keys()) == {"active_cat", "empty_cat_1", "empty_cat_2"}

    assert metrics.category_metrics["active_cat"]["count"] == 1
    assert metrics.category_metrics["active_cat"]["mean_reciprocal_rank"] == 1.0

    empty1 = metrics.category_metrics["empty_cat_1"]
    assert empty1["count"] == 0
    assert empty1["mean_reciprocal_rank"] == 0.0
    assert empty1["precision_at_k"] == {1: 0.0, 5: 0.0}
    assert empty1["recall_at_k"] == {1: 0.0, 5: 0.0}

    empty2 = metrics.category_metrics["empty_cat_2"]
    assert empty2["count"] == 0
    assert empty2["mean_reciprocal_rank"] == 0.0


def test_no_categories_leaves_category_metrics_none():
    """If no cases provide category and no categories are specified, category_metrics is None."""
    cases = [
        {"query": "q1", "expected_chunk_ids": ["c1"]},
        {"query": "q2", "expected_chunk_ids": ["c2"]},
    ]

    def mock_retrieve(query: str, limit: int = 5) -> SearchResult:
        return _make_dummy_search_result(["c1"])

    metrics = evaluate_retrieval(cases, mock_retrieve, k_values=(1,))
    assert metrics.category_metrics is None


# ===========================================================================
# 2. Baseline vs Quality-Filtered Comparison Logic
# ===========================================================================

def test_compare_retrieval_metrics_improvement():
    """Compare baseline to filtered where metrics strictly improved."""
    baseline = RetrievalEvaluationMetrics(
        total_queries=10,
        mean_reciprocal_rank=0.50,
        recall_at_k={1: 0.40, 5: 0.70},
        precision_at_k={1: 0.40, 5: 0.20},
        empty_retrievals=2,
        unanswerable_queries=2,
        unanswerable_rejections=1,
        unanswerable_rejection_rate=0.50,
    )
    filtered = RetrievalEvaluationMetrics(
        total_queries=10,
        mean_reciprocal_rank=0.75,
        recall_at_k={1: 0.60, 5: 0.80},
        precision_at_k={1: 0.60, 5: 0.30},
        empty_retrievals=3,
        unanswerable_queries=2,
        unanswerable_rejections=2,
        unanswerable_rejection_rate=1.00,
    )

    comparison = compare_retrieval_metrics(baseline, filtered)

    assert comparison["net_effect"] == "improved"
    assert comparison["has_regressions"] is False
    assert comparison["regressions_count"] == 0
    assert comparison["improvements_count"] > 0
    assert comparison["mrr_delta"] == 0.25
    assert comparison["recall_at_k_deltas"] == {1: 0.20, 5: 0.10}
    assert comparison["precision_at_k_deltas"] == {1: 0.20, 5: 0.10}
    assert comparison["unanswerable_rejection_rate_delta"] == 0.50
    assert comparison["regressions"] == []


def test_compare_retrieval_metrics_degradation():
    """Compare baseline to filtered where metrics degraded honestly reports regressions."""
    baseline = RetrievalEvaluationMetrics(
        total_queries=10,
        mean_reciprocal_rank=0.80,
        recall_at_k={1: 0.80, 3: 0.90},
        precision_at_k={1: 0.80, 3: 0.40},
        empty_retrievals=0,
        unanswerable_queries=2,
        unanswerable_rejections=2,
        unanswerable_rejection_rate=1.00,
    )
    filtered = RetrievalEvaluationMetrics(
        total_queries=10,
        mean_reciprocal_rank=0.60,
        recall_at_k={1: 0.60, 3: 0.70},
        precision_at_k={1: 0.60, 3: 0.30},
        empty_retrievals=2,
        unanswerable_queries=2,
        unanswerable_rejections=1,
        unanswerable_rejection_rate=0.50,
    )

    comparison = compare_retrieval_metrics(baseline, filtered)

    assert comparison["net_effect"] == "degraded"
    assert comparison["has_regressions"] is True
    assert comparison["regressions_count"] >= 4
    assert comparison["improvements_count"] == 0
    assert comparison["mrr_delta"] == -0.20
    assert comparison["recall_at_k_deltas"][1] == -0.20
    assert comparison["unanswerable_rejection_rate_delta"] == -0.50

    reg_metrics = [r["metric"] for r in comparison["regressions"]]
    assert "mean_reciprocal_rank" in reg_metrics
    assert "recall@1" in reg_metrics
    assert "precision@1" in reg_metrics
    assert "unanswerable_rejection_rate" in reg_metrics


def test_compare_retrieval_metrics_mixed_and_unchanged():
    """Verify 'mixed' classification when some metrics improve while others degrade, and 'unchanged'."""
    # Mixed case: MRR improved, but Recall@5 dropped due to tight filtering
    baseline = RetrievalEvaluationMetrics(
        total_queries=10,
        mean_reciprocal_rank=0.50,
        recall_at_k={1: 0.40, 5: 0.90},
        precision_at_k={1: 0.40, 5: 0.20},
        empty_retrievals=0,
    )
    filtered = RetrievalEvaluationMetrics(
        total_queries=10,
        mean_reciprocal_rank=0.70,  # Improved
        recall_at_k={1: 0.60, 5: 0.70},  # Recall@5 degraded!
        precision_at_k={1: 0.60, 5: 0.30},
        empty_retrievals=1,
    )

    comp_mixed = compare_retrieval_metrics(baseline, filtered)
    assert comp_mixed["net_effect"] == "mixed"
    assert comp_mixed["has_regressions"] is True
    assert comp_mixed["regressions_count"] > 0
    assert comp_mixed["improvements_count"] > 0
    assert any(r["metric"] == "recall@5" for r in comp_mixed["regressions"])

    # Unchanged case
    comp_same = compare_retrieval_metrics(baseline, baseline)
    assert comp_same["net_effect"] == "unchanged"
    assert comp_same["has_regressions"] is False
    assert comp_same["mrr_delta"] == 0.0


def test_compare_retrieval_metrics_per_category():
    """Verify per-category comparison delta calculation."""
    baseline = RetrievalEvaluationMetrics(
        total_queries=4,
        mean_reciprocal_rank=0.50,
        recall_at_k={1: 0.50},
        precision_at_k={1: 0.50},
        empty_retrievals=0,
        category_metrics={
            "exact": {"count": 2, "mean_reciprocal_rank": 0.5, "precision_at_k": {1: 0.5}, "recall_at_k": {1: 0.5}},
            "semantic": {"count": 2, "mean_reciprocal_rank": 0.5, "precision_at_k": {1: 0.5}, "recall_at_k": {1: 0.5}},
        },
    )
    filtered = RetrievalEvaluationMetrics(
        total_queries=4,
        mean_reciprocal_rank=0.75,
        recall_at_k={1: 0.75},
        precision_at_k={1: 0.75},
        empty_retrievals=0,
        category_metrics={
            "exact": {"count": 2, "mean_reciprocal_rank": 1.0, "precision_at_k": {1: 1.0}, "recall_at_k": {1: 1.0}},
            "semantic": {"count": 2, "mean_reciprocal_rank": 0.5, "precision_at_k": {1: 0.5}, "recall_at_k": {1: 0.5}},
        },
    )

    comparison = compare_retrieval_metrics(baseline, filtered)
    cat_comp = comparison["category_comparisons"]
    assert "exact" in cat_comp
    assert "semantic" in cat_comp
    assert cat_comp["exact"]["mrr_delta"] == 0.50
    assert cat_comp["exact"]["precision_at_1_delta"] == 0.50
    assert cat_comp["semantic"]["mrr_delta"] == 0.00


# ===========================================================================
# 3. Ground-Truth Relevance Label Isolation
# ===========================================================================

def test_ground_truth_relevance_labels_strictly_isolated_from_retrieval():
    """Verify that retrieval callers never receive ground-truth relevance labels as inputs."""
    calls: list[tuple[str, int | None]] = []

    def spying_retrieve_fn(query: str, limit: int = 5) -> SearchResult:
        calls.append((query, limit))
        return _make_dummy_search_result(["c1"])

    cases = [
        {"query": "attention scaling", "expected_chunk_ids": ["c1", "secret_gt_id_123"], "category": "test"},
        {"query": "unanswerable topic", "expected_chunk_ids": [], "category": "empty"},
    ]

    metrics = evaluate_retrieval(cases, spying_retrieve_fn, k_values=(1, 3))

    assert metrics.total_queries == 2
    assert len(calls) == 2

    # Check that retrieve_fn was called with query text only
    for query_arg, limit_arg in calls:
        assert isinstance(query_arg, str)
        assert "secret_gt_id_123" not in query_arg
        assert isinstance(limit_arg, int)


# ===========================================================================
# 4. Provenance Preservation Across Reports
# ===========================================================================

def test_evaluate_production_retrieval_preserves_provenance_and_compares_baseline():
    """Verify evaluate_production_retrieval generates both baseline and quality-filtered metrics."""
    # 1. Custom illustrative dataset
    ds_illustrative = ResearchRetrievalDataset(
        dataset_id="ill_ds",
        description="illustrative sample",
        chunks=[
            {"chunk_id": "c1", "paper_id": "p1", "text": "Scaled Dot-Product Attention equation."},
            {"chunk_id": "c2", "paper_id": "p2", "text": "FlashAttention IO-aware tiling."},
        ],
        queries=[
            {"query_id": "q1", "query": "Scaled Dot-Product", "expected_chunk_ids": ["c1"], "category": "exact"},
            {"query_id": "q2", "query": "Unknown topic", "expected_chunk_ids": [], "category": "empty"},
        ],
        is_illustrative_only=True,
    )

    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.SYNTHETIC,
        dataset=ds_illustrative,
        quality_policy=RetrievalQualityPolicy(min_score=0.35, max_results=3),
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert report.dataset_info is not None
    assert report.dataset_info["provenance"] == "illustrative"

    assert report.baseline_metrics is not None
    assert report.baseline_metrics.total_queries == 2

    assert report.metrics is not None
    assert report.metrics.total_queries == 2

    assert report.quality_comparison is not None
    assert report.quality_comparison["dataset_provenance"] == "illustrative"
    assert "net_effect" in report.quality_comparison
    assert "mrr_delta" in report.quality_comparison
    assert "regressions" in report.quality_comparison

    # 2. Benchmark dataset
    ds_benchmark = ResearchRetrievalDataset(
        dataset_id="bench_ds",
        description="real benchmark",
        chunks=ds_illustrative.chunks,
        queries=ds_illustrative.queries,
        is_illustrative_only=False,
    )
    cfg_bench = ProductionEvaluationConfig(
        mode=EvaluationMode.SYNTHETIC,
        dataset=ds_benchmark,
    )
    report_bench = evaluate_production_retrieval(cfg_bench)
    assert report_bench.dataset_info["provenance"] == "benchmark"
    assert report_bench.quality_comparison["dataset_provenance"] == "benchmark"

    # 3. Unknown provenance dataset
    ds_unknown = ResearchRetrievalDataset(
        dataset_id="unknown_ds",
        description="unknown provenance",
        chunks=ds_illustrative.chunks,
        queries=ds_illustrative.queries,
        is_illustrative_only=None,
    )
    cfg_unknown = ProductionEvaluationConfig(
        mode=EvaluationMode.SYNTHETIC,
        dataset=ds_unknown,
    )
    report_unknown = evaluate_production_retrieval(cfg_unknown)
    assert report_unknown.dataset_info["provenance"] == "unknown"
    assert report_unknown.quality_comparison["dataset_provenance"] == "unknown"


# ===========================================================================
# 5. CLI Output & Serialization Integration
# ===========================================================================

def test_eval_cli_serializes_baseline_and_quality_comparison(tmp_path, capsys):
    """Verify eval CLI outputs baseline metrics and quality comparison to JSON."""
    report_path = str(tmp_path / "eval_report.json")

    exit_code, report, json_text = run_eval_cli([
        "--mode", "synthetic",
        "--output", report_path,
        "--min-score", "0.35",
    ])

    assert exit_code == 0
    assert report.status == "completed"
    assert report.baseline_metrics is not None
    assert report.quality_comparison is not None

    data = json.loads(json_text)
    assert "baseline_metrics" in data
    assert data["baseline_metrics"]["total_queries"] == 13
    assert "quality_comparison" in data
    assert "net_effect" in data["quality_comparison"]
    assert "mrr_delta" in data["quality_comparison"]
    assert "regressions" in data["quality_comparison"]
    assert data["quality_comparison"]["dataset_provenance"] == "illustrative"

    # Verify stdout summary contains quality filter effect
    captured = capsys.readouterr().out
    assert "Quality Filter Effect:" in captured
    assert "MRR Delta:" in captured
