"""Tests for Milestone 12C-36: Curated Real-World Benchmark Pilot.

Verifies:
1. Pilot benchmark dataset loading and schema validation from JSON and JSONL.
2. Verified citation metadata: traceable titles, DOIs, URLs, publication years.
3. Complete manual review audit trail: reviewer identity and notes on every query.
4. Evaluation pipeline execution: baseline vs. quality-filtered retrieval.
5. Per-category metric reporting and unanswerable query behavior.
6. CLI integration and structured evaluation reporting.
"""

from __future__ import annotations

import json
import os
import pytest

from nexora.retrieval.dataset import (
    ResearchRetrievalDataset,
    get_pilot_benchmark_dataset,
    load_retrieval_dataset,
    validate_dataset_dict,
)
from nexora.retrieval.eval_cli import run_eval_cli
from nexora.retrieval.production_eval import (
    EvaluationMode,
    ProductionEvaluationConfig,
    evaluate_production_retrieval,
)


JSON_PATH = os.path.join("examples", "real_world_pilot_benchmark.json")
JSONL_PATH = os.path.join("examples", "real_world_pilot_benchmark.jsonl")


def test_pilot_benchmark_helper_function():
    """Verify get_pilot_benchmark_dataset returns valid manually reviewed benchmark."""
    ds = get_pilot_benchmark_dataset()
    assert isinstance(ds, ResearchRetrievalDataset)
    assert ds.dataset_id == "real-world-pilot-benchmark-v1"
    assert ds.version == "1.0.0"
    assert ds.is_illustrative_only is False
    assert ds.provenance == "manually_reviewed_benchmark"
    assert ds.benchmark_tier == "manually_reviewed_benchmark"
    assert ds.is_manually_reviewed is True
    assert len(ds.chunks) == 6
    assert len(ds.queries) == 7

    # Check verified source papers metadata
    expected_papers = {"arxiv_1706.03762", "arxiv_2106.09685", "arxiv_2205.14135"}
    observed_papers = {c["paper_id"] for c in ds.chunks}
    assert observed_papers == expected_papers

    for chunk in ds.chunks:
        assert chunk["doi"] is not None and chunk["doi"].startswith("10.48550/arXiv.")
        assert chunk["url"] is not None and chunk["url"].startswith("https://arxiv.org/abs/")
        assert chunk["title"] is not None and len(chunk["title"]) > 5
        assert isinstance(chunk["publication_year"], int)
        assert chunk["publication_year"] in (2017, 2021, 2022)
        assert len(chunk["text"]) > 50

    # Check every query has verified manual review
    for query in ds.queries:
        assert query["reviewed"] is True
        assert query["reviewer"] == "curator_saiyed_simnan"
        assert len(query["review_notes"]) > 10


def test_pilot_benchmark_files_exist_and_match():
    """Verify examples/ JSON and JSONL files exist and match get_pilot_benchmark_dataset."""
    assert os.path.exists(JSON_PATH)
    assert os.path.exists(JSONL_PATH)

    ds_json = load_retrieval_dataset(JSON_PATH)
    ds_jsonl = load_retrieval_dataset(JSONL_PATH)
    ds_helper = get_pilot_benchmark_dataset()

    assert ds_json.dataset_id == ds_jsonl.dataset_id == ds_helper.dataset_id
    assert ds_json.benchmark_tier == "manually_reviewed_benchmark"
    assert ds_jsonl.benchmark_tier == "manually_reviewed_benchmark"
    assert ds_json.is_manually_reviewed is True
    assert ds_jsonl.is_manually_reviewed is True
    assert len(ds_json.chunks) == len(ds_jsonl.chunks) == 6
    assert len(ds_json.queries) == len(ds_jsonl.queries) == 7


def test_pilot_benchmark_query_categories():
    """Verify queries cover exact_term, semantic, and unanswerable categories."""
    ds = get_pilot_benchmark_dataset()
    categories = {q.get("category") for q in ds.queries}
    assert "exact_term" in categories
    assert "semantic" in categories
    assert "unanswerable_empty" in categories

    exact_queries = [q for q in ds.queries if q.get("category") == "exact_term"]
    semantic_queries = [q for q in ds.queries if q.get("category") == "semantic"]
    unans_queries = [q for q in ds.queries if q.get("category") == "unanswerable_empty"]

    assert len(exact_queries) == 3
    assert len(semantic_queries) == 3
    assert len(unans_queries) == 1
    assert unans_queries[0]["expected_chunk_ids"] == []


def test_evaluate_pilot_benchmark_synthetic_mode():
    """Execute evaluation harness on pilot benchmark and verify metric reporting."""
    ds = get_pilot_benchmark_dataset()
    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.SYNTHETIC,
        dataset=ds,
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert report.dataset_info is not None
    assert report.dataset_info["name"] == "real-world-pilot-benchmark-v1"
    assert report.dataset_info["benchmark_tier"] == "manually_reviewed_benchmark"
    assert report.dataset_info["review_summary"]["is_fully_reviewed"] is True

    # Check metrics existence
    assert report.metrics is not None
    assert report.metrics.total_queries == 7
    assert report.metrics.mean_reciprocal_rank >= 0.0

    # Check per-category metrics breakdown
    cat_metrics = report.metrics.category_metrics
    assert cat_metrics is not None
    assert "exact_term" in cat_metrics
    assert "semantic" in cat_metrics
    assert "unanswerable_empty" in cat_metrics

    # Check baseline vs filtered comparison
    assert report.baseline_metrics is not None
    assert report.quality_comparison is not None
    qc = report.quality_comparison
    assert qc["dataset_provenance"] == "manually_reviewed_benchmark"
    assert "mrr_delta" in qc
    assert "net_effect" in qc


def test_pilot_benchmark_eval_cli(tmp_path, capsys):
    """Run CLI against pilot benchmark JSON and verify summary and report output."""
    out_file = str(tmp_path / "pilot_report.json")
    exit_code, report, _ = run_eval_cli([
        "--mode", "synthetic",
        "--dataset", JSON_PATH,
        "--output", out_file,
    ])
    assert exit_code == 0
    assert report.status == "completed"

    captured = capsys.readouterr().out
    assert "(manually reviewed benchmark)" in captured
    assert "v1.0.0" in captured
    assert "Dataset: real-world-pilot-benchmark-v1" in captured

    # Verify serialized report
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "completed"
    assert data["dataset_info"]["benchmark_tier"] == "manually_reviewed_benchmark"
    assert data["dataset_info"]["review_summary"]["is_fully_reviewed"] is True
    assert data["dataset_info"]["chunks_count"] == 6
    assert data["dataset_info"]["queries_count"] == 7
