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


# ===========================================================================
# 7. Real Embedding Evaluation Pilot Tests (Milestone 12C-37)
# ===========================================================================

def test_pilot_benchmark_production_mode_unavailable_without_credentials(monkeypatch):
    """When credentials are not set, production evaluation of pilot benchmark reports unavailable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_EMBEDDING_API_KEY", raising=False)

    ds = get_pilot_benchmark_dataset()
    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        api_key=None,
        dataset=ds,
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "unavailable"
    assert report.mode == "production"
    assert report.provider_name == "openai"
    assert report.model_name == "text-embedding-3-small"
    assert report.dimension == 1536
    assert report.is_semantic is True
    # Crucial guarantee: never substitute synthetic or hash embeddings when real credentials missing
    assert report.is_synthetic is False
    assert report.is_live is False
    assert report.provider_live is False
    assert "External model credentials are unavailable" in report.infrastructure_limitation
    assert report.unavailable_dependencies == ["openai_credentials"]
    assert report.metrics is None
    assert report.baseline_metrics is None
    assert report.dataset_info["benchmark_tier"] == "manually_reviewed_benchmark"
    assert report.dataset_info["review_summary"]["is_fully_reviewed"] is True


def test_pilot_benchmark_configured_pipeline_hash_provider():
    """Configured pipeline evaluation with hash provider reports honestly without semantic pretense."""
    from nexora.retrieval.quality import RetrievalQualityPolicy

    ds = get_pilot_benchmark_dataset()
    policy = RetrievalQualityPolicy(min_score=0.35, max_results=5)
    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.CONFIGURED_PIPELINE,
        provider_name="hash",
        quality_policy=policy,
        dataset=ds,
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert report.provider_name == "hash"
    assert report.is_semantic is False
    assert report.is_synthetic is False
    assert report.is_live is False
    assert report.provider_live is False
    assert "deterministic text hashing" in report.infrastructure_limitation

    # Under hash retrieval with min_score=0.35, all queries return empty retrievals
    assert report.metrics is not None
    assert report.metrics.total_queries == 7
    assert report.metrics.empty_retrievals == 7
    assert report.metrics.mean_reciprocal_rank == 0.0

    # Quality comparison against baseline is recorded honestly
    assert report.baseline_metrics is not None
    assert report.quality_comparison is not None
    assert report.quality_comparison["dataset_provenance"] == "manually_reviewed_benchmark"
    assert "regressions" in report.quality_comparison


def test_pilot_benchmark_configured_pipeline_openai_without_credentials_fails(monkeypatch):
    """Configured pipeline mode with provider='openai' and no key fails cleanly with informative report."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_EMBEDDING_API_KEY", raising=False)

    ds = get_pilot_benchmark_dataset()
    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.CONFIGURED_PIPELINE,
        provider_name="openai",
        api_key=None,
        dataset=ds,
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "failed"
    assert report.provider_name == "openai"
    assert report.is_live is False
    assert report.provider_live is False
    assert "openai_provider" in report.unavailable_dependencies
    assert report.metrics is None


def test_pilot_benchmark_production_mode_api_failure_handled_cleanly():
    """Simulated API failure during real OpenAI call produces truthful failed report without crash."""
    from unittest.mock import MagicMock
    from nexora.embedding.config import EmbeddingConfig
    from nexora.embedding.openai_provider import OpenAIEmbeddingProvider

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = RuntimeError("Simulated 401 Unauthorized from OpenAI API")

    emb_config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        api_key="sk-test-key",
        dimension=1536,
    )
    provider = OpenAIEmbeddingProvider(emb_config, client=mock_client)

    ds = get_pilot_benchmark_dataset()
    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        embedding_provider=provider,
        dataset=ds,
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "failed"
    assert report.is_live is False
    assert report.provider_live is False
    assert "Embedding or indexing call failed" in report.error
    assert "OpenAI embedding request failed (RuntimeError)" in report.error
    assert report.metrics is None


def test_pilot_benchmark_simulated_double_demarcation():
    """Simulated test doubles exercise the 1536-d pipeline but are demarcated as non-live measurements."""
    try:
        from tests.test_production_retrieval_evaluation import SimulatedProductionOpenAIProvider
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, ".")
        from tests.test_production_retrieval_evaluation import SimulatedProductionOpenAIProvider

    provider = SimulatedProductionOpenAIProvider(dimension=1536)
    ds = get_pilot_benchmark_dataset()
    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        embedding_provider=provider,
        dataset=ds,
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert report.is_synthetic is True
    assert report.is_live is False
    assert report.provider_live is False
    assert "simulated" in report.provider_name.lower()
    assert "Simulated test double" in report.infrastructure_limitation
    assert report.dataset_info["benchmark_tier"] == "manually_reviewed_benchmark"
    assert report.metrics is not None
    assert report.quality_comparison is not None


def test_pilot_benchmark_ground_truth_isolation():
    """Ground-truth expected chunk IDs and review notes are never leaked into retrieval queries."""
    from nexora.retrieval.evaluation import evaluate_retrieval
    from nexora.retrieval.models import RetrievedChunk, SearchResult

    captured_queries: list[str] = []

    def spy_retrieve(query: str, limit: int = 5) -> SearchResult:
        captured_queries.append(query)
        chunk = RetrievedChunk(
            document_id="doc_arxiv_1706.03762",
            chunk_id="chunk_1706_03762_01",
            text="Transformer architecture",
            score=0.9,
            metadata={"paper_id": "arxiv_1706.03762"},
        )
        return SearchResult(query=query, results=[chunk])

    ds = get_pilot_benchmark_dataset()
    metrics = evaluate_retrieval(ds.queries, spy_retrieve)

    assert metrics.total_queries == 7
    assert len(captured_queries) == 7

    for q_text, query_obj in zip(captured_queries, ds.queries):
        assert q_text == query_obj["query"]
        for exp_id in query_obj["expected_chunk_ids"]:
            assert exp_id not in q_text
        if query_obj.get("reviewer"):
            assert query_obj["reviewer"] not in q_text
        if query_obj.get("review_notes"):
            assert query_obj["review_notes"] not in q_text


def test_pilot_benchmark_cli_production_unavailable_and_fail_on_unavailable(tmp_path, capsys, monkeypatch):
    """CLI evaluates pilot benchmark in production mode, reporting unavailable and respecting fail_on_unavailable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_EMBEDDING_API_KEY", raising=False)

    out_file = str(tmp_path / "prod_unavail.json")

    # 1. Without --fail-on-unavailable: exit code 0, status unavailable
    exit_code, report, _ = run_eval_cli([
        "--mode", "production",
        "--dataset", JSON_PATH,
        "--output", out_file,
    ])
    assert exit_code == 0
    assert report.status == "unavailable"
    assert report.provider_live is False
    assert report.metrics is None

    captured = capsys.readouterr().out
    assert "Status: unavailable" in captured
    assert "Limitation: External model credentials are unavailable" in captured

    # 2. With --fail-on-unavailable: exit code 1
    exit_code_fail, report_fail, _ = run_eval_cli([
        "--mode", "production",
        "--dataset", JSON_PATH,
        "--fail-on-unavailable",
    ])
    assert exit_code_fail == 1
    assert report_fail.status == "unavailable"


def test_pilot_benchmark_cli_configured_pipeline_with_provider_flag(tmp_path, capsys):
    """CLI configured_pipeline mode with --provider hash evaluates pilot benchmark."""
    out_file = str(tmp_path / "pipeline_report.json")
    exit_code, report, _ = run_eval_cli([
        "--mode", "configured_pipeline",
        "--provider", "hash",
        "--dataset", JSON_PATH,
        "--output", out_file,
    ])
    assert exit_code == 0
    assert report.status == "completed"
    assert report.provider_name == "hash"
    assert report.is_semantic is False

    captured = capsys.readouterr().out
    assert "Provider: hash (live=False)" in captured
    assert "(manually reviewed benchmark)" in captured
    assert "Limitation:" in captured
    assert "deterministic text hashing" in captured

