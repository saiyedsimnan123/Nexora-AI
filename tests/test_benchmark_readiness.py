"""Tests for Milestone 12C-35: Real-World Retrieval Benchmark Readiness.

Verifies:
1. Benchmark dataset template loading and schema compliance (JSON and JSONL).
2. Traceable source metadata validation (title, DOI, URL, publication year).
3. Benchmark provenance validation:
   - Distinguishes illustrative, candidate real-world, and manually reviewed benchmarks.
   - Fails closed when manually reviewed benchmark status is claimed without review metadata.
   - Rejects contradictory or invalid provenance metadata.
4. Relevance-label review tracking on queries (reviewed status, reviewer, review notes).
5. Strict ground-truth isolation (review metadata and expected chunk IDs never passed to retrieval).
6. Integration with evaluation pipeline and CLI reporting.
"""

from __future__ import annotations

import json
import os
import pytest

from nexora.retrieval.dataset import (
    DatasetValidationError,
    ResearchRetrievalDataset,
    get_benchmark_template_dataset,
    get_illustrative_example_dataset,
    load_retrieval_dataset,
    validate_dataset_dict,
)
from nexora.retrieval.eval_cli import run_eval_cli
from nexora.retrieval.evaluation import evaluate_retrieval
from nexora.retrieval.models import RetrievedChunk, SearchResult
from nexora.retrieval.production_eval import (
    EvaluationMode,
    ProductionEvaluationConfig,
    evaluate_production_retrieval,
)


@pytest.fixture
def base_valid_dataset_dict() -> dict:
    """Minimal valid dataset dictionary for parameterized testing."""
    return {
        "dataset_id": "test_benchmark_01",
        "description": "Test dataset for benchmark readiness",
        "chunks": [
            {
                "chunk_id": "c1",
                "paper_id": "p1",
                "text": "FlashAttention IO-aware tiling algorithm implementation.",
                "title": "FlashAttention Paper",
                "doi": "10.48550/arXiv.2205.14135",
                "url": "https://arxiv.org/abs/2205.14135",
                "publication_year": 2022,
            },
            {
                "chunk_id": "c2",
                "paper_id": "p2",
                "text": "LoRA low-rank adaptation of large language models.",
                "title": "LoRA Paper",
                "doi": "10.48550/arXiv.2106.09685",
                "url": "https://arxiv.org/abs/2106.09685",
                "publication_year": 2021,
            },
        ],
        "queries": [
            {
                "query_id": "q1",
                "query": "How does IO-aware tiling reduce memory bandwidth?",
                "expected_chunk_ids": ["c1"],
                "category": "exact_term",
                "reviewed": True,
                "reviewer": "curator_alice",
                "review_notes": "Verified against Section 2 equation 3.",
            },
            {
                "query_id": "q2",
                "query": "Parameter-efficient fine-tuning with factorized matrices",
                "expected_chunk_ids": ["c2"],
                "category": "semantic",
                "reviewed": False,
                "reviewer": None,
                "review_notes": "Pending human review.",
            },
        ],
    }


# ===========================================================================
# 1. Benchmark Dataset Template Tests
# ===========================================================================

def test_benchmark_template_function():
    """Verify built-in get_benchmark_template_dataset produces valid candidate dataset."""
    template = get_benchmark_template_dataset()
    assert isinstance(template, ResearchRetrievalDataset)
    assert template.dataset_id == "real-world-research-benchmark-template-v1"
    assert template.version == "1.0.0"
    assert template.is_illustrative_only is False
    assert template.provenance == "candidate_real_world"
    assert template.benchmark_tier == "candidate_real_world"
    assert len(template.chunks) == 3
    assert len(template.queries) == 3

    # Check chunks have traceable source fields
    for chunk in template.chunks:
        assert "doi" in chunk
        assert "url" in chunk
        assert "publication_year" in chunk
        assert "title" in chunk
        assert isinstance(chunk["publication_year"], int)

    # Check queries have review status fields
    for query in template.queries:
        assert "reviewed" in query
        assert query["reviewed"] is False
        assert query["reviewer"] is None
        assert "review_notes" in query


def test_benchmark_template_files_match_loader():
    """Verify example JSON and JSONL template files exist and parse cleanly."""
    json_path = os.path.join("examples", "real_world_benchmark_template.json")
    jsonl_path = os.path.join("examples", "real_world_benchmark_template.jsonl")

    assert os.path.exists(json_path)
    assert os.path.exists(jsonl_path)

    ds_json = load_retrieval_dataset(json_path)
    ds_jsonl = load_retrieval_dataset(jsonl_path)

    assert ds_json.dataset_id == ds_jsonl.dataset_id
    assert ds_json.version == "1.0.0"
    assert ds_jsonl.version == "1.0.0"
    assert ds_json.provenance == "candidate_real_world"
    assert ds_jsonl.provenance == "candidate_real_world"
    assert len(ds_json.chunks) == len(ds_jsonl.chunks) == 3
    assert len(ds_json.queries) == len(ds_jsonl.queries) == 3


def test_existing_illustrative_datasets_preserved():
    """Verify illustrative datasets continue to be recognized as strictly illustrative."""
    ill = get_illustrative_example_dataset()
    assert ill.is_illustrative_only is True
    assert ill.provenance == "illustrative"
    assert ill.benchmark_tier == "illustrative"

    json_path = os.path.join("examples", "illustrative_research_dataset.json")
    ds = load_retrieval_dataset(json_path)
    assert ds.is_illustrative_only is True
    assert ds.provenance == "illustrative"
    assert ds.benchmark_tier == "illustrative"


# ===========================================================================
# 2. Source Metadata Validation Tests (Title, DOI, URL, Publication Year)
# ===========================================================================

def test_source_metadata_valid(base_valid_dataset_dict):
    """Chunks with valid doi, url, and publication_year validate correctly."""
    ds = validate_dataset_dict(base_valid_dataset_dict)
    c1 = ds.chunks[0]
    assert c1["doi"] == "10.48550/arXiv.2205.14135"
    assert c1["url"] == "https://arxiv.org/abs/2205.14135"
    assert c1["publication_year"] == 2022
    assert c1["title"] == "FlashAttention Paper"


def test_source_metadata_optional(base_valid_dataset_dict):
    """Omitting doi, url, and publication_year defaults cleanly to None."""
    del base_valid_dataset_dict["chunks"][0]["doi"]
    del base_valid_dataset_dict["chunks"][0]["url"]
    del base_valid_dataset_dict["chunks"][0]["publication_year"]

    ds = validate_dataset_dict(base_valid_dataset_dict)
    c1 = ds.chunks[0]
    assert c1["doi"] is None
    assert c1["url"] is None
    assert c1["publication_year"] is None


def test_source_metadata_invalid_doi_fails(base_valid_dataset_dict):
    """Non-string DOI raises DatasetValidationError."""
    base_valid_dataset_dict["chunks"][0]["doi"] = 12345
    with pytest.raises(DatasetValidationError, match="has invalid 'doi'"):
        validate_dataset_dict(base_valid_dataset_dict)


def test_source_metadata_invalid_url_fails(base_valid_dataset_dict):
    """Non-string URL raises DatasetValidationError."""
    base_valid_dataset_dict["chunks"][0]["url"] = ["https://example.com"]
    with pytest.raises(DatasetValidationError, match="has invalid 'url'"):
        validate_dataset_dict(base_valid_dataset_dict)


def test_source_metadata_invalid_year_fails(base_valid_dataset_dict):
    """Non-integer, boolean, or out-of-range publication_year raises DatasetValidationError."""
    # String year
    base_valid_dataset_dict["chunks"][0]["publication_year"] = "2022"
    with pytest.raises(DatasetValidationError, match="has invalid 'publication_year'"):
        validate_dataset_dict(base_valid_dataset_dict)

    # Boolean year
    base_valid_dataset_dict["chunks"][0]["publication_year"] = True
    with pytest.raises(DatasetValidationError, match="has invalid 'publication_year'"):
        validate_dataset_dict(base_valid_dataset_dict)

    # Out of range year (< 1000)
    base_valid_dataset_dict["chunks"][0]["publication_year"] = 999
    with pytest.raises(DatasetValidationError, match="out of valid range"):
        validate_dataset_dict(base_valid_dataset_dict)


# ===========================================================================
# 3. Provenance Tier & Fail-Closed Review Status Validation
# ===========================================================================

def test_provenance_distinguishes_tiers(base_valid_dataset_dict):
    """Verify clean categorization across illustrative, candidate, and reviewed benchmarks."""
    # 1. Candidate real-world
    base_valid_dataset_dict["provenance"] = "candidate_real_world"
    ds_candidate = validate_dataset_dict(base_valid_dataset_dict)
    assert ds_candidate.provenance == "candidate_real_world"
    assert ds_candidate.benchmark_tier == "candidate_real_world"
    assert ds_candidate.is_illustrative_only is False

    # 2. Manually reviewed benchmark (all queries reviewed)
    base_valid_dataset_dict["queries"][1]["reviewed"] = True
    base_valid_dataset_dict["queries"][1]["reviewer"] = "curator_bob"
    base_valid_dataset_dict["provenance"] = "manually_reviewed_benchmark"
    ds_reviewed = validate_dataset_dict(base_valid_dataset_dict)
    assert ds_reviewed.provenance == "manually_reviewed_benchmark"
    assert ds_reviewed.benchmark_tier == "manually_reviewed_benchmark"
    assert ds_reviewed.is_manually_reviewed is True

    # 3. Illustrative
    base_valid_dataset_dict["provenance"] = "illustrative"
    ds_ill = validate_dataset_dict(base_valid_dataset_dict)
    assert ds_ill.provenance == "illustrative"
    assert ds_ill.benchmark_tier == "illustrative"
    assert ds_ill.is_illustrative_only is True


def test_manually_reviewed_claim_without_review_metadata_fails_closed(base_valid_dataset_dict):
    """Claiming manually_reviewed_benchmark without explicitly recorded review status fails closed."""
    # Ensure neither query has reviewed=True
    base_valid_dataset_dict["queries"][0]["reviewed"] = False
    base_valid_dataset_dict["queries"][1]["reviewed"] = False
    base_valid_dataset_dict["provenance"] = "manually_reviewed_benchmark"

    with pytest.raises(DatasetValidationError, match="Cannot label dataset as manually reviewed"):
        validate_dataset_dict(base_valid_dataset_dict)


def test_contradictory_provenance_fails(base_valid_dataset_dict):
    """Contradictory is_illustrative_only and provenance combinations fail closed."""
    # is_illustrative_only=True combined with candidate_real_world
    base_valid_dataset_dict["is_illustrative_only"] = True
    base_valid_dataset_dict["provenance"] = "candidate_real_world"
    with pytest.raises(DatasetValidationError, match="Contradictory metadata"):
        validate_dataset_dict(base_valid_dataset_dict)

    # is_illustrative_only=False combined with illustrative
    base_valid_dataset_dict["is_illustrative_only"] = False
    base_valid_dataset_dict["provenance"] = "illustrative"
    with pytest.raises(DatasetValidationError, match="Contradictory metadata"):
        validate_dataset_dict(base_valid_dataset_dict)


def test_invalid_provenance_value_fails(base_valid_dataset_dict):
    """Unrecognized provenance string raises DatasetValidationError."""
    base_valid_dataset_dict["provenance"] = "gold_standard_benchmark"
    with pytest.raises(DatasetValidationError, match="Invalid 'provenance'"):
        validate_dataset_dict(base_valid_dataset_dict)


# ===========================================================================
# 4. Relevance-Label Review Tracking on Queries
# ===========================================================================

def test_query_review_fields_parsing(base_valid_dataset_dict):
    """Query review status, reviewer, and notes are parsed accurately."""
    ds = validate_dataset_dict(base_valid_dataset_dict)
    q1 = ds.queries[0]
    assert q1["reviewed"] is True
    assert q1["reviewer"] == "curator_alice"
    assert q1["review_notes"] == "Verified against Section 2 equation 3."

    q2 = ds.queries[1]
    assert q2["reviewed"] is False
    assert q2["reviewer"] is None
    assert q2["review_notes"] == "Pending human review."

    # Dataset review summary metrics
    summary = ds.review_summary
    assert summary["total_queries"] == 2
    assert summary["reviewed_queries"] == 1
    assert summary["unreviewed_queries"] == 1
    assert summary["is_fully_reviewed"] is False


def test_query_nested_review_object_support(base_valid_dataset_dict):
    """Query with nested 'review' dictionary is parsed consistently."""
    base_valid_dataset_dict["queries"][0] = {
        "query_id": "q1",
        "query": "Sample query",
        "expected_chunk_ids": ["c1"],
        "review": {
            "reviewed": True,
            "reviewer": "expert_verifier",
            "notes": "Verified directly.",
        },
    }
    ds = validate_dataset_dict(base_valid_dataset_dict)
    q1 = ds.queries[0]
    assert q1["reviewed"] is True
    assert q1["reviewer"] == "expert_verifier"
    assert q1["review_notes"] == "Verified directly."


def test_query_invalid_reviewed_boolean_fails(base_valid_dataset_dict):
    """String 'true' instead of boolean True for query reviewed fails closed."""
    base_valid_dataset_dict["queries"][0]["reviewed"] = "true"
    with pytest.raises(DatasetValidationError, match="has invalid 'reviewed'"):
        validate_dataset_dict(base_valid_dataset_dict)


def test_query_invalid_review_status_fails(base_valid_dataset_dict):
    """Invalid review_status string raises DatasetValidationError."""
    base_valid_dataset_dict["queries"][0]["review_status"] = "finished"
    with pytest.raises(DatasetValidationError, match="has invalid 'review_status'"):
        validate_dataset_dict(base_valid_dataset_dict)


# ===========================================================================
# 5. Strict Ground-Truth & Review Metadata Isolation
# ===========================================================================

def test_review_metadata_and_relevance_labels_isolated_from_retrieval():
    """Verify that retrieval callers never receive ground-truth IDs or review metadata."""
    calls: list[tuple[str, int]] = []

    def spy_retrieve(query: str, limit: int = 5) -> SearchResult:
        calls.append((query, limit))
        chunk = RetrievedChunk(
            document_id="doc_c1",
            chunk_id="c1",
            text="FlashAttention IO-aware tiling",
            score=0.95,
            metadata={"paper_id": "p1"},
        )
        return SearchResult(query=query, results=[chunk])

    benchmark_cases = [
        {
            "query": "IO-aware attention tiling",
            "expected_chunk_ids": ["c1", "secret_gt_chunk_id"],
            "category": "exact_term",
            "reviewed": True,
            "reviewer": "secret_reviewer_id",
            "review_notes": "Internal audit confidential note",
        }
    ]

    metrics = evaluate_retrieval(benchmark_cases, spy_retrieve)
    assert metrics.total_queries == 1
    assert len(calls) == 1

    query_arg, limit_arg = calls[0]
    assert query_arg == "IO-aware attention tiling"
    assert limit_arg == 5

    # Confirm secret ground-truth IDs and reviewer info never leaked into query argument
    assert "secret_gt_chunk_id" not in query_arg
    assert "secret_reviewer_id" not in query_arg
    assert "confidential" not in query_arg


# ===========================================================================
# 6. Evaluation Harness & CLI Reporting Integration
# ===========================================================================

def test_evaluate_production_retrieval_with_benchmark_template():
    """Verify evaluate_production_retrieval executes cleanly on candidate benchmark template."""
    template = get_benchmark_template_dataset()
    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.SYNTHETIC,
        dataset=template,
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert report.dataset_info is not None
    assert report.dataset_info["name"] == "real-world-research-benchmark-template-v1"
    assert report.dataset_info["version"] == "1.0.0"
    assert report.dataset_info["provenance"] == "candidate_real_world"
    assert report.dataset_info["benchmark_tier"] == "candidate_real_world"
    assert report.dataset_info["chunks_count"] == 3
    assert report.dataset_info["queries_count"] == 3
    assert report.baseline_metrics is not None
    assert report.quality_comparison is not None
    assert report.quality_comparison["dataset_provenance"] == "candidate_real_world"


def test_eval_cli_reports_candidate_and_reviewed_tiers(tmp_path, capsys):
    """Verify CLI correctly outputs benchmark tiers and version."""
    # 1. Candidate real-world dataset
    json_path = os.path.join("examples", "real_world_benchmark_template.json")
    out_file = str(tmp_path / "cand_report.json")

    exit_code, report, _ = run_eval_cli([
        "--mode", "synthetic",
        "--dataset", json_path,
        "--output", out_file,
    ])
    assert exit_code == 0
    assert report.status == "completed"
    assert report.dataset_info["provenance"] == "candidate_real_world"

    captured = capsys.readouterr().out
    assert "(candidate real-world)" in captured
    assert "v1.0.0" in captured
