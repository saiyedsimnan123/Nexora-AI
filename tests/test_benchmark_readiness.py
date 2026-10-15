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


# ===========================================================================
# 7. Strict Benchmark Review Integrity Fix Tests (Milestone 12C-35.1)
# ===========================================================================

def test_partial_review_rejected_for_reviewed_benchmark(base_valid_dataset_dict):
    """A dataset with partial query reviews cannot claim manually_reviewed_benchmark."""
    # q1 is reviewed by alice, q2 is unreviewed
    base_valid_dataset_dict["queries"][0]["reviewed"] = True
    base_valid_dataset_dict["queries"][0]["reviewer"] = "curator_alice"
    base_valid_dataset_dict["queries"][1]["reviewed"] = False
    base_valid_dataset_dict["queries"][1]["reviewer"] = None
    base_valid_dataset_dict["provenance"] = "manually_reviewed_benchmark"

    with pytest.raises(DatasetValidationError, match="every query's relevance labels"):
        validate_dataset_dict(base_valid_dataset_dict)

    # In candidate mode, partial reviews validate but cannot promote to manually_reviewed_benchmark tier
    base_valid_dataset_dict["provenance"] = "candidate_real_world"
    ds = validate_dataset_dict(base_valid_dataset_dict)
    assert ds.benchmark_tier == "candidate_real_world"
    assert ds.is_manually_reviewed is False
    assert ds.reviewed_queries_count == 1
    assert ds.unreviewed_queries_count == 1


def test_missing_reviewer_identity_rejected(base_valid_dataset_dict):
    """A query marked reviewed=True without a valid, non-empty reviewer fails validation."""
    # None reviewer
    base_valid_dataset_dict["queries"][0]["reviewed"] = True
    base_valid_dataset_dict["queries"][0]["reviewer"] = None
    with pytest.raises(DatasetValidationError, match="missing a valid, non-empty 'reviewer'"):
        validate_dataset_dict(base_valid_dataset_dict)

    # Empty string reviewer
    base_valid_dataset_dict["queries"][0]["reviewer"] = ""
    with pytest.raises(DatasetValidationError, match="missing a valid, non-empty 'reviewer'"):
        validate_dataset_dict(base_valid_dataset_dict)

    # Whitespace-only string reviewer
    base_valid_dataset_dict["queries"][0]["reviewer"] = "   "
    with pytest.raises(DatasetValidationError, match="missing a valid, non-empty 'reviewer'"):
        validate_dataset_dict(base_valid_dataset_dict)

    # In nested review object
    base_valid_dataset_dict["queries"][0] = {
        "query_id": "q1",
        "query": "Sample query",
        "expected_chunk_ids": ["c1"],
        "review": {
            "reviewed": True,
            "reviewer": "   ",
        },
    }
    with pytest.raises(DatasetValidationError, match="missing a valid, non-empty 'reviewer'"):
        validate_dataset_dict(base_valid_dataset_dict)


def test_all_queries_reviewed_including_unanswerable_succeeds(base_valid_dataset_dict):
    """When all queries (including unanswerable) have confirmed reviews with reviewers, classification succeeds."""
    # Add an unanswerable query
    base_valid_dataset_dict["queries"].append({
        "query_id": "q3_unanswerable",
        "query": "Query with no relevant evidence in the corpus",
        "expected_chunk_ids": [],
        "category": "unanswerable",
        "reviewed": True,
        "reviewer": "curator_charlie",
        "review_notes": "Confirmed corpus contains no relevant evidence.",
    })
    # Mark q2 as reviewed too
    base_valid_dataset_dict["queries"][1]["reviewed"] = True
    base_valid_dataset_dict["queries"][1]["reviewer"] = "curator_bob"
    base_valid_dataset_dict["provenance"] = "manually_reviewed_benchmark"

    ds = validate_dataset_dict(base_valid_dataset_dict)
    assert ds.provenance == "manually_reviewed_benchmark"
    assert ds.benchmark_tier == "manually_reviewed_benchmark"
    assert ds.is_manually_reviewed is True
    assert ds.reviewed_queries_count == 3
    assert ds.unreviewed_queries_count == 0

    summary = ds.review_summary
    assert summary["total_queries"] == 3
    assert summary["reviewed_queries"] == 3
    assert summary["unreviewed_queries"] == 0
    assert summary["is_fully_reviewed"] is True
    assert summary["benchmark_tier"] == "manually_reviewed_benchmark"


def test_unanswerable_query_unreviewed_blocks_benchmark_tier(base_valid_dataset_dict):
    """Unanswerable query left unreviewed strictly blocks manually_reviewed_benchmark status."""
    # Answerable queries are reviewed
    base_valid_dataset_dict["queries"][0]["reviewed"] = True
    base_valid_dataset_dict["queries"][0]["reviewer"] = "curator_alice"
    base_valid_dataset_dict["queries"][1]["reviewed"] = True
    base_valid_dataset_dict["queries"][1]["reviewer"] = "curator_bob"

    # Add unreviewed unanswerable query
    base_valid_dataset_dict["queries"].append({
        "query_id": "q3_unanswerable",
        "query": "Query with no evidence",
        "expected_chunk_ids": [],
        "reviewed": False,
        "reviewer": None,
    })

    base_valid_dataset_dict["provenance"] = "manually_reviewed_benchmark"
    with pytest.raises(DatasetValidationError, match="q3_unanswerable"):
        validate_dataset_dict(base_valid_dataset_dict)

    # In candidate mode, remains candidate_real_world
    base_valid_dataset_dict["provenance"] = "candidate_real_world"
    ds = validate_dataset_dict(base_valid_dataset_dict)
    assert ds.benchmark_tier == "candidate_real_world"
    assert ds.is_manually_reviewed is False
    assert ds.reviewed_queries_count == 2
    assert ds.unreviewed_queries_count == 1


def test_json_and_jsonl_parity_for_strict_review_enforcement(tmp_path):
    """Verify JSON and JSONL both enforce strict review validation identically."""
    # 1. Dataset dict with partial review
    partial_data = {
        "dataset_id": "test_parity_partial",
        "provenance": "manually_reviewed_benchmark",
        "is_illustrative_only": False,
        "chunks": [
            {"chunk_id": "c1", "paper_id": "p1", "text": "Some text content for testing"}
        ],
        "queries": [
            {
                "query_id": "q1",
                "query": "Query one",
                "expected_chunk_ids": ["c1"],
                "reviewed": True,
                "reviewer": "auditor_1",
            },
            {
                "query_id": "q2",
                "query": "Query two unreviewed",
                "expected_chunk_ids": [],
                "reviewed": False,
            },
        ],
    }

    # Write JSON
    json_file = str(tmp_path / "partial.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(partial_data, f)

    # Write JSONL
    jsonl_file = str(tmp_path / "partial.jsonl")
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metadata", "dataset_id": "test_parity_partial", "provenance": "manually_reviewed_benchmark", "is_illustrative_only": False}) + "\n")
        f.write(json.dumps({"type": "chunk", "chunk_id": "c1", "paper_id": "p1", "text": "Some text content for testing"}) + "\n")
        f.write(json.dumps({"type": "query", "query_id": "q1", "query": "Query one", "expected_chunk_ids": ["c1"], "reviewed": True, "reviewer": "auditor_1"}) + "\n")
        f.write(json.dumps({"type": "query", "query_id": "q2", "query": "Query two unreviewed", "expected_chunk_ids": [], "reviewed": False}) + "\n")

    # Both must fail closed on loading
    with pytest.raises(DatasetValidationError, match="every query's relevance labels"):
        load_retrieval_dataset(json_file)

    with pytest.raises(DatasetValidationError, match="every query's relevance labels"):
        load_retrieval_dataset(jsonl_file)

    # 2. Both succeed when all queries reviewed
    full_data = dict(partial_data)
    full_data["queries"] = [
        {
            "query_id": "q1",
            "query": "Query one",
            "expected_chunk_ids": ["c1"],
            "reviewed": True,
            "reviewer": "auditor_1",
        },
        {
            "query_id": "q2",
            "query": "Query two reviewed",
            "expected_chunk_ids": [],
            "reviewed": True,
            "reviewer": "auditor_2",
        },
    ]

    full_json = str(tmp_path / "full.json")
    with open(full_json, "w", encoding="utf-8") as f:
        json.dump(full_data, f)

    full_jsonl = str(tmp_path / "full.jsonl")
    with open(full_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metadata", "dataset_id": "test_parity_full", "provenance": "manually_reviewed_benchmark", "is_illustrative_only": False}) + "\n")
        f.write(json.dumps({"type": "chunk", "chunk_id": "c1", "paper_id": "p1", "text": "Some text content for testing"}) + "\n")
        f.write(json.dumps({"type": "query", "query_id": "q1", "query": "Query one", "expected_chunk_ids": ["c1"], "reviewed": True, "reviewer": "auditor_1"}) + "\n")
        f.write(json.dumps({"type": "query", "query_id": "q2", "query": "Query two reviewed", "expected_chunk_ids": [], "reviewed": True, "reviewer": "auditor_2"}) + "\n")

    ds_j = load_retrieval_dataset(full_json)
    ds_jl = load_retrieval_dataset(full_jsonl)

    assert ds_j.benchmark_tier == "manually_reviewed_benchmark"
    assert ds_jl.benchmark_tier == "manually_reviewed_benchmark"
    assert ds_j.is_manually_reviewed is True
    assert ds_jl.is_manually_reviewed is True


def test_serialized_report_and_cli_summary_never_overstate_review_status(tmp_path, capsys):
    """CLI and serialized reports never describe candidate/partially reviewed datasets as manually reviewed."""
    # Partially reviewed dataset in candidate tier
    dataset_dict = {
        "dataset_id": "dataset_partial_review",
        "description": "Candidate dataset with 1 of 2 queries reviewed",
        "provenance": "candidate_real_world",
        "is_illustrative_only": False,
        "chunks": [
            {"chunk_id": "c1", "paper_id": "p1", "text": "Chunk text 1"},
            {"chunk_id": "c2", "paper_id": "p2", "text": "Chunk text 2"},
        ],
        "queries": [
            {
                "query_id": "q1",
                "query": "Query 1",
                "expected_chunk_ids": ["c1"],
                "reviewed": True,
                "reviewer": "auditor_1",
            },
            {
                "query_id": "q2",
                "query": "Query 2",
                "expected_chunk_ids": ["c2"],
                "reviewed": False,
            },
        ],
    }

    ds_path = str(tmp_path / "candidate_partial.json")
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f)

    out_report = str(tmp_path / "report_partial.json")
    exit_code, report, _ = run_eval_cli([
        "--mode", "synthetic",
        "--dataset", ds_path,
        "--output", out_report,
    ])
    assert exit_code == 0
    captured = capsys.readouterr().out

    # Must NOT overstate as manually reviewed benchmark
    assert "(manually reviewed benchmark)" not in captured
    assert "(candidate real-world)" in captured

    # Inspect serialized report
    with open(out_report, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    d_info = report_data["dataset_info"]
    assert d_info["benchmark_tier"] == "candidate_real_world"
    assert d_info["review_summary"]["reviewed_queries"] == 1
    assert d_info["review_summary"]["unreviewed_queries"] == 1
    assert d_info["review_summary"]["is_fully_reviewed"] is False

    # Now run on fully reviewed dataset
    dataset_dict["provenance"] = "manually_reviewed_benchmark"
    dataset_dict["queries"][1]["reviewed"] = True
    dataset_dict["queries"][1]["reviewer"] = "auditor_2"
    full_ds_path = str(tmp_path / "fully_reviewed.json")
    with open(full_ds_path, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f)

    out_full_report = str(tmp_path / "report_full.json")
    exit_code, report_full, _ = run_eval_cli([
        "--mode", "synthetic",
        "--dataset", full_ds_path,
        "--output", out_full_report,
    ])
    assert exit_code == 0
    captured_full = capsys.readouterr().out

    assert "(manually reviewed benchmark)" in captured_full
    with open(out_full_report, "r", encoding="utf-8") as f:
        full_report_data = json.load(f)

    d_full_info = full_report_data["dataset_info"]
    assert d_full_info["benchmark_tier"] == "manually_reviewed_benchmark"
    assert d_full_info["review_summary"]["is_fully_reviewed"] is True
    assert d_full_info["review_summary"]["reviewed_queries"] == 2
    assert d_full_info["review_summary"]["unreviewed_queries"] == 0

