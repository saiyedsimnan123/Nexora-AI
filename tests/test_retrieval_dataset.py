"""Tests for dataset-driven retrieval evaluation (Milestone 12C-33).

Covers:
- JSON and JSONL dataset parsing and schema validation
- Fail-closed structural and referential integrity (missing fields, duplicate IDs, nonexistent chunk references)
- Handling of answerable, unanswerable, and multi-chunk queries
- Preservation of ground-truth relevance isolation
- Illustrative example datasets verification
- Integration with evaluate_production_retrieval()
- Reproducible CLI execution with --dataset / -d
"""

import json
import os
import tempfile
import pytest

from nexora.retrieval.dataset import (
    DatasetValidationError,
    ResearchRetrievalDataset,
    get_illustrative_example_dataset,
    load_retrieval_dataset,
    load_retrieval_dataset_from_jsonl,
    validate_dataset_dict,
)
from nexora.retrieval.eval_cli import run_eval_cli
from nexora.retrieval.production_eval import (
    EvaluationMode,
    ProductionEvaluationConfig,
    evaluate_production_retrieval,
)


@pytest.fixture
def sample_valid_dataset_dict():
    return {
        "dataset_id": "test_ds_01",
        "description": "Test research retrieval dataset",
        "is_illustrative_only": True,
        "chunks": [
            {
                "chunk_id": "c1",
                "paper_id": "p1",
                "text": "First chunk discussing transformer attention heads.",
            },
            {
                "chunk_id": "c2",
                "paper_id": "p1",
                "text": "Second chunk discussing transformer feedforward layers.",
            },
            {
                "chunk_id": "c3",
                "paper_id": "p2",
                "text": "Third chunk discussing low-rank adaptation matrices.",
            },
        ],
        "queries": [
            {
                "query_id": "q1",
                "query": "transformer attention mechanisms",
                "expected_chunk_ids": ["c1"],
                "category": "exact_term",
                "target_paper_ids": ["p1"],
            },
            {
                "query_id": "q2",
                "query": "low rank adaptation adapters",
                "expected_chunk_ids": ["c3"],
                "category": "semantic",
                "target_paper_ids": ["p2"],
            },
            {
                "query_id": "q3",
                "query": "quantum topological braiding",
                "expected_chunk_ids": [],
                "category": "unanswerable_empty",
            },
        ],
    }


# ===========================================================================
# 1. Schema & Structural Validation Tests
# ===========================================================================

def test_valid_dataset_dict(sample_valid_dataset_dict):
    ds = validate_dataset_dict(sample_valid_dataset_dict)
    assert isinstance(ds, ResearchRetrievalDataset)
    assert ds.dataset_id == "test_ds_01"
    assert ds.name == "test_ds_01"
    assert len(ds.chunks) == 3
    assert len(ds.queries) == 3
    assert ds.is_illustrative_only is True


def test_missing_chunks_fails():
    with pytest.raises(DatasetValidationError, match="missing required key 'chunks'"):
        validate_dataset_dict({"queries": [{"query": "test", "expected_chunk_ids": []}]})


def test_empty_chunks_fails():
    with pytest.raises(DatasetValidationError, match="'chunks' list must not be empty"):
        validate_dataset_dict({"chunks": [], "queries": [{"query": "test", "expected_chunk_ids": []}]})


def test_missing_queries_fails():
    with pytest.raises(DatasetValidationError, match="missing required key 'queries'"):
        validate_dataset_dict({
            "chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "text"}]
        })


def test_empty_queries_fails():
    with pytest.raises(DatasetValidationError, match="'queries' list must not be empty"):
        validate_dataset_dict({
            "chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "text"}],
            "queries": [],
        })


def test_chunk_missing_required_fields():
    # Missing chunk_id
    with pytest.raises(DatasetValidationError, match="missing required field 'chunk_id'"):
        validate_dataset_dict({
            "chunks": [{"paper_id": "p1", "text": "sample text"}],
            "queries": [{"query": "q", "expected_chunk_ids": []}],
        })

    # Missing paper_id
    with pytest.raises(DatasetValidationError, match="missing required field 'paper_id'"):
        validate_dataset_dict({
            "chunks": [{"chunk_id": "c1", "text": "sample text"}],
            "queries": [{"query": "q", "expected_chunk_ids": []}],
        })

    # Missing text
    with pytest.raises(DatasetValidationError, match="missing required field 'text'"):
        validate_dataset_dict({
            "chunks": [{"chunk_id": "c1", "paper_id": "p1"}],
            "queries": [{"query": "q", "expected_chunk_ids": []}],
        })

    # Empty text
    with pytest.raises(DatasetValidationError, match="invalid 'text': must be a non-empty string"):
        validate_dataset_dict({
            "chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "   "}],
            "queries": [{"query": "q", "expected_chunk_ids": []}],
        })


def test_duplicate_chunk_id_fails():
    with pytest.raises(DatasetValidationError, match="Duplicate chunk_id 'c1'"):
        validate_dataset_dict({
            "chunks": [
                {"chunk_id": "c1", "paper_id": "p1", "text": "text 1"},
                {"chunk_id": "c1", "paper_id": "p1", "text": "text 2"},
            ],
            "queries": [{"query": "q", "expected_chunk_ids": []}],
        })


def test_duplicate_query_id_fails():
    with pytest.raises(DatasetValidationError, match="Duplicate query_id 'q1'"):
        validate_dataset_dict({
            "chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "text"}],
            "queries": [
                {"query_id": "q1", "query": "q1 text", "expected_chunk_ids": ["c1"]},
                {"query_id": "q1", "query": "q2 text", "expected_chunk_ids": ["c1"]},
            ],
        })


def test_query_duplicate_expected_chunk_ids_fails():
    with pytest.raises(DatasetValidationError, match="duplicate expected_chunk_id 'c1'"):
        validate_dataset_dict({
            "chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "text"}],
            "queries": [
                {"query_id": "q1", "query": "q1 text", "expected_chunk_ids": ["c1", "c1"]},
            ],
        })


# ===========================================================================
# 2. Referential Integrity Tests
# ===========================================================================

def test_query_references_nonexistent_chunk():
    with pytest.raises(DatasetValidationError, match="references nonexistent chunk_id 'c999'"):
        validate_dataset_dict({
            "chunks": [{"chunk_id": "c1", "paper_id": "p1", "text": "text"}],
            "queries": [
                {"query": "find stuff", "expected_chunk_ids": ["c999"]},
            ],
        })


def test_unanswerable_query_with_empty_expected_chunk_ids(sample_valid_dataset_dict):
    ds = validate_dataset_dict(sample_valid_dataset_dict)
    unanswerable = [q for q in ds.queries if len(q["expected_chunk_ids"]) == 0]
    assert len(unanswerable) == 1
    assert unanswerable[0]["query_id"] == "q3"
    assert unanswerable[0]["category"] == "unanswerable_empty"


# ===========================================================================
# 3. Multi-format Loaders (JSON and JSONL)
# ===========================================================================

def test_load_json_dataset(tmp_path, sample_valid_dataset_dict):
    file_path = tmp_path / "dataset.json"
    file_path.write_text(json.dumps(sample_valid_dataset_dict), encoding="utf-8")

    ds = load_retrieval_dataset(file_path)
    assert ds.dataset_id == "test_ds_01"
    assert len(ds.chunks) == 3
    assert len(ds.queries) == 3


def test_load_jsonl_dataset(tmp_path):
    lines = [
        json.dumps({"type": "metadata", "dataset_id": "jsonl_ds", "description": "JSONL test", "is_illustrative_only": True}),
        json.dumps({"type": "chunk", "chunk_id": "ck1", "paper_id": "pA", "text": "Chunk A"}),
        json.dumps({"type": "chunk", "chunk_id": "ck2", "paper_id": "pB", "text": "Chunk B"}),
        json.dumps({"type": "query", "query_id": "qk1", "query": "Find Chunk A", "expected_chunk_ids": ["ck1"]}),
        json.dumps({"type": "query", "query_id": "qk2", "query": "Unanswerable", "expected_chunk_ids": []}),
    ]
    file_path = tmp_path / "dataset.jsonl"
    file_path.write_text("\n".join(lines), encoding="utf-8")

    ds = load_retrieval_dataset(file_path)
    assert ds.dataset_id == "jsonl_ds"
    assert len(ds.chunks) == 2
    assert len(ds.queries) == 2


def test_load_malformed_json_fails(tmp_path):
    file_path = tmp_path / "bad.json"
    file_path.write_text("{ unquoted_key: 123 ", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Malformed JSON"):
        load_retrieval_dataset(file_path)


def test_load_empty_file_fails(tmp_path):
    file_path = tmp_path / "empty.json"
    file_path.write_text("   \n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Dataset file is empty"):
        load_retrieval_dataset(file_path)


def test_load_nonexistent_file_fails():
    with pytest.raises(DatasetValidationError, match="Dataset file not found"):
        load_retrieval_dataset("nonexistent_path_xyz_123.json")


def test_illustrative_example_datasets_valid():
    # Verify built-in illustrative dataset function
    builtin_ds = get_illustrative_example_dataset()
    assert builtin_ds.is_illustrative_only is True
    assert len(builtin_ds.chunks) == 6
    assert len(builtin_ds.queries) == 4

    # Verify example JSON file
    json_path = os.path.join("examples", "illustrative_research_dataset.json")
    assert os.path.exists(json_path)
    json_ds = load_retrieval_dataset(json_path)
    assert json_ds.is_illustrative_only is True
    assert len(json_ds.chunks) == 6
    assert len(json_ds.queries) == 4

    # Verify example JSONL file
    jsonl_path = os.path.join("examples", "illustrative_research_dataset.jsonl")
    assert os.path.exists(jsonl_path)
    jsonl_ds = load_retrieval_dataset(jsonl_path)
    assert jsonl_ds.is_illustrative_only is True
    assert len(jsonl_ds.chunks) == 6
    assert len(jsonl_ds.queries) == 4


# ===========================================================================
# 4. evaluate_production_retrieval Integration Tests
# ===========================================================================

def test_evaluate_production_retrieval_with_custom_dataset(sample_valid_dataset_dict):
    ds = validate_dataset_dict(sample_valid_dataset_dict)
    config = ProductionEvaluationConfig(
        mode=EvaluationMode.SYNTHETIC,
        dataset=ds,
    )

    report = evaluate_production_retrieval(config)
    assert report.status == "completed"
    assert report.dataset_info is not None
    assert report.dataset_info["name"] == "test_ds_01"
    assert report.dataset_info["chunks_count"] == 3
    assert report.dataset_info["queries_count"] == 3
    assert report.dataset_info["unanswerable_queries_count"] == 1
    assert report.corpus_info["chunks_count"] == 3
    assert report.corpus_info["papers_count"] == 2
    assert report.paper_scope_isolation is not None
    assert report.paper_scope_isolation["target_paper_id"] == "p1"
    assert report.paper_scope_isolation["isolation_successful"] is True
    assert report.metrics is not None
    assert report.metrics.total_queries == 3


def test_evaluate_production_retrieval_default_preserves_builtin_benchmark():
    config = ProductionEvaluationConfig(
        mode=EvaluationMode.SYNTHETIC,
    )
    report = evaluate_production_retrieval(config)
    assert report.status == "completed"
    assert report.dataset_info is not None
    assert report.dataset_info["name"] == "builtin_synthetic_benchmark"
    assert report.corpus_info["papers_count"] == 4
    assert report.corpus_info["chunks_count"] == 15
    assert report.metrics.total_queries == 13


def test_evaluate_production_retrieval_invalid_dataset_fails_gracefully():
    config = ProductionEvaluationConfig(
        mode=EvaluationMode.SYNTHETIC,
        dataset={"chunks": []},  # Invalid dataset
    )
    report = evaluate_production_retrieval(config)
    assert report.status == "failed"
    assert "Invalid evaluation dataset" in report.infrastructure_limitation
    assert report.dataset_info is not None
    assert report.metrics is None


# ===========================================================================
# 5. CLI Integration Tests
# ===========================================================================

def test_cli_with_valid_dataset_json(tmp_path, sample_valid_dataset_dict):
    ds_file = tmp_path / "cli_dataset.json"
    ds_file.write_text(json.dumps(sample_valid_dataset_dict), encoding="utf-8")

    out_file = tmp_path / "cli_report.json"
    exit_code, report, json_text = run_eval_cli([
        "--mode", "synthetic",
        "--dataset", str(ds_file),
        "--output", str(out_file),
    ])

    assert exit_code == 0
    assert report.status == "completed"
    assert report.dataset_info["name"] == "test_ds_01"
    assert report.dataset_info["chunks_count"] == 3
    assert os.path.exists(out_file)

    saved_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved_data["dataset_info"]["name"] == "test_ds_01"
    assert saved_data["metrics"]["total_queries"] == 3


def test_cli_with_invalid_dataset_exits_with_error(tmp_path):
    bad_ds_file = tmp_path / "bad_dataset.json"
    bad_ds_file.write_text(
        json.dumps({
            "chunks": [{"chunk_id": "c1", "paper_id": "p1"}],  # Missing required field 'text'
            "queries": [{"query": "test query", "expected_chunk_ids": []}],
        }),
        encoding="utf-8",
    )

    exit_code, report, _ = run_eval_cli([
        "--mode", "synthetic",
        "--dataset", str(bad_ds_file),
    ])

    assert exit_code == 1
    assert report.status == "failed"
    assert "missing required field 'text'" in report.error
