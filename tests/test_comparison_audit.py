"""Tests for Cross-Paper Comparison Quality & Calibration Audit.

Validates deterministic decision boundaries, safety guardrails, and precision/recall
metrics across difficult research scenarios:
1. Split-dataset win / loss trade-offs
2. Small gains within experimental noise / lack of significance
3. Disparate non-overlapping benchmarks & differing evaluation protocols
4. Accuracy vs latency / compute overhead trade-offs
5. Contradictory findings between credible papers
6. Missing and duplicated evidence handling
7. Prompt injection defense and instruction override neutralization
8. Unsupported numerical claims exceeding cited excerpts
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from nexora.research.calibration import (
    AUDIT_EVALUATION_DATASET,
    ComparisonAuditCase,
    ComparisonAuditResult,
    run_comparison_audit,
)
from nexora.research.comparison import (
    COMPARISON_MATRIX_DIMENSIONS,
    ConclusionClassification,
    CrossPaperComparisonResult,
    MatrixPaperFinding,
)
from nexora.research.cross_paper_service import CrossPaperComparisonService
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import VectorRecord
from tests.test_cross_paper_comparison import ControllableComparisonLLMProvider


def _get_case_by_id(case_id: str) -> ComparisonAuditCase:
    for c in AUDIT_EVALUATION_DATASET:
        if c.case_id == case_id:
            return c
    raise KeyError(f"Case {case_id} not found in audit dataset")


def test_audit_case_01_demo_improve_shared_benchmark():
    """Case 1: Demonstrated improvement on verified shared benchmark."""
    case = _get_case_by_id("case_01_demo_improve_shared_benchmark")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.correct_classifications == 1
    assert result.accuracy == 1.0
    eval_item = result.case_evaluations[0]
    assert eval_item["passed"] is True
    assert eval_item["actual_conclusion"] == ConclusionClassification.DEMONSTRATED_IMPROVEMENT.value
    assert eval_item["actual_comparability"] == "comparable"


def test_audit_case_02_split_dataset_win_loss():
    """Case 2: Split wins across benchmarks must override superiority to Conflicting evidence."""
    case = _get_case_by_id("case_02_split_dataset_win_loss")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    assert result.false_superiority_claims_prevented == 1
    assert result.false_superiority_claims_leaked == 0
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.CONFLICTING_EVIDENCE.value


def test_audit_case_03_accuracy_latency_tradeoff():
    """Case 3: Accuracy gains accompanied by latency regression must classify as Conflicting evidence."""
    case = _get_case_by_id("case_03_accuracy_latency_tradeoff")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    assert result.false_superiority_claims_prevented == 1
    assert result.false_superiority_claims_leaked == 0
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.CONFLICTING_EVIDENCE.value


def test_audit_case_04_contradictory_findings():
    """Case 4: Directly contradictory empirical findings between papers -> Conflicting evidence."""
    case = _get_case_by_id("case_04_contradictory_findings")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.CONFLICTING_EVIDENCE.value


def test_audit_case_05_disparate_benchmarks_incomparable():
    """Case 5: Disparate non-overlapping benchmarks must classify as Limited evidence and incomparable."""
    case = _get_case_by_id("case_05_disparate_benchmarks_incomparable")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    assert result.false_superiority_claims_prevented == 1
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.LIMITED_EVIDENCE.value
    assert eval_item["actual_comparability"] == "incomparable"


def test_audit_case_06_experimental_noise_marginal():
    """Case 6: Small improvements within experimental noise override to Limited evidence."""
    case = _get_case_by_id("case_06_experimental_noise_marginal")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    assert result.false_superiority_claims_prevented == 1
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.LIMITED_EVIDENCE.value


def test_audit_case_07_differing_evaluation_protocols():
    """Case 7: Differing evaluation protocols (5-shot vs 0-shot) must be flagged as Limited evidence."""
    case = _get_case_by_id("case_07_differing_evaluation_protocols")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    assert result.false_superiority_claims_prevented == 1
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.LIMITED_EVIDENCE.value
    assert eval_item["actual_comparability"] == "incomparable"


def test_audit_case_08_missing_evidence_all_papers():
    """Case 8: All papers lack evidence chunks -> Insufficient evidence short-circuit."""
    case = _get_case_by_id("case_08_missing_evidence_all_papers")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.INSUFFICIENT_EVIDENCE.value
    assert eval_item["actual_comparability"] == "insufficient_evidence"


def test_audit_case_09_solo_paper_evidence_no_comparator():
    """Case 9: Never infer superiority from one paper alone when comparator lacks evidence."""
    case = _get_case_by_id("case_09_solo_paper_evidence_no_comparator")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    assert result.false_superiority_claims_prevented == 1
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.INSUFFICIENT_EVIDENCE.value
    assert eval_item["actual_comparability"] == "insufficient_evidence"


def test_audit_case_10_no_reported_results_or_metrics():
    """Case 10: Theoretical papers with zero empirical metrics must classify as Insufficient evidence."""
    case = _get_case_by_id("case_10_no_reported_results_or_metrics")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    assert result.false_superiority_claims_prevented == 1
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.INSUFFICIENT_EVIDENCE.value


def test_audit_case_11_prompt_injection_defense():
    """Case 11: Prompt injections inside paper text must be neutralized without false superiority."""
    case = _get_case_by_id("case_11_prompt_injection_defense")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    assert result.false_superiority_claims_prevented == 1
    assert result.false_superiority_claims_leaked == 0
    assert result.safety_critical_errors == 0
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.LIMITED_EVIDENCE.value


def test_audit_case_12_unsupported_numerical_claims():
    """Case 12: Unsupported numerical claims mark finding as uncertain and block Demonstrated improvement."""
    case = _get_case_by_id("case_12_unsupported_numerical_claims")
    result = run_comparison_audit(dataset=[case])

    assert result.total_cases == 1
    assert result.accuracy == 1.0
    assert result.false_superiority_claims_prevented == 1
    eval_item = result.case_evaluations[0]
    assert eval_item["actual_conclusion"] == ConclusionClassification.LIMITED_EVIDENCE.value


def test_audit_evidence_deduplication():
    """Verifies that duplicated evidence chunks are deterministically deduplicated during comparison."""
    pid1 = str(uuid4())
    pid2 = str(uuid4())

    store = InMemoryVectorStore()
    # Add duplicate chunk IDs and duplicate content for pid1
    store._records["c1"] = VectorRecord(
        id="c1",
        vector=[1.0, 0.0],
        payload={"paper_id": pid1, "text": "Common method details.", "chunk_index": 0},
    )
    store._records["c1_dup"] = VectorRecord(
        id="c1",  # duplicate ID
        vector=[1.0, 0.0],
        payload={"paper_id": pid1, "text": "Common method details.", "chunk_index": 0},
    )
    store._records["c2"] = VectorRecord(
        id="c2",
        vector=[1.0, 0.0],
        payload={"paper_id": pid2, "text": "Comparator method details.", "chunk_index": 0},
    )

    mock_json = json.dumps({
        "matrix": {
            dim: {
                "comparison_summary": "Summary",
                "key_differences": [],
                "key_similarities": [],
                "per_paper": {
                    pid1: {"summary": "Summary 1", "status": "reported", "evidence_labels": ["[E1]"], "details": {}},
                    pid2: {"summary": "Summary 2", "status": "reported", "evidence_labels": ["[E1]"], "details": {}},
                },
            }
            for dim in COMPARISON_MATRIX_DIMENSIONS
        },
        "comparability_status": "comparable",
        "conclusion": "Limited evidence",
        "conclusion_reasoning": "Deduplication audit test.",
        "overall_summary": "Deduplication audit summary.",
    })

    service = CrossPaperComparisonService(
        ControllableComparisonLLMProvider(response_text=mock_json),
        vector_store=store,
    )
    result = service.compare([pid1, pid2])
    assert result.conclusion == "Limited evidence"
    # Ensure sources were cleanly populated without duplicate chunk IDs
    for dim in COMPARISON_MATRIX_DIMENSIONS:
        srcs = result.matrix[dim].per_paper[pid1].sources
        chunk_ids = [s.chunk_id for s in srcs]
        assert len(chunk_ids) == len(set(chunk_ids))


def test_run_comparison_audit_benchmark():
    """Execute full 12-case calibration audit benchmark and assert 100% calibration."""
    result: ComparisonAuditResult = run_comparison_audit()

    assert result.total_cases == 12
    assert result.correct_classifications == 12
    assert result.accuracy == 1.0

    # Safety-critical invariant: 0 leaked false superiority claims across all adversarial cases
    assert result.false_superiority_claims_leaked == 0
    assert result.safety_critical_errors == 0
    assert result.false_superiority_claims_prevented >= 8

    # Precision & recall per class
    for cls_name in (
        ConclusionClassification.DEMONSTRATED_IMPROVEMENT.value,
        ConclusionClassification.LIMITED_EVIDENCE.value,
        ConclusionClassification.CONFLICTING_EVIDENCE.value,
        ConclusionClassification.INSUFFICIENT_EVIDENCE.value,
    ):
        assert result.per_class_precision[cls_name] == 1.0
        assert result.per_class_recall[cls_name] == 1.0

    # Ensure result serializes cleanly to dict
    data = result.to_dict()
    assert data["total_cases"] == 12
    assert data["accuracy"] == 1.0
    assert data["false_superiority_claims_leaked"] == 0
    assert len(data["case_evaluations"]) == 12
    assert len(data["known_limitations"]) >= 3
