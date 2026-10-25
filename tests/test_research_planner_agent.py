"""Comprehensive unit tests for Stage 4 Step 3: Research Planner Agent.

Verifies:
1. Valid structured research plan generation across all 7 dimensions.
2. Distinguishing evidence-backed findings from proposed recommendations.
3. Rejection of empty, whitespace, and oversized research goals.
4. Authorized collection access succeeds.
5. Unauthorized or cross-tenant collection access fails closed.
6. Safe structured errors on model or parsing failures.
7. Prevention of falsely labeling proposed ideas as verified evidence.
8. 100% offline execution without live API credentials.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from nexora.agents.models import (
    PLAN_STANDARD_DIMENSIONS,
    AgentContext,
    PlannerServiceError,
    ResearchPlan,
    ResearchPlanDimension,
)
from nexora.agents.planner import ResearchPlannerAgent, ResearchPlannerPromptBuilder
from nexora.api.errors import (
    DocumentNotFoundError,
    ForbiddenError,
    InvalidRequestError,
)
from nexora.llm.models import LLMResponse
from nexora.research.synthesis import (
    LiteratureSynthesisReport,
    SynthesisFinding,
    SynthesisLimitation,
)


# ===========================================================================
# 1. Input Validation & Bounds Tests
# ===========================================================================

def test_planner_rejects_empty_and_oversized_goals():
    """Verify ResearchPlannerAgent rejects empty, whitespace, non-string, or oversized goals."""
    mock_llm = MagicMock()
    planner = ResearchPlannerAgent(mock_llm)

    # Empty goal
    with pytest.raises(InvalidRequestError, match="must not be empty"):
        planner.plan("")

    # Whitespace-only goal
    with pytest.raises(InvalidRequestError, match="must not be empty"):
        planner.plan("   \n\t  ")

    # Non-string goal
    with pytest.raises(InvalidRequestError, match="must not be empty"):
        planner.plan(None)  # type: ignore

    # Oversized goal (> 2000 chars)
    with pytest.raises(InvalidRequestError, match="exceeds maximum length"):
        planner.plan("A" * 2001)

    # Malformed collection_id
    with pytest.raises(InvalidRequestError, match="must be a valid UUID"):
        planner.plan("Valid goal", collection_id="not-a-valid-uuid")


# ===========================================================================
# 2. Deterministic Mock Plan Generation (Offline CI)
# ===========================================================================

def test_planner_deterministic_mock_all_seven_dimensions():
    """Verify that deterministic mock mode produces a complete ResearchPlan with all 7 dimensions."""
    mock_llm = MagicMock()
    type(mock_llm).__name__ = "MockLLMService"

    planner = ResearchPlannerAgent(mock_llm)
    goal = "Investigate sub-quadratic state space models for efficient sequence processing"

    plan = planner.plan(goal)

    assert isinstance(plan, ResearchPlan)
    assert plan.is_mock is True
    assert "sub-quadratic state space" in plan.title.lower()
    assert len(plan.research_questions) >= 3
    assert plan.has_all_standard_dimensions() is True
    assert len(plan.dimensions) == 7

    # Verify each required dimension exists and has substantive content
    for dim_name in PLAN_STANDARD_DIMENSIONS:
        dim = plan.get_dimension(dim_name)
        assert dim is not None
        assert dim.name == dim_name
        assert len(dim.title) > 0
        assert len(dim.description) > 0
        assert isinstance(dim.details, dict)
        assert len(dim.details) > 0

    # Verify that ungrounded conceptual goals mark methodology & experiments as hypotheses
    methodology = plan.get_dimension("methodology")
    assert methodology.is_grounded is False
    assert methodology.details.get("recommendation_type") == "proposed_hypothesis"

    dataset_dim = plan.get_dimension("dataset_strategy")
    assert dataset_dim.is_grounded is False
    assert "recommended_datasets" in dataset_dim.details

    experiment_dim = plan.get_dimension("experiment_plan")
    assert experiment_dim.is_grounded is False
    assert "stages" in experiment_dim.details
    assert len(experiment_dim.details["stages"]) == 3

    metrics_dim = plan.get_dimension("evaluation_metrics")
    assert metrics_dim.is_grounded is False
    assert "primary_metrics" in metrics_dim.details

    # When no collection is provided, citations must be empty (zero hallucination)
    assert plan.citations == []
    assert plan.paper_ids == []


# ===========================================================================
# 3. Authorized vs Unauthorized Collection Access
# ===========================================================================

def test_planner_authorized_collection_grounds_literature_and_gaps():
    """Verify authorized collection access grounds literature and gap dimensions in real evidence."""
    col_id = uuid.uuid4()
    user_id = uuid.uuid4()
    paper_1 = str(uuid.uuid4())
    paper_2 = str(uuid.uuid4())

    mock_llm = MagicMock()
    type(mock_llm).__name__ = "MockLLMService"

    mock_auth_svc = MagicMock()
    mock_auth_svc.verify_collection_ownership.return_value = (MagicMock(), MagicMock())

    # Build fake literature synthesis report
    mock_finding = SynthesisFinding(
        claim="Mamba achieves linear scaling without standard attention matrix [E1].",
        status="reported",
        evidence_labels=["[E1]"],
        sources=[{"chunk_id": "c1", "text": "Mamba achieves linear scaling..."}],
        supporting_paper_ids=[paper_1],
    )
    mock_limitation = SynthesisLimitation(
        description="Selective scans require specialized SRAM kernel tiling.",
        affected_paper_ids=[paper_1, paper_2],
        evidence_labels=["[E1]"],
    )
    mock_synthesis = LiteratureSynthesisReport(
        collection_id=str(col_id),
        topic="Efficient Transformers",
        common_findings=[mock_finding],
        recurring_limitations=[mock_limitation],
        paper_ids=[paper_1, paper_2],
        is_grounded=True,
    )

    mock_trend_svc = MagicMock()
    mock_trend_svc.synthesize_collection_literature.return_value = mock_synthesis

    planner = ResearchPlannerAgent(
        mock_llm,
        trend_service=mock_trend_svc,
    )

    context = AgentContext(auth_enforced=True, user_id=user_id, auth_service=mock_auth_svc)
    plan = planner.plan("Design sub-quadratic architectures", collection_id=col_id, context=context)

    # Verify authorization check occurred
    mock_auth_svc.verify_collection_ownership.assert_called_once_with(col_id, user_id)
    mock_trend_svc.synthesize_collection_literature.assert_called_once()

    assert plan.collection_id == str(col_id)
    assert plan.paper_ids == [paper_1, paper_2]

    # Literature strategy and gaps are grounded in evidence
    lit_dim = plan.get_dimension("literature_strategy")
    assert lit_dim.is_grounded is True
    assert "[E1]" in lit_dim.evidence_labels
    assert lit_dim.supporting_paper_ids == [paper_1, paper_2]

    gaps_dim = plan.get_dimension("documented_gaps")
    assert gaps_dim.is_grounded is True
    assert "[E1]" in gaps_dim.evidence_labels

    # Methodology and experiments remain proposals (not falsely labeled as verified)
    method_dim = plan.get_dimension("methodology")
    assert method_dim.is_grounded is False
    assert method_dim.details["recommendation_type"] == "proposed_hypothesis"


def test_planner_unauthorized_cross_tenant_collection_fails_closed():
    """Verify cross-tenant collection access is blocked before any service is called."""
    col_id = uuid.uuid4()
    user_id = uuid.uuid4()

    mock_llm = MagicMock()
    type(mock_llm).__name__ = "MockLLMService"

    mock_auth_svc = MagicMock()
    mock_auth_svc.verify_collection_ownership.side_effect = ForbiddenError(
        "Access to the requested collection is forbidden."
    )

    mock_trend_svc = MagicMock()
    planner = ResearchPlannerAgent(mock_llm, trend_service=mock_trend_svc)

    context = AgentContext(auth_enforced=True, user_id=user_id, auth_service=mock_auth_svc)

    with pytest.raises(ForbiddenError, match="Access to the requested collection is forbidden"):
        planner.plan("Analyze proprietary models", collection_id=col_id, context=context)

    # Underlying trend service must never be invoked on unauthorized access
    mock_trend_svc.synthesize_collection_literature.assert_not_called()


def test_planner_missing_collection_raises_not_found():
    """Verify requesting a non-existent collection raises DocumentNotFoundError."""
    col_id = uuid.uuid4()
    mock_llm = MagicMock()
    type(mock_llm).__name__ = "MockLLMService"

    mock_auth_svc = MagicMock()
    mock_auth_svc.verify_collection_ownership.side_effect = DocumentNotFoundError(
        f"Collection {col_id} not found."
    )

    planner = ResearchPlannerAgent(mock_llm)
    context = AgentContext(auth_enforced=True, user_id=uuid.uuid4(), auth_service=mock_auth_svc)

    with pytest.raises(DocumentNotFoundError, match="not found"):
        planner.plan("Investigate missing collection", collection_id=col_id, context=context)


# ===========================================================================
# 4. Live LLM Generation & Parsing Tests
# ===========================================================================

def _create_valid_plan_json() -> str:
    """Helper returning a complete, valid JSON research plan covering all 7 dimensions."""
    return json.dumps({
        "title": "Sub-Quadratic Sequence Modeling Plan",
        "research_problem": "Transformer attention scales with O(N^2) memory complexity.",
        "research_questions": [
            "Can state space duality achieve linear scaling on 100k context?",
            "What accuracy trade-offs occur on associative recall tasks?"
        ],
        "dimensions": {
            "problem_and_objectives": {
                "title": "Problem Statement & Objectives",
                "description": "Develop sub-quadratic attention alternatives.",
                "details": {"core_problem": "Quadratic memory bottleneck"},
                "is_grounded": False,
                "evidence_labels": []
            },
            "literature_strategy": {
                "title": "Literature Strategy",
                "description": "Prior work on selective state spaces and linear attention.",
                "details": {"key_themes": ["State Space Models", "Linear Attention"]},
                "is_grounded": True,
                "evidence_labels": ["[E1]"]
            },
            "dataset_strategy": {
                "title": "Dataset Strategy",
                "description": "Pretrain on Pile and evaluate on GLUE and LongBench.",
                "details": {"recommended_datasets": ["Pile", "GLUE", "LongBench"]},
                "is_grounded": False,
                "evidence_labels": []
            },
            "methodology": {
                "title": "Proposed Methodology",
                "description": "Gated linear state spaces combined with hardware-aware SRAM tiling.",
                "details": {
                    "approach": "Hybrid selective scan",
                    "baseline_models": ["Transformer", "Mamba-1"]
                },
                "is_grounded": False,
                "evidence_labels": []
            },
            "experiment_plan": {
                "title": "Experiment Plan",
                "description": "Three-stage protocol: unit benchmark, pretraining sweep, ablation.",
                "details": {"stages": ["Stage 1", "Stage 2", "Stage 3"]},
                "is_grounded": False,
                "evidence_labels": []
            },
            "evaluation_metrics": {
                "title": "Evaluation Metrics",
                "description": "Measure throughput (tokens/sec), peak VRAM, and perplexity.",
                "details": {"primary_metrics": ["Perplexity", "Throughput", "VRAM"]},
                "is_grounded": False,
                "evidence_labels": []
            },
            "documented_gaps": {
                "title": "Documented Gaps",
                "description": "Existing SSMs degrade on multi-hop associative recall.",
                "details": {"addressed_limitations": ["Associative recall degradation"]},
                "is_grounded": True,
                "evidence_labels": ["[E1]"]
            }
        }
    })


class FakeLiveLLMService:
    """Explicitly non-mock test double that generates LLMResponse objects."""

    def __init__(self, response_text: str = "", side_effect: Exception | None = None) -> None:
        self.response_text = response_text
        self.side_effect = side_effect
        self._is_mock = False

    def generate(self, messages: list[Any], **kwargs: Any) -> LLMResponse:
        if self.side_effect:
            raise self.side_effect
        return LLMResponse(text=self.response_text, model="live-gpt-4o")


def test_planner_live_llm_json_parsing_success():
    """Verify ResearchPlannerAgent parses valid JSON from LLM into a complete ResearchPlan."""
    valid_json = _create_valid_plan_json()
    live_llm = FakeLiveLLMService(response_text=f"```json\n{valid_json}\n```")

    planner = ResearchPlannerAgent(live_llm)
    col_id = str(uuid.uuid4())
    plan = planner.plan(
        "Overcoming associative recall degradation in SSMs",
        collection_id=col_id,
    )

    assert plan.is_mock is False
    assert plan.title == "Sub-Quadratic Sequence Modeling Plan"
    assert len(plan.research_questions) == 2
    assert plan.has_all_standard_dimensions() is True
    assert plan.dimensions["methodology"].is_grounded is False
    assert plan.dimensions["literature_strategy"].is_grounded is True


def test_planner_live_llm_malformed_json_raises_service_error():
    """Verify malformed JSON from model raises PlannerServiceError with clean explanation."""
    live_llm = FakeLiveLLMService(response_text="```json\n{unclosed_json: 'bad'...\n```")

    planner = ResearchPlannerAgent(live_llm)
    with pytest.raises(PlannerServiceError, match="Failed to parse research plan JSON"):
        planner.plan("Goal causing malformed output")


def test_planner_live_llm_missing_dimension_raises_service_error():
    """Verify model output missing any of the 7 standard dimensions raises PlannerServiceError."""
    data = json.loads(_create_valid_plan_json())
    # Delete one mandatory dimension
    del data["dimensions"]["experiment_plan"]

    live_llm = FakeLiveLLMService(response_text=json.dumps(data))

    planner = ResearchPlannerAgent(live_llm)
    with pytest.raises(PlannerServiceError, match="missing required dimension: 'experiment_plan'"):
        planner.plan("Goal with incomplete dimension plan")


def test_planner_llm_network_failure_raises_service_error():
    """Verify upstream provider network exception wraps cleanly in PlannerServiceError."""
    live_llm = FakeLiveLLMService(side_effect=RuntimeError("OpenAI connection timeout"))

    planner = ResearchPlannerAgent(live_llm)
    with pytest.raises(PlannerServiceError, match="Planner LLM generation failed: RuntimeError"):
        planner.plan("Goal causing connection error")


def test_planner_enforces_hypotheses_not_labeled_as_verified():
    """Verify that model output attempting to label methodology or experiments as verified is corrected."""
    data = json.loads(_create_valid_plan_json())
    # Adversarial model claim: methodology is an established fact
    data["dimensions"]["methodology"]["is_grounded"] = True
    data["dimensions"]["experiment_plan"]["is_grounded"] = True

    live_llm = FakeLiveLLMService(response_text=json.dumps(data))

    planner = ResearchPlannerAgent(live_llm)
    col_id = str(uuid.uuid4())
    plan = planner.plan("Goal with overclaiming model", collection_id=col_id)

    # Safeguard: Methodology and experiment plan must NEVER be is_grounded: true
    method_dim = plan.get_dimension("methodology")
    assert method_dim.is_grounded is False
    assert method_dim.details.get("recommendation_type") == "proposed_hypothesis"

    exp_dim = plan.get_dimension("experiment_plan")
    assert exp_dim.is_grounded is False


def test_planner_to_dict_serialization():
    """Verify ResearchPlan serializes cleanly to dict."""
    mock_llm = MagicMock()
    type(mock_llm).__name__ = "MockLLMService"

    planner = ResearchPlannerAgent(mock_llm)
    plan = planner.plan("Investigate sequence model scaling")
    d = plan.to_dict()

    assert "plan_id" in d
    assert "title" in d
    assert "research_problem" in d
    assert "research_questions" in d
    assert "dimensions" in d
    assert len(d["dimensions"]) == 7
    for dim_name in PLAN_STANDARD_DIMENSIONS:
        assert dim_name in d["dimensions"]
        assert "name" in d["dimensions"][dim_name]
        assert "is_grounded" in d["dimensions"][dim_name]
