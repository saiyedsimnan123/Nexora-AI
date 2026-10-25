"""Research Planner Agent: synthesizes evidence-grounded research plans across 7 dimensions (Stage 4)."""

from __future__ import annotations

import copy
import datetime
import json
import re
from typing import Any, Sequence
from uuid import UUID, uuid4

from nexora.agents.models import (
    PLAN_STANDARD_DIMENSIONS,
    AgentContext,
    PlannerServiceError,
    ResearchPlan,
    ResearchPlanDimension,
)
from nexora.agents.prompt import _clean_json_text
from nexora.api.errors import (
    DocumentNotFoundError,
    ForbiddenError,
    InvalidRequestError,
)
from nexora.llm.models import LLMMessage
from nexora.retrieval.evidence import extract_citations, filter_unsupplied_citations

MAX_GOAL_CHARACTERS = 2000


class ResearchPlannerPromptBuilder:
    """Builds prompts for evidence-grounded research plan synthesis."""

    def build_system_message(self) -> LLMMessage:
        content = (
            "You are Nexora AI's autonomous Research Planner Agent.\n"
            "Your objective is to synthesize a structured, rigorous, evidence-grounded research plan "
            "for a researcher based on their research problem and any available literature evidence.\n\n"
            "MANDATORY RESEARCH PLAN SCHEMA:\n"
            "You must return ONLY a valid JSON object conforming to the following structure:\n"
            "{\n"
            '  "title": "<Concise research project title>",\n'
            '  "research_problem": "<Formal problem statement>",\n'
            '  "research_questions": ["<Question 1>", "<Question 2>", "<Question 3>"],\n'
            '  "dimensions": {\n'
            '    "problem_and_objectives": {\n'
            '      "title": "Problem Statement & Primary Objectives",\n'
            '      "description": "<Detailed description of core problem and expected contributions>",\n'
            '      "details": {\n'
            '        "core_problem": "<Problem>",\n'
            '        "primary_objectives": ["<Objective 1>", "<Objective 2>"],\n'
            '        "expected_contributions": ["<Contribution 1>", "<Contribution 2>"]\n'
            '      },\n'
            '      "is_grounded": false,\n'
            '      "evidence_labels": []\n'
            '    },\n'
            '    "literature_strategy": {\n'
            '      "title": "Literature Strategy & Theoretical Foundations",\n'
            '      "description": "<Synthesized review of foundational prior work and theoretical baselines>",\n'
            '      "details": {\n'
            '        "key_themes": ["<Theme 1>", "<Theme 2>"],\n'
            '        "theoretical_basis": "<Basis>"\n'
            '      },\n'
            '      "is_grounded": true,\n'
            '      "evidence_labels": ["<Cite [E#] labels present in provided evidence>"]\n'
            '    },\n'
            '    "dataset_strategy": {\n'
            '      "title": "Dataset Selection & Preprocessing Strategy",\n'
            '      "description": "<Recommended benchmark datasets, data splits, and preprocessing>",\n'
            '      "details": {\n'
            '        "recommended_datasets": ["<Dataset 1>", "<Dataset 2>"],\n'
            '        "data_splits": "<Splits>",\n'
            '        "rationale": "<Why selected>"\n'
            '      },\n'
            '      "is_grounded": false,\n'
            '      "evidence_labels": []\n'
            '    },\n'
            '    "methodology": {\n'
            '      "title": "Proposed Algorithmic Methodology",\n'
            '      "description": "<Proposed approach, baseline comparisons, and modifications>",\n'
            '      "details": {\n'
            '        "approach": "<Approach>",\n'
            '        "baseline_models": ["<Baseline 1>", "<Baseline 2>"],\n'
            '        "proposed_modifications": ["<Mod 1>", "<Mod 2>"],\n'
            '        "recommendation_type": "proposed_hypothesis"\n'
            '      },\n'
            '      "is_grounded": false,\n'
            '      "evidence_labels": []\n'
            '    },\n'
            '    "experiment_plan": {\n'
            '      "title": "Experimental Protocol & Stages",\n'
            '      "description": "<Phased execution stages, control setups, and validation>",\n'
            '      "details": {\n'
            '        "stages": ["<Stage 1>", "<Stage 2>", "<Stage 3>"],\n'
            '        "compute_requirements": "<Compute>",\n'
            '        "ablation_studies": ["<Ablation 1>"]\n'
            '      },\n'
            '      "is_grounded": false,\n'
            '      "evidence_labels": []\n'
            '    },\n'
            '    "evaluation_metrics": {\n'
            '      "title": "Quantitative Evaluation Metrics",\n'
            '      "description": "<Primary and secondary quantitative metrics and statistical testing>",\n'
            '      "details": {\n'
            '        "primary_metrics": ["<Metric 1>", "<Metric 2>"],\n'
            '        "secondary_metrics": ["<Metric 3>"],\n'
            '        "statistical_tests": "<Significance tests>"\n'
            '      },\n'
            '      "is_grounded": false,\n'
            '      "evidence_labels": []\n'
            '    },\n'
            '    "documented_gaps": {\n'
            '      "title": "Documented Gaps & Addressed Limitations",\n'
            '      "description": "<Specific acknowledged limitations in literature that this plan addresses>",\n'
            '      "details": {\n'
            '        "addressed_limitations": ["<Limitation 1>"],\n'
            '        "open_questions": ["<Question 1>"]\n'
            '      },\n'
            '      "is_grounded": true,\n'
            '      "evidence_labels": ["<Cite [E#] labels present in provided evidence>"]\n'
            '    }\n'
            '  }\n'
            "}\n\n"
            "CRITICAL RESEARCH INTEGRITY RULES:\n"
            "1. You must populate ALL seven standard dimensions. Do not omit any dimension.\n"
            "2. Distinguish established evidence from proposed hypotheses:\n"
            "   - Mark empirical findings from literature as is_grounded: true with verified [E#] citations.\n"
            "   - Mark recommended methodologies, architectures, and experiments as is_grounded: false "
            "     and set recommendation_type: 'proposed_hypothesis'. Never present hypotheses as established facts.\n"
            "   - Mark unverified questions as requires_investigation.\n"
            "3. NEVER fabricate citations or paper IDs. Only cite [E#] labels present in the provided evidence.\n"
            "4. Do not include markdown formatting or commentary outside the JSON object."
        )
        return LLMMessage(role="system", content=content)

    def build_user_message(self, goal: str, evidence_context_str: str | None = None) -> LLMMessage:
        user_text = f"RESEARCH GOAL:\n{goal.strip()}\n\n"
        if evidence_context_str and evidence_context_str.strip():
            user_text += (
                "=== BEGIN UNTRUSTED EVIDENTIARY FOUNDATION ===\n"
                f"{evidence_context_str.strip()}\n"
                "=== END UNTRUSTED EVIDENTIARY FOUNDATION ===\n\n"
            )
        else:
            user_text += "Note: No prior collection evidence provided. Formulate exploratory research questions and methodology hypotheses.\n\n"

        user_text += "Synthesize a complete, structured ResearchPlan JSON object addressing all seven dimensions."
        return LLMMessage(role="user", content=user_text)


class ResearchPlannerAgent:
    """Autonomous Research Planner Agent that translates research goals into evidence-grounded research plans."""

    def __init__(
        self,
        llm_service: Any,
        *,
        trend_service: Any = None,
        gap_service: Any = None,
        session_factory: Any = None,
        prompt_builder: ResearchPlannerPromptBuilder | None = None,
    ) -> None:
        if not callable(getattr(llm_service, "generate", None)):
            raise TypeError("llm_service must provide generate(messages, ...)")
        self._llm = llm_service
        self._trend_service = trend_service
        self._gap_service = gap_service
        self._session_factory = session_factory
        self._prompt_builder = prompt_builder or ResearchPlannerPromptBuilder()

    def _is_mock_mode(self) -> bool:
        """Detect whether the configured LLM is an offline mock or test generator."""
        if getattr(self._llm, "_is_mock", None) is False:
            return False
        if getattr(self._llm, "is_mock", None) is True or getattr(self._llm, "_is_mock", None) is True:
            return True
        llm_cls = type(self._llm).__name__
        if "mock" in llm_cls.lower():
            return True
        # Check provider if not a mock object
        provider = getattr(self._llm, "_provider", None)
        if provider is not None and not type(provider).__name__.lower().startswith("magicmock"):
            if "mock" in type(provider).__name__.lower():
                return True
        return False

    def plan(
        self,
        goal: str,
        *,
        collection_id: str | UUID | None = None,
        context: AgentContext | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ResearchPlan:
        """Generate a structured, evidence-grounded research plan.

        Args:
            goal: Natural language research problem statement or objective.
            collection_id: Optional collection UUID to ground the plan in verified literature.
            context: Security and caller authorization context.
            model: Optional model override.
            temperature: Sampling temperature.
            max_output_tokens: Token generation budget.

        Raises:
            InvalidRequestError: On empty or oversized research goals, or malformed UUIDs.
            ForbiddenError: On cross-tenant access to unauthorized collections.
            DocumentNotFoundError: If the requested collection does not exist.
            PlannerServiceError: On model failure or schema validation errors.
        """
        # 1. Validate inputs
        if not isinstance(goal, str) or not goal.strip():
            raise InvalidRequestError("goal must not be empty or whitespace-only.")
        if len(goal) > MAX_GOAL_CHARACTERS:
            raise InvalidRequestError(
                f"goal exceeds maximum length of {MAX_GOAL_CHARACTERS} characters (received {len(goal)})."
            )

        col_uuid: UUID | None = None
        col_id_str: str | None = None
        if collection_id is not None:
            try:
                col_uuid = UUID(str(collection_id).strip())
                col_id_str = str(col_uuid)
            except (ValueError, AttributeError):
                raise InvalidRequestError(f"Invalid collection_id {collection_id!r}: must be a valid UUID.")

        # 2. Verify authorization fail-closed
        ctx = context if context is not None else AgentContext(auth_enforced=False)
        if col_uuid is not None and ctx.auth_enforced:
            ctx.ensure_authorized()
            assert ctx.user_id is not None
            assert ctx.auth_service is not None
            ctx.auth_service.verify_collection_ownership(col_uuid, ctx.user_id)

        # 3. Gather collection evidence foundation if available
        synthesis_report: Any = None
        evidence_text: str = ""
        known_sources: list[dict[str, Any]] = []
        collection_paper_ids: list[str] = []

        if col_uuid is not None and self._trend_service is not None:
            try:
                synthesis_report = self._trend_service.synthesize_collection_literature(
                    col_uuid, topic=goal
                )
                collection_paper_ids = getattr(synthesis_report, "paper_ids", [])
                # Format summary of findings as evidentiary context
                evidence_lines = []
                findings_list = getattr(synthesis_report, "common_findings", getattr(synthesis_report, "findings", []))
                for idx, finding in enumerate(findings_list):
                    lbl = f"[E{idx + 1}]"
                    evidence_lines.append(f"{lbl} Claim: {finding.claim}")
                    for s in getattr(finding, "sources", []):
                        known_sources.append(s)

                for lim in getattr(synthesis_report, "recurring_limitations", []):
                    evidence_lines.append(f"Documented Limitation: {lim.description}")

                evidence_text = "\n".join(evidence_lines)
            except (ForbiddenError, DocumentNotFoundError):
                raise
            except Exception:
                synthesis_report = None

        # 4. Deterministic Mock Path (for offline CI / test environments)
        if self._is_mock_mode():
            return self._generate_deterministic_mock_plan(
                goal=goal,
                collection_id=col_id_str,
                paper_ids=collection_paper_ids,
                synthesis_report=synthesis_report,
                sources=known_sources,
            )

        # 5. Live LLM Generation Path
        system_msg = self._prompt_builder.build_system_message()
        user_msg = self._prompt_builder.build_user_message(goal, evidence_text)
        messages = [system_msg, user_msg]

        try:
            response = self._llm.generate(
                messages,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise PlannerServiceError(f"Planner LLM generation failed: {type(exc).__name__}") from exc

        # 6. Parse and validate structured output
        return self._parse_and_validate_plan(
            raw_text=response.text,
            goal=goal,
            collection_id=col_id_str,
            collection_paper_ids=collection_paper_ids,
            known_sources=known_sources,
            is_mock=False,
        )

    def _generate_deterministic_mock_plan(
        self,
        goal: str,
        collection_id: str | None,
        paper_ids: list[str],
        synthesis_report: Any,
        sources: list[dict[str, Any]],
    ) -> ResearchPlan:
        """Construct a deterministic, validated research plan for offline execution."""
        clean_goal = goal.strip().rstrip(".")
        title = f"Research Plan: {clean_goal[:70]}"
        problem = f"Investigating systematic scientific approaches to address: {clean_goal}."
        questions = [
            f"What are the empirical baselines and constraints associated with {clean_goal}?",
            f"How can novel algorithmic architectures improve performance over existing approaches?",
            f"What evaluation benchmarks and significance metrics rigorously validate improvements?",
        ]

        dimensions: dict[str, ResearchPlanDimension] = {}

        # 1. problem_and_objectives
        dimensions["problem_and_objectives"] = ResearchPlanDimension(
            name="problem_and_objectives",
            title="Problem Statement & Objectives",
            description=f"Primary scientific objective: advance state-of-the-art methodology for {clean_goal}.",
            details={
                "core_problem": clean_goal,
                "primary_objectives": [
                    f"Establish reproducible experimental baselines for {clean_goal}.",
                    "Formulate novel algorithmic improvements with sub-quadratic complexity.",
                    "Demonstrate statistically significant empirical gains on standard benchmarks.",
                ],
                "expected_contributions": [
                    "Formal algorithmic formulation and open-source implementation.",
                    "Comprehensive ablation analysis isolating key architectural drivers.",
                ],
            },
            is_grounded=True if synthesis_report is not None else False,
            evidence_labels=["[E1]"] if synthesis_report is not None else [],
            supporting_paper_ids=paper_ids[:2] if paper_ids else [],
        )

        # 2. literature_strategy
        findings_list = getattr(synthesis_report, "common_findings", getattr(synthesis_report, "findings", [])) if synthesis_report is not None else []
        if findings_list:
            findings_summary = "; ".join(f.claim for f in findings_list[:3])
            lit_labels = [f"[E{i+1}]" for i in range(min(3, len(findings_list)))]
            dimensions["literature_strategy"] = ResearchPlanDimension(
                name="literature_strategy",
                title="Literature Strategy & Theoretical Foundations",
                description=f"Synthesized evidence from collection: {findings_summary}.",
                details={
                    "key_themes": ["Linear attention mechanisms", "State space duality", "Hardware-aware kernel tiling"],
                    "theoretical_basis": "Sequence modeling without quadratic attention bottlenecks.",
                },
                is_grounded=True,
                evidence_labels=lit_labels,
                supporting_paper_ids=paper_ids,
            )
        else:
            dimensions["literature_strategy"] = ResearchPlanDimension(
                name="literature_strategy",
                title="Literature Strategy & Theoretical Foundations",
                description="Theoretical survey of transformer variants and efficient sequence modeling.",
                details={
                    "key_themes": ["Attention recurrence", "Sub-quadratic approximation", "Structured state spaces"],
                    "theoretical_basis": "Proposed theoretical framework based on linear recurrence.",
                    "recommendation_type": "proposed_hypothesis",
                },
                is_grounded=False,
                evidence_labels=[],
                supporting_paper_ids=[],
            )

        # 3. dataset_strategy (Recommendation, NOT established evidence)
        dimensions["dataset_strategy"] = ResearchPlanDimension(
            name="dataset_strategy",
            title="Dataset Selection & Preprocessing Strategy",
            description="Recommended standard benchmarks for evaluation across standard tasks.",
            details={
                "recommended_datasets": ["GLUE Benchmark", "Pile Long-Context", "WMT Translation"],
                "data_splits": "Standard 80/10/10 train/validation/test splits with deterministic seeding.",
                "rationale": "Enables direct comparative evaluation against published baseline metrics.",
                "recommendation_type": "recommended_benchmark",
            },
            is_grounded=False,
            evidence_labels=[],
            supporting_paper_ids=[],
        )

        # 4. methodology (Proposed hypothesis)
        dimensions["methodology"] = ResearchPlanDimension(
            name="methodology",
            title="Proposed Algorithmic Methodology",
            description="Proposed hybrid architecture combining selective state spaces with local attention.",
            details={
                "approach": "Hybrid selective scan mechanism operating in linear time.",
                "baseline_models": ["Standard Transformer", "FlashAttention-2", "Mamba-1"],
                "proposed_modifications": ["Dynamic chunk tiling", "Selective state gating"],
                "recommendation_type": "proposed_hypothesis",
            },
            is_grounded=False,
            evidence_labels=[],
            supporting_paper_ids=[],
        )

        # 5. experiment_plan (Phased experimental protocol)
        dimensions["experiment_plan"] = ResearchPlanDimension(
            name="experiment_plan",
            title="Experimental Protocol & Phased Stages",
            description="Phased execution protocol across three controlled development stages.",
            details={
                "stages": [
                    "Stage 1: Reproduce published baseline benchmarks under identical GPU environments.",
                    "Stage 2: Implement proposed algorithmic modifications and conduct hyperparameter sweep.",
                    "Stage 3: Full ablation study isolating state dimension, chunk size, and learning rate.",
                ],
                "compute_requirements": "8x NVIDIA A100 (80GB) cluster for 48 GPU hours.",
                "ablation_studies": ["Kernel chunk size sensitivity", "Gating parameter initialization"],
                "recommendation_type": "proposed_protocol",
            },
            is_grounded=False,
            evidence_labels=[],
            supporting_paper_ids=[],
        )

        # 6. evaluation_metrics (Quantitative metrics)
        dimensions["evaluation_metrics"] = ResearchPlanDimension(
            name="evaluation_metrics",
            title="Quantitative Evaluation Metrics",
            description="Rigorous empirical evaluation combining perplexity, speedup, and memory overhead.",
            details={
                "primary_metrics": ["Validation Perplexity", "Tokens/sec throughput", "Peak VRAM memory"],
                "secondary_metrics": ["Downstream task accuracy (GLUE score)", "Zero-shot transfer"],
                "statistical_tests": "Paired two-tailed t-test (p < 0.05) across 5 random seeds.",
                "recommendation_type": "recommended_metrics",
            },
            is_grounded=False,
            evidence_labels=[],
            supporting_paper_ids=[],
        )

        # 7. documented_gaps
        if synthesis_report is not None and getattr(synthesis_report, "recurring_limitations", []):
            lim_desc = "; ".join(l.description for l in synthesis_report.recurring_limitations[:2])
            dimensions["documented_gaps"] = ResearchPlanDimension(
                name="documented_gaps",
                title="Documented Gaps & Addressed Limitations",
                description=f"Addressed author-acknowledged limitations: {lim_desc}.",
                details={
                    "addressed_limitations": [l.description for l in synthesis_report.recurring_limitations[:3]],
                    "open_questions": getattr(synthesis_report, "open_questions", []),
                    "recommendation_type": "established_evidence",
                },
                is_grounded=True,
                evidence_labels=["[E1]"],
                supporting_paper_ids=paper_ids,
            )
        else:
            dimensions["documented_gaps"] = ResearchPlanDimension(
                name="documented_gaps",
                title="Documented Gaps & Addressed Limitations",
                description="Addresses documented limitations in long-context quadratic scaling.",
                details={
                    "addressed_limitations": [
                        "Quadratic memory complexity of standard self-attention.",
                        "Hardware-bound memory transfer bottlenecks between HBM and SRAM.",
                    ],
                    "open_questions": [
                        "Does selective state gating preserve associative recall on multi-hop reasoning?",
                    ],
                    "recommendation_type": "proposed_hypothesis",
                },
                is_grounded=False,
                evidence_labels=[],
                supporting_paper_ids=[],
            )

        return ResearchPlan(
            plan_id=str(uuid4()),
            title=title,
            research_problem=problem,
            research_questions=questions,
            dimensions=dimensions,
            collection_id=collection_id,
            paper_ids=list(paper_ids),
            citations=sources[:5],
            is_mock=True,
        )

    def _parse_and_validate_plan(
        self,
        raw_text: str,
        goal: str,
        collection_id: str | None,
        collection_paper_ids: list[str],
        known_sources: list[dict[str, Any]],
        is_mock: bool,
    ) -> ResearchPlan:
        """Parse, clean, and validate raw JSON output into a verified ResearchPlan."""
        cleaned = _clean_json_text(raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PlannerServiceError(f"Failed to parse research plan JSON from model: {exc}") from exc

        if not isinstance(data, dict):
            raise PlannerServiceError("Model output for research plan must be a JSON object.")

        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            title = f"Research Plan: {goal[:60]}"

        problem = data.get("research_problem")
        if not isinstance(problem, str) or not problem.strip():
            problem = f"Systematic investigation to address: {goal}"

        questions = data.get("research_questions", [])
        if not isinstance(questions, list) or not questions:
            questions = [f"How can we systematically investigate: {goal}?"]
        clean_questions = [str(q).strip() for q in questions if str(q).strip()]

        raw_dimensions = data.get("dimensions")
        if not isinstance(raw_dimensions, dict):
            raise PlannerServiceError("Research plan JSON must contain a 'dimensions' dictionary.")

        dimensions: dict[str, ResearchPlanDimension] = {}
        plan_citations: list[dict[str, Any]] = []

        for dim_name in PLAN_STANDARD_DIMENSIONS:
            dim_dict = raw_dimensions.get(dim_name)
            if not isinstance(dim_dict, dict):
                raise PlannerServiceError(
                    f"Generated research plan is missing required dimension: {dim_name!r}."
                )

            d_title = dim_dict.get("title")
            if not isinstance(d_title, str) or not d_title.strip():
                d_title = dim_name.replace("_", " ").title()

            d_desc = dim_dict.get("description")
            if not isinstance(d_desc, str) or not d_desc.strip():
                d_desc = "Analysis and strategy for this research dimension."

            d_details = dim_dict.get("details", {})
            if not isinstance(d_details, dict):
                d_details = {}

            # Evidence labels and citation grounding verification
            raw_labels = dim_dict.get("evidence_labels", [])
            clean_labels: list[str] = []
            if isinstance(raw_labels, list):
                for lbl in raw_labels:
                    if isinstance(lbl, str) and lbl.startswith("[E") and lbl.endswith("]"):
                        clean_labels.append(lbl)

            # Distinguish established findings from proposed hypotheses:
            # Only literature_strategy and documented_gaps can be is_grounded: true
            # and ONLY if verified evidence labels are present.
            is_grounded = bool(dim_dict.get("is_grounded", False))
            if dim_name in ("methodology", "dataset_strategy", "experiment_plan", "evaluation_metrics"):
                # Methodologies, future datasets, and experiment protocols are inherently proposals
                is_grounded = False
                if "recommendation_type" not in d_details:
                    d_details["recommendation_type"] = "proposed_hypothesis"
            elif dim_name in ("literature_strategy", "documented_gaps"):
                if not clean_labels or not collection_id:
                    # No evidence provided or no citations present
                    is_grounded = False
                    clean_labels = []

            # Populate supporting paper IDs
            supporting_pids = []
            if is_grounded and collection_paper_ids:
                supporting_pids = list(collection_paper_ids)

            dimensions[dim_name] = ResearchPlanDimension(
                name=dim_name,
                title=d_title,
                description=d_desc,
                details=d_details,
                evidence_labels=clean_labels,
                supporting_paper_ids=supporting_pids,
                is_grounded=is_grounded,
            )

        if collection_id and known_sources:
            plan_citations = known_sources[:10]

        return ResearchPlan(
            plan_id=str(uuid4()),
            title=title,
            research_problem=problem,
            research_questions=clean_questions,
            dimensions=dimensions,
            collection_id=collection_id,
            paper_ids=list(collection_paper_ids),
            citations=plan_citations,
            is_mock=is_mock,
        )
