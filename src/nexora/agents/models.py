"""Agent data models, execution traces, tool contracts, and research plan schemas (Stage 4)."""

from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence
from uuid import UUID, uuid4

_SECRET_MARKERS = ("api_key", "apikey", "authorization", "secret", "password", "token")


class AgentStatus(str, Enum):
    """Lifecycle status of an agent execution trace."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"


class AgentStepStatus(str, Enum):
    """Outcome status of an individual agent step."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentExecutionError(RuntimeError):
    """Raised when an unrecoverable agent execution failure occurs."""


class UnknownToolError(KeyError):
    """Raised when an unrecognized tool name is requested."""


class InvalidToolArgumentError(ValueError):
    """Raised when tool arguments fail validation schemas."""


class AgentSecurityError(PermissionError):
    """Raised when an operation violates tenant boundaries or authorization checks."""


class InvalidActionError(ValueError):
    """Raised when model returns an unparseable or unsupported action."""


class PlannerServiceError(RuntimeError):
    """Raised when research plan synthesis or validation fails."""


@dataclass(frozen=True)
class AgentAction:
    """Parsed operational action from an agent reasoning step."""

    action_type: str  # "call_tool" | "final_answer"
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    final_answer: str | None = None
    thought: str | None = None

    def __post_init__(self) -> None:
        if self.action_type not in ("call_tool", "final_answer"):
            raise ValueError(f"Unsupported action_type: {self.action_type!r}")
        if self.action_type == "call_tool":
            if not isinstance(self.tool_name, str) or not self.tool_name.strip():
                raise ValueError("call_tool action requires a non-empty tool_name")
            if not isinstance(self.tool_input, dict):
                raise ValueError("call_tool action requires a tool_input dict")
        elif self.action_type == "final_answer":
            if not isinstance(self.final_answer, str):
                raise ValueError("final_answer action requires string answer")


def scrub_sensitive_dict(value: Any) -> Any:
    """Recursively redact sensitive credential keys from logging and traces."""
    if isinstance(value, dict):
        scrubbed = {}
        for k, v in value.items():
            if isinstance(k, str) and any(marker in k.lower() for marker in _SECRET_MARKERS):
                scrubbed[k] = "[REDACTED]"
            else:
                scrubbed[k] = scrub_sensitive_dict(v)
        return scrubbed
    elif isinstance(value, (list, tuple)):
        return [scrub_sensitive_dict(v) for v in value]
    return value


@dataclass
class AgentContext:
    """Security and execution context for an agent run.

    Guarantees that tenant boundaries cannot be spoofed by model inputs.
    """

    user_id: UUID | None = None
    workspace_id: UUID | None = None
    auth_enforced: bool = True
    session_factory: Any = None
    auth_service: Any = None

    def __post_init__(self) -> None:
        if self.user_id is not None and not isinstance(self.user_id, UUID):
            try:
                self.user_id = UUID(str(self.user_id).strip())
            except (ValueError, AttributeError):
                raise ValueError("user_id must be a valid UUID or None")
        if self.workspace_id is not None and not isinstance(self.workspace_id, UUID):
            try:
                self.workspace_id = UUID(str(self.workspace_id).strip())
            except (ValueError, AttributeError):
                raise ValueError("workspace_id must be a valid UUID or None")

    @property
    def is_authenticated(self) -> bool:
        """True if a valid user_id is present."""
        return self.user_id is not None

    def ensure_authorized(self) -> None:
        """Fail closed if authentication is enforced and user_id or auth_service is missing."""
        if not self.auth_enforced:
            return
        if self.user_id is None:
            raise AgentSecurityError(
                "Missing authenticated user context: operation requires an authenticated caller."
            )
        if self.auth_service is None:
            raise AgentSecurityError(
                "Missing authorization service: cannot verify caller permissions."
            )


@dataclass
class ToolDefinition:
    """Public specification of an available agent tool."""

    name: str
    description: str
    parameters_schema: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Tool description must be a non-empty string.")
        if not isinstance(self.parameters_schema, dict):
            raise TypeError("parameters_schema must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters_schema": copy.deepcopy(self.parameters_schema),
        }


@dataclass
class ToolResult:
    """Structured output returned by an individual tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    summary: str = ""
    error: str | None = None
    error_type: str | None = None
    evidence_labels: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str):
            raise TypeError("tool_name must be a string")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")
        if not isinstance(self.summary, str):
            self.summary = str(self.summary)
        # Cap summary to prevent memory exhaustion
        if len(self.summary) > 2000:
            self.summary = self.summary[:2000] + "... [truncated]"
        if self.duration_ms < 0:
            self.duration_ms = 0.0

    @classmethod
    def ok(
        cls,
        tool_name: str,
        data: Any,
        summary: str,
        evidence_labels: list[str] | None = None,
        duration_ms: float = 0.0,
    ) -> ToolResult:
        """Factory for a successful tool execution."""
        return cls(
            tool_name=tool_name,
            success=True,
            data=data,
            summary=summary,
            evidence_labels=list(evidence_labels or []),
            duration_ms=duration_ms,
        )

    @classmethod
    def failure(
        cls,
        tool_name: str,
        error: str,
        error_type: str = "execution_error",
        duration_ms: float = 0.0,
    ) -> ToolResult:
        """Factory for a failed tool execution."""
        return cls(
            tool_name=tool_name,
            success=False,
            error=str(error),
            error_type=str(error_type),
            summary=f"Failed ({error_type}): {error}",
            duration_ms=duration_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": scrub_sensitive_dict(self.data),
            "summary": self.summary,
            "error": self.error,
            "error_type": self.error_type,
            "evidence_labels": list(self.evidence_labels),
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class AgentStep:
    """Operational record of a single action taken by an agent.

    Deliberately records operational parameters rather than model-specific internal reasoning.
    """

    step_index: int
    tool_name: str
    tool_input: dict[str, Any]
    output_summary: str
    status: str = AgentStepStatus.SUCCESS.value
    duration_ms: float = 0.0
    error: str | None = None
    evidence_labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not isinstance(self.tool_name, str):
            raise TypeError("tool_name must be a string")
        if not isinstance(self.tool_input, dict):
            raise TypeError("tool_input must be a dict")
        # Scrub tool_input of any credentials
        self.tool_input = scrub_sensitive_dict(self.tool_input)
        if len(self.output_summary) > 2000:
            self.output_summary = self.output_summary[:2000] + "... [truncated]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "tool_input": copy.deepcopy(self.tool_input),
            "output_summary": self.output_summary,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "evidence_labels": list(self.evidence_labels),
        }


@dataclass
class AgentTrace:
    """Audit and execution trace recording the full lifecycle of an agent run."""

    trace_id: str
    goal: str
    status: str = AgentStatus.PENDING.value
    steps: list[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_steps: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = str(uuid4())
        self.metadata = scrub_sensitive_dict(self.metadata)

    def add_step(self, step: AgentStep) -> None:
        """Append an operational step to the trace and update summary metrics."""
        if not isinstance(step, AgentStep):
            raise TypeError("step must be an AgentStep")
        self.steps.append(step)
        self.total_steps = len(self.steps)
        self.total_duration_ms += step.duration_ms

    def mark_completed(self) -> None:
        self.status = AgentStatus.COMPLETED.value

    def mark_failed(self, error_message: str) -> None:
        self.status = AgentStatus.FAILED.value
        self.error = error_message

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_steps": self.total_steps,
            "error": self.error,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ResearchPlanDimension:
    """A structured dimension within an evidence-grounded research plan."""

    name: str
    title: str
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    evidence_labels: list[str] = field(default_factory=list)
    supporting_paper_ids: list[str] = field(default_factory=list)
    is_grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "details": copy.deepcopy(self.details),
            "evidence_labels": list(self.evidence_labels),
            "supporting_paper_ids": list(self.supporting_paper_ids),
            "is_grounded": self.is_grounded,
        }


PLAN_STANDARD_DIMENSIONS = (
    "problem_and_objectives",
    "literature_strategy",
    "dataset_strategy",
    "methodology",
    "experiment_plan",
    "evaluation_metrics",
    "documented_gaps",
)


@dataclass
class ResearchPlan:
    """Evidence-grounded structured research plan generated by the Research Planner Agent."""

    plan_id: str
    title: str
    research_problem: str
    research_questions: list[str] = field(default_factory=list)
    dimensions: dict[str, ResearchPlanDimension] = field(default_factory=dict)
    collection_id: str | None = None
    paper_ids: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    is_mock: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "research_problem": self.research_problem,
            "research_questions": list(self.research_questions),
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "collection_id": self.collection_id,
            "paper_ids": list(self.paper_ids),
            "citations": copy.deepcopy(self.citations),
            "created_at": self.created_at,
            "is_mock": self.is_mock,
        }

    def get_dimension(self, name: str) -> ResearchPlanDimension | None:
        """Retrieve a plan dimension by name."""
        return self.dimensions.get(name)

    def has_all_standard_dimensions(self) -> bool:
        """Verify that all 7 standard research dimensions are populated."""
        return all(dim in self.dimensions for dim in PLAN_STANDARD_DIMENSIONS)


@dataclass
class AgentRunResult:
    """Outcome of an end-to-end agent orchestration execution."""

    trace: AgentTrace
    final_answer: str | None
    status: str
    total_steps: int
    duration_ms: float
    evidence_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.to_dict(),
            "final_answer": self.final_answer,
            "status": self.status,
            "total_steps": self.total_steps,
            "duration_ms": round(self.duration_ms, 2),
            "evidence_labels": list(self.evidence_labels),
        }

