"""Bounded agent orchestration engine with execution limits, timeouts, and loop detection (Stage 4)."""

from __future__ import annotations

import json
import time
from typing import Any, Sequence
from uuid import uuid4

from nexora.agents.models import (
    AgentAction,
    AgentContext,
    AgentRunResult,
    AgentSecurityError,
    AgentStatus,
    AgentStep,
    AgentStepStatus,
    AgentTrace,
    InvalidActionError,
)
from nexora.agents.prompt import AgentPromptBuilder, parse_agent_action
from nexora.agents.tools import ToolRegistry
from nexora.llm.models import LLMMessage

DEFAULT_MAX_STEPS = 5
HARD_MAX_STEPS = 10
DEFAULT_TIMEOUT_SECONDS = 30.0
HARD_TIMEOUT_SECONDS = 30.0


class AgentOrchestrator:
    """Bounded multi-step research agent orchestrator.

    Executes an iterative Reason-Act-Observe loop with strict safety controls:
    1. Maximum step bounds: default 5 steps, hard ceiling of 10.
    2. Execution deadline checking: default 30.0s, hard ceiling of 30.0s. Checked
       before and after every LLM generation and tool execution.
    3. Loop detection: tracks fingerprints of executed tool calls and aborts on
       repeated identical calls with the same arguments.
    4. Controlled task lifecycle: runs synchronously in the caller's thread/task
       without spawning unmanaged detached background threads.
    5. Tenant authorization: never trusts model-generated identifiers; all calls
       are validated against caller's authenticated AgentContext.
    """

    def __init__(
        self,
        llm_service: Any,
        tool_registry: ToolRegistry,
        *,
        default_max_steps: int = DEFAULT_MAX_STEPS,
        hard_max_steps: int = HARD_MAX_STEPS,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        hard_timeout_seconds: float = HARD_TIMEOUT_SECONDS,
        prompt_builder: AgentPromptBuilder | None = None,
    ) -> None:
        if not callable(getattr(llm_service, "generate", None)):
            raise TypeError("llm_service must provide generate(messages, ...)")
        if not isinstance(tool_registry, ToolRegistry):
            raise TypeError("tool_registry must be an instance of ToolRegistry")

        if hard_max_steps < 1 or hard_max_steps > HARD_MAX_STEPS:
            raise ValueError(f"hard_max_steps must be between 1 and {HARD_MAX_STEPS}.")
        if default_max_steps < 1 or default_max_steps > hard_max_steps:
            raise ValueError(f"default_max_steps must be between 1 and {hard_max_steps}.")

        if hard_timeout_seconds <= 0 or hard_timeout_seconds > HARD_TIMEOUT_SECONDS:
            raise ValueError(f"hard_timeout_seconds must be between 0 and {HARD_TIMEOUT_SECONDS}s.")
        if default_timeout_seconds <= 0 or default_timeout_seconds > hard_timeout_seconds:
            raise ValueError(f"default_timeout_seconds must be between 0 and {hard_timeout_seconds}s.")

        self._llm = llm_service
        self._tool_registry = tool_registry
        self._default_max_steps = default_max_steps
        self._hard_max_steps = hard_max_steps
        self._default_timeout_seconds = default_timeout_seconds
        self._hard_timeout_seconds = hard_timeout_seconds
        self._prompt_builder = prompt_builder or AgentPromptBuilder()

    def run(
        self,
        goal: str,
        *,
        context: AgentContext | None = None,
        max_steps: int | None = None,
        timeout_seconds: float | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AgentRunResult:
        """Execute a goal-driven agent run within bounded execution guardrails."""
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string.")

        # Resolve and validate limits
        effective_max_steps = self._default_max_steps if max_steps is None else max_steps
        if effective_max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if effective_max_steps > self._hard_max_steps:
            raise ValueError(
                f"max_steps ({effective_max_steps}) cannot exceed hard limit of {self._hard_max_steps}."
            )

        effective_timeout = (
            self._default_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if effective_timeout <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if effective_timeout > self._hard_timeout_seconds:
            raise ValueError(
                f"timeout_seconds ({effective_timeout}) cannot exceed hard limit of {self._hard_timeout_seconds}s."
            )

        ctx = context if context is not None else AgentContext(auth_enforced=False)

        start_time = time.perf_counter()
        deadline = start_time + effective_timeout

        trace = AgentTrace(
            trace_id=str(uuid4()),
            goal=goal.strip(),
            status=AgentStatus.RUNNING.value,
        )

        # 1. Enforce security context pre-flight
        try:
            ctx.ensure_authorized()
        except AgentSecurityError as exc:
            trace.mark_failed(str(exc))
            total_dur = (time.perf_counter() - start_time) * 1000
            trace.total_duration_ms = total_dur
            return AgentRunResult(
                trace=trace,
                final_answer=None,
                status=trace.status,
                total_steps=0,
                duration_ms=total_dur,
            )

        # 2. Build initial conversation messages
        messages: list[LLMMessage] = [
            self._prompt_builder.build_system_message(self._tool_registry.list_tools(), ctx),
            self._prompt_builder.build_user_message(goal),
        ]

        seen_action_fingerprints: set[str] = set()
        step_count = 0
        all_evidence_labels: list[str] = []
        final_answer: str | None = None

        def is_deadline_exceeded() -> bool:
            return time.perf_counter() >= deadline

        # 3. Iterative execution loop
        while step_count < effective_max_steps:
            # Check deadline before calling model
            if is_deadline_exceeded():
                trace.status = AgentStatus.TIMED_OUT.value
                trace.error = f"Execution deadline exceeded ({effective_timeout:.2f}s timeout)."
                break

            step_start = time.perf_counter()

            # Call model
            try:
                response = self._llm.generate(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as exc:
                trace.mark_failed(f"LLM generation failed: {type(exc).__name__}")
                break

            # Check deadline after calling model
            if is_deadline_exceeded():
                trace.status = AgentStatus.TIMED_OUT.value
                trace.error = f"Execution deadline exceeded ({effective_timeout:.2f}s timeout)."
                break

            # Parse action
            try:
                action = parse_agent_action(response.text)
            except InvalidActionError as exc:
                step_dur = (time.perf_counter() - step_start) * 1000
                step = AgentStep(
                    step_index=step_count,
                    tool_name="unrecognized_action",
                    tool_input={},
                    output_summary=f"Invalid action from model: {exc}",
                    status=AgentStepStatus.FAILED.value,
                    duration_ms=step_dur,
                    error=str(exc),
                )
                trace.add_step(step)
                trace.mark_failed(f"Invalid agent action format: {exc}")
                break

            # Branch A: Final Answer
            if action.action_type == "final_answer":
                final_answer = action.final_answer
                trace.mark_completed()
                break

            # Branch B: Call Tool
            assert action.action_type == "call_tool"
            tool_name = action.tool_name or ""
            tool_input = action.tool_input or {}

            # Check tool registration in allowlist
            if not self._tool_registry.has(tool_name):
                step_dur = (time.perf_counter() - step_start) * 1000
                step = AgentStep(
                    step_index=step_count,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    output_summary=f"Tool {tool_name!r} is not registered in the allowlist.",
                    status=AgentStepStatus.FAILED.value,
                    duration_ms=step_dur,
                    error=f"Unknown tool {tool_name!r}",
                )
                trace.add_step(step)
                trace.mark_failed(f"Attempted to call unknown tool {tool_name!r}.")
                break

            # Loop detection: identical tool + identical arguments
            try:
                fingerprint = f"{tool_name}::{json.dumps(tool_input, sort_keys=True)}"
            except Exception:
                fingerprint = f"{tool_name}::{str(tool_input)}"

            if fingerprint in seen_action_fingerprints:
                step_dur = (time.perf_counter() - step_start) * 1000
                step = AgentStep(
                    step_index=step_count,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    output_summary=f"Loop protection triggered: duplicate call to {tool_name!r}.",
                    status=AgentStepStatus.FAILED.value,
                    duration_ms=step_dur,
                    error="loop_detected",
                )
                trace.add_step(step)
                trace.mark_failed(
                    f"Loop protection triggered: identical call to {tool_name!r} repeated with identical arguments."
                )
                break

            seen_action_fingerprints.add(fingerprint)

            # Check deadline before tool execution
            if is_deadline_exceeded():
                trace.status = AgentStatus.TIMED_OUT.value
                trace.error = f"Execution deadline exceeded ({effective_timeout:.2f}s timeout) before tool execution."
                break

            # Execute tool through registry
            tool_result = self._tool_registry.execute(tool_name, tool_input, ctx)
            step_dur = (time.perf_counter() - step_start) * 1000

            step = AgentStep(
                step_index=step_count,
                tool_name=tool_name,
                tool_input=tool_input,
                output_summary=tool_result.summary,
                status=AgentStepStatus.SUCCESS.value if tool_result.success else AgentStepStatus.FAILED.value,
                duration_ms=step_dur,
                error=tool_result.error,
                evidence_labels=tool_result.evidence_labels,
            )
            trace.add_step(step)
            all_evidence_labels.extend(tool_result.evidence_labels)
            step_count += 1

            # Check deadline after tool execution
            if is_deadline_exceeded():
                trace.status = AgentStatus.TIMED_OUT.value
                trace.error = f"Execution deadline exceeded ({effective_timeout:.2f}s timeout) after tool execution."
                break

            # If tool failed on authorization, abort run immediately (security fail-closed)
            if not tool_result.success and tool_result.error_type in ("forbidden", "unauthorized"):
                trace.mark_failed(f"Authorization check failed during {tool_name}: {tool_result.error}")
                break

            # Feed observation back into conversation
            messages.append(self._prompt_builder.build_assistant_message(action))
            messages.append(
                self._prompt_builder.build_observation_message(step.step_index, tool_name, tool_result)
            )

        # 4. Handle step limit exhaustion
        if step_count >= effective_max_steps and trace.status == AgentStatus.RUNNING.value:
            trace.status = AgentStatus.MAX_STEPS_EXCEEDED.value
            trace.error = (
                f"Maximum execution steps ({effective_max_steps}) reached without receiving a final answer."
            )

        total_dur = (time.perf_counter() - start_time) * 1000
        trace.total_duration_ms = total_dur

        return AgentRunResult(
            trace=trace,
            final_answer=final_answer,
            status=trace.status,
            total_steps=trace.total_steps,
            duration_ms=total_dur,
            evidence_labels=list(dict.fromkeys(all_evidence_labels)),
        )
