"""Research query orchestration: query -> EvidenceContext -> ResearchAnswer."""

from __future__ import annotations

from typing import Any

from nexora.research.models import ResearchAnswer


class ResearchQueryService:
    """Thin orchestrator over two already-built services.

    ``evidence_builder.build(query)`` already runs retrieval and quality
    filtering internally (EvidenceContextBuilder -> RetrievalQualityService ->
    RetrievalService), so this class only chains the builder to the answer
    service. The number of retrieved candidates is therefore set by the
    RetrievalQualityPolicy, not by this class.
    """

    def __init__(self, evidence_builder: Any, answer_service: Any) -> None:
        if not callable(getattr(evidence_builder, "build", None)):
            raise TypeError("evidence_builder must provide build(query)")
        if not callable(getattr(answer_service, "answer", None)):
            raise TypeError("answer_service must provide answer(query, context, ...)")
        self._evidence_builder = evidence_builder
        self._answer_service = answer_service

    def ask(
        self,
        query: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ResearchAnswer:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        context = self._evidence_builder.build(query)
        return self._answer_service.answer(
            query,
            context,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
