"""Thin orchestration: EvidenceContext -> prompt -> LLMService -> ResearchAnswer."""

from __future__ import annotations

from typing import Any

from nexora.llm.prompt import ResearchPromptBuilder
from nexora.research.models import ResearchAnswer
from nexora.retrieval.evidence import EvidenceContext, filter_unsupplied_citations


class ResearchAnswerService:
    def __init__(
        self,
        llm_service: Any,
        *,
        prompt_builder: ResearchPromptBuilder | None = None,
        filter_invalid_citations: bool = True,
    ) -> None:
        if not callable(getattr(llm_service, "generate", None)):
            raise TypeError("llm_service must provide generate(messages, ...)")
        if prompt_builder is None:
            prompt_builder = ResearchPromptBuilder()
        elif not isinstance(prompt_builder, ResearchPromptBuilder):
            raise TypeError("prompt_builder must be a ResearchPromptBuilder")
        self._llm = llm_service
        self._prompt_builder = prompt_builder
        self._filter_invalid_citations = filter_invalid_citations

    def answer(
        self,
        query: str,
        context: EvidenceContext,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ResearchAnswer:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if not isinstance(context, EvidenceContext):
            raise TypeError("context must be an EvidenceContext")

        messages = self._prompt_builder.build(query, context)
        response = self._llm.generate(
            messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        text = response.text
        if self._filter_invalid_citations:
            text = filter_unsupplied_citations(text, len(context.items))

        return ResearchAnswer(
            text=text,
            query=query,
            model=response.model,
            usage=response.usage,
            evidence_count=len(context.items),
            raw=response.raw,
        )
