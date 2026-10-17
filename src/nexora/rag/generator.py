"""Answer generators for the end-to-end RAG pipeline.

Provides both offline deterministic mock generation and live LLM integration:
- AnswerGenerator: Protocol for generation components.
- DeterministicMockGenerator: 100% offline generator producing verifiable, grounded answers
  citing retrieved evidence without network calls, SDKs, or API keys.
- LLMAnswerGenerator: Adapter delegating to LLMService (e.g. OpenAI or Gemini).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from nexora.llm.prompt import ResearchPromptBuilder
from nexora.retrieval.evidence import EvidenceContext, filter_unsupplied_citations


@dataclass(frozen=True)
class GeneratedAnswer:
    """Raw output from an AnswerGenerator prior to pipeline verification."""

    text: str
    model: str
    is_mock: bool
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.is_mock, bool):
            raise TypeError("is_mock must be a boolean")


@runtime_checkable
class AnswerGenerator(Protocol):
    """Protocol for RAG answer generation providers."""

    @property
    def is_mock(self) -> bool:
        """True if the generator produces mock/offline synthetic responses."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier of the model or generator strategy."""
        ...

    def generate(
        self,
        query: str,
        context: EvidenceContext,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GeneratedAnswer:
        """Produce an answer grounded in the supplied EvidenceContext."""
        ...


class DeterministicMockGenerator:
    """100% offline, deterministic research answer generator.

    Produces clear, grounded answers that attribute key claims to retrieved [E#]
    citations, enabling complete testability in CI and offline development.
    """

    def __init__(self, model_name: str = "offline-deterministic-mock-v1") -> None:
        self._model_name = model_name

    @property
    def is_mock(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._model_name

    def _extract_core_sentence(self, text: str, query: str = "") -> str:
        """Extract the most substantive and query-relevant sentence from evidence text."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return text.strip()

        if query:
            stopwords = {
                "what", "is", "the", "how", "does", "do", "of", "in", "on", "a", "an",
                "and", "to", "for", "with", "by", "are", "from", "that", "this", "or",
            }
            query_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower())) - stopwords
            if query_words:
                best_sentence = sentences[0]
                best_score = -1
                for s in sentences:
                    s_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", s.lower()))
                    overlap = len(query_words & s_words)
                    if overlap > best_score:
                        best_score = overlap
                        best_sentence = s
                if best_score > 0:
                    return best_sentence

        return sentences[0]

    def generate(
        self,
        query: str,
        context: EvidenceContext,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GeneratedAnswer:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(context, EvidenceContext):
            raise TypeError("context must be an EvidenceContext")

        effective_model = model or self._model_name

        if not context.items:
            return GeneratedAnswer(
                text="I could not find sufficient evidence in the indexed research papers to answer this question.",
                model=effective_model,
                is_mock=True,
                usage={"prompt_tokens": 10, "completion_tokens": 18, "total_tokens": 28},
            )

        # Select active items: always include top retrieved candidate, plus other query-relevant candidates
        stopwords = {
            "what", "is", "the", "how", "does", "do", "of", "in", "on", "a", "an",
            "and", "to", "for", "with", "by", "are", "from", "that", "this", "or",
        }
        query_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower())) - stopwords

        if query_words and len(context.items) > 1:
            scored_candidates: list[tuple[int, int, EvidenceItem]] = []
            for rank, item in enumerate(context.items):
                item_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", (item.text + " " + (item.metadata.get("title") or "")).lower()))
                overlap = len(query_words & item_words)
                scored_candidates.append((overlap, rank, item))

            # Always preserve the top retrieved candidate
            selected = [context.items[0]]
            selected_ids = {context.items[0].chunk_id}

            # Append candidates with substantial query overlap
            max_overlap = max(c[0] for c in scored_candidates)
            threshold = max(2, max_overlap // 2) if max_overlap >= 2 else 1

            for overlap, rank, item in scored_candidates[1:]:
                if overlap >= threshold and item.chunk_id not in selected_ids:
                    selected.append(item)
                    selected_ids.add(item.chunk_id)
                if len(selected) >= 3:
                    break

            active_items = selected
        else:
            active_items = context.items[:3]

        # Synthesize grounded answer citing retrieved evidence items
        statements: list[str] = []
        for item in active_items:
            paper_title = item.metadata.get("title") or item.document_id
            core_finding = self._extract_core_sentence(item.text, query)
            statements.append(f"According to {paper_title} {item.label}, {core_finding}")

        synthesized_text = " ".join(statements)
        total_words = len(synthesized_text.split())

        return GeneratedAnswer(
            text=synthesized_text,
            model=effective_model,
            is_mock=True,
            usage={
                "prompt_tokens": len(query.split()) + sum(len(i.text.split()) for i in active_items),
                "completion_tokens": total_words,
                "total_tokens": total_words + 50,
            },
        )



class LLMAnswerGenerator:
    """Delegates answer generation to an LLMService with citation verification."""

    def __init__(
        self,
        llm_service: Any,
        *,
        model_name: str = "gpt-4o-mini",
        prompt_builder: ResearchPromptBuilder | None = None,
        filter_invalid_citations: bool = True,
    ) -> None:
        if not callable(getattr(llm_service, "generate", None)):
            raise TypeError("llm_service must provide generate(messages, ...)")
        self._llm = llm_service
        self._model_name = model_name
        self._prompt_builder = prompt_builder or ResearchPromptBuilder()
        self._filter_invalid_citations = filter_invalid_citations

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        query: str,
        context: EvidenceContext,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GeneratedAnswer:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(context, EvidenceContext):
            raise TypeError("context must be an EvidenceContext")

        effective_model = model or self._model_name

        if not context.items:
            return GeneratedAnswer(
                text="I could not find sufficient evidence in the indexed research papers to answer this question.",
                model=effective_model,
                is_mock=False,
                usage={"prompt_tokens": 0, "completion_tokens": 18, "total_tokens": 18},
            )

        messages = self._prompt_builder.build(query, context)
        response = self._llm.generate(
            messages,
            model=effective_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        text = response.text
        if self._filter_invalid_citations:
            text = filter_unsupplied_citations(text, len(context.items))

        return GeneratedAnswer(
            text=text,
            model=response.model or effective_model,
            is_mock=False,
            usage=response.usage,
            raw=response.raw,
        )
