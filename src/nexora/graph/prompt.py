"""Prompt builders and templates for knowledge graph extraction."""

from __future__ import annotations

from typing import Sequence

from nexora.llm.models import LLMMessage
from nexora.retrieval.evidence import EvidenceContext

GRAPH_EXTRACTION_SYSTEM_INSTRUCTION = """You are a rigorous scientific knowledge extraction specialist.
Your task is to extract canonical scientific entities and grounded relationships from the provided research paper excerpts.

CRITICAL RULES:
1. ENTITY TYPES:
   - "concept": Core scientific topics, phenomena, paradigms (e.g., "Self-Attention", "Overfitting").
   - "method": Specific algorithms, architectures, systems, techniques (e.g., "Transformer", "FlashAttention-2", "LoRA").
   - "dataset": Datasets, benchmarks, corpora (e.g., "WMT 2014 English-to-German", "GLUE", "ImageNet").
   - "metric": Evaluation metrics, scores (e.g., "BLEU-4", "Accuracy", "Perplexity").
   - "other": Other domain-specific scientific entities.

2. RELATIONSHIPS:
   - Identify relationships between the target research paper and entities, or between two entities.
   - Permitted relationship types:
     * "proposes": Paper introduces/proposes a new method or concept.
     * "uses_dataset": Paper trains, fine-tunes, or evaluates on a dataset.
     * "evaluates_method": Paper implements, benchmarks, or evaluates an existing method.
     * "discusses": Paper analyzes, surveys, or discusses a concept without proposing it.
     * "reports_metric": Paper reports performance on a specific metric.
     * "extends": Paper or method builds upon / extends another paper or method.
     * "contradicts": Paper challenges or contradicts conclusions of another.
     * "relates_to": General relationship between two concepts/methods.

3. STRICT EVIDENCE GROUNDING:
   - Every extracted relationship MUST cite its source chunk using the exact citation label (e.g. "[cite: 0]").
   - Include a concise, direct quote from that chunk in "evidence_quote".
   - Assign a "confidence" float between 0.0 and 1.0.
   - Never extrapolate or claim a paper uses a method unless the excerpt explicitly states it was used.

4. OUTPUT FORMAT:
   - Output ONLY valid JSON matching this schema:
   {
     "entities": [
       {
         "name": "Exact Name",
         "entity_type": "concept|method|dataset|metric|other",
         "description": "Brief summary",
         "aliases": ["Alias 1"]
       }
     ],
     "relationships": [
       {
         "source_name": "Paper Title or Source Entity",
         "source_type": "paper|entity",
         "target_name": "Target Entity or Paper",
         "target_type": "paper|entity",
         "relationship_type": "proposes|uses_dataset|evaluates_method|discusses|reports_metric|extends|relates_to",
         "evidence_label": "[cite: 0]",
         "evidence_quote": "Direct text quote supporting this link",
         "confidence": 0.95
       }
     ]
   }
"""


class GraphExtractionPromptBuilder:
    """Builds prompt messages for scientific knowledge graph extraction."""

    def __init__(self, *, system_instruction: str | None = None) -> None:
        self._system_instruction = (
            GRAPH_EXTRACTION_SYSTEM_INSTRUCTION
            if system_instruction is None
            else system_instruction
        )

    def build(
        self,
        paper_id: str,
        context: EvidenceContext,
        *,
        title: str | None = None,
        abstract: str | None = None,
    ) -> list[LLMMessage]:
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise TypeError("paper_id must be a non-empty string")
        if not isinstance(context, EvidenceContext):
            raise TypeError("context must be an EvidenceContext")

        title_header = f"Paper Title: {title}\n" if title else ""
        abstract_header = f"Abstract: {abstract}\n\n" if abstract else "\n"

        chunks_text_parts: list[str] = []
        for i, item in enumerate(context.items):
            page_info = f" (Page {item.page_number})" if item.page_number else ""
            chunks_text_parts.append(
                f"[cite: {i}]{page_info}:\n{item.text.strip()}\n"
            )

        evidence_formatted = "\n".join(chunks_text_parts)

        user_content = (
            f"TARGET RESEARCH PAPER:\n"
            f"Paper ID: {paper_id}\n"
            f"{title_header}"
            f"{abstract_header}"
            f"RETRIEVED EVIDENCE CHUNKS:\n"
            f"{evidence_formatted}\n\n"
            f"TASK: Extract all grounded scientific entities and relationships supported by the above evidence. "
            f"Output ONLY valid JSON."
        )

        return [
            LLMMessage(role="system", content=self._system_instruction),
            LLMMessage(role="user", content=user_content),
        ]
