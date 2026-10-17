"""Prompt builder for cross-paper comparison and evidence synthesis."""

from __future__ import annotations

import json
from typing import Any

from nexora.llm.context import EvidenceContextFormatter
from nexora.llm.models import LLMMessage
from nexora.retrieval.evidence import EvidenceContext

CROSS_PAPER_COMPARISON_SYSTEM_INSTRUCTION = (
    "You are an expert scientific literature synthesis and cross-paper comparison system.\n"
    "Your objective is to compare two or more research papers and construct a rigorous 7-dimension comparison matrix:\n"
    "1. methods: architectures, algorithms, mechanisms, model techniques\n"
    "2. datasets: benchmarks, training corpora, evaluation sets, data domains\n"
    "3. baselines: prior models, competitor systems, ablation variants\n"
    "4. metrics: quantitative evaluation targets (e.g. accuracy, perplexity, latency, efficiency)\n"
    "5. results: measured findings, empirical results, performance numbers\n"
    "6. limitations: compute constraints, assumptions, bottlenecks, weaknesses\n"
    "7. comparability: thorough assessment of whether the experimental setups, datasets, baselines, and metrics allow direct, fair head-to-head comparison.\n\n"
    "INTEGRITY AND EVIDENCE RULES:\n"
    "1. Base findings ONLY on the provided evidence for each respective paper. Never invent facts.\n"
    "2. Evidence isolation: Never attribute one paper's findings or chunks to another paper.\n"
    "3. Cite supporting evidence chunks for each paper using '[E#]' tags corresponding to that paper's chunks.\n"
    "4. If a dimension or detail is not reported in a paper's evidence, explicitly set status to 'not_reported' and summary to 'Not reported in the provided evidence.' Do not guess.\n"
    "5. Comparability and Conclusion Rules:\n"
    "   - If papers evaluate on completely disparate benchmarks without overlapping datasets/metrics or direct baseline comparison, mark comparability_status as 'incomparable' or 'partially_comparable'.\n"
    "   - NEVER infer superiority from one paper alone. If Paper A has results on Benchmark X and Paper B did not evaluate on Benchmark X, do NOT conclude Paper A outperforms Paper B.\n"
    "   - Overall conclusion MUST be classified strictly as one of:\n"
    "     * 'Demonstrated improvement': Directly comparable evaluations on common benchmarks with reproducible or substantiated metrics showing one approach outperforms baselines across shared metrics without conflicting regressions.\n"
    "     * 'Limited evidence': Only narrow, single-benchmark, or non-overlapping evaluation setups; positive claims made but insufficient general head-to-head evidence.\n"
    "     * 'Conflicting evidence': One paper reports gains while another or alternative metrics/baselines report regression, parity, or trade-offs.\n"
    "     * 'Insufficient evidence': Missing metrics, lack of reported evaluation data, or incomparable setups where no conclusion can be drawn.\n\n"
    "REQUIRED SCHEMA:\n"
    "Return a valid JSON object matching:\n"
    "{\n"
    '  "matrix": {\n'
    '    "methods": {\n'
    '      "comparison_summary": "...",\n'
    '      "key_differences": ["..."],\n'
    '      "key_similarities": ["..."],\n'
    '      "per_paper": {\n'
    '        "<paper_id>": {"summary": "...", "status": "reported|not_reported|uncertain", "evidence_labels": ["[E1]"], "details": {...}}\n'
    '      }\n'
    '    },\n'
    '    "datasets": { ... },\n'
    '    "baselines": { ... },\n'
    '    "metrics": { ... },\n'
    '    "results": { ... },\n'
    '    "limitations": { ... },\n'
    '    "comparability": { ... }\n'
    '  },\n'
    '  "comparability_status": "comparable|partially_comparable|incomparable|insufficient_evidence",\n'
    '  "conclusion": "Demonstrated improvement|Limited evidence|Conflicting evidence|Insufficient evidence",\n'
    '  "conclusion_reasoning": "...",\n'
    '  "overall_summary": "..."\n'
    "}\n\n"
    "Output ONLY the JSON object."
)


class CrossPaperComparisonPromptBuilder:
    """Builds prompt messages for cross-paper comparison and 7-dimension synthesis."""

    def __init__(self, *, system_instruction: str | None = None) -> None:
        self._system_instruction = (
            CROSS_PAPER_COMPARISON_SYSTEM_INSTRUCTION
            if system_instruction is None
            else system_instruction
        )
        self._formatter = EvidenceContextFormatter()

    def build(
        self,
        paper_ids: list[str],
        per_paper_contexts: dict[str, EvidenceContext],
        paper_titles: dict[str, str | None],
        *,
        query: str | None = None,
    ) -> list[LLMMessage]:
        sections: list[str] = []

        if query and query.strip():
            sections.append(f"RESEARCH QUESTION / COMPARISON FOCUS:\n{query.strip()}\n")

        sections.append("PAPERS AND RETRIEVED EVIDENCE:")

        for pid in paper_ids:
            title = paper_titles.get(pid) or "Untitled"
            ctx = per_paper_contexts.get(pid)
            if ctx is None or not ctx.items:
                evidence_text = "No evidence was retrieved for this paper."
            else:
                formatted_items: list[str] = []
                for num, item in enumerate(ctx.items, start=1):
                    page_info = f", Page {item.page_number}" if item.page_number else ""
                    formatted_items.append(
                        f"[E{num}] (Document: {item.document_id}, Chunk: {item.chunk_id}{page_info}):\n{item.text}"
                    )
                evidence_text = "\n\n".join(formatted_items)

            paper_block = (
                f"--- PAPER ID: {pid} ---\n"
                f"Title: {title}\n"
                f"Evidence Chunks:\n{evidence_text}\n"
            )
            sections.append(paper_block)

        sections.append(
            "TASK:\n"
            "Build the structured 7-dimension comparison matrix (methods, datasets, baselines, metrics, "
            "results, limitations, comparability), determine comparability_status, and classify the overall "
            "conclusion strictly into one of ('Demonstrated improvement', 'Limited evidence', 'Conflicting evidence', "
            "'Insufficient evidence'). Output ONLY valid JSON."
        )

        user_content = "\n\n".join(sections)

        return [
            LLMMessage(role="system", content=self._system_instruction),
            LLMMessage(role="user", content=user_content),
        ]
