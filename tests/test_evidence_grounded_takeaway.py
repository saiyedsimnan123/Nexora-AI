"""Regression tests for evidence-grounded research takeaway synthesis.

Verifies:
1. Grounding in cited evidence: Only claims supported by retrieved passages appear.
2. Direct attribution: Every factual claim is associated with its [E#] citation.
3. Tripartite distinction:
   - Reported findings
   - Interpretations derived from findings
   - Claims that cannot be established from evidence
4. Prohibition on unsupported metrics: Never introduces throughput, latency, accuracy,
   or benchmark scores unless explicitly present in the retrieved evidence.
5. Preservation of baselines, numerical values, units, and experimental context
   (e.g. 10,000x, 3x, trainable parameters, GPU memory, GPT-3 175B with Adam).
6. Qualified language: Uses 'the authors report' instead of implying independent verification.
7. Explicit insufficiency: Clearly states what cannot be concluded when evidence is missing.
8. Non-empty, concise, and meaningful output.
9. Preservation of paper-specific evidence grouping and citations.
"""

from __future__ import annotations

import pytest

from nexora.rag.models import AttributedSource, RAGResponse
from nexora.rag.pipeline import create_offline_rag_pipeline
from nexora.rag.takeaway import (
    PROHIBITED_UNSUPPORTED_METRICS,
    EvidenceGroundedTakeaway,
    generate_evidence_grounded_takeaway,
)


def _make_source(
    label: str,
    text: str,
    title: str = "LoRA: Low-Rank Adaptation of Large Language Models",
    paper_id: str = "p-hu-2021",
) -> AttributedSource:
    return AttributedSource(
        label=label,
        document_id="doc-01",
        chunk_id=f"chunk-{label.strip('[]')}",
        paper_id=paper_id,
        paper_title=title,
        score=0.95,
        text=text,
        text_excerpt=text[:100] + "...",
    )


# ---------------------------------------------------------------------------
# 1. Example Case: LoRA Parameter & Memory Reduction vs Unsupported Metrics
# ---------------------------------------------------------------------------

def test_lora_takeaway_supports_parameter_and_memory_without_unsupported_metrics():
    """Verify example behavior: LoRA evidence reporting 10,000x parameter reduction

    and 3x GPU memory reduction compared to full fine-tuning with Adam produces
    a takeaway explaining those two improvements without claiming unsubstantiated
    accuracy or throughput improvements.
    """
    lora_text = (
        "Parameter Efficiency: By freezing pre-trained model weights W0 and training only "
        "small factorized matrices A and B, LoRA reduces trainable parameters by 10,000x and "
        "GPU memory consumption by 3x compared to full fine-tuning of GPT-3 175B with Adam."
    )
    source = _make_source("[E1]", lora_text)

    takeaway: EvidenceGroundedTakeaway = generate_evidence_grounded_takeaway(
        query="What efficiency gains does LoRA provide?",
        sources=[source],
    )

    text = takeaway.takeaway_text
    lowered = text.lower()

    # 1. Qualified language
    assert "the authors" in lowered or "authors of lora" in lowered
    assert "report" in lowered

    # 2. Preserves numerical values, units, and comparison baseline
    assert "10,000" in text or "10000" in text
    assert "3x" in text or "3 times" in text
    assert "parameter" in lowered
    assert "memory" in lowered
    assert "[E1]" in text

    # 3. Strictly does NOT claim throughput or accuracy improvements
    # Check that throughput or accuracy are NOT asserted as improvements
    assert "throughput improvement" not in lowered
    assert "accuracy improvement" not in lowered
    assert "latency improvement" not in lowered

    # 4. Explicitly declares unestablished claims
    assert len(takeaway.unestablished_claims) > 0
    unestablished = " ".join(takeaway.unestablished_claims).lower()
    assert "throughput" in unestablished or "accuracy" in unestablished or "latency" in unestablished
    assert "cannot be established" in unestablished or "not reported" in unestablished

    # 5. Conclusion classification
    assert takeaway.conclusion == "Demonstrated improvement"


def test_no_unsupported_metrics_introduced_when_absent_from_evidence():
    """Verify that unsupported metrics (throughput, latency, task accuracy) are NEVER

    claimed when the evidence only discusses architectural decomposition.
    """
    decomp_text = (
        "Low-Rank Decomposition: Large models possess low intrinsic dimension during adaptation. "
        "Parameter updates are represented by low-rank matrix decomposition Delta W = B * A where rank r << min(d, k)."
    )
    source = _make_source("[E1]", decomp_text)

    takeaway = generate_evidence_grounded_takeaway("How does LoRA decompose weights?", [source])
    lowered = takeaway.takeaway_text.lower()

    # Prohibited claims must not be asserted as findings
    for metric in ("throughput", "speedup", "accuracy", "bleu", "latency"):
        # If the metric word appears at all, it MUST only be in an unestablished declaration
        if metric in lowered:
            assert "cannot be established" in lowered or "not reported" in lowered


def test_flashattention_wallclock_speedup_accurately_grounded():
    """When wall-clock speedup IS explicitly in the text, it is reported with exact numbers,

    while unmentioned parameter reduction and accuracy are flagged as unestablished.
    """
    flash_text = (
        "Speedup Results: FlashAttention yields 2-4x wall-clock speedup on BERT and GPT-2 models "
        "while scaling to 64k sequences with strictly linear memory footprint relative to sequence length."
    )
    source = _make_source(
        "[E1]",
        flash_text,
        title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        paper_id="p-dao-2022",
    )

    takeaway = generate_evidence_grounded_takeaway("What speedup does FlashAttention achieve?", [source])
    text = takeaway.takeaway_text
    lowered = text.lower()

    # Reported metrics are present
    assert "2-4x" in text
    assert "wall-clock" in lowered
    assert "linear memory" in lowered
    assert "[E1]" in text

    # Unestablished metrics are flagged
    unestablished = " ".join(takeaway.unestablished_claims).lower()
    assert "throughput" in unestablished or "accuracy" in unestablished or "cannot be established" in unestablished


# ---------------------------------------------------------------------------
# 2. Tripartite Distinction: Findings, Interpretations, Unestablished Claims
# ---------------------------------------------------------------------------

def test_tripartite_distinction_structure():
    """Verify reported_findings, interpretations, and unestablished_claims are distinctly populated."""
    text = (
        "Parameter Efficiency: By freezing pre-trained model weights W0 and training only small factorized matrices A and B, "
        "LoRA reduces trainable parameters by 10,000x and GPU memory consumption by 3x on large foundation models."
    )
    source = _make_source("[E1]", text)

    takeaway = generate_evidence_grounded_takeaway("Summarize LoRA efficiency.", [source])

    # 1. Reported findings must have citation and qualified language
    assert len(takeaway.reported_findings) > 0
    assert "[E1]" in takeaway.reported_findings[0]
    assert "the authors" in takeaway.reported_findings[0].lower()

    # 2. Interpretations must be clearly prefixed as interpretation
    assert len(takeaway.interpretations) > 0
    assert "interpretation" in takeaway.interpretations[0].lower()

    # 3. Unestablished claims must explicitly declare missing evidence
    assert len(takeaway.unestablished_claims) > 0
    assert "cannot be established" in takeaway.unestablished_claims[0].lower() or "not reported" in takeaway.unestablished_claims[0].lower()


# ---------------------------------------------------------------------------
# 3. Insufficient and Missing Evidence Handling
# ---------------------------------------------------------------------------

def test_empty_sources_yields_insufficient_evidence():
    """Empty source list produces an explicit insufficient evidence takeaway."""
    takeaway = generate_evidence_grounded_takeaway("What is the throughput of LoRA on TPUs?", [])

    assert takeaway.conclusion == "Insufficient evidence"
    assert "insufficient" in takeaway.takeaway_text.lower() or "cannot be established" in takeaway.takeaway_text.lower()
    assert len(takeaway.takeaway_text.strip()) > 0
    assert len(takeaway.unestablished_claims) > 0


def test_explicit_insufficient_flag():
    """When is_insufficient=True is passed, takeaway declares lack of empirical evidence."""
    source = _make_source("[E1]", "Irrelevant passing mention.")
    takeaway = generate_evidence_grounded_takeaway(
        "Does LoRA increase inference throughput?",
        [source],
        is_insufficient=True,
    )

    assert takeaway.conclusion == "Insufficient evidence"
    assert "insufficient" in takeaway.takeaway_text.lower()


# ---------------------------------------------------------------------------
# 4. Conflicting Evidence and Trade-offs
# ---------------------------------------------------------------------------

def test_tradeoffs_and_conflicts_classified_correctly():
    """When excerpts report trade-offs between memory and complexity, conclusion reflects trade-offs."""
    s1 = _make_source(
        "[E1]",
        "Standard self-attention exhibits quadratic O(n^2) computational and memory complexity with sequence length.",
        title="Attention Is All You Need",
        paper_id="p-vaswani-2017",
    )
    s2 = _make_source(
        "[E2]",
        "Trade-off: Exchanging additional recomputation in the backward pass enables linear memory footprint.",
        title="FlashAttention",
        paper_id="p-dao-2022",
    )

    takeaway = generate_evidence_grounded_takeaway("Compare attention memory trade-offs.", [s1, s2])
    assert takeaway.conclusion in ("Conflicting evidence", "Demonstrated improvement")
    assert "[E1]" in takeaway.takeaway_text or "[E2]" in takeaway.takeaway_text


# ---------------------------------------------------------------------------
# 5. Full End-to-End Pipeline Integration
# ---------------------------------------------------------------------------

def test_pipeline_generates_grounded_takeaway_for_lora():
    """End-to-end integration test: RAG pipeline on LoRA query returns structured

    takeaway without unsupported throughput or accuracy claims.
    """
    pipeline = create_offline_rag_pipeline()
    response: RAGResponse = pipeline.answer("What parameter reduction does LoRA achieve?")

    assert response.status == "success"
    assert response.is_grounded is True
    assert len(response.sources) > 0

    # Verify takeaway field exists on RAGResponse
    assert hasattr(response, "takeaway")
    assert response.takeaway is not None
    assert len(response.takeaway.strip()) > 0

    takeaway_lower = response.takeaway.lower()
    # Qualified language
    assert "the authors" in takeaway_lower or "report" in takeaway_lower
    # Supported claims
    assert "10,000" in response.takeaway or "parameter" in takeaway_lower

    # Verify unestablished claims in comparison_of_findings
    assert response.comparison_of_findings is not None
    unest = response.comparison_of_findings.get("unestablished_claims", [])
    assert len(unest) > 0

    # Ensure to_dict contains takeaway
    d = response.to_dict()
    assert "takeaway" in d
    assert d["takeaway"] == response.takeaway
