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


# ---------------------------------------------------------------------------
# 6. Critical Evidence Isolation & Leakage Prevention Tests
# ---------------------------------------------------------------------------

def test_lora_query_with_lora_and_flashattention_both_retrieved_excludes_flashattention():
    """Observed production-preview bug regression test:

    Research query:
    'What parameter reduction and memory efficiency does LoRA provide compared with full fine-tuning?'

    Even when FlashAttention chunks occupy [E1] and [E2] in retrieval context,
    the takeaway MUST isolate LoRA evidence exclusively, cite [E3], and strictly
    exclude FlashAttention.
    """
    s1_flash = AttributedSource(
        label="[E1]",
        document_id="doc_arxiv_2205.14135",
        chunk_id="chunk_2205_14135_01",
        paper_id="arxiv_2205.14135",
        paper_title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        score=0.92,
        text="Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length.",
        text_excerpt="Transformers are slow and memory-hungry...",
    )
    s2_flash = AttributedSource(
        label="[E2]",
        document_id="doc_arxiv_2205.14135",
        chunk_id="chunk_2205_14135_02",
        paper_id="arxiv_2205.14135",
        paper_title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        score=0.88,
        text="We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention.",
        text_excerpt="We analyze the IO complexity...",
    )
    s3_lora = AttributedSource(
        label="[E3]",
        document_id="doc_arxiv_2106.09685",
        chunk_id="chunk_2106_09685_02",
        paper_id="arxiv_2106.09685",
        paper_title="LoRA: Low-Rank Adaptation of Large Language Models",
        score=0.81,
        text="Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.",
        text_excerpt="Compared to GPT-3 175B fine-tuned with Adam...",
    )

    query = "What parameter reduction and memory efficiency does LoRA provide compared with full fine-tuning?"
    takeaway = generate_evidence_grounded_takeaway(query, [s1_flash, s2_flash, s3_lora])

    expected_takeaway = (
        "The authors of LoRA report that, compared with GPT-3 175B full fine-tuning using Adam, "
        "LoRA reduces trainable parameters by 10,000 times and GPU memory requirements by 3 times [E3]. "
        "These results demonstrate reported efficiency improvements under the stated comparison configuration. "
        "They do not establish universal improvements across all models or tasks."
    )

    assert takeaway.takeaway_text == expected_takeaway
    # Strict isolation: Zero FlashAttention leakage
    assert "flashattention" not in takeaway.takeaway_text.lower()
    assert "[E1]" not in takeaway.takeaway_text
    assert "[E2]" not in takeaway.takeaway_text
    assert "[E3]" in takeaway.takeaway_text


def test_transformer_query_with_multiple_papers_retrieved():
    """Verify Transformer query excludes LoRA and FlashAttention passages."""
    s1_lora = _make_source("[E1]", "LoRA freezes pre-trained weights W0.", title="LoRA", paper_id="p-lora")
    s2_flash = _make_source("[E2]", "FlashAttention tiles blocks in SRAM.", title="FlashAttention", paper_id="p-flash")
    s3_trans = _make_source(
        "[E3]",
        "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task.",
        title="Attention Is All You Need",
        paper_id="p-trans",
    )

    query = "What BLEU score does Attention Is All You Need achieve?"
    takeaway = generate_evidence_grounded_takeaway(query, [s1_lora, s2_flash, s3_trans])

    assert "28.4 BLEU" in takeaway.takeaway_text
    assert "[E3]" in takeaway.takeaway_text
    assert "LoRA" not in takeaway.takeaway_text
    assert "FlashAttention" not in takeaway.takeaway_text


def test_irrelevant_distractor_evidence_strictly_excluded():
    """Verify distractor evidence sharing superficial terms is filtered out."""
    s1_distractor = AttributedSource(
        label="[E1]",
        document_id="doc_distractor",
        chunk_id="chunk_distractor_01",
        paper_id="p-distractor-2020",
        paper_title="Biological and Hardware Near-Match Distractors",
        score=0.99,
        text="Biological Attention in Primate Vision: Studies visual saccades and selective attention in macaque visual cortex V1 and V4.",
        text_excerpt="Biological Attention in Primate Vision...",
    )
    s2_lora = _make_source(
        "[E2]",
        "Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        paper_id="p-lora",
    )

    query = "What parameter reduction does LoRA achieve?"
    takeaway = generate_evidence_grounded_takeaway(query, [s1_distractor, s2_lora])

    assert "10,000 times" in takeaway.takeaway_text
    assert "[E2]" in takeaway.takeaway_text
    assert "primate" not in takeaway.takeaway_text.lower()
    assert "saccades" not in takeaway.takeaway_text.lower()


def test_multiple_papers_in_comparative_query_maintain_isolated_boundaries():
    """When query is explicitly comparative, each paper is represented separately with its own citation."""
    s1_lora = _make_source(
        "[E1]",
        "Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        paper_id="p-lora",
    )
    s2_flash = _make_source(
        "[E2]",
        "Speedup Results: FlashAttention yields 2-4x wall-clock speedup on BERT and GPT-2 models while scaling to 64k sequences with strictly linear memory footprint relative to sequence length.",
        title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        paper_id="p-flash",
    )

    query = "Compare the efficiency of LoRA and FlashAttention"
    takeaway = generate_evidence_grounded_takeaway(query, [s1_lora, s2_flash])

    assert "The authors of LoRA" in takeaway.takeaway_text
    assert "[E1]" in takeaway.takeaway_text
    assert "The authors of FlashAttention" in takeaway.takeaway_text
    assert "[E2]" in takeaway.takeaway_text
    assert "universal" in takeaway.takeaway_text


def test_citation_id_consistency_across_answer_and_takeaway():
    """Verify original citation IDs are preserved without renumbering."""
    s = _make_source(
        "[E7]",
        "Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.",
    )
    takeaway = generate_evidence_grounded_takeaway("How does LoRA reduce parameters?", [s])

    assert "[E7]" in takeaway.takeaway_text
    assert "[E1]" not in takeaway.takeaway_text


def test_offline_pipeline_full_integration_excludes_leakage():
    """Verify that full create_offline_rag_pipeline() executes the LoRA query with zero FlashAttention leakage."""
    pipeline = create_offline_rag_pipeline()
    res = pipeline.answer("What parameter reduction and memory efficiency does LoRA provide compared with full fine-tuning?")

    assert res.status == "success"
    # Takeaway must NOT contain FlashAttention
    assert "flashattention" not in res.takeaway.lower()
    assert "the authors of lora" in res.takeaway.lower()
    assert "universal" in res.takeaway.lower()

    # Evidence by paper must only contain LoRA
    paper_titles = [p["paper_title"].lower() for p in res.evidence_by_paper]
    assert all("flashattention" not in pt for pt in paper_titles)


# ---------------------------------------------------------------------------
# 7. Quantitative Finding Prioritization & Anti-Hallucination Regression Tests
# ---------------------------------------------------------------------------

def test_lora_takeaway_prioritizes_quantitative_e3_over_motivation_e1():
    """Regression test for E1 motivation repetition vs E3 quantitative findings:

    When both the general motivation chunk [E1] and the quantitative findings chunk [E3]
    are retrieved:
    1. The takeaway directly answers the user's inquiry regarding parameter reduction and memory efficiency.
    2. It includes both quantitative findings: 10,000x fewer trainable parameters and 3x lower GPU memory requirement.
    3. It preserves comparison context: GPT-3 175B full fine-tuning with Adam.
    4. It cites [E3] for both numerical findings and does NOT attribute them to [E1].
    5. It does not repeat only the general motivation ('full fine-tuning becomes less feasible').
    6. It does not claim unsupported throughput, latency, or accuracy improvements.
    7. Single-paper evidence isolation is maintained.
    """
    s1_motivation = AttributedSource(
        label="[E1]",
        document_id="doc_arxiv_2106.09685",
        chunk_id="chunk_2106_09685_01",
        paper_id="arxiv_2106.09685",
        paper_title="LoRA: Low-Rank Adaptation of Large Language Models",
        score=0.91,
        text=(
            "An important paradigm of natural language processing consists of large-scale pre-training on general domain data "
            "and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model "
            "parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, "
            "each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained "
            "model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly "
            "reducing the number of trainable parameters for downstream tasks."
        ),
        text_excerpt="An important paradigm of natural language processing...",
    )
    s3_quantitative = AttributedSource(
        label="[E3]",
        document_id="doc_arxiv_2106.09685",
        chunk_id="chunk_2106_09685_02",
        paper_id="arxiv_2106.09685",
        paper_title="LoRA: Low-Rank Adaptation of Large Language Models",
        score=0.82,
        text=(
            "Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times "
            "and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on "
            "RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, "
            "unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency "
            "in language model adaptation, which sheds light on the efficacy of LoRA."
        ),
        text_excerpt="Compared to GPT-3 175B fine-tuned with Adam...",
    )

    query = "What parameter reduction and memory efficiency does LoRA provide compared with full fine-tuning?"
    takeaway = generate_evidence_grounded_takeaway(query, [s1_motivation, s3_quantitative])

    text = takeaway.takeaway_text
    lowered = text.lower()

    # 1. Directly answers inquiry
    assert "the authors of lora report" in lowered

    # 2. Both quantitative findings included
    assert "10,000 times" in text or "10,000x" in text or "10000" in text
    assert "3 times" in text or "3x" in text
    assert "parameter" in lowered
    assert "gpu memory" in lowered or "memory requirement" in lowered

    # 3. Preserves comparison context
    assert "gpt-3 175b" in lowered
    assert "adam" in lowered

    # 4. Attributed to [E3], not misattributed to [E1]
    assert "[E3]" in text
    # The primary quantitative claim in takeaway_text must cite [E3]
    assert "3 times [E3]" in text or "requirements by 3 times [E3]" in text or "Adam, LoRA reduces trainable parameters by 10,000 times and GPU memory requirements by 3 times [E3]" in text

    # 5. Does not merely repeat general motivation from [E1]
    assert "full fine-tuning becomes less feasible [e1]" not in lowered
    assert "prohibitively expensive [e1]" not in lowered

    # 6. Tripartite distinction
    assert len(takeaway.reported_findings) >= 1
    assert "[E3]" in takeaway.reported_findings[0]
    assert len(takeaway.interpretations) >= 1
    assert len(takeaway.unestablished_claims) >= 1

    # 7. No unsupported metrics claimed as improvements
    assert "throughput improvement" not in lowered
    assert "accuracy improvement" not in lowered
    assert "latency improvement" not in lowered


def test_takeaway_does_not_invent_quantitative_findings_when_evidence_is_absent():
    """Regression test ensuring takeaway does not hallucinate 10,000x or 3x when evidence lacks them:

    When evidence consists exclusively of a motivation passage lacking numerical reductions:
    1. Takeaway does not claim 10,000x parameter reduction or 3x GPU memory reduction.
    2. Takeaway does not claim comparison with Adam when Adam is absent.
    3. Unestablished claims explicitly note that specific quantitative reductions or memory metrics cannot be established.
    """
    s1_motivation_only = AttributedSource(
        label="[E1]",
        document_id="doc_arxiv_2106.09685",
        chunk_id="chunk_2106_09685_01",
        paper_id="arxiv_2106.09685",
        paper_title="LoRA: Low-Rank Adaptation of Large Language Models",
        score=0.91,
        text=(
            "An important paradigm of natural language processing consists of large-scale pre-training on general domain data "
            "and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model "
            "parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, "
            "each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained "
            "model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly "
            "reducing the number of trainable parameters for downstream tasks."
        ),
        text_excerpt="An important paradigm of natural language processing...",
    )

    query = "What parameter reduction and memory efficiency does LoRA provide compared with full fine-tuning?"
    takeaway = generate_evidence_grounded_takeaway(query, [s1_motivation_only])

    text = takeaway.takeaway_text
    lowered = text.lower()

    # Must NOT invent numbers or comparison algorithm not present in the excerpt
    assert "10,000" not in text
    assert "10000" not in text
    assert "3x" not in lowered
    assert "3 times" not in lowered
    assert "adam" not in lowered

    # Must explicitly state in unestablished claims that quantitative reductions or GPU memory are not established
    unest_str = " ".join(takeaway.unestablished_claims).lower()
    assert "cannot be established" in unest_str or "not reported" in unest_str
    assert "gpu memory" in unest_str or "quantitative" in unest_str
