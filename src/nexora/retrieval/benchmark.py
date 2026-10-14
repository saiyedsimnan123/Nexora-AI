"""Deterministic benchmark dataset and evaluation harness for research retrieval (Milestone 12C-30).

Provides a small, reproducible, deterministic benchmark dataset of research questions with known
relevant and irrelevant evidence chunks, spanning:
- Exact-term queries
- Semantic concept queries
- Multi-paper comparative queries
- Misleading near-matches (distractors sharing superficial terms)
- Unanswerable / empty ground-truth queries

Evaluates retrieval quality deterministically without external network calls or LLM judge dependency.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from nexora.retrieval.evaluation import (
    RetrievalEvaluationMetrics,
    evaluate_retrieval,
)
from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord

# ---------------------------------------------------------------------------
# Benchmark Paper Constants
# ---------------------------------------------------------------------------
PAPER_TRANSFORMER = "p-vaswani-2017"
PAPER_FLASHATTN = "p-dao-2022"
PAPER_LORA = "p-hu-2021"
PAPER_DISTRACTOR = "p-distractor-2020"

DOC_TRANSFORMER = "doc-transformer-01"
DOC_FLASHATTN = "doc-flashattn-01"
DOC_LORA = "doc-lora-01"
DOC_DISTRACTOR = "doc-distractor-01"

BENCHMARK_DIMENSION = 64

# Concept feature keys used for deterministic semantic projection
_FEATURE_KEYS = [
    # Transformer & Self-Attention
    "attention_formula",
    "scaled_dot_product",
    "key_dimension_dk",
    "multi_head_attention",
    "representation_subspaces",
    "quadratic_complexity",
    "wmt_translation_benchmark",
    # FlashAttention & IO Efficiency
    "io_aware_attention",
    "sram_tiling",
    "gpu_hbm_bandwidth",
    "online_softmax",
    "recomputation_backward",
    "wall_clock_speedup",
    # LoRA & Parameter-Efficient Adaptation
    "low_rank_decomposition",
    "frozen_weights_w0",
    "matrix_factors_b_a",
    "rank_r_bottleneck",
    "adapter_memory_efficiency",
    "attention_projection_weights",
    # Distractor & Misleading Domains
    "biological_vision_primate",
    "eye_tracking_saccades",
    "cpu_cache_hierarchy",
    "intel_vtune_latency",
    "hyperspectral_tensor_svd",
    "image_denoising_tucker",
    "cnn_edge_quantization",
    # Unanswerable Out-of-Domain
    "quantum_annealing_ising",
    "crispr_cas9_guide_rna",
]

_FEATURE_INDEX = {k: idx for idx, k in enumerate(_FEATURE_KEYS)}

# ---------------------------------------------------------------------------
# Deterministic Semantic Embedding Provider
# ---------------------------------------------------------------------------
class BenchmarkSemanticEmbeddingService:
    """Deterministic, dense semantic embedding provider for the retrieval benchmark.

    Projects text into a 64-dimensional normalized vector space based on exact lexical
    cues and semantic concept associations. Completely deterministic, pure standard library,
    and requires no GPU or network access.
    """

    def __init__(self, dimension: int = BENCHMARK_DIMENSION) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _extract_feature_weights(self, text: str) -> list[float]:
        lower = text.lower()
        weights = [0.0] * self._dimension

        # Concept rules with associated keywords and coordinate indices
        rules: list[tuple[str, list[str], float]] = [
            # Transformer
            ("scaled_dot_product", ["scaled dot-product", "dot product", "softmax(qk^t", "softmax(qk"], 3.5),
            ("attention_formula", ["attention(q, k, v)", "compatibility of queries", "queries with keys"], 3.0),
            ("key_dimension_dk", ["d_k", "dimension d_k", "scaling factor", "small gradients"], 3.0),
            ("multi_head_attention", ["multi-head", "multi head", "heads"], 3.5),
            ("representation_subspaces", ["representation subspaces", "different representation", "subspaces simultaneously"], 3.5),
            ("quadratic_complexity", ["quadratic", "o(n^2)", "o(n^2 * d)", "computational and memory complexity", "scalability bottleneck"], 3.5),
            ("wmt_translation_benchmark", ["wmt 2014", "bleu", "english-to-german", "p100"], 2.5),

            # FlashAttention
            ("io_aware_attention", ["io-aware", "io awareness", "hbm", "high bandwidth memory", "memory accesses"], 3.5),
            ("sram_tiling", ["sram", "tiling", "fast sram", "tiling query", "sram accesses"], 3.5),
            ("gpu_hbm_bandwidth", ["gpu memory", "gpu hbm", "hbm access overhead", "io bandwidth"], 3.0),
            ("online_softmax", ["online softmax", "running normalizer", "intermediate attention matrices", "numerical instability"], 3.5),
            ("recomputation_backward", ["backward pass", "recomputed", "recomputation", "on the fly"], 3.0),
            ("wall_clock_speedup", ["speedup", "wall-clock", "bert", "gpt-2", "64k"], 2.5),

            # LoRA
            ("low_rank_decomposition", ["low-rank", "low rank", "intrinsic dimension", "delta w = b * a", "decomposition"], 3.5),
            ("frozen_weights_w0", ["freeze", "freezing", "frozen", "pre-trained model weights"], 3.0),
            ("matrix_factors_b_a", ["matrix a", "matrix b", "b * a", "small factorized matrices"], 3.0),
            ("adapter_memory_efficiency", ["parameter-efficient", "parameter efficiency", "trainable parameters", "foundation models"], 3.5),
            ("attention_projection_weights", ["projection matrices", "projection weights", "targets transformer", "target multi-head", "adapters target", "w_q", "w_k", "w_v", "w_o", "query and value weights"], 4.0),

            # Distractors
            ("biological_vision_primate", ["biological", "primate", "macaque", "visual cortex", "v1 and v4"], 4.0),
            ("eye_tracking_saccades", ["eye-tracking", "saccades", "saccadic", "scanpaths", "infrared"], 4.0),
            ("cpu_cache_hierarchy", ["cpu", "l1", "l2", "l3", "cache coherence", "x86", "ddr4 dram"], 4.0),
            ("intel_vtune_latency", ["intel vtune", "multicore", "latency profiling"], 3.5),
            ("hyperspectral_tensor_svd", ["hyperspectral", "remote sensing", "remote-sensing", "tensor svd"], 4.0),
            ("image_denoising_tucker", ["tucker decomposition", "image denoising", "gaussian noise"], 3.5),
            ("cnn_edge_quantization", ["convolutional neural networks", "resnet-50", "dsp", "8-bit integer", "battery"], 4.0),

            # Unanswerable Out-of-Domain
            ("quantum_annealing_ising", ["quantum annealing", "ising spin", "spin glass", "ground states"], 5.0),
            ("crispr_cas9_guide_rna", ["crispr", "cas9", "guide rna", "ribonucleoprotein", "cleavage"], 5.0),
        ]

        for feature_name, keywords, weight in rules:
            if feature_name in _FEATURE_INDEX:
                idx = _FEATURE_INDEX[feature_name]
                for kw in keywords:
                    if kw in lower:
                        weights[idx] += weight

        # Generic overlap support for common terms across concepts
        if "attention" in lower:
            weights[0] += 0.5
            weights[3] += 0.5
            weights[7] += 0.5
            weights[19] += 0.3  # Biological attention gets partial generic overlap
        if "memory" in lower:
            weights[5] += 0.5
            weights[9] += 0.5
            weights[17] += 0.5
            weights[21] += 0.3  # CPU memory gets partial generic overlap

        return weights

    def embed_text(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")

        weights = self._extract_feature_weights(text)
        norm = math.sqrt(sum(w * w for w in weights))
        if norm == 0.0:
            # Deterministic small uniform vector for completely unknown vocabulary
            uniform_val = 1.0 / math.sqrt(self._dimension)
            return [uniform_val] * self._dimension
        return [w / norm for w in weights]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Benchmark Corpus Definitions (4 Papers, 15 Chunks)
# ---------------------------------------------------------------------------
_RAW_CHUNKS: list[dict[str, Any]] = [
    # Paper 1: Transformer (Attention Is All You Need)
    {
        "chunk_id": "chunk_p1_c1",
        "document_id": DOC_TRANSFORMER,
        "paper_id": PAPER_TRANSFORMER,
        "title": "Attention Is All You Need",
        "filename": "vaswani_2017.pdf",
        "chunk_index": 0,
        "text": (
            "Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. "
            "We compute the compatibility of queries with keys scaled by the inverse square root "
            "of key dimension d_k to avoid small gradients in large dimensions."
        ),
    },
    {
        "chunk_id": "chunk_p1_c2",
        "document_id": DOC_TRANSFORMER,
        "paper_id": PAPER_TRANSFORMER,
        "title": "Attention Is All You Need",
        "filename": "vaswani_2017.pdf",
        "chunk_index": 1,
        "text": (
            "Multi-Head Attention: Instead of performing a single attention function, we project "
            "queries, keys, and values h times with learned linear projections to d_k, d_k, and d_v dimensions. "
            "This allows the model to jointly attend to information from different representation subspaces simultaneously."
        ),
    },
    {
        "chunk_id": "chunk_p1_c3",
        "document_id": DOC_TRANSFORMER,
        "paper_id": PAPER_TRANSFORMER,
        "title": "Attention Is All You Need",
        "filename": "vaswani_2017.pdf",
        "chunk_index": 2,
        "text": (
            "Computational and Memory Complexity: Self-attention layers have O(n^2 * d) per-layer complexity "
            "where n is sequence length. The quadratic memory complexity of the attention matrix forms "
            "a fundamental scalability bottleneck for long sequences."
        ),
    },
    {
        "chunk_id": "chunk_p1_c4",
        "document_id": DOC_TRANSFORMER,
        "paper_id": PAPER_TRANSFORMER,
        "title": "Attention Is All You Need",
        "filename": "vaswani_2017.pdf",
        "chunk_index": 3,
        "text": (
            "Training Setup: Evaluated on the WMT 2014 English-to-German dataset with 4.5 million sentence pairs. "
            "Trained with 8 NVIDIA P100 GPUs for 3.5 days, achieving a new state-of-the-art BLEU score of 28.4."
        ),
    },

    # Paper 2: FlashAttention (Dao et al., 2022)
    {
        "chunk_id": "chunk_p2_c1",
        "document_id": DOC_FLASHATTN,
        "paper_id": PAPER_FLASHATTN,
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "filename": "dao_2022.pdf",
        "chunk_index": 0,
        "text": (
            "IO-Aware Attention: Standard attention materializes intermediate N x N attention matrices in GPU High Bandwidth Memory (HBM). "
            "FlashAttention computes exact attention with O(N) memory accesses by tiling query, key, and value blocks in fast SRAM."
        ),
    },
    {
        "chunk_id": "chunk_p2_c2",
        "document_id": DOC_FLASHATTN,
        "paper_id": PAPER_FLASHATTN,
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "filename": "dao_2022.pdf",
        "chunk_index": 1,
        "text": (
            "Tiling and Online Softmax: FlashAttention uses online softmax scaling across blocks in SRAM without materializing "
            "intermediate attention matrices to HBM, maintaining running normalizers to ensure numerical instability is prevented."
        ),
    },
    {
        "chunk_id": "chunk_p2_c3",
        "document_id": DOC_FLASHATTN,
        "paper_id": PAPER_FLASHATTN,
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "filename": "dao_2022.pdf",
        "chunk_index": 2,
        "text": (
            "Speedup Results: FlashAttention yields 2-4x wall-clock speedup on BERT and GPT-2 models while scaling to 64k sequences "
            "with strictly linear memory footprint relative to sequence length."
        ),
    },
    {
        "chunk_id": "chunk_p2_c4",
        "document_id": DOC_FLASHATTN,
        "paper_id": PAPER_FLASHATTN,
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "filename": "dao_2022.pdf",
        "chunk_index": 3,
        "text": (
            "Backward Pass Recomputation: During the backward pass, attention matrices are recomputed on the fly from SRAM "
            "rather than read from HBM, exchanging computation for substantially reduced IO bandwidth."
        ),
    },

    # Paper 3: LoRA (Hu et al., 2021)
    {
        "chunk_id": "chunk_p3_c1",
        "document_id": DOC_LORA,
        "paper_id": PAPER_LORA,
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "filename": "hu_2021.pdf",
        "chunk_index": 0,
        "text": (
            "Low-Rank Decomposition: Large models possess low intrinsic dimension during adaptation. Parameter updates "
            "are represented by low-rank matrix decomposition Delta W = B * A where rank r << min(d, k)."
        ),
    },
    {
        "chunk_id": "chunk_p3_c2",
        "document_id": DOC_LORA,
        "paper_id": PAPER_LORA,
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "filename": "hu_2021.pdf",
        "chunk_index": 1,
        "text": (
            "Parameter Efficiency: By freezing pre-trained model weights W0 and training only small factorized matrices A and B, "
            "LoRA reduces trainable parameters by 10,000x and GPU memory consumption by 3x on large foundation models."
        ),
    },
    {
        "chunk_id": "chunk_p3_c3",
        "document_id": DOC_LORA,
        "paper_id": PAPER_LORA,
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "filename": "hu_2021.pdf",
        "chunk_index": 2,
        "text": (
            "Attention Projection Weights: LoRA targets transformer attention projection matrices W_q, W_k, W_v, and W_o. "
            "Adapting query and value weights yields competitive performance while keeping rank r as small as 1, 2, or 4."
        ),
    },

    # Paper 4: Distractors & Misleading Near-Matches
    {
        "chunk_id": "chunk_p4_c1",
        "document_id": DOC_DISTRACTOR,
        "paper_id": PAPER_DISTRACTOR,
        "title": "Biological and Hardware Near-Match Distractors",
        "filename": "distractors_2020.pdf",
        "chunk_index": 0,
        "text": (
            "Biological Attention in Primate Vision: Studies visual saccades and selective attention in macaque visual cortex "
            "V1 and V4 during natural scene viewing recorded with eye-tracking infrared scanpaths."
        ),
    },
    {
        "chunk_id": "chunk_p4_c2",
        "document_id": DOC_DISTRACTOR,
        "paper_id": PAPER_DISTRACTOR,
        "title": "Biological and Hardware Near-Match Distractors",
        "filename": "distractors_2020.pdf",
        "chunk_index": 1,
        "text": (
            "Hardware Memory Hierarchy in Multicore CPUs: Evaluates L1, L2, and L3 cache coherence and x86 DDR4 DRAM latency profiling "
            "using Intel VTune on multicore processor architectures."
        ),
    },
    {
        "chunk_id": "chunk_p4_c3",
        "document_id": DOC_DISTRACTOR,
        "paper_id": PAPER_DISTRACTOR,
        "title": "Biological and Hardware Near-Match Distractors",
        "filename": "distractors_2020.pdf",
        "chunk_index": 2,
        "text": (
            "Low-Rank Tensor SVD for Hyperspectral Image Denoising: Decomposes 3D remote-sensing imagery tensors using Tucker decomposition "
            "to filter additive Gaussian noise."
        ),
    },
    {
        "chunk_id": "chunk_p4_c4",
        "document_id": DOC_DISTRACTOR,
        "paper_id": PAPER_DISTRACTOR,
        "title": "Biological and Hardware Near-Match Distractors",
        "filename": "distractors_2020.pdf",
        "chunk_index": 3,
        "text": (
            "Energy Efficient Convolutional Neural Networks on Embedded Edge DSPs: Quantizes ResNet-50 weights to 8-bit integer formats "
            "to reduce battery consumption on mobile devices."
        ),
    },
]


def build_benchmark_corpus(
    embedding_service: BenchmarkSemanticEmbeddingService | None = None,
) -> list[VectorRecord]:
    """Build and return the deterministic benchmark vector records."""
    emb = embedding_service or BenchmarkSemanticEmbeddingService()
    records: list[VectorRecord] = []
    for raw in _RAW_CHUNKS:
        vector = emb.embed_text(raw["text"])
        payload = {
            "chunk_id": raw["chunk_id"],
            "document_id": raw["document_id"],
            "paper_id": raw["paper_id"],
            "title": raw["title"],
            "filename": raw["filename"],
            "chunk_index": raw["chunk_index"],
            "text": raw["text"],
        }
        records.append(
            VectorRecord(
                id=raw["chunk_id"],
                vector=vector,
                payload=payload,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Benchmark Queries (13 Queries across 5 Categories)
# ---------------------------------------------------------------------------
_BENCHMARK_CASES: list[dict[str, Any]] = [
    # 1. Exact-Term Queries
    {
        "query": "What is the scaling factor in Scaled Dot-Product Attention to prevent small gradients?",
        "expected_chunk_ids": ["chunk_p1_c1"],
        "category": "exact_term",
        "target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "query": "How does online softmax scaling enable FlashAttention tiling in SRAM?",
        "expected_chunk_ids": ["chunk_p2_c2"],
        "category": "exact_term",
        "target_paper_ids": [PAPER_FLASHATTN],
    },
    {
        "query": "What low-rank matrices represent weight updates in LoRA formulation Delta W = B * A?",
        "expected_chunk_ids": ["chunk_p3_c1"],
        "category": "exact_term",
        "target_paper_ids": [PAPER_LORA],
    },

    # 2. Semantic Concept Queries
    {
        "query": "How do modern architectures attend across different representation subspaces simultaneously?",
        "expected_chunk_ids": ["chunk_p1_c2"],
        "category": "semantic",
        "target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "query": "Technique to reduce GPU High Bandwidth Memory access overhead during matrix calculations",
        "expected_chunk_ids": ["chunk_p2_c1"],
        "category": "semantic",
        "target_paper_ids": [PAPER_FLASHATTN],
    },
    {
        "query": "Adapting massive foundation models by freezing weights and training small factorized matrices",
        "expected_chunk_ids": ["chunk_p3_c2"],
        "category": "semantic",
        "target_paper_ids": [PAPER_LORA],
    },

    # 3. Multi-Paper Comparative Queries
    {
        "query": "Comparing quadratic memory complexity in self-attention with linear IO-aware tiling solutions",
        "expected_chunk_ids": ["chunk_p1_c3", "chunk_p2_c1"],
        "category": "multi_paper",
        "target_paper_ids": [PAPER_TRANSFORMER, PAPER_FLASHATTN],
    },
    {
        "query": "How do parameter-efficient adapters target multi-head attention projection weights during fine-tuning?",
        "expected_chunk_ids": ["chunk_p1_c2", "chunk_p3_c3"],
        "category": "multi_paper",
        "target_paper_ids": [PAPER_TRANSFORMER, PAPER_LORA],
    },

    # 4. Misleading Near-Matches (Testing Distractor Discrimination)
    {
        "query": "Mechanisms of visual saccadic attention and eye-tracking scanpaths in primates",
        "expected_chunk_ids": ["chunk_p4_c1"],
        "category": "misleading_near_match",
        "target_paper_ids": [PAPER_DISTRACTOR],
    },
    {
        "query": "Evaluating L1 and L2 cache latency and DRAM memory hierarchy on CPU processors",
        "expected_chunk_ids": ["chunk_p4_c2"],
        "category": "misleading_near_match",
        "target_paper_ids": [PAPER_DISTRACTOR],
    },
    {
        "query": "Tensor singular value decomposition for denoising hyperspectral remote sensing",
        "expected_chunk_ids": ["chunk_p4_c3"],
        "category": "misleading_near_match",
        "target_paper_ids": [PAPER_DISTRACTOR],
    },

    # 5. Empty Results / Unanswerable Queries
    {
        "query": "Quantum annealing algorithms for finding ground states in Ising spin glass models",
        "expected_chunk_ids": [],
        "category": "unanswerable_empty",
        "target_paper_ids": None,
    },
    {
        "query": "CRISPR-Cas9 guide RNA off-target cleavage prevention using modified ribonucleoproteins",
        "expected_chunk_ids": [],
        "category": "unanswerable_empty",
        "target_paper_ids": None,
    },
]


def get_benchmark_cases() -> list[dict[str, Any]]:
    """Return a fresh copy of the 13 deterministic benchmark cases."""
    import copy
    return copy.deepcopy(_BENCHMARK_CASES)


def run_retrieval_benchmark(
    retrieve_fn: Callable[[str, int], SearchResult] | Callable[[str], SearchResult],
    *,
    k_values: Sequence[int] = (1, 3, 5),
    benchmark_cases: Sequence[dict[str, Any]] | None = None,
) -> RetrievalEvaluationMetrics:
    """Run deterministic retrieval evaluation on the benchmark suite.

    Accepts any retrieve function or service providing retrieve(query, limit).
    """
    cases = benchmark_cases if benchmark_cases is not None else get_benchmark_cases()
    return evaluate_retrieval(cases, retrieve_fn, k_values=k_values)
