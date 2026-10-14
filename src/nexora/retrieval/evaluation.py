"""Deterministic evaluation suite for retrieval relevance and citation correctness.

Provides quantitative, reproducible metrics for retrieval and citation grounding:
- Retrieval: Recall@K, Precision@K, Mean Reciprocal Rank (MRR).
- Citations: Citation Validity Rate, Evidence Coverage, Invalid Citation Rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from nexora.retrieval.evidence import EvidenceContext, verify_citations
from nexora.retrieval.models import SearchResult


@dataclass(frozen=True)
class RetrievalEvaluationMetrics:
    """Quantitative retrieval metrics over an evaluation benchmark."""

    total_queries: int
    mean_reciprocal_rank: float
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    empty_retrievals: int


@dataclass(frozen=True)
class CitationEvaluationMetrics:
    """Quantitative citation grounding metrics over an evaluation benchmark."""

    total_answers: int
    total_citations: int
    valid_citations_count: int
    invalid_citations_count: int
    citation_validity_rate: float
    evidence_coverage_rate: float
    grounded_answers_rate: float


def compute_reciprocal_rank(
    retrieved_ids: Sequence[str], relevant_ids: set[str]
) -> float:
    """Compute Reciprocal Rank: 1 / (rank of first relevant chunk), or 0.0."""
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_precision_recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: set[str], k: int
) -> tuple[float, float]:
    """Compute (Precision@K, Recall@K)."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    precision = hits / k
    recall = (hits / len(relevant_ids)) if relevant_ids else 0.0
    return precision, recall


def evaluate_retrieval(
    benchmark_cases: Sequence[dict[str, Any]],
    retrieve_fn: Callable[[str, int], SearchResult] | Callable[[str], SearchResult],
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> RetrievalEvaluationMetrics:
    """Run deterministic retrieval evaluation over a sequence of benchmark cases.

    Each case must be a dict with:
        - "query": str
        - "expected_chunk_ids": Sequence[str] (ground-truth relevant chunk IDs)
    """
    if not benchmark_cases:
        return RetrievalEvaluationMetrics(
            total_queries=0,
            mean_reciprocal_rank=0.0,
            recall_at_k={k: 0.0 for k in k_values},
            precision_at_k={k: 0.0 for k in k_values},
            empty_retrievals=0,
        )

    max_k = max(k_values)
    reciprocal_ranks: list[float] = []
    p_at_k_sums: dict[int, float] = {k: 0.0 for k in k_values}
    r_at_k_sums: dict[int, float] = {k: 0.0 for k in k_values}
    empty_count = 0

    for case in benchmark_cases:
        query = case["query"]
        relevant_ids = set(case["expected_chunk_ids"])

        try:
            result = retrieve_fn(query, limit=max_k)  # type: ignore[call-arg]
        except TypeError:
            result = retrieve_fn(query)

        retrieved_chunk_ids = [chunk.chunk_id for chunk in result.results]
        if not retrieved_chunk_ids:
            empty_count += 1

        rr = compute_reciprocal_rank(retrieved_chunk_ids, relevant_ids)
        reciprocal_ranks.append(rr)

        for k in k_values:
            prec, rec = compute_precision_recall_at_k(retrieved_chunk_ids, relevant_ids, k)
            p_at_k_sums[k] += prec
            r_at_k_sums[k] += rec

    num_cases = len(benchmark_cases)
    mrr = sum(reciprocal_ranks) / num_cases
    p_at_k = {k: p_at_k_sums[k] / num_cases for k in k_values}
    r_at_k = {k: r_at_k_sums[k] / num_cases for k in k_values}

    return RetrievalEvaluationMetrics(
        total_queries=num_cases,
        mean_reciprocal_rank=round(mrr, 4),
        recall_at_k={k: round(v, 4) for k, v in r_at_k.items()},
        precision_at_k={k: round(v, 4) for k, v in p_at_k.items()},
        empty_retrievals=empty_count,
    )


def evaluate_citations(
    benchmark_cases: Sequence[dict[str, Any]],
) -> CitationEvaluationMetrics:
    """Evaluate citation validity and grounding over a benchmark of answer-context pairs.

    Each case must be a dict with:
        - "text": str (or "answer": ResearchAnswer)
        - "context": EvidenceContext
    """
    if not benchmark_cases:
        return CitationEvaluationMetrics(
            total_answers=0,
            total_citations=0,
            valid_citations_count=0,
            invalid_citations_count=0,
            citation_validity_rate=1.0,
            evidence_coverage_rate=0.0,
            grounded_answers_rate=1.0,
        )

    total_citations = 0
    total_valid = 0
    total_invalid = 0
    coverage_ratios: list[float] = []
    grounded_answers = 0

    for case in benchmark_cases:
        text = case["answer"].text if "answer" in case else case["text"]
        context = case["context"]

        res = verify_citations(text, context)
        n_cited = len(res.cited_labels)
        total_citations += n_cited
        total_valid += len(res.valid_citations)
        total_invalid += len(res.invalid_citations)

        if res.is_fully_grounded:
            grounded_answers += 1

        n_evidence = len(context.items)
        if n_evidence > 0:
            coverage = len(set(res.valid_citations)) / n_evidence
        else:
            coverage = 1.0 if n_cited == 0 else 0.0
        coverage_ratios.append(coverage)

    num_answers = len(benchmark_cases)
    val_rate = (total_valid / total_citations) if total_citations > 0 else 1.0
    cov_rate = sum(coverage_ratios) / num_answers if num_answers > 0 else 0.0
    grounded_rate = grounded_answers / num_answers if num_answers > 0 else 1.0

    return CitationEvaluationMetrics(
        total_answers=num_answers,
        total_citations=total_citations,
        valid_citations_count=total_valid,
        invalid_citations_count=total_invalid,
        citation_validity_rate=round(val_rate, 4),
        evidence_coverage_rate=round(cov_rate, 4),
        grounded_answers_rate=round(grounded_rate, 4),
    )
