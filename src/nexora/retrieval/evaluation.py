"""Deterministic evaluation suite for retrieval relevance and citation correctness.

Provides quantitative, reproducible metrics for retrieval and citation grounding:
- Retrieval: Recall@K, Precision@K, Mean Reciprocal Rank (MRR).
- Citations: Citation Validity Rate (reference integrity), Evidence Coverage Rate,
  and Grounded Answers Rate (reference-valid answers).

NOTE ON CITATION METRIC SEMANTICS:
Citation metrics evaluate syntactic reference integrity (whether citation labels
like [E1] resolve to chunks present in the supplied EvidenceContext). They do NOT
measure semantic claim support, natural language entailment, or the factual truth
of generated statements.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
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
    """Quantitative citation reference metrics over an evaluation benchmark.

    NOTE ON CITATION SEMANTICS:
    `citation_validity_rate` and `grounded_answers_rate` measure citation-reference
    integrity: whether cited labels map to items in the supplied evidence context.
    They do not evaluate natural language claim entailment or full factual groundedness.
    """

    total_answers: int
    total_citations: int
    valid_citations_count: int
    invalid_citations_count: int
    citation_validity_rate: float
    evidence_coverage_rate: float
    grounded_answers_rate: float

    @property
    def reference_grounded_answers_rate(self) -> float:
        """Alias for grounded_answers_rate clarifying reference-level validity."""
        return self.grounded_answers_rate


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


def _resolve_retrieve_caller(
    retrieve_fn: Callable[..., Any]
) -> Callable[[str, int], SearchResult]:
    """Inspect retrieve_fn once upfront and return an adapter accepting (query, limit).

    Determines the calling convention via signature inspection upfront without swallowing
    runtime TypeErrors raised inside the callback.
    """
    if not callable(retrieve_fn):
        raise TypeError(f"retrieve_fn must be callable, got {type(retrieve_fn).__name__}")

    try:
        sig = inspect.signature(retrieve_fn)
        params = list(sig.parameters.values())
    except (ValueError, TypeError):
        def _default_caller(q: str, limit: int) -> SearchResult:
            return retrieve_fn(q, limit=limit)
        return _default_caller

    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

    accepts_limit_kw = has_var_keyword or any(
        p.name == "limit" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for p in params
    )

    positional_count = sum(
        1 for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )

    if accepts_limit_kw:
        def _kw_caller(q: str, limit: int) -> SearchResult:
            return retrieve_fn(q, limit=limit)
        return _kw_caller

    if has_var_positional or positional_count >= 2:
        def _pos_caller(q: str, limit: int) -> SearchResult:
            return retrieve_fn(q, limit)
        return _pos_caller

    def _unary_caller(q: str, _limit: int) -> SearchResult:
        return retrieve_fn(q)
    return _unary_caller


def _validate_k_values(k_values: Any) -> tuple[int, ...]:
    if k_values is None or isinstance(k_values, (str, bytes)) or not hasattr(k_values, "__iter__"):
        raise TypeError("k_values must be a sequence of positive integers")
    k_list = list(k_values)
    if not k_list:
        raise ValueError("k_values must not be empty")
    for k in k_list:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"Each k in k_values must be an integer, got {type(k).__name__}")
        if k <= 0:
            raise ValueError(f"Each k in k_values must be a positive integer, got {k}")
    return tuple(k_list)


def _validate_benchmark_cases(cases: Any) -> list[dict[str, Any]]:
    if cases is None or isinstance(cases, (str, bytes)) or not hasattr(cases, "__iter__"):
        raise TypeError("benchmark_cases must be a sequence of dictionaries")
    validated: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TypeError(f"benchmark_cases[{i}] must be a dict, got {type(case).__name__}")
        if "query" not in case:
            raise KeyError(f"benchmark_cases[{i}] missing required 'query' key")
        query = case["query"]
        if not isinstance(query, str):
            raise TypeError(f"benchmark_cases[{i}]['query'] must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError(f"benchmark_cases[{i}]['query'] must not be empty or whitespace-only")
        if "expected_chunk_ids" not in case:
            raise KeyError(f"benchmark_cases[{i}] missing required 'expected_chunk_ids' key")
        expected_ids = case["expected_chunk_ids"]
        if expected_ids is None or isinstance(expected_ids, (str, bytes)) or not hasattr(expected_ids, "__iter__"):
            raise TypeError(f"benchmark_cases[{i}]['expected_chunk_ids'] must be an iterable of string chunk IDs")
        for j, cid in enumerate(expected_ids):
            if not isinstance(cid, str):
                raise TypeError(f"benchmark_cases[{i}]['expected_chunk_ids'][{j}] must be a string, got {type(cid).__name__}")
        validated.append(case)
    return validated


def evaluate_retrieval(
    benchmark_cases: Sequence[dict[str, Any]],
    retrieve_fn: Callable[[str, int], SearchResult] | Callable[[str], SearchResult],
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> RetrievalEvaluationMetrics:
    """Run deterministic retrieval evaluation over a sequence of benchmark cases.

    Each case must be a dict with:
        - "query": str
        - "expected_chunk_ids": Sequence[str] (ground-truth relevant chunk IDs)
    """
    valid_k_values = _validate_k_values(k_values)
    valid_cases = _validate_benchmark_cases(benchmark_cases)

    if not valid_cases:
        return RetrievalEvaluationMetrics(
            total_queries=0,
            mean_reciprocal_rank=0.0,
            recall_at_k={k: 0.0 for k in valid_k_values},
            precision_at_k={k: 0.0 for k in valid_k_values},
            empty_retrievals=0,
        )

    caller = _resolve_retrieve_caller(retrieve_fn)
    max_k = max(valid_k_values)
    reciprocal_ranks: list[float] = []
    p_at_k_sums: dict[int, float] = {k: 0.0 for k in valid_k_values}
    r_at_k_sums: dict[int, float] = {k: 0.0 for k in valid_k_values}
    empty_count = 0

    for case in valid_cases:
        query = case["query"]
        relevant_ids = set(case["expected_chunk_ids"])

        result = caller(query, max_k)
        if not isinstance(result, SearchResult):
            raise TypeError(
                f"retrieve_fn must return a SearchResult, got {type(result).__name__}"
            )

        retrieved_chunk_ids = [chunk.chunk_id for chunk in result.results]
        if not retrieved_chunk_ids:
            empty_count += 1

        rr = compute_reciprocal_rank(retrieved_chunk_ids, relevant_ids)
        reciprocal_ranks.append(rr)

        for k in valid_k_values:
            prec, rec = compute_precision_recall_at_k(retrieved_chunk_ids, relevant_ids, k)
            p_at_k_sums[k] += prec
            r_at_k_sums[k] += rec

    num_cases = len(valid_cases)
    mrr = sum(reciprocal_ranks) / num_cases
    p_at_k = {k: p_at_k_sums[k] / num_cases for k in valid_k_values}
    r_at_k = {k: r_at_k_sums[k] / num_cases for k in valid_k_values}

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
    """Evaluate citation validity and reference grounding over answer-context pairs.

    NOTE: Measures reference integrity (syntactic mapping to supplied evidence).
    Does not measure semantic claim entailment or factual truth.

    Each case must be a dict with:
        - "text": str (or "answer": ResearchAnswer / object with .text)
        - "context": EvidenceContext
    """
    if benchmark_cases is None or isinstance(benchmark_cases, (str, bytes)) or not hasattr(benchmark_cases, "__iter__"):
        raise TypeError("benchmark_cases must be a sequence of dictionaries")

    cases_list = list(benchmark_cases)
    if not cases_list:
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

    for i, case in enumerate(cases_list):
        if not isinstance(case, dict):
            raise TypeError(f"benchmark_cases[{i}] must be a dict, got {type(case).__name__}")
        if "context" not in case:
            raise KeyError(f"benchmark_cases[{i}] missing required 'context' key")
        context = case["context"]
        if not isinstance(context, EvidenceContext):
            raise TypeError(f"benchmark_cases[{i}]['context'] must be an EvidenceContext, got {type(context).__name__}")

        if "answer" in case:
            answer_obj = case["answer"]
            text = getattr(answer_obj, "text", None)
            if not isinstance(text, str):
                raise TypeError(f"benchmark_cases[{i}]['answer'].text must be a string")
        elif "text" in case:
            text = case["text"]
            if not isinstance(text, str):
                raise TypeError(f"benchmark_cases[{i}]['text'] must be a string")
        else:
            raise KeyError(f"benchmark_cases[{i}] must contain 'text' or 'answer'")

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

    num_answers = len(cases_list)
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
