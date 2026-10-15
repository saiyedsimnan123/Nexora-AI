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
    unanswerable_queries: int = 0
    unanswerable_rejections: int = 0
    unanswerable_rejection_rate: float = 1.0
    category_metrics: dict[str, Any] | None = None


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
    """Compute Reciprocal Rank: 1 / (rank of first relevant chunk), or 0.0.

    Behavior for empty or unanswerable cases:
    - If relevant_ids is empty (unanswerable query), returns 0.0 (no relevant target).
    - If retrieved_ids is empty, returns 0.0 (no hits).
    """
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_precision_recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: set[str], k: int
) -> tuple[float, float]:
    """Compute (Precision@K, Recall@K).

    Behavior for empty or unanswerable cases:
    - If k <= 0, raises ValueError.
    - If relevant_ids is empty:
      - If retrieved_ids is empty (clean rejection), returns (1.0, 1.0) because 0 false positives were retrieved
        and 100% of the 0 relevant items were omitted.
      - If retrieved_ids has elements (false alarms), returns (0.0, 0.0) as all retrieved items are irrelevant.
    - If relevant_ids is non-empty but retrieved_ids is empty: returns (0.0, 0.0).
    - Otherwise returns standard hits / k and hits / len(relevant_ids).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    if not relevant_ids:
        if not top_k:
            return 1.0, 1.0
        return 0.0, 0.0
    precision = hits / k
    recall = hits / len(relevant_ids)
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
    categories: Sequence[str] | None = None,
) -> RetrievalEvaluationMetrics:
    """Run deterministic retrieval evaluation over a sequence of benchmark cases.

    Each case must be a dict with:
        - "query": str
        - "expected_chunk_ids": Sequence[str] (ground-truth relevant chunk IDs)
        - "category": optional str category label
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
    unanswerable_count = 0
    unanswerable_rejections = 0

    explicit_category_seen = False
    category_buckets: dict[str, list[tuple[float, dict[int, float], dict[int, float]]]] = {}

    if categories is not None:
        for cat in categories:
            clean_cat = str(cat).strip()
            if clean_cat and clean_cat not in category_buckets:
                category_buckets[clean_cat] = []

    for case in valid_cases:
        query = case["query"]
        relevant_ids = set(case["expected_chunk_ids"])
        raw_cat = case.get("category")
        if raw_cat is not None and str(raw_cat).strip():
            category = str(raw_cat).strip()
            explicit_category_seen = True
        else:
            category = "uncategorized"

        result = caller(query, max_k)
        if not isinstance(result, SearchResult):
            raise TypeError(
                f"retrieve_fn must return a SearchResult, got {type(result).__name__}"
            )

        retrieved_chunk_ids = [chunk.chunk_id for chunk in result.results]
        if not retrieved_chunk_ids:
            empty_count += 1

        if not relevant_ids:
            unanswerable_count += 1
            if not retrieved_chunk_ids:
                unanswerable_rejections += 1

        rr = compute_reciprocal_rank(retrieved_chunk_ids, relevant_ids)
        reciprocal_ranks.append(rr)

        case_p: dict[int, float] = {}
        case_r: dict[int, float] = {}
        for k in valid_k_values:
            prec, rec = compute_precision_recall_at_k(retrieved_chunk_ids, relevant_ids, k)
            p_at_k_sums[k] += prec
            r_at_k_sums[k] += rec
            case_p[k] = prec
            case_r[k] = rec

        if category not in category_buckets:
            category_buckets[category] = []
        category_buckets[category].append((rr, case_p, case_r))

    num_cases = len(valid_cases)
    mrr = sum(reciprocal_ranks) / num_cases
    p_at_k = {k: p_at_k_sums[k] / num_cases for k in valid_k_values}
    r_at_k = {k: r_at_k_sums[k] / num_cases for k in valid_k_values}
    unans_rate = (unanswerable_rejections / unanswerable_count) if unanswerable_count > 0 else 1.0

    cat_metrics: dict[str, Any] = {}
    should_include_categories = explicit_category_seen or (categories is not None)

    if should_include_categories:
        for cat_name, entries in category_buckets.items():
            if cat_name == "uncategorized" and not entries and (categories is None or "uncategorized" not in categories):
                continue
            n_cat = len(entries)
            if n_cat > 0:
                cat_mrr = sum(e[0] for e in entries) / n_cat
                cat_p = {k: sum(e[1][k] for e in entries) / n_cat for k in valid_k_values}
                cat_r = {k: sum(e[2][k] for e in entries) / n_cat for k in valid_k_values}
            else:
                cat_mrr = 0.0
                cat_p = {k: 0.0 for k in valid_k_values}
                cat_r = {k: 0.0 for k in valid_k_values}

            cat_metrics[cat_name] = {
                "count": n_cat,
                "mean_reciprocal_rank": round(cat_mrr, 4),
                "precision_at_k": {k: round(v, 4) for k, v in cat_p.items()},
                "recall_at_k": {k: round(v, 4) for k, v in cat_r.items()},
            }

    return RetrievalEvaluationMetrics(
        total_queries=num_cases,
        mean_reciprocal_rank=round(mrr, 4),
        recall_at_k={k: round(v, 4) for k, v in r_at_k.items()},
        precision_at_k={k: round(v, 4) for k, v in p_at_k.items()},
        empty_retrievals=empty_count,
        unanswerable_queries=unanswerable_count,
        unanswerable_rejections=unanswerable_rejections,
        unanswerable_rejection_rate=round(unans_rate, 4),
        category_metrics=cat_metrics if cat_metrics else None,
    )


def compare_retrieval_metrics(
    baseline: RetrievalEvaluationMetrics,
    filtered: RetrievalEvaluationMetrics,
) -> dict[str, Any]:
    """Compare baseline retrieval against quality-filtered retrieval honestly.

    Computes exact metric deltas across MRR, Precision@K, Recall@K, unanswerable
    rejection rate, and per-category metrics.

    Identifies regressions and improvements explicitly without implying improvement
    when metrics degraded.
    """
    mrr_delta = round(filtered.mean_reciprocal_rank - baseline.mean_reciprocal_rank, 4)
    unans_delta = round(
        filtered.unanswerable_rejection_rate - baseline.unanswerable_rejection_rate, 4
    )
    empty_delta = filtered.empty_retrievals - baseline.empty_retrievals

    k_values = sorted(set(baseline.recall_at_k.keys()) | set(filtered.recall_at_k.keys()))
    recall_deltas: dict[int, float] = {}
    precision_deltas: dict[int, float] = {}

    for k in k_values:
        b_rec = baseline.recall_at_k.get(k, 0.0)
        f_rec = filtered.recall_at_k.get(k, 0.0)
        recall_deltas[k] = round(f_rec - b_rec, 4)

        b_prec = baseline.precision_at_k.get(k, 0.0)
        f_prec = filtered.precision_at_k.get(k, 0.0)
        precision_deltas[k] = round(f_prec - b_prec, 4)

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    if mrr_delta < 0:
        regressions.append({
            "metric": "mean_reciprocal_rank",
            "baseline": baseline.mean_reciprocal_rank,
            "filtered": filtered.mean_reciprocal_rank,
            "delta": mrr_delta,
            "description": f"MRR degraded by {abs(mrr_delta):.4f}",
        })
    elif mrr_delta > 0:
        improvements.append({
            "metric": "mean_reciprocal_rank",
            "baseline": baseline.mean_reciprocal_rank,
            "filtered": filtered.mean_reciprocal_rank,
            "delta": mrr_delta,
            "description": f"MRR improved by {mrr_delta:.4f}",
        })

    for k in k_values:
        r_d = recall_deltas[k]
        if r_d < 0:
            regressions.append({
                "metric": f"recall@{k}",
                "baseline": baseline.recall_at_k.get(k, 0.0),
                "filtered": filtered.recall_at_k.get(k, 0.0),
                "delta": r_d,
                "description": f"Recall@{k} dropped by {abs(r_d):.4f}",
            })
        elif r_d > 0:
            improvements.append({
                "metric": f"recall@{k}",
                "baseline": baseline.recall_at_k.get(k, 0.0),
                "filtered": filtered.recall_at_k.get(k, 0.0),
                "delta": r_d,
                "description": f"Recall@{k} improved by {r_d:.4f}",
            })

        p_d = precision_deltas[k]
        if p_d < 0:
            regressions.append({
                "metric": f"precision@{k}",
                "baseline": baseline.precision_at_k.get(k, 0.0),
                "filtered": filtered.precision_at_k.get(k, 0.0),
                "delta": p_d,
                "description": f"Precision@{k} dropped by {abs(p_d):.4f}",
            })
        elif p_d > 0:
            improvements.append({
                "metric": f"precision@{k}",
                "baseline": baseline.precision_at_k.get(k, 0.0),
                "filtered": filtered.precision_at_k.get(k, 0.0),
                "delta": p_d,
                "description": f"Precision@{k} improved by {p_d:.4f}",
            })

    if unans_delta < 0:
        regressions.append({
            "metric": "unanswerable_rejection_rate",
            "baseline": baseline.unanswerable_rejection_rate,
            "filtered": filtered.unanswerable_rejection_rate,
            "delta": unans_delta,
            "description": f"Unanswerable rejection rate dropped by {abs(unans_delta):.4f}",
        })
    elif unans_delta > 0:
        improvements.append({
            "metric": "unanswerable_rejection_rate",
            "baseline": baseline.unanswerable_rejection_rate,
            "filtered": filtered.unanswerable_rejection_rate,
            "delta": unans_delta,
            "description": f"Unanswerable rejection rate improved by {unans_delta:.4f}",
        })

    has_regressions = len(regressions) > 0
    has_improvements = len(improvements) > 0

    if has_improvements and not has_regressions:
        net_effect = "improved"
    elif has_regressions and not has_improvements:
        net_effect = "degraded"
    elif has_improvements and has_regressions:
        net_effect = "mixed"
    else:
        net_effect = "unchanged"

    # Per-category comparisons
    category_comparisons: dict[str, Any] = {}
    b_cats = baseline.category_metrics or {}
    f_cats = filtered.category_metrics or {}
    all_cats = sorted(set(b_cats.keys()) | set(f_cats.keys()))

    for cat in all_cats:
        b_c = b_cats.get(cat, {})
        f_c = f_cats.get(cat, {})
        b_mrr = b_c.get("mean_reciprocal_rank", 0.0)
        f_mrr = f_c.get("mean_reciprocal_rank", 0.0)
        cat_mrr_delta = round(f_mrr - b_mrr, 4)

        b_p1 = b_c.get("precision_at_k", {}).get(1, 0.0)
        f_p1 = f_c.get("precision_at_k", {}).get(1, 0.0)
        b_r1 = b_c.get("recall_at_k", {}).get(1, 0.0)
        f_r1 = f_c.get("recall_at_k", {}).get(1, 0.0)

        category_comparisons[cat] = {
            "count": f_c.get("count", b_c.get("count", 0)),
            "baseline_mrr": b_mrr,
            "filtered_mrr": f_mrr,
            "mrr_delta": cat_mrr_delta,
            "baseline_precision_at_1": b_p1,
            "filtered_precision_at_1": f_p1,
            "precision_at_1_delta": round(f_p1 - b_p1, 4),
            "baseline_recall_at_1": b_r1,
            "filtered_recall_at_1": f_r1,
            "recall_at_1_delta": round(f_r1 - b_r1, 4),
        }

    return {
        "net_effect": net_effect,
        "has_regressions": has_regressions,
        "regressions_count": len(regressions),
        "improvements_count": len(improvements),
        "mrr_delta": mrr_delta,
        "recall_at_k_deltas": recall_deltas,
        "precision_at_k_deltas": precision_deltas,
        "unanswerable_rejection_rate_delta": unans_delta,
        "empty_retrievals_delta": empty_delta,
        "regressions": regressions,
        "improvements": improvements,
        "category_comparisons": category_comparisons,
    }


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


@dataclass(frozen=True)
class AnswerGroundingEvaluationMetrics:
    """Quantitative metrics for research-answer grounding over an evaluation benchmark.

    SEPARATION OF EVALUATION CONCEPTS:
    1. Retrieval Relevance: MRR, Recall@K, Precision@K evaluate finding relevant chunks.
    2. Citation Reference Integrity: Syntactic resolution of citation labels to evidence.
    3. Claim-to-Evidence Grounding: Entailment, contradiction, and sufficiency of factual statements.

    NOTE ON SCIENTIFIC TRUTH:
    These metrics assess agreement between generated text claims and cited context chunks.
    They do NOT guarantee absolute real-world factual correctness or eliminate all model error.
    """

    total_answers: int
    total_claims: int
    total_citations: int
    valid_citations_count: int
    invalid_citations_count: int
    supported_claims_count: int
    contradicted_claims_count: int
    insufficient_claims_count: int
    unresolved_claims_count: int
    citation_validity_rate: float
    claim_coverage_rate: float
    supported_claim_rate: float
    contradicted_claim_rate: float
    insufficient_claim_rate: float
    unresolved_claim_rate: float
    fully_grounded_answer_rate: float
    zero_evidence_handled_rate: float
    empty_claims_answers_count: int


def evaluate_answer_grounding(
    benchmark_cases: Sequence[dict[str, Any]],
    verifier: Any = None,
) -> AnswerGroundingEvaluationMetrics:
    """Evaluate claim-level answer grounding over a suite of answer-context pairs.

    Each case in benchmark_cases must be a dict with:
        - "context": EvidenceContext
        - "text": str OR "answer": ResearchAnswer (or object with .text)
        - optional "query": str
        - optional "expected_paper_id": str

    verifier: An instance providing verify_answer(text, context, query),
              defaulting to ClaimToEvidenceVerifier().
    """
    if benchmark_cases is None or isinstance(benchmark_cases, (str, bytes)) or not hasattr(benchmark_cases, "__iter__"):
        raise TypeError("benchmark_cases must be a sequence of dictionaries")

    cases_list = list(benchmark_cases)
    if not cases_list:
        return AnswerGroundingEvaluationMetrics(
            total_answers=0,
            total_claims=0,
            total_citations=0,
            valid_citations_count=0,
            invalid_citations_count=0,
            supported_claims_count=0,
            contradicted_claims_count=0,
            insufficient_claims_count=0,
            unresolved_claims_count=0,
            citation_validity_rate=1.0,
            claim_coverage_rate=0.0,
            supported_claim_rate=0.0,
            contradicted_claim_rate=0.0,
            insufficient_claim_rate=0.0,
            unresolved_claim_rate=0.0,
            fully_grounded_answer_rate=1.0,
            zero_evidence_handled_rate=1.0,
            empty_claims_answers_count=0,
        )

    if verifier is None:
        from nexora.research.verification import ClaimToEvidenceVerifier
        verifier = ClaimToEvidenceVerifier()

    total_claims = 0
    total_citations = 0
    total_valid_citations = 0
    total_invalid_citations = 0
    total_supported = 0
    total_contradicted = 0
    total_insufficient = 0
    total_unresolved = 0

    covered_claims_count = 0
    fully_grounded_answers = 0
    empty_claims_answers = 0

    zero_ev_total = 0
    zero_ev_handled = 0
    total_evidence_across_all = 0

    for i, case in enumerate(cases_list):
        if not isinstance(case, dict):
            raise TypeError(f"benchmark_cases[{i}] must be a dict, got {type(case).__name__}")
        if "context" not in case:
            raise KeyError(f"benchmark_cases[{i}] missing required 'context' key")
        context = case["context"]
        if not isinstance(context, EvidenceContext):
            raise TypeError(
                f"benchmark_cases[{i}]['context'] must be an EvidenceContext, got {type(context).__name__}"
            )

        if "answer" in case:
            answer_obj = case["answer"]
            text = getattr(answer_obj, "text", None)
            if not isinstance(text, str):
                raise TypeError(f"benchmark_cases[{i}]['answer'].text must be a string")
            query = getattr(answer_obj, "query", "") or case.get("query", "")
        elif "text" in case:
            text = case["text"]
            if not isinstance(text, str):
                raise TypeError(f"benchmark_cases[{i}]['text'] must be a string")
            query = case.get("query", "")
        else:
            raise KeyError(f"benchmark_cases[{i}] must contain 'text' or 'answer'")

        num_ev = len(context.items)
        total_evidence_across_all += num_ev

        result = verifier.verify_answer(text, context, query=query)

        n_claims = result.total_claims
        total_claims += n_claims

        if n_claims == 0:
            empty_claims_answers += 1

        if num_ev == 0:
            zero_ev_total += 1
            if result.is_grounded:
                zero_ev_handled += 1

        total_supported += result.supported_claims_count
        total_contradicted += result.contradicted_claims_count
        total_insufficient += result.insufficient_claims_count
        total_unresolved += result.unresolved_claims_count

        for c in result.claims:
            total_citations += len(c.cited_labels)
            total_valid_citations += (len(c.cited_labels) - len(c.unresolved_labels))
            total_invalid_citations += len(c.unresolved_labels)
            if len(c.sources) > 0:
                covered_claims_count += 1

        if result.is_grounded:
            fully_grounded_answers += 1

    num_answers = len(cases_list)

    if total_citations > 0:
        val_rate = total_valid_citations / total_citations
    else:
        val_rate = 1.0 if total_evidence_across_all == 0 else 0.0

    cov_rate = (covered_claims_count / total_claims) if total_claims > 0 else 0.0
    supp_rate = (total_supported / total_claims) if total_claims > 0 else 0.0
    contra_rate = (total_contradicted / total_claims) if total_claims > 0 else 0.0
    insuff_rate = (total_insufficient / total_claims) if total_claims > 0 else 0.0
    unres_rate = (total_unresolved / total_claims) if total_claims > 0 else 0.0

    grounded_rate = (fully_grounded_answers / num_answers) if num_answers > 0 else 1.0
    zero_ev_rate = (zero_ev_handled / zero_ev_total) if zero_ev_total > 0 else 1.0

    return AnswerGroundingEvaluationMetrics(
        total_answers=num_answers,
        total_claims=total_claims,
        total_citations=total_citations,
        valid_citations_count=total_valid_citations,
        invalid_citations_count=total_invalid_citations,
        supported_claims_count=total_supported,
        contradicted_claims_count=total_contradicted,
        insufficient_claims_count=total_insufficient,
        unresolved_claims_count=total_unresolved,
        citation_validity_rate=round(val_rate, 4),
        claim_coverage_rate=round(cov_rate, 4),
        supported_claim_rate=round(supp_rate, 4),
        contradicted_claim_rate=round(contra_rate, 4),
        insufficient_claim_rate=round(insuff_rate, 4),
        unresolved_claim_rate=round(unres_rate, 4),
        fully_grounded_answer_rate=round(grounded_rate, 4),
        zero_evidence_handled_rate=round(zero_ev_rate, 4),
        empty_claims_answers_count=empty_claims_answers,
    )

