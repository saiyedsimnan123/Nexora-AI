"""Reproducible retrieval evaluation CLI for Nexora AI (Milestone 12C-32).

Provides a command-line interface to execute retrieval evaluation across:
1. Synthetic benchmark mode (offline deterministic CI benchmark)
2. Configured application pipeline mode (current application configuration)
3. Live production mode (exercises genuine external embedding API and vector store)

Produces trustworthy, structured JSON reports with truthful live/synthetic demarcation,
separate provider and vector store live status, and full provenance/metric tracking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from nexora.retrieval.production_eval import (
    EvaluationMode,
    ProductionEvaluationConfig,
    ProductionEvaluationReport,
    _get_corpus_metadata,
    _get_policy_metadata,
    evaluate_production_retrieval,
    format_vector_store_label,
)
from nexora.retrieval.quality import RetrievalQualityPolicy


def parse_k_values(k_str: str) -> tuple[int, ...]:
    """Parse comma-separated positive integer k values (e.g. '1,3,5')."""
    if not isinstance(k_str, str) or not k_str.strip():
        raise argparse.ArgumentTypeError("k-values cannot be empty")
    tokens = [t.strip() for t in k_str.split(",") if t.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("k-values cannot be empty")
    parsed: list[int] = []
    for token in tokens:
        try:
            val = int(token)
            if val <= 0:
                raise ValueError("must be positive")
            parsed.append(val)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid k value '{token}': must be a positive integer"
            ) from exc
    return tuple(parsed)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for retrieval evaluation CLI."""
    parser = argparse.ArgumentParser(
        prog="nexora-eval",
        description="Nexora AI Reproducible Retrieval Evaluation CLI (Milestone 12C-32)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["synthetic", "configured_pipeline", "production"],
        default="synthetic",
        help="Evaluation mode: 'synthetic' (offline deterministic), 'configured_pipeline', or 'production' (live external API).",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="File path to write the structured JSON evaluation report. If omitted or '-', prints to stdout.",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Path to custom evaluation dataset (.json or .jsonl). If omitted, uses built-in synthetic benchmark.",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for live production external provider (or uses OPENAI_API_KEY / GEMINI_API_KEY / NEXORA_EMBEDDING_API_KEY).",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for external embedding endpoint (or uses OPENAI_BASE_URL / GEMINI_BASE_URL / NEXORA_EMBEDDING_BASE_URL).",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Embedding provider name (e.g. 'openai', 'gemini', or 'hash'). Overrides configured default.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="Model name for production evaluation.",
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=1536,
        help="Embedding vector dimension.",
    )

    parser.add_argument(
        "--vector-store",
        choices=["in_memory", "qdrant"],
        default="in_memory",
        help="Vector store type to evaluate: 'in_memory' (local evaluation) or 'qdrant' (persistent vector store).",
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant server URL (e.g. http://localhost:6333 or from NEXORA_QDRANT_URL).",
    )

    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="Qdrant API key (or from NEXORA_QDRANT_API_KEY).",
    )

    parser.add_argument(
        "--retriever",
        choices=["vector", "bm25", "hybrid"],
        default="vector",
        help="Retrieval method: 'vector' (semantic vector search), 'bm25' (Okapi BM25 lexical search), or 'hybrid' (combined vector + BM25).",
    )

    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5",
        help="Comma-separated k values for retrieval metrics (e.g. '1,3,5').",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Minimum similarity score threshold for quality policy (use negative to disable).",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum retrieved candidates returned by quality policy.",
    )

    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="Exit with non-zero exit code if evaluation status is 'unavailable' or 'failed'.",
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON without indentation.",
    )

    parser.add_argument(
        "--smoke-test",
        "--check-provider",
        action="store_true",
        dest="smoke_test",
        help="Perform provider readiness smoke test and print readiness report.",
    )

    return parser


def _handle_output(output_path: str | None, json_text: str, report: ProductionEvaluationReport) -> None:
    """Write JSON to file or stdout with human-readable summary if outputting to file."""
    if output_path and output_path != "-":
        path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json_text)
            fh.write("\n")
        print(f"Report written to: {path}")
        method = (report.evaluation_conditions or {}).get("retrieval_method", "vector")
        method_str = f" | Method: {method}" if method and method != "vector" else ""
        print(
            f"Mode: {report.mode} | Status: {report.status} | "
            f"Provider: {report.provider_name} (live={report.provider_live}) | "
            f"Store: {report.vector_store_type} (live={report.vector_store_live}){method_str}"
        )
        if report.dataset_info is not None:
            d_name = report.dataset_info.get("name") or "custom"
            d_chunks = report.dataset_info.get("chunks_count", 0)
            d_queries = report.dataset_info.get("queries_count", 0)
            d_illustrative = report.dataset_info.get("is_illustrative_only")
            d_provenance = report.dataset_info.get("provenance")
            d_tier = report.dataset_info.get("benchmark_tier")
            review_summary = report.dataset_info.get("review_summary") or {}
            is_fully_reviewed = review_summary.get("is_fully_reviewed")

            # Review integrity: Never display manually reviewed benchmark unless tier is confirmed
            # and every query's review is verified (cannot overstate partial reviews).
            if d_tier == "manually_reviewed_benchmark" and (
                is_fully_reviewed is True or review_summary.get("reviewed_queries", 0) == d_queries
            ):
                ill_str = " (manually reviewed benchmark)"
            elif d_tier == "candidate_real_world" or d_provenance in ("candidate", "candidate_real_world"):
                ill_str = " (candidate real-world)"
            elif d_tier == "illustrative" or d_illustrative is True or d_provenance == "illustrative":
                ill_str = " (illustrative)"
            elif d_illustrative is False or d_provenance == "benchmark":
                ill_str = " (benchmark)"
            else:
                ill_str = " (unknown provenance)"
            d_ver = report.dataset_info.get("version")
            ver_str = f" v{d_ver}" if d_ver else ""
            print(f"Dataset: {d_name}{ver_str}{ill_str} | Chunks: {d_chunks} | Queries: {d_queries}")
        if report.metrics is not None:
            mrr = report.metrics.mean_reciprocal_rank
            r1 = report.metrics.recall_at_k.get(1, 0.0)
            p1 = report.metrics.precision_at_k.get(1, 0.0)
            print(
                f"MRR: {mrr:.4f} | Recall@1: {r1:.4f} | Precision@1: {p1:.4f} | "
                f"Rejection: {report.metrics.unanswerable_rejection_rate:.1%}"
            )
        if report.quality_comparison is not None:
            qc = report.quality_comparison
            net_effect = qc.get("net_effect", "unchanged")
            mrr_delta = qc.get("mrr_delta", 0.0)
            sign = "+" if mrr_delta > 0 else ""
            print(
                f"Quality Filter Effect: {net_effect.upper()} | "
                f"MRR Delta: {sign}{mrr_delta:.4f} | "
                f"Regressions: {qc.get('regressions_count', 0)} | "
                f"Improvements: {qc.get('improvements_count', 0)}"
            )
            regressions = qc.get("regressions", [])
            if regressions:
                for reg in regressions[:3]:
                    print(f"  [REGRESSION] {reg.get('description', reg.get('metric'))}")
        if report.infrastructure_limitation:
            print(f"Limitation: {report.infrastructure_limitation}")
        if report.error:
            print(f"Error: {report.error}")
    else:
        print(json_text)


def run_eval_cli(argv: Sequence[str] | None = None) -> tuple[int, Any, str]:
    """Execute evaluation CLI and return (exit_code, report, json_string)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.smoke_test:
        prov_name = (args.provider or os.getenv("NEXORA_EMBEDDING_PROVIDER") or "gemini").strip().lower()
        if prov_name == "gemini":
            from nexora.embedding.config import ENV_API_KEY, EmbeddingConfig
            from nexora.embedding.gemini_provider import check_gemini_readiness

            target_model = args.model
            if target_model in ("text-embedding-3-small", "text-embedding-004", "default"):
                target_model = "gemini-embedding-2"
            target_dim = args.dimension if args.dimension != 1536 else 768

            api_key = (
                args.api_key
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("NEXORA_GEMINI_API_KEY")
                or os.getenv(ENV_API_KEY)
            )
            if api_key:
                api_key = str(api_key).strip().strip("'\"")
            base_url = (
                args.base_url
                or os.getenv("GEMINI_BASE_URL")
                or os.getenv("NEXORA_EMBEDDING_BASE_URL")
            )

            emb_cfg = EmbeddingConfig(
                provider="gemini",
                model=target_model,
                api_key=api_key,
                base_url=base_url,
                dimension=target_dim,
            )
            readiness = check_gemini_readiness(emb_cfg)
            json_text = json.dumps(readiness.to_dict(), indent=None if args.compact else 2)
            if args.output and args.output != "-":
                path = os.path.abspath(args.output)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(json_text + "\n")
                print(f"Readiness report written to: {path}")
                print(
                    f"Provider: gemini | Status: {readiness.status} | "
                    f"Ready: {readiness.is_ready} | Live: {readiness.is_live} | "
                    f"Simulated: {readiness.is_simulated}"
                )
                if readiness.error:
                    print(f"Detail: {readiness.error}")
            else:
                print(json_text)

            if readiness.is_ready:
                exit_code = 0
            elif readiness.status == "unavailable":
                exit_code = 1 if args.fail_on_unavailable else 0
            else:
                exit_code = 1
            return exit_code, readiness, json_text

    mode = EvaluationMode(args.mode)
    k_vals = parse_k_values(args.k_values)

    min_score = args.min_score if (args.min_score is not None and args.min_score >= 0.0) else None
    policy = RetrievalQualityPolicy(
        min_score=min_score,
        max_results=args.max_results,
        max_results_per_paper=2,
        max_results_per_document=2,
    )

    # 1. Dataset loading and validation if --dataset is specified
    loaded_dataset = None
    if args.dataset:
        try:
            from nexora.retrieval.dataset import load_retrieval_dataset

            loaded_dataset = load_retrieval_dataset(args.dataset)
        except Exception as exc:
            policy_info = _get_policy_metadata(policy)
            report = ProductionEvaluationReport(
                mode=mode.value,
                status="failed",
                provider_name="unknown",
                model_name=args.model,
                dimension=args.dimension,
                vector_store_type=args.vector_store,
                is_semantic=False,
                is_synthetic=False,
                is_live=False,
                provider_live=False,
                vector_store_live=False,
                infrastructure_limitation=f"Invalid evaluation dataset: {exc}",
                error=f"Dataset error: {exc}",
                unavailable_dependencies=[],
                corpus_info={"papers_count": 0, "chunks_count": 0, "papers": []},
                policy_settings=policy_info,
                dataset_info={"path": args.dataset, "error": str(exc), "provenance": "invalid"},
            )
            json_text = json.dumps(report.to_dict(), indent=None if args.compact else 2)
            _handle_output(args.output, json_text, report)
            return 1, report, json_text

    vector_store = None
    if args.vector_store == "qdrant":
        qdrant_url = args.qdrant_url or os.getenv("NEXORA_QDRANT_URL")
        if not qdrant_url:
            report = ProductionEvaluationReport(
                mode=mode.value,
                status="unavailable",
                provider_name="unknown",
                model_name=args.model,
                dimension=args.dimension,
                vector_store_type="qdrant",
                is_semantic=False,
                is_synthetic=False,
                is_live=False,
                provider_live=False,
                vector_store_live=False,
                infrastructure_limitation="Qdrant vector store requested but NEXORA_QDRANT_URL is not set.",
                error="Missing Qdrant URL configuration.",
                unavailable_dependencies=["qdrant_url"],
                corpus_info=_get_corpus_metadata(),
                policy_settings=_get_policy_metadata(policy),
            )
            json_text = json.dumps(report.to_dict(), indent=None if args.compact else 2)
            _handle_output(args.output, json_text, report)
            exit_code = 1 if args.fail_on_unavailable else 0
            return exit_code, report, json_text

        try:
            from nexora.retrieval.qdrant import QdrantVectorStore
            from nexora.retrieval.qdrant_config import QdrantConfig

            q_cfg = QdrantConfig(
                url=qdrant_url,
                api_key=args.qdrant_api_key or os.getenv("NEXORA_QDRANT_API_KEY"),
            )
            vector_store = QdrantVectorStore(q_cfg, dimension=args.dimension)
            vector_store.ensure_collection()
        except Exception as exc:
            report = ProductionEvaluationReport(
                mode=mode.value,
                status="failed",
                provider_name="unknown",
                model_name=args.model,
                dimension=args.dimension,
                vector_store_type="qdrant",
                is_semantic=False,
                is_synthetic=False,
                is_live=False,
                provider_live=False,
                vector_store_live=False,
                infrastructure_limitation=f"Failed to initialize Qdrant store: {exc}",
                error=f"Qdrant connection failure: {exc}",
                unavailable_dependencies=["qdrant_connection"],
                corpus_info=_get_corpus_metadata(),
                policy_settings=_get_policy_metadata(policy),
            )
            json_text = json.dumps(report.to_dict(), indent=None if args.compact else 2)
            _handle_output(args.output, json_text, report)
            return 1, report, json_text

    model_name = args.model
    dimension = args.dimension
    if args.provider and args.provider.strip().lower() == "gemini":
        if model_name in ("text-embedding-3-small", "text-embedding-004"):
            model_name = "gemini-embedding-2"
        if dimension == 1536:
            dimension = 768

    config = ProductionEvaluationConfig(
        mode=mode,
        vector_store=vector_store,
        quality_policy=policy,
        k_values=k_vals,
        api_key=str(args.api_key).strip().strip("'\"") if args.api_key else None,
        base_url=args.base_url,
        provider_name=args.provider,
        model_name=model_name,
        dimension=dimension,
        dataset=loaded_dataset,
        dataset_path=args.dataset,
        retrieval_method=args.retriever,
    )

    report = evaluate_production_retrieval(config)
    json_text = json.dumps(report.to_dict(), indent=None if args.compact else 2)
    _handle_output(args.output, json_text, report)

    if report.status == "completed":
        exit_code = 0
    elif report.status == "unavailable":
        exit_code = 1 if args.fail_on_unavailable else 0
    else:  # "failed"
        exit_code = 1

    return exit_code, report, json_text


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning integer exit code."""
    try:
        exit_code, _, _ = run_eval_cli(argv)
        return exit_code
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:
        sys.stderr.write(f"nexora-eval: error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
