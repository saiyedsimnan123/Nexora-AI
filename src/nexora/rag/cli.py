"""Command-line interface for the end-to-end RAG pipeline.

Allows users to ask research questions, retrieve supporting evidence,
and view grounded answers with full source attribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from nexora.rag.models import RAGResponse
from nexora.rag.pipeline import create_offline_rag_pipeline

SAMPLE_QUESTIONS: list[str] = [
    "What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation?",
    "How does Low-Rank Adaptation (LoRA) freeze pre-trained weights and reduce trainable parameters?",
    "How does IO-aware tiling reduce memory reads and writes between GPU HBM and on-chip SRAM in FlashAttention?",
    "Parameter-efficient adaptation with rank decomposition matrices for large foundation models",
    "Wall-clock training speedup and IO complexity for exact attention on long sequences",
    "Dispensing with recurrence and convolutions in sequence transduction architectures",
]


def format_rag_response(response: RAGResponse) -> str:
    """Format a RAGResponse into a clean, human-readable terminal report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("                 Nexora AI -- End-to-End RAG Pipeline")
    lines.append("=" * 78)
    lines.append(f"Question:       {response.query}")
    lines.append(f"Retrieval Mode: {response.retrieval_mode}")
    lines.append(f"Status:         {response.status.upper()}")
    mock_str = "Yes (Offline Deterministic Mock)" if response.is_mock else "No (Live LLM Model)"
    lines.append(f"Mock Answer:    {mock_str}")
    lines.append(f"Model:          {response.model}")
    lines.append("-" * 78)

    if response.status == "error":
        lines.append(f"ERROR: {response.error}")
        lines.append("-" * 78)
        return "\n".join(lines)

    lines.append("\nANSWER:")
    lines.append(response.answer)
    lines.append("")

    lines.append("-" * 78)
    lines.append(f"SUPPORTING EVIDENCE & SOURCE ATTRIBUTION ({len(response.sources)} retrieved chunks):")
    lines.append("-" * 78)

    if not response.sources:
        lines.append("No evidence sources retrieved.")
    else:
        for src in response.sources:
            is_cited = any(c.chunk_id == src.chunk_id for c in response.cited_sources)
            cited_tag = " [CITED IN ANSWER]" if is_cited else ""
            lines.append(f"{src.label}{cited_tag}")
            lines.append(f"  Paper:    {src.paper_title}")
            lines.append(f"  Chunk ID: {src.chunk_id} | Relevance Score: {src.score:.4f}")
            lines.append(f"  Excerpt:  \"{src.text_excerpt}\"")
            lines.append("")

    lines.append("-" * 78)
    grounded_tag = "VERIFIED GROUNDED (All citations point to retrieved evidence)" if response.is_grounded else "WARNING: UNGROUNDED (Contains unverified citations)"
    lines.append(f"Reference Integrity: {grounded_tag}")
    lines.append("=" * 78)

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the RAG pipeline."""
    parser = argparse.ArgumentParser(
        prog="nexora.rag.cli",
        description="Nexora AI -- End-to-End RAG Pipeline CLI",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Research question to answer directly.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Research question (alternative to positional argument).",
    )
    parser.add_argument(
        "--retriever",
        "-r",
        choices=["bm25", "hybrid", "vector"],
        default="bm25",
        help="Retrieval strategy: 'bm25' (default for offline benchmark), 'hybrid', or 'vector'.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of evidence chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Path to custom JSON/JSONL dataset file (default: curated pilot benchmark).",
    )
    parser.add_argument(
        "--sample-questions",
        action="store_true",
        help="Display curated sample research questions and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw structured JSON response instead of formatted text.",
    )
    return parser


def run_rag_cli(argv: Sequence[str] | None = None) -> int:
    """Run the RAG command-line interface.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sample_questions:
        print("\nCurated Sample Questions (Pilot Benchmark Corpus):")
        print("=" * 78)
        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            print(f"  {i}. {q}")
        print("=" * 78)
        return 0

    if args.top_k <= 0:
        print(f"Error: --top-k must be a positive integer, got {args.top_k}", file=sys.stderr)
        return 2

    # Initialize offline pipeline
    try:
        pipeline = create_offline_rag_pipeline(
            dataset_path=args.dataset,
            default_mode=args.retriever,
        )
    except Exception as exc:
        print(f"Error initializing RAG pipeline: {exc}", file=sys.stderr)
        return 1

    query = args.query or args.question

    # If query supplied directly, run and print result
    if query:
        response = pipeline.answer(
            query,
            retrieval_mode=args.retriever,
            top_k=args.top_k,
        )
        if args.json:
            print(json.dumps(response.to_dict(), indent=2))
        else:
            print(format_rag_response(response))
        return 0 if response.status != "error" else 1

    # If running interactively, prompt user in a loop
    if sys.stdin.isatty():
        print("\n" + "=" * 78)
        print("         Nexora AI -- Interactive End-to-End RAG Shell (Offline Mode)")
        print("=" * 78)
        print("Ask questions about Attention Is All You Need, LoRA, and FlashAttention.")
        print("Type 'samples' for suggested questions, or 'exit' / 'q' to quit.\n")

        while True:
            try:
                user_input = input("Research Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("q", "quit", "exit"):
                print("Exiting.")
                break
            if user_input.lower() in ("samples", "sample", "help"):
                print("\nCurated Sample Questions:")
                for i, q in enumerate(SAMPLE_QUESTIONS, 1):
                    print(f"  {i}. {q}")
                print()
                continue

            response = pipeline.answer(
                user_input,
                retrieval_mode=args.retriever,
                top_k=args.top_k,
            )
            print()
            if args.json:
                print(json.dumps(response.to_dict(), indent=2))
            else:
                print(format_rag_response(response))
            print()
        return 0

    # Non-interactive fallback when no query was passed: run default question
    default_q = SAMPLE_QUESTIONS[1]
    response = pipeline.answer(
        default_q,
        retrieval_mode=args.retriever,
        top_k=args.top_k,
    )
    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
    else:
        print(format_rag_response(response))
    return 0


if __name__ == "__main__":
    sys.exit(run_rag_cli())
