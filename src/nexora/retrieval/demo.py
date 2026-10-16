"""Offline BM25 Retrieval Demo for Nexora AI.

Provides a simple, beginner-friendly, 100% offline retrieval interface:
- Uses Okapi BM25 lexical ranking (standard library only).
- Zero external API keys, credentials, internet access, or dependencies needed.
- Loads the curated pilot research corpus (Attention Is All You Need, LoRA, FlashAttention).
- Displays top matching paper chunks with paper titles, scores, and text excerpts.
- Supports single query via CLI, positional argument, or interactive question loop.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Sequence

from nexora.retrieval.bm25 import BM25Index
from nexora.retrieval.dataset import (
    get_pilot_benchmark_dataset,
    load_retrieval_dataset,
)
from nexora.retrieval.models import RetrievedChunk, SearchResult

SAMPLE_QUESTIONS: list[str] = [
    "What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation?",
    "How does Low-Rank Adaptation (LoRA) freeze pre-trained weights and reduce trainable parameters?",
    "How does IO-aware tiling reduce memory reads and writes between GPU HBM and on-chip SRAM in FlashAttention?",
    "Parameter-efficient adaptation with rank decomposition matrices for large foundation models",
    "Wall-clock training speedup and IO complexity for exact attention on long sequences",
    "Dispensing with recurrence and convolutions in sequence transduction architectures",
]


def load_demo_index(
    dataset_path: str | None = None,
) -> tuple[BM25Index, dict[str, Any]]:
    """Load research paper chunks into an in-memory BM25 index.

    Args:
        dataset_path: Optional path to a JSON or JSONL research dataset file.
            If None, loads the curated pilot benchmark dataset.

    Returns:
        A tuple of (BM25Index, corpus_summary_dict).
    """
    if dataset_path:
        path = os.path.abspath(dataset_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        dataset = load_retrieval_dataset(path)
        corpus_name = dataset.dataset_id or os.path.basename(path)
        chunks = dataset.chunks
    else:
        dataset = get_pilot_benchmark_dataset()
        corpus_name = dataset.dataset_id
        chunks = dataset.chunks

    index = BM25Index()
    indexed_count = index.index_chunks(chunks)

    # Collect unique papers and titles
    papers: dict[str, str] = {}
    for c in chunks:
        if isinstance(c, dict):
            p_id = str(c.get("paper_id") or c.get("document_id") or "")
            title = str(c.get("title") or p_id)
        else:
            p_id = str(getattr(c, "paper_id", getattr(c, "document_id", "")))
            title = str(getattr(c, "title", p_id))
        if p_id and title:
            papers[p_id] = title

    corpus_info: dict[str, Any] = {
        "dataset_name": corpus_name,
        "indexed_chunks": indexed_count,
        "papers_count": len(papers),
        "papers": papers,
    }

    return index, corpus_info


def search_demo(index: BM25Index, query: str, limit: int = 5) -> SearchResult:
    """Execute an Okapi BM25 search against the indexed corpus.

    Args:
        index: The initialized BM25Index.
        query: The user question or search phrase.
        limit: Maximum number of top matching chunks to return.

    Returns:
        SearchResult containing matching RetrievedChunk instances ordered by score descending.
    """
    if not isinstance(query, str) or not query.strip():
        return SearchResult(query=str(query or ""), results=[])
    return index.search(query.strip(), limit=limit)


def format_retrieval_results(
    result: SearchResult,
    limit: int = 5,
    max_text_chars: int = 350,
) -> str:
    """Format search results into a clean, human-readable terminal presentation.

    Args:
        result: The SearchResult containing matching chunks.
        limit: Maximum number of chunks to display.
        max_text_chars: Maximum characters of chunk text to show before truncation.

    Returns:
        Formatted multi-line string.
    """
    lines: list[str] = []
    lines.append("-" * 78)
    lines.append(f"Question: {result.query}")
    lines.append("-" * 78)

    matching_chunks = [c for c in result.chunks if c.score > 0.0][:limit]

    if not matching_chunks:
        lines.append("No matching paper chunks found in the corpus.")
        lines.append("(All BM25 term relevance scores are 0.0 for this query).")
        lines.append("-" * 78)
        return "\n".join(lines)

    lines.append(f"Found {len(matching_chunks)} matching chunk(s) (Top {limit}):\n")

    for rank, chunk in enumerate(matching_chunks, start=1):
        meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        paper_title = meta.get("title") or chunk.document_id or "Unknown Paper"
        paper_id = meta.get("paper_id", "")
        id_suffix = f" [{paper_id}]" if paper_id else ""

        # Format excerpt with clean truncation
        text = chunk.text.strip().replace("\n", " ")
        if len(text) > max_text_chars:
            excerpt = text[:max_text_chars].rsplit(" ", 1)[0] + "..."
        else:
            excerpt = text

        lines.append(f"[{rank}] Relevance Score: {chunk.score:.4f}")
        lines.append(f"    Paper: {paper_title}{id_suffix}")
        lines.append(f"    Chunk ID: {chunk.chunk_id}")
        lines.append(f"    Excerpt: \"{excerpt}\"")
        lines.append("")

    lines.append("-" * 78)
    return "\n".join(lines)


def build_demo_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the retrieval demo."""
    parser = argparse.ArgumentParser(
        prog="nexora.retrieval.demo",
        description="Nexora AI — Simple Offline BM25 Retrieval Demo (100% Offline, Zero API Keys)",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Optional research question to search for directly.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Research question to search for (alternative to positional argument).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top matching paper chunks to display (default: 5).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Optional path to custom JSON/JSONL dataset file (default: pilot benchmark corpus).",
    )
    parser.add_argument(
        "--sample-questions",
        action="store_true",
        help="List curated sample questions from the benchmark corpus and exit.",
    )
    return parser


def run_demo(argv: Sequence[str] | None = None) -> int:
    """Run the offline BM25 retrieval demo.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code: 0 on success, non-zero on error.
    """
    parser = build_demo_parser()
    args = parser.parse_args(argv)

    if args.sample_questions:
        print("\nCurated Sample Questions (Supported by the Pilot Benchmark Corpus):")
        print("=" * 78)
        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            print(f"  {i}. {q}")
        print("=" * 78)
        return 0

    if args.top_k <= 0:
        print(f"Error: --top-k must be a positive integer, got {args.top_k}", file=sys.stderr)
        return 2

    try:
        index, corpus_info = load_demo_index(args.dataset)
    except Exception as exc:
        print(f"Error loading corpus: {exc}", file=sys.stderr)
        return 1

    query = args.query or args.question

    # Banner printout
    print("\n" + "=" * 78)
    print("                Nexora AI -- Offline BM25 Retrieval Demo")
    print("=" * 78)
    print(f"Corpus:         {corpus_info['dataset_name']}")
    print(f"Research Papers: {corpus_info['papers_count']} paper(s) indexed")
    for p_id, p_title in corpus_info["papers"].items():
        print(f"  - {p_title} ({p_id})")
    print(f"Indexed Chunks: {corpus_info['indexed_chunks']}")
    print("Offline Mode:   100% Offline (Zero external API calls, zero credentials)")
    print("=" * 78 + "\n")

    # If a specific query was provided on the CLI, run it and return
    if query:
        result = search_demo(index, query, limit=args.top_k)
        print(format_retrieval_results(result, limit=args.top_k))
        return 0

    # If running in an interactive terminal, enter an interactive question loop
    if sys.stdin.isatty():
        print("Tip: Type a question about Attention, LoRA, or FlashAttention.")
        print("     Type 'samples' to see example questions, or 'q' / 'exit' to quit.\n")
        while True:
            try:
                user_input = input("Enter your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting demo.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("q", "quit", "exit"):
                print("Exiting demo.")
                break
            if user_input.lower() in ("samples", "sample", "help"):
                print("\nCurated Sample Questions:")
                for i, q in enumerate(SAMPLE_QUESTIONS, 1):
                    print(f"  {i}. {q}")
                print()
                continue

            result = search_demo(index, user_input, limit=args.top_k)
            print()
            print(format_retrieval_results(result, limit=args.top_k))
            print()
        return 0

    # Non-interactive fallback when no query was passed: run default sample question
    default_q = SAMPLE_QUESTIONS[1]
    print(f"No query specified. Running default sample question:\n\"{default_q}\"\n")
    result = search_demo(index, default_q, limit=args.top_k)
    print(format_retrieval_results(result, limit=args.top_k))
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
