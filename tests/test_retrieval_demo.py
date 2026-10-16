"""Tests for the offline BM25 retrieval demo module.

Verifies:
1. Default and custom dataset index loading.
2. Error handling for nonexistent datasets.
3. BM25 search helper behavior on valid and empty queries.
4. Output formatting for matching and unanswerable/empty queries.
5. CLI execution: --query, positional questions, --sample-questions, --top-k.
6. Interactive question loop execution with mocked inputs.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from nexora.retrieval.bm25 import BM25Index
from nexora.retrieval.demo import (
    SAMPLE_QUESTIONS,
    build_demo_parser,
    format_retrieval_results,
    load_demo_index,
    run_demo,
    search_demo,
)
from nexora.retrieval.models import RetrievedChunk, SearchResult


# ---------------------------------------------------------------------------
# 1. Index Loading Tests
# ---------------------------------------------------------------------------
def test_load_demo_index_default():
    """Verify loading the default pilot benchmark index."""
    index, info = load_demo_index()
    assert isinstance(index, BM25Index)
    assert info["indexed_chunks"] == 6
    assert info["papers_count"] == 3
    assert "Attention Is All You Need" in info["papers"].values()
    assert "LoRA: Low-Rank Adaptation of Large Language Models" in info["papers"].values()
    assert "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" in info["papers"].values()


def test_load_demo_index_custom_path():
    """Verify loading an explicit valid dataset path."""
    index, info = load_demo_index("examples/real_world_pilot_benchmark.json")
    assert isinstance(index, BM25Index)
    assert info["indexed_chunks"] == 6
    assert info["papers_count"] == 3


def test_load_demo_index_nonexistent_file():
    """Verify FileNotFoundError when dataset path does not exist."""
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_demo_index("nonexistent_path_to_dataset.json")


# ---------------------------------------------------------------------------
# 2. Search Helper Tests
# ---------------------------------------------------------------------------
def test_search_demo_empty_or_whitespace_query():
    """Verify empty or whitespace query returns empty results cleanly."""
    index, _ = load_demo_index()
    res_empty = search_demo(index, "")
    assert isinstance(res_empty, SearchResult)
    assert len(res_empty.chunks) == 0

    res_spaces = search_demo(index, "   \t \n  ")
    assert len(res_spaces.chunks) == 0


def test_search_demo_matches_relevant_papers():
    """Verify search returns relevant chunks ordered by relevance score."""
    index, _ = load_demo_index()
    res = search_demo(index, "FlashAttention HBM SRAM GPU tiling", limit=5)
    assert len(res.chunks) > 0
    top = res.chunks[0]
    assert top.score > 0.0
    assert "FlashAttention" in (top.metadata.get("title") or "")
    assert top.chunk_id in ("chunk_2205_14135_01", "chunk_2205_14135_02")


# ---------------------------------------------------------------------------
# 3. Output Formatting Tests
# ---------------------------------------------------------------------------
def test_format_retrieval_results_with_matches():
    """Verify output formatting contains score, title, chunk ID, and excerpt."""
    chunk = RetrievedChunk(
        document_id="doc_lora",
        chunk_id="c_lora_01",
        text="Low-Rank Adaptation freezes pre-trained weights and adds trainable matrices.",
        score=8.5432,
        metadata={"title": "LoRA Paper", "paper_id": "arxiv_2106.09685"},
    )
    result = SearchResult(query="How does LoRA freeze weights?", results=[chunk])
    output = format_retrieval_results(result, limit=5)

    assert "Question: How does LoRA freeze weights?" in output
    assert "Relevance Score: 8.5432" in output
    assert "Paper: LoRA Paper [arxiv_2106.09685]" in output
    assert "Chunk ID: c_lora_01" in output
    assert "Excerpt:" in output
    assert "Low-Rank Adaptation freezes" in output


def test_format_retrieval_results_no_matches():
    """Verify formatting when zero chunks match."""
    result = SearchResult(query="Unrelated quantum query", results=[])
    output = format_retrieval_results(result, limit=5)

    assert "Question: Unrelated quantum query" in output
    assert "No matching paper chunks found in the corpus." in output
    assert "(All BM25 term relevance scores are 0.0 for this query)." in output


# ---------------------------------------------------------------------------
# 4. CLI Argument Parsing and Execution Tests
# ---------------------------------------------------------------------------
def test_build_demo_parser_defaults():
    """Verify parser arguments and defaults."""
    parser = build_demo_parser()
    args = parser.parse_args([])
    assert args.question is None
    assert args.query is None
    assert args.top_k == 5
    assert args.dataset is None
    assert args.sample_questions is False


def test_run_demo_cli_sample_questions(capsys):
    """Verify --sample-questions prints sample questions and exits with 0."""
    exit_code = run_demo(["--sample-questions"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Curated Sample Questions" in captured
    for q in SAMPLE_QUESTIONS:
        assert q in captured


def test_run_demo_cli_invalid_top_k(capsys):
    """Verify non-positive --top-k returns error code 2."""
    exit_code = run_demo(["--top-k", "0"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "--top-k must be a positive integer" in err


def test_run_demo_cli_with_query_flag(capsys):
    """Verify running with --query flag produces formatted results."""
    exit_code = run_demo(["--query", "What BLEU score did Transformer achieve?", "--top-k", "3"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Nexora AI -- Offline BM25 Retrieval Demo" in captured
    assert "Attention Is All You Need" in captured
    assert "28.4 BLEU" in captured


def test_run_demo_cli_with_positional_question(capsys):
    """Verify running with a positional question argument."""
    exit_code = run_demo(["How does LoRA freeze pre-trained weights?"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "LoRA: Low-Rank Adaptation of Large Language Models" in captured
    assert "chunk_2106_09685_01" in captured


def test_run_demo_cli_non_interactive_fallback_without_query(capsys):
    """Verify running without arguments non-interactively falls back to default sample query."""
    with patch("sys.stdin.isatty", return_value=False):
        exit_code = run_demo([])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "No query specified. Running default sample question:" in captured
    assert "LoRA" in captured


def test_run_demo_cli_interactive_loop(capsys):
    """Verify interactive question loop with simulated terminal inputs."""
    user_inputs = ["samples", "FlashAttention tiling", "exit"]
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=user_inputs):
        exit_code = run_demo([])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Curated Sample Questions:" in captured
    assert "FlashAttention" in captured
    assert "Exiting demo." in captured
