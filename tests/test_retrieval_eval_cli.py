"""Tests for reproducible retrieval evaluation CLI (Milestone 12C-32).

Verifies:
1. CLI argument parsing, defaults, and validation.
2. Deterministic synthetic evaluation runs offline in CI and writes structured JSON.
3. Configured pipeline evaluation reporting honest hash/non-semantic metrics.
4. Production mode missing credentials reporting unavailable with exit code semantics.
5. Production mode API failure handling with explicit failure reporting and zero false live claims.
6. Decoupled provider-live and vector-store-live demarcation.
7. Output handling to files and stdout.
8. Qdrant vector store CLI configuration and error handling.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nexora.embedding.base import EmbeddingProvider
from nexora.embedding.config import EmbeddingConfig
from nexora.embedding.openai_provider import EmbeddingProviderError, OpenAIEmbeddingProvider
from nexora.retrieval.benchmark import BenchmarkSemanticEmbeddingService
from nexora.retrieval.eval_cli import (
    build_parser,
    main,
    parse_k_values,
    run_eval_cli,
)
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.production_eval import (
    EvaluationMode,
    ProductionEvaluationConfig,
    evaluate_production_retrieval,
)


# ---------------------------------------------------------------------------
# 1. Argument Parsing & Validation
# ---------------------------------------------------------------------------
def test_parse_k_values_valid():
    """Valid comma-separated k-values parse into tuples of positive ints."""
    assert parse_k_values("1,3,5") == (1, 3, 5)
    assert parse_k_values(" 1 , 5 , 10 ") == (1, 5, 10)
    assert parse_k_values("5") == (5,)


def test_parse_k_values_invalid():
    """Invalid or negative k values raise ArgumentTypeError."""
    with pytest.raises(Exception):
        parse_k_values("")
    with pytest.raises(Exception):
        parse_k_values("0,1,3")
    with pytest.raises(Exception):
        parse_k_values("-5,1")
    with pytest.raises(Exception):
        parse_k_values("abc,5")


def test_cli_parser_defaults():
    """Default CLI arguments match the expected defaults."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.mode == "synthetic"
    assert args.output is None
    assert args.model == "text-embedding-3-small"
    assert args.dimension == 1536
    assert args.vector_store == "in_memory"
    assert args.k_values == "1,3,5"
    assert args.min_score == 0.35
    assert args.max_results == 5
    assert args.fail_on_unavailable is False


# ---------------------------------------------------------------------------
# 2. Deterministic Synthetic Evaluation via CLI
# ---------------------------------------------------------------------------
def test_deterministic_synthetic_evaluation_cli(tmp_path):
    """Running CLI in synthetic mode produces deterministic report and valid JSON file."""
    out_file = tmp_path / "synthetic_report.json"
    exit_code, report, json_text = run_eval_cli([
        "--mode", "synthetic",
        "--output", str(out_file),
    ])

    assert exit_code == 0
    assert report.mode == "synthetic"
    assert report.status == "completed"
    assert report.provider_name == "benchmark_synthetic"
    assert report.model_name == "concept_projection_64d"
    assert report.is_synthetic is True
    assert report.is_live is False
    assert report.provider_live is False
    assert report.vector_store_live is False
    assert report.vector_store_type == "in_memory (local evaluation)"
    assert report.error is None
    assert report.unavailable_dependencies == []

    # Metrics verify deterministic benchmark performance
    assert report.metrics is not None
    assert report.metrics.total_queries == 13
    assert report.metrics.mean_reciprocal_rank >= 0.75
    assert report.metrics.unanswerable_rejection_rate == 1.0

    # Output file was written and is valid JSON
    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)

    assert loaded["timestamp"] is not None
    assert "version" in loaded["code_version"]
    assert loaded["mode"] == "synthetic"
    assert loaded["status"] == "completed"
    assert loaded["provider_live"] is False
    assert loaded["vector_store_live"] is False
    assert loaded["metrics"]["total_queries"] == 13


# ---------------------------------------------------------------------------
# 3. Configured Application Pipeline Mode via CLI
# ---------------------------------------------------------------------------
def test_configured_pipeline_evaluation_cli(tmp_path):
    """Configured pipeline mode honestly reports non-semantic hash provider."""
    out_file = tmp_path / "pipeline_report.json"
    exit_code, report, json_text = run_eval_cli([
        "--mode", "configured_pipeline",
        "--output", str(out_file),
    ])

    assert exit_code == 0
    assert report.mode == "configured_pipeline"
    assert report.status == "completed"
    assert report.provider_name == "hash"
    assert report.is_semantic is False
    assert report.is_synthetic is False
    assert report.provider_live is False
    assert report.vector_store_live is False
    assert report.infrastructure_limitation is not None
    assert "deterministic text hashing" in report.infrastructure_limitation

    assert out_file.exists()


# ---------------------------------------------------------------------------
# 4. Production Mode Missing Credentials via CLI
# ---------------------------------------------------------------------------
def test_production_mode_missing_credentials_cli(monkeypatch, tmp_path):
    """Production mode without credentials writes report with status unavailable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_EMBEDDING_API_KEY", raising=False)

    out_file = tmp_path / "prod_unavailable.json"
    exit_code, report, json_text = run_eval_cli([
        "--mode", "production",
        "--output", str(out_file),
    ])

    assert exit_code == 0
    assert report.mode == "production"
    assert report.status == "unavailable"
    assert report.provider_name == "openai"
    assert report.provider_live is False
    assert report.vector_store_live is False
    assert report.is_synthetic is False
    assert report.infrastructure_limitation is not None
    assert "External model credentials are unavailable" in report.infrastructure_limitation
    assert "openai_credentials" in report.unavailable_dependencies
    assert report.metrics is None

    # When --fail-on-unavailable is passed, returns exit code 1
    fail_code, _, _ = run_eval_cli([
        "--mode", "production",
        "--fail-on-unavailable",
    ])
    assert fail_code == 1


# ---------------------------------------------------------------------------
# 5. Production Mode API Failure Handling
# ---------------------------------------------------------------------------
def test_production_mode_api_failure_handling(monkeypatch):
    """When external embedding API fails during evaluation, reports failed status honestly without claiming live quality."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key-for-test")

    # Mock OpenAI provider that raises an API error during embed_text
    mock_provider = MagicMock(spec=EmbeddingProvider)
    mock_provider.name = "openai"
    mock_provider.provider = "openai"
    mock_provider.model = "text-embedding-3-small"
    mock_provider.dimension = 1536
    mock_provider.embed_text.side_effect = RuntimeError("OpenAI API error: Rate limit reached (429)")

    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        embedding_provider=mock_provider,
    )
    report = evaluate_production_retrieval(cfg)

    # Must report status="failed" and NOT claim live execution
    assert report.status == "failed"
    assert report.provider_live is False
    assert report.is_live is False
    assert report.vector_store_live is False
    assert report.error is not None
    assert "OpenAI API error: Rate limit reached (429)" in report.error
    assert report.metrics is None
    assert report.paper_scope_isolation is None
    assert report.threshold_calibration is None


# ---------------------------------------------------------------------------
# 6. Decoupled Provider-Live and Vector-Store-Live Status
# ---------------------------------------------------------------------------
def test_decoupled_provider_and_vector_store_live_status():
    """Verify provider_live and vector_store_live are independent and never confused."""
    # Case 1: In-Memory store is never marked live even if provider were live
    report = evaluate_production_retrieval(ProductionEvaluationConfig(mode=EvaluationMode.SYNTHETIC))
    assert report.provider_live is False
    assert report.vector_store_live is False
    assert report.vector_store_type == "in_memory (local evaluation)"

    # Case 2: Injected simulated provider is never marked live
    class SimulatedProvider(EmbeddingProvider):
        def __init__(self):
            self.provider = "openai"
            self.name = "openai"
            self.model = "text-embedding-3-small"
            self.dimension = 64
            self.is_simulated = True
            self._backend = BenchmarkSemanticEmbeddingService()
        def embed_text(self, text: str):
            return self._backend.embed_text(text)
        def embed_texts(self, texts: list[str]):
            return [self.embed_text(t) for t in texts]

    report2 = evaluate_production_retrieval(ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        embedding_provider=SimulatedProvider(),
    ))
    assert report2.status == "completed"
    assert report2.is_synthetic is True
    assert report2.provider_live is False
    assert report2.vector_store_live is False
    assert "simulated" in report2.provider_name.lower()
    assert report2.vector_store_type == "in_memory (local evaluation)"


# ---------------------------------------------------------------------------
# 7. Qdrant Vector Store CLI Mode
# ---------------------------------------------------------------------------
def test_qdrant_vector_store_cli_missing_url(monkeypatch):
    """When Qdrant is requested but no URL is provided, report unavailable explicitly."""
    monkeypatch.delenv("NEXORA_QDRANT_URL", raising=False)
    exit_code, report, _ = run_eval_cli([
        "--mode", "synthetic",
        "--vector-store", "qdrant",
    ])

    assert exit_code == 0
    assert report.status == "unavailable"
    assert "qdrant_url" in report.unavailable_dependencies
    assert report.vector_store_live is False
    assert "NEXORA_QDRANT_URL is not set" in report.infrastructure_limitation


def test_qdrant_vector_store_cli_connection_failure():
    """When Qdrant fails connection, report failed status with error details."""
    exit_code, report, _ = run_eval_cli([
        "--mode", "synthetic",
        "--vector-store", "qdrant",
        "--qdrant-url", "http://127.0.0.1:59999",  # unreachable port
    ])

    assert exit_code == 1
    assert report.status == "failed"
    assert report.vector_store_live is False
    assert report.error is not None
    assert "Qdrant connection failure" in report.error or "Failed to initialize Qdrant store" in report.infrastructure_limitation


# ---------------------------------------------------------------------------
# 8. CLI main() Function & SystemExit Handling
# ---------------------------------------------------------------------------
def test_main_help_returns_zero(capsys):
    """main(['--help']) displays help and exits with code 0."""
    code = main(["--help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Nexora AI Reproducible Retrieval Evaluation CLI" in captured.out


def test_main_invalid_argument_returns_code_two(capsys):
    """main() with invalid arguments returns code 2."""
    code = main(["--unknown-argument-xyz"])
    assert code == 2


def test_main_synthetic_stdout_outputs_valid_json(capsys):
    """main(['--mode', 'synthetic', '--compact']) writes valid JSON to stdout and returns 0."""
    code = main(["--mode", "synthetic", "--compact"])
    assert code == 0
    captured = capsys.readouterr()
    loaded = json.loads(captured.out)
    assert loaded["mode"] == "synthetic"
    assert loaded["status"] == "completed"
    assert loaded["is_synthetic"] is True
    assert "metrics" in loaded
