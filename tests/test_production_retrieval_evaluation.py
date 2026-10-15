"""Tests for production retrieval evaluation and honest limitation reporting (Milestone 12C-31).

Verifies:
1. Production mode fails open and reports limitation explicitly when credentials are unavailable.
2. No silent substitution of synthetic embeddings when external models are unavailable.
3. Production mode execution with injected/configured production providers.
4. Configured application pipeline evaluation reporting honest metrics and semantic status.
5. Strict separation of synthetic benchmark results from real pipeline results.
6. Threshold calibration using actual measured results.
7. Paper-scope isolation measurement and leakage verification.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from nexora.embedding.base import EmbeddingProvider
from nexora.embedding.config import EmbeddingConfig
from nexora.embedding.factory import create_embedding_provider
from nexora.embedding.hash_provider import HashEmbeddingProvider
from nexora.retrieval.benchmark import (
    DOC_TRANSFORMER,
    PAPER_FLASHATTN,
    PAPER_TRANSFORMER,
    BenchmarkSemanticEmbeddingService,
)
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.production_eval import (
    EvaluationMode,
    ProductionEvaluationConfig,
    ProductionEvaluationReport,
    calibrate_retrieval_thresholds,
    check_production_credentials,
    evaluate_paper_scope_isolation,
    evaluate_production_retrieval,
)
from nexora.retrieval.quality import RetrievalQualityPolicy, RetrievalQualityService
from nexora.retrieval.service import RetrievalService


# ---------------------------------------------------------------------------
# Test Double: Simulated Production OpenAI Provider (1536 Dimensions)
# ---------------------------------------------------------------------------
class SimulatedProductionOpenAIProvider(EmbeddingProvider):
    """Simulates real OpenAI text-embedding-3-small provider (1536-d unit vectors)."""

    def __init__(self, dimension: int = 1536) -> None:
        self._dimension = dimension
        self.provider = "openai"
        self.name = "openai"
        self.model = "text-embedding-3-small"
        self.call_count = 0
        self._synthetic_backend = BenchmarkSemanticEmbeddingService(dimension=64)
        self.is_simulated = True

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        self.call_count += 1
        base_64 = self._synthetic_backend.embed_text(text)
        # Pad deterministically to 1536 dimensions while preserving unit norm
        padded = base_64 + [0.0] * (self._dimension - len(base_64))
        return padded

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# 1. Credential Check & Honest Limitation Reporting
# ---------------------------------------------------------------------------
def test_production_mode_fails_open_when_credentials_unavailable(monkeypatch):
    """When credentials are not configured, report limitation explicitly without synthetic substitution."""
    # Ensure neither API key is set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_EMBEDDING_API_KEY", raising=False)

    has_creds, msg = check_production_credentials(api_key=None)
    assert has_creds is False
    assert "External model credentials are unavailable" in msg

    cfg = ProductionEvaluationConfig(mode=EvaluationMode.PRODUCTION, api_key=None)
    report = evaluate_production_retrieval(cfg)

    # Must report status="unavailable"
    assert report.status == "unavailable"
    assert report.mode == "production"
    assert report.provider_name == "openai"
    assert report.model_name == "text-embedding-3-small"
    assert report.is_semantic is True
    # Crucial requirement: DO NOT silently substitute synthetic embedding!
    assert report.is_synthetic is False
    assert report.is_live is False
    assert report.vector_store_type == "uninitialized"
    assert report.infrastructure_limitation is not None
    assert "External model credentials are unavailable" in report.infrastructure_limitation
    assert report.metrics is None
    assert report.paper_scope_isolation is None
    assert report.threshold_calibration is None


# ---------------------------------------------------------------------------
# 2. Production Evaluation Control-Flow Testing with Mocked Provider
# ---------------------------------------------------------------------------
def test_production_mode_executes_with_injected_provider():
    """When simulated production provider is injected for control-flow testing,
    verify complete execution pipeline while ensuring metrics are flagged as simulated/synthetic."""
    prod_provider = SimulatedProductionOpenAIProvider()
    quality_policy = RetrievalQualityPolicy(min_score=0.35, max_results=5, max_results_per_paper=2)

    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        embedding_provider=prod_provider,
        quality_policy=quality_policy,
        k_values=(1, 3, 5),
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert report.mode == "production"
    assert report.provider_name == "openai (simulated)"
    assert report.model_name == "text-embedding-3-small"
    assert report.dimension == 1536
    assert report.is_semantic is True
    # Crucial integrity requirement: simulated provider must NOT claim genuine live production!
    assert report.is_live is False
    assert report.is_synthetic is True
    assert report.vector_store_type == "in_memory (local evaluation)"
    assert report.infrastructure_limitation is not None
    assert "Simulated test double" in report.infrastructure_limitation
    assert report.metrics is not None
    assert report.metrics.total_queries == 13
    assert report.metrics.unanswerable_rejection_rate == 1.0
    assert report.metrics.mean_reciprocal_rank >= 0.75
    assert report.paper_scope_isolation is not None
    assert report.paper_scope_isolation["isolation_successful"] is True
    assert len(report.threshold_calibration) == 4


# ---------------------------------------------------------------------------
# 2b. Regression Test: Simulated/Synthetic Provider Cannot Be Reported as Live Production
# ---------------------------------------------------------------------------
def test_simulated_or_synthetic_provider_cannot_be_reported_as_genuine_live_production():
    """Regression test: simulated or synthetic embeddings cannot be reported as genuine live production.

    Requirement (Milestone 12C-31.1):
    Do not report metrics from SimulatedProductionOpenAIProvider or any synthetic backend
    as real OpenAI or production embedding measurements.
    """
    # 1. Simulated OpenAI provider injected into PRODUCTION mode
    simulated_provider = SimulatedProductionOpenAIProvider()
    cfg1 = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        embedding_provider=simulated_provider,
    )
    report1 = evaluate_production_retrieval(cfg1)

    assert report1.is_live is False
    assert report1.is_synthetic is True
    assert "simulated" in report1.provider_name.lower()
    assert report1.vector_store_type == "in_memory (local evaluation)"
    assert report1.infrastructure_limitation is not None
    assert "Simulated test double" in report1.infrastructure_limitation
    assert report1.evaluation_conditions is not None
    assert report1.evaluation_conditions["is_live_provider"] is False
    assert report1.evaluation_conditions["is_live_vector_store"] is False

    # 2. BenchmarkSynthetic provider injected into PRODUCTION mode
    synthetic_provider = BenchmarkSemanticEmbeddingService()
    cfg2 = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        embedding_provider=synthetic_provider,
    )
    report2 = evaluate_production_retrieval(cfg2)

    assert report2.is_live is False
    assert report2.is_synthetic is True
    assert "simulated" in report2.provider_name.lower() or "synthetic" in report2.provider_name.lower()
    assert report2.vector_store_type == "in_memory (local evaluation)"
    assert report2.infrastructure_limitation is not None
    assert report2.evaluation_conditions["is_live_provider"] is False

    # 3. OpenAIEmbeddingProvider with a mock client injected into PRODUCTION mode
    mock_client = MagicMock()
    item = MagicMock()
    item.embedding = [0.05] * 1536
    item.index = 0
    mock_client.embeddings.create.return_value = MagicMock(data=[item])
    config = EmbeddingConfig(provider="openai", model="text-embedding-3-small", api_key="sk-test", dimension=1536)
    from nexora.embedding.openai_provider import OpenAIEmbeddingProvider
    mock_openai_provider = OpenAIEmbeddingProvider(config, client=mock_client)
    cfg3 = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        embedding_provider=mock_openai_provider,
    )
    report3 = evaluate_production_retrieval(cfg3)

    assert report3.is_live is False
    assert report3.is_synthetic is True
    assert "simulated" in report3.provider_name.lower()
    assert report3.vector_store_type == "in_memory (local evaluation)"
    assert report3.infrastructure_limitation is not None
    assert report3.evaluation_conditions["is_live_provider"] is False


# ---------------------------------------------------------------------------
# 3. Configured Application Pipeline Mode (Hash Provider)
# ---------------------------------------------------------------------------
def test_configured_pipeline_mode_reports_honestly():
    """Verify that configured pipeline mode runs on actual default provider and notes non-semantic hash."""
    # Default pipeline uses HashEmbeddingProvider
    cfg = ProductionEvaluationConfig(mode=EvaluationMode.CONFIGURED_PIPELINE)
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert report.mode == "configured_pipeline"
    assert report.provider_name == "hash"
    assert report.is_semantic is False
    assert report.is_synthetic is False
    assert report.is_live is False
    assert report.vector_store_type == "in_memory (local evaluation)"
    # Reports honest limitation about hash non-semantic nature
    assert "deterministic text hashing and does not model semantic" in report.infrastructure_limitation

    # Metrics on hash provider reflect non-semantic nature (low question-to-passage similarity)
    assert report.metrics is not None
    assert report.metrics.total_queries == 13


# ---------------------------------------------------------------------------
# 4. Synthetic Mode Is Strictly Separated
# ---------------------------------------------------------------------------
def test_synthetic_mode_is_clearly_demarcated():
    """Synthetic mode is explicitly flagged as synthetic and isolated from production."""
    cfg = ProductionEvaluationConfig(mode=EvaluationMode.SYNTHETIC)
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert report.mode == "synthetic"
    assert report.provider_name == "benchmark_synthetic"
    assert report.is_synthetic is True
    assert report.is_semantic is True
    assert report.is_live is False
    assert report.vector_store_type == "in_memory (local evaluation)"
    assert report.metrics is not None
    assert report.metrics.mean_reciprocal_rank >= 0.75


# ---------------------------------------------------------------------------
# 5. Paper-Scope Isolation Measurement
# ---------------------------------------------------------------------------
def test_paper_scope_isolation_measurement():
    """Test paper-scope isolation measurement comparing unscoped vs scoped retrieval."""
    provider = BenchmarkSemanticEmbeddingService()
    store = InMemoryVectorStore(dimension=provider.dimension)

    # Ingest chunks from two papers
    c1 = VectorRecord("c1", provider.embed_text("Self-attention complexity"), {"paper_id": PAPER_TRANSFORMER, "text": "Self-attention complexity"})
    c2 = VectorRecord("c2", provider.embed_text("FlashAttention IO complexity"), {"paper_id": PAPER_FLASHATTN, "text": "FlashAttention IO complexity"})
    store.upsert([c1, c2])

    retrieval_service = RetrievalService(provider, store)
    isolation = evaluate_paper_scope_isolation(
        retrieval_service,
        target_paper_id=PAPER_TRANSFORMER,
        query="complexity in attention mechanisms",
    )

    # Unscoped should contain chunks from other paper (leakage)
    assert isolation["unscoped_retrieved_count"] == 2
    assert isolation["unscoped_cross_paper_chunks"] == 1
    assert isolation["unscoped_leakage_rate"] == 0.5

    # Scoped must have zero cross-paper leakage
    assert isolation["scoped_retrieved_count"] == 1
    assert isolation["scoped_cross_paper_chunks"] == 0
    assert isolation["scoped_leakage_rate"] == 0.0
    assert isolation["isolation_successful"] is True


# ---------------------------------------------------------------------------
# 6. Threshold Calibration with Actual Measurements
# ---------------------------------------------------------------------------
def test_threshold_calibration_measured_results():
    """Verify that calibrate_retrieval_thresholds produces comparable before-and-after measurements."""
    provider = BenchmarkSemanticEmbeddingService()
    store = InMemoryVectorStore(dimension=provider.dimension)

    from nexora.retrieval.benchmark import build_benchmark_corpus, get_benchmark_cases
    store.upsert(build_benchmark_corpus(provider))
    retrieval_service = RetrievalService(provider, store)

    cases = get_benchmark_cases()
    results = calibrate_retrieval_thresholds(
        retrieval_service,
        cases,
        candidate_thresholds=(None, 0.20, 0.35, 0.50),
    )

    assert len(results) == 4

    # Baseline with threshold None has 0% unanswerable rejection (false alarms)
    base = results[0]
    assert base["min_score"] is None
    assert base["unanswerable_rejection_rate"] == 0.0
    assert base["empty_retrievals"] == 0

    # Calibrated threshold 0.35 achieves 100% clean unanswerable rejection
    calibrated = results[2]
    assert calibrated["min_score"] == 0.35
    assert calibrated["unanswerable_rejection_rate"] == 1.0
    assert calibrated["empty_retrievals"] == 2


# ---------------------------------------------------------------------------
# 7. Serialization and Dictionary Conversion
# ---------------------------------------------------------------------------
def test_production_evaluation_report_to_dict():
    """Verify to_dict produces a complete, serializable dictionary."""
    report = evaluate_production_retrieval(ProductionEvaluationConfig(mode=EvaluationMode.SYNTHETIC))
    data = report.to_dict()

    assert data["mode"] == "synthetic"
    assert data["status"] == "completed"
    assert data["is_live"] is False
    assert data["is_synthetic"] is True
    assert data["vector_store_type"] == "in_memory (local evaluation)"
    assert "evaluation_conditions" in data
    assert "metrics" in data
    assert "total_queries" in data["metrics"]
    assert "paper_scope_isolation" in data
    assert "threshold_calibration" in data
