"""Production retrieval evaluation harness for Nexora AI (Milestone 12C-31).

Evaluates the real application retrieval path—including configured embedding providers,
vector stores, and retrieval-quality policies—with a fixed, labeled research corpus.

Explicitly separates:
- Production mode (exercises real external embedding models like OpenAI text-embedding-3-small)
- Configured application pipeline mode (exercises configured local or hash providers)
- Synthetic benchmark mode (deterministic CI testing)

Strictly fails open to honesty: if external model credentials or vector infrastructure
are unavailable, reports that limitation explicitly without silently substituting synthetic embeddings.
"""

from __future__ import annotations

import copy
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Sequence

from nexora.embedding.base import EmbeddingProvider
from nexora.embedding.config import ENV_API_KEY, EmbeddingConfig
from nexora.embedding.factory import create_embedding_provider
from nexora.embedding.hash_provider import HashEmbeddingProvider
from nexora.embedding.service import EmbeddingService
from nexora.retrieval.benchmark import (
    DOC_DISTRACTOR,
    DOC_FLASHATTN,
    DOC_LORA,
    DOC_TRANSFORMER,
    PAPER_DISTRACTOR,
    PAPER_FLASHATTN,
    PAPER_LORA,
    PAPER_TRANSFORMER,
    BenchmarkSemanticEmbeddingService,
    get_benchmark_cases,
)
from nexora.retrieval.evaluation import (
    RetrievalEvaluationMetrics,
    compute_precision_recall_at_k,
    compute_reciprocal_rank,
    evaluate_retrieval,
)
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.quality import RetrievalQualityPolicy, RetrievalQualityService
from nexora.retrieval.service import RetrievalService
from nexora.retrieval.store import VectorStore


def _get_code_identity() -> dict[str, Any]:
    """Retrieve version and git commit hash if available."""
    try:
        import nexora
        version = getattr(nexora, "__version__", "0.1.0")
    except Exception:
        version = "0.1.0"
    commit = None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            commit = res.stdout.strip()
    except Exception:
        commit = None
    return {
        "version": version,
        "commit": commit,
    }


class EvaluationMode(str, Enum):
    """Mode under which retrieval evaluation is executed."""

    PRODUCTION = "production"
    CONFIGURED_PIPELINE = "configured_pipeline"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class ProductionEvaluationConfig:
    """Configuration for production retrieval evaluation."""

    mode: EvaluationMode = EvaluationMode.PRODUCTION
    embedding_provider: Any = None
    embedding_service: Any = None
    vector_store: VectorStore | None = None
    quality_policy: RetrievalQualityPolicy | None = None
    k_values: tuple[int, ...] = (1, 3, 5)
    api_key: str | None = None
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536


@dataclass(frozen=True)
class ProductionEvaluationReport:
    """Comprehensive, honest report of retrieval evaluation results."""

    mode: str
    status: str  # "completed", "unavailable", "failed"
    provider_name: str
    model_name: str
    dimension: int | None
    vector_store_type: str
    is_semantic: bool
    is_synthetic: bool
    infrastructure_limitation: str | None
    corpus_info: dict[str, Any]
    policy_settings: dict[str, Any]
    is_live: bool = False
    provider_live: bool = False
    vector_store_live: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    code_version: dict[str, Any] = field(default_factory=_get_code_identity)
    error: str | None = None
    unavailable_dependencies: list[str] = field(default_factory=list)
    evaluation_conditions: dict[str, Any] | None = None
    metrics: RetrievalEvaluationMetrics | None = None
    paper_scope_isolation: dict[str, Any] | None = None
    threshold_calibration: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation."""
        metrics_dict = None
        if self.metrics is not None:
            metrics_dict = {
                "total_queries": self.metrics.total_queries,
                "mean_reciprocal_rank": self.metrics.mean_reciprocal_rank,
                "recall_at_k": self.metrics.recall_at_k,
                "precision_at_k": self.metrics.precision_at_k,
                "empty_retrievals": self.metrics.empty_retrievals,
                "unanswerable_queries": self.metrics.unanswerable_queries,
                "unanswerable_rejections": self.metrics.unanswerable_rejections,
                "unanswerable_rejection_rate": self.metrics.unanswerable_rejection_rate,
                "category_metrics": self.metrics.category_metrics,
            }
        return {
            "timestamp": self.timestamp,
            "code_version": self.code_version,
            "mode": self.mode,
            "status": self.status,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "vector_store_type": self.vector_store_type,
            "is_semantic": self.is_semantic,
            "is_synthetic": self.is_synthetic,
            "is_live": self.is_live,
            "provider_live": self.provider_live,
            "vector_store_live": self.vector_store_live,
            "infrastructure_limitation": self.infrastructure_limitation,
            "error": self.error,
            "unavailable_dependencies": self.unavailable_dependencies,
            "corpus_info": self.corpus_info,
            "policy_settings": self.policy_settings,
            "evaluation_conditions": self.evaluation_conditions,
            "metrics": metrics_dict,
            "paper_scope_isolation": self.paper_scope_isolation,
            "threshold_calibration": self.threshold_calibration,
        }


def format_vector_store_label(vector_store: VectorStore | None) -> str:
    """Format an honest vector store description distinguishing live from local evaluation."""
    if vector_store is None:
        return "uninitialized"
    store_cls_name = type(vector_store).__name__
    if isinstance(vector_store, InMemoryVectorStore) or store_cls_name == "InMemoryVectorStore":
        return "in_memory (local evaluation)"
    if store_cls_name == "QdrantVectorStore":
        client = getattr(vector_store, "_client", None)
        client_cls = type(client).__name__
        if "Mock" in client_cls or "Fake" in client_cls:
            return "qdrant (simulated local evaluation)"
        return "qdrant (live persistent)"
    return f"{store_cls_name} (local evaluation)"


def is_genuine_live_qdrant_store(vector_store: VectorStore | None) -> bool:
    """Return True if vector_store is a genuine live Qdrant store (not in-memory or mock)."""
    if vector_store is None:
        return False
    if isinstance(vector_store, InMemoryVectorStore):
        return False
    store_cls_name = type(vector_store).__name__
    if store_cls_name != "QdrantVectorStore":
        return False
    client = getattr(vector_store, "_client", None)
    if client is None:
        return False
    client_cls = type(client).__name__
    if any(k in client_cls for k in ("Mock", "Fake", "Stub")):
        return False
    config = getattr(vector_store, "_config", None)
    url = getattr(config, "url", "")
    if not url or url == ":memory:":
        return False
    return True


def is_simulated_or_synthetic_provider(provider: Any) -> bool:
    """Return True if provider is a mock, simulation, or synthetic test double."""
    if provider is None:
        return False
    if getattr(provider, "is_simulated", False) or getattr(provider, "is_synthetic", False):
        return True
    if hasattr(provider, "_synthetic_backend"):
        return True
    cls_name = type(provider).__name__
    if any(k in cls_name for k in ("Simulated", "Mock", "Fake", "Synthetic", "Double", "Stub")):
        return True
    if isinstance(provider, BenchmarkSemanticEmbeddingService):
        return True
    provider_id = str(getattr(provider, "provider", getattr(provider, "name", ""))).lower()
    from nexora.embedding.openai_provider import OpenAIEmbeddingProvider
    if provider_id == "openai" and not isinstance(provider, OpenAIEmbeddingProvider):
        return True
    if isinstance(provider, OpenAIEmbeddingProvider):
        client = getattr(provider, "_client", None)
        if client is not None:
            client_cls = type(client).__name__
            if any(k in client_cls for k in ("Mock", "Fake", "Stub", "MagicMock")):
                return True
    return False


def is_genuine_live_openai_provider(provider: Any) -> bool:
    """Return True if provider is a live, genuine OpenAI provider with real client and credentials."""
    from nexora.embedding.openai_provider import OpenAIEmbeddingProvider
    if not isinstance(provider, OpenAIEmbeddingProvider):
        return False
    if is_simulated_or_synthetic_provider(provider):
        return False
    config = getattr(provider, "_config", None)
    if config is None or not getattr(config, "has_api_key", False):
        return False
    client = getattr(provider, "_client", None)
    if client is None:
        return False
    return True


def check_production_credentials(api_key: str | None = None) -> tuple[bool, str | None]:
    """Check if production external model credentials are configured.

    Returns (is_available, error_message_or_none).
    """
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv(ENV_API_KEY)
    if not key or not str(key).strip():
        return False, (
            "External model credentials are unavailable. Set OPENAI_API_KEY or "
            f"{ENV_API_KEY} to exercise the real OpenAI embedding provider."
        )
    return True, None


def _get_corpus_metadata() -> dict[str, Any]:
    return {
        "papers_count": 4,
        "chunks_count": 15,
        "papers": [
            {"paper_id": PAPER_TRANSFORMER, "title": "Attention Is All You Need", "chunks": 4},
            {"paper_id": PAPER_FLASHATTN, "title": "FlashAttention: Fast and Memory-Efficient Exact Attention", "chunks": 4},
            {"paper_id": PAPER_LORA, "title": "LoRA: Low-Rank Adaptation of Large Language Models", "chunks": 3},
            {"paper_id": PAPER_DISTRACTOR, "title": "Biological and Hardware Near-Match Distractors", "chunks": 4},
        ],
    }


def _get_policy_metadata(policy: RetrievalQualityPolicy | None) -> dict[str, Any]:
    if policy is None:
        return {"min_score": None, "max_results": 10, "max_results_per_paper": None, "allowed_paper_ids": None}
    return {
        "min_score": policy.min_score,
        "max_results": policy.max_results,
        "max_results_per_document": policy.max_results_per_document,
        "max_results_per_paper": policy.max_results_per_paper,
        "allowed_paper_ids": list(policy.allowed_paper_ids) if policy.allowed_paper_ids else None,
    }


def evaluate_paper_scope_isolation(
    retrieval_service: RetrievalService,
    target_paper_id: str = PAPER_TRANSFORMER,
    query: str = "Attention mechanisms and computational complexity",
) -> dict[str, Any]:
    """Measure paper-scope isolation by comparing unscoped vs scoped retrieval."""
    # 1. Unscoped retrieval
    unscoped_res = retrieval_service.retrieve(query, limit=10)
    unscoped_papers = [c.metadata.get("paper_id") for c in unscoped_res.results]
    unscoped_leakage_count = sum(1 for p in unscoped_papers if p != target_paper_id)

    # 2. Scoped retrieval with allowed_paper_ids
    scoped_policy = RetrievalQualityPolicy(allowed_paper_ids=(target_paper_id,))
    scoped_service = RetrievalQualityService(retrieval_service, scoped_policy)
    scoped_res = scoped_service.retrieve(query)
    scoped_papers = [c.metadata.get("paper_id") for c in scoped_res.results]
    scoped_leakage_count = sum(1 for p in scoped_papers if p != target_paper_id)

    return {
        "target_paper_id": target_paper_id,
        "query": query,
        "unscoped_retrieved_count": len(unscoped_res.results),
        "unscoped_cross_paper_chunks": unscoped_leakage_count,
        "unscoped_leakage_rate": round(unscoped_leakage_count / len(unscoped_res.results), 4) if unscoped_res.results else 0.0,
        "scoped_retrieved_count": len(scoped_res.results),
        "scoped_cross_paper_chunks": scoped_leakage_count,
        "scoped_leakage_rate": round(scoped_leakage_count / len(scoped_res.results), 4) if scoped_res.results else 0.0,
        "isolation_successful": (scoped_leakage_count == 0),
    }


def calibrate_retrieval_thresholds(
    retrieval_service: RetrievalService,
    benchmark_cases: Sequence[dict[str, Any]],
    candidate_thresholds: Sequence[float | None] = (None, 0.20, 0.35, 0.50),
    *,
    k_values: Sequence[int] = (1, 3, 5),
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Calibrate retrieval min_score thresholds using actual measured results.

    Tests each threshold deterministically and records the impact on:
    - Mean Reciprocal Rank (MRR)
    - Recall@1 and Precision@1
    - Unanswerable clean rejection rate
    """
    calibration_results: list[dict[str, Any]] = []

    for threshold in candidate_thresholds:
        policy = RetrievalQualityPolicy(
            min_score=threshold,
            max_results=max_results,
            max_results_per_paper=2,
            max_results_per_document=2,
        )
        service = RetrievalQualityService(retrieval_service, policy)

        metrics = evaluate_retrieval(
            benchmark_cases,
            lambda q, limit=max_results: service.retrieve(q),
            k_values=k_values,
        )

        calibration_results.append({
            "min_score": threshold,
            "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
            "precision_at_1": metrics.precision_at_k.get(1, 0.0),
            "recall_at_1": metrics.recall_at_k.get(1, 0.0),
            "recall_at_3": metrics.recall_at_k.get(3, 0.0),
            "unanswerable_rejection_rate": metrics.unanswerable_rejection_rate,
            "empty_retrievals": metrics.empty_retrievals,
        })

    return calibration_results


def evaluate_production_retrieval(
    config: ProductionEvaluationConfig | None = None,
) -> ProductionEvaluationReport:
    """Execute retrieval evaluation in the specified mode with explicit limitations.

    If Mode is PRODUCTION and external credentials or vector infrastructure are unavailable,
    reports that limitation explicitly without silently substituting synthetic embeddings.
    """
    cfg = config or ProductionEvaluationConfig()
    corpus_info = _get_corpus_metadata()
    policy_info = _get_policy_metadata(cfg.quality_policy)

    # -----------------------------------------------------------------------
    # 1. Production Mode: Requires real external model credentials
    # -----------------------------------------------------------------------
    candidate_provider_live = False
    if cfg.mode == EvaluationMode.PRODUCTION:
        if cfg.embedding_provider is not None:
            provider = cfg.embedding_provider
            if is_simulated_or_synthetic_provider(provider):
                # Injected test double or synthetic provider: do not report as genuine production!
                candidate_provider_live = False
                is_synthetic = True
                is_semantic = True
                base_name = getattr(provider, "name", getattr(provider, "provider", "openai"))
                provider_name = f"{base_name} (simulated)" if "(simulated)" not in str(base_name) else str(base_name)
                model_name = getattr(provider, "model", cfg.model_name)
                dim = getattr(provider, "dimension", cfg.dimension)
                simulated_limitation = (
                    "Simulated test double: metrics reflect deterministic mock embeddings, "
                    "not genuine live production API measurements."
                )
            else:
                candidate_provider_live = is_genuine_live_openai_provider(provider)
                is_synthetic = False
                is_semantic = True
                provider_name = getattr(provider, "name", getattr(provider, "provider", "openai"))
                model_name = getattr(provider, "model", cfg.model_name)
                dim = getattr(provider, "dimension", cfg.dimension)
                simulated_limitation = None
        else:
            has_credentials, limitation_msg = check_production_credentials(cfg.api_key)
            if not has_credentials:
                # Explicit limitation reporting: DO NOT silently substitute synthetic embedding!
                return ProductionEvaluationReport(
                    mode=cfg.mode.value,
                    status="unavailable",
                    provider_name="openai",
                    model_name=cfg.model_name,
                    dimension=cfg.dimension,
                    vector_store_type=format_vector_store_label(cfg.vector_store),
                    is_semantic=True,
                    is_synthetic=False,
                    is_live=False,
                    provider_live=False,
                    vector_store_live=False,
                    infrastructure_limitation=limitation_msg,
                    error=None,
                    unavailable_dependencies=["openai_credentials"],
                    corpus_info=corpus_info,
                    policy_settings=policy_info,
                    evaluation_conditions={
                        "is_live_provider": False,
                        "is_live_vector_store": False,
                        "credentials_available": False,
                    },
                    metrics=None,
                    paper_scope_isolation=None,
                    threshold_calibration=None,
                )

            # Build real production embedding provider
            try:
                emb_config = EmbeddingConfig(
                    provider="openai",
                    model=cfg.model_name,
                    api_key=cfg.api_key or os.getenv("OPENAI_API_KEY") or os.getenv(ENV_API_KEY),
                    dimension=cfg.dimension,
                )
                provider = create_embedding_provider(emb_config)
                candidate_provider_live = True
                is_synthetic = False
                is_semantic = True
                provider_name = "openai"
                model_name = cfg.model_name
                dim = cfg.dimension
                simulated_limitation = None
            except Exception as exc:
                return ProductionEvaluationReport(
                    mode=cfg.mode.value,
                    status="failed",
                    provider_name="openai",
                    model_name=cfg.model_name,
                    dimension=cfg.dimension,
                    vector_store_type=format_vector_store_label(cfg.vector_store),
                    is_semantic=True,
                    is_synthetic=False,
                    is_live=False,
                    provider_live=False,
                    vector_store_live=False,
                    infrastructure_limitation=str(exc),
                    error=f"Failed to initialize OpenAI provider: {exc}",
                    unavailable_dependencies=["openai_provider"],
                    corpus_info=corpus_info,
                    policy_settings=policy_info,
                    evaluation_conditions={
                        "is_live_provider": False,
                        "is_live_vector_store": False,
                    },
                    metrics=None,
                    paper_scope_isolation=None,
                    threshold_calibration=None,
                )

    # -----------------------------------------------------------------------
    # 2. Configured Pipeline Mode: Uses whatever pipeline is currently configured
    # -----------------------------------------------------------------------
    elif cfg.mode == EvaluationMode.CONFIGURED_PIPELINE:
        if cfg.embedding_provider is not None:
            provider = cfg.embedding_provider
        elif cfg.embedding_service is not None:
            provider = cfg.embedding_service
        else:
            # Default configured provider from application config
            emb_config = EmbeddingConfig()
            provider = create_embedding_provider(emb_config)

        candidate_provider_live = False
        is_synthetic = False
        if isinstance(provider, HashEmbeddingProvider):
            provider_name = "hash"
            model_name = getattr(provider, "model", "hash-sha256")
            is_semantic = False
        else:
            provider_name = getattr(provider, "provider", getattr(provider, "name", type(provider).__name__))
            model_name = getattr(provider, "model", "default")
            is_semantic = (str(provider_name).lower() == "openai")
            if is_semantic and is_genuine_live_openai_provider(provider):
                candidate_provider_live = True
        dim = getattr(provider, "dimension", None)
        simulated_limitation = None

    # -----------------------------------------------------------------------
    # 3. Synthetic Mode: Offline deterministic CI benchmark
    # -----------------------------------------------------------------------
    else:
        provider = BenchmarkSemanticEmbeddingService()
        provider_name = "benchmark_synthetic"
        model_name = "concept_projection_64d"
        dim = 64
        is_semantic = True
        is_synthetic = True
        candidate_provider_live = False
        simulated_limitation = None

    # Assemble pipeline pieces
    try:
        embedding_service = (
            cfg.embedding_service
            if cfg.embedding_service is not None
            else EmbeddingService(provider)
        )

        vector_store = (
            cfg.vector_store
            if cfg.vector_store is not None
            else InMemoryVectorStore(dimension=dim)
        )
    except Exception as exc:
        return ProductionEvaluationReport(
            mode=cfg.mode.value,
            status="failed",
            provider_name=str(provider_name),
            model_name=str(model_name),
            dimension=dim,
            vector_store_type=format_vector_store_label(cfg.vector_store),
            is_semantic=is_semantic,
            is_synthetic=is_synthetic,
            is_live=False,
            provider_live=False,
            vector_store_live=False,
            infrastructure_limitation=f"Failed to assemble embedding service or vector store: {exc}",
            error=f"Service initialization failed: {exc}",
            unavailable_dependencies=["embedding_service"],
            corpus_info=corpus_info,
            policy_settings=policy_info,
            evaluation_conditions={"is_live_provider": False, "is_live_vector_store": False},
            metrics=None,
            paper_scope_isolation=None,
            threshold_calibration=None,
        )

    if hasattr(vector_store, "ensure_collection"):
        try:
            vector_store.ensure_collection()
        except Exception as exc:
            return ProductionEvaluationReport(
                mode=cfg.mode.value,
                status="failed",
                provider_name=str(provider_name),
                model_name=str(model_name),
                dimension=dim,
                vector_store_type=format_vector_store_label(vector_store),
                is_semantic=is_semantic,
                is_synthetic=is_synthetic,
                is_live=False,
                provider_live=False,
                vector_store_live=False,
                infrastructure_limitation=f"Failed to initialize vector store collection: {exc}",
                error=f"Vector store collection creation failed: {exc}",
                unavailable_dependencies=["vector_store"],
                corpus_info=corpus_info,
                policy_settings=policy_info,
                evaluation_conditions={"is_live_provider": False, "is_live_vector_store": False},
                metrics=None,
                paper_scope_isolation=None,
                threshold_calibration=None,
            )

    # Ingest fixed labeled benchmark corpus
    from nexora.retrieval.benchmark import _RAW_CHUNKS
    corpus_records = []
    actual_embeddings_completed = 0
    try:
        for raw in _RAW_CHUNKS:
            vec = embedding_service.embed_text(raw["text"])
            actual_embeddings_completed += 1
            payload = {
                "chunk_id": raw["chunk_id"],
                "document_id": raw["document_id"],
                "paper_id": raw["paper_id"],
                "title": raw["title"],
                "filename": raw["filename"],
                "chunk_index": raw["chunk_index"],
                "text": raw["text"],
            }
            corpus_records.append(VectorRecord(id=raw["chunk_id"], vector=vec, payload=payload))

        vector_store.upsert(corpus_records)
    except Exception as exc:
        vector_store_label = format_vector_store_label(vector_store)
        return ProductionEvaluationReport(
            mode=cfg.mode.value,
            status="failed",
            provider_name=str(provider_name),
            model_name=str(model_name),
            dimension=dim,
            vector_store_type=vector_store_label,
            is_semantic=is_semantic,
            is_synthetic=is_synthetic,
            is_live=False,
            provider_live=False,
            vector_store_live=False,
            infrastructure_limitation=f"Execution failed during corpus embedding/indexing: {exc}",
            error=f"Embedding or indexing call failed: {exc}",
            unavailable_dependencies=[],
            corpus_info=corpus_info,
            policy_settings=policy_info,
            evaluation_conditions={
                "is_live_provider": False,
                "is_live_vector_store": False,
                "vector_store_type": vector_store_label,
                "actual_embeddings_completed": actual_embeddings_completed,
            },
            metrics=None,
            paper_scope_isolation=None,
            threshold_calibration=None,
        )

    # Truthful live status: Mark provider live ONLY after actual embedding calls complete successfully!
    provider_live = candidate_provider_live and (actual_embeddings_completed > 0)
    vector_store_live = is_genuine_live_qdrant_store(vector_store)
    is_live = provider_live

    retrieval_service = RetrievalService(embedding_service, vector_store, cache_queries=True)

    # Apply quality policy if configured, otherwise evaluate raw retrieval
    policy = cfg.quality_policy
    if policy is not None:
        quality_service = RetrievalQualityService(retrieval_service, policy)
        retrieve_fn: Callable[[str], SearchResult] = quality_service.retrieve
    else:
        retrieve_fn = lambda q, limit=5: retrieval_service.retrieve(q, limit=limit)

    # Execute deterministic queries
    try:
        benchmark_cases = get_benchmark_cases()
        metrics = evaluate_retrieval(benchmark_cases, retrieve_fn, k_values=cfg.k_values)

        # Measure paper scope isolation
        isolation_res = evaluate_paper_scope_isolation(retrieval_service, target_paper_id=PAPER_TRANSFORMER)

        # Calibrate thresholds
        calibration_res = calibrate_retrieval_thresholds(
            retrieval_service,
            benchmark_cases,
            candidate_thresholds=(None, 0.20, 0.35, 0.50),
            k_values=cfg.k_values,
        )
    except Exception as exc:
        vector_store_label = format_vector_store_label(vector_store)
        return ProductionEvaluationReport(
            mode=cfg.mode.value,
            status="failed",
            provider_name=str(provider_name),
            model_name=str(model_name),
            dimension=dim,
            vector_store_type=vector_store_label,
            is_semantic=is_semantic,
            is_synthetic=is_synthetic,
            is_live=False,
            provider_live=False,
            vector_store_live=False,
            infrastructure_limitation=f"Execution failed during retrieval query evaluation: {exc}",
            error=f"Retrieval query evaluation failed: {exc}",
            unavailable_dependencies=[],
            corpus_info=corpus_info,
            policy_settings=_get_policy_metadata(policy),
            evaluation_conditions={
                "is_live_provider": False,
                "is_live_vector_store": False,
                "vector_store_type": vector_store_label,
                "actual_embeddings_completed": actual_embeddings_completed,
            },
            metrics=None,
            paper_scope_isolation=None,
            threshold_calibration=None,
        )

    limitation_text = simulated_limitation
    if not is_semantic:
        hash_limitation = (
            f"The '{provider_name}' provider uses deterministic text hashing and does not model "
            "semantic conceptual similarity. Production deployment requires semantic embeddings."
        )
        limitation_text = (
            f"{simulated_limitation}; {hash_limitation}"
            if simulated_limitation
            else hash_limitation
        )

    vector_store_label = format_vector_store_label(vector_store)
    eval_conditions = {
        "is_live_provider": provider_live,
        "is_live_vector_store": vector_store_live,
        "vector_store_type": vector_store_label,
        "cache_queries": True,
        "k_values": list(cfg.k_values),
        "total_benchmark_cases": len(benchmark_cases),
        "corpus_chunks_count": len(corpus_records),
        "actual_embeddings_completed": actual_embeddings_completed,
    }

    return ProductionEvaluationReport(
        mode=cfg.mode.value,
        status="completed",
        provider_name=str(provider_name),
        model_name=str(model_name),
        dimension=dim,
        vector_store_type=vector_store_label,
        is_semantic=is_semantic,
        is_synthetic=is_synthetic,
        is_live=is_live,
        provider_live=provider_live,
        vector_store_live=vector_store_live,
        infrastructure_limitation=limitation_text,
        error=None,
        unavailable_dependencies=[],
        corpus_info=corpus_info,
        policy_settings=_get_policy_metadata(policy),
        evaluation_conditions=eval_conditions,
        metrics=metrics,
        paper_scope_isolation=isolation_res,
        threshold_calibration=calibration_res,
    )


if __name__ == "__main__":
    import sys
    from nexora.retrieval.eval_cli import main
    sys.exit(main())
