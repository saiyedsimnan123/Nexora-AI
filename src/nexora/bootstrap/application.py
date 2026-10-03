"""Composition root: the one place where concrete services are assembled.

Nothing here runs at import time. Concrete infrastructure (the Qdrant client)
is created here, explicitly, only when the caller does not inject one; the
real client's constructor may contact the server, so tests always inject a
fake. ensure_collection() is never called here.
"""

from __future__ import annotations

from typing import Any

from nexora.api.service import ResearchApplicationService
from nexora.config.settings import AppSettings, ConfigurationError
from nexora.llm.service import LLMService, create_llm_provider
from nexora.research.query import ResearchQueryService
from nexora.research.service import ResearchAnswerService
from nexora.retrieval.evidence import EvidenceContextBuilder
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.qdrant import QdrantVectorStore
from nexora.retrieval.qdrant_config import QdrantConfig
from nexora.retrieval.quality import RetrievalQualityService
from nexora.retrieval.service import RetrievalService
from nexora.retrieval.store import VectorStore


def _check_dimension(label: str, component: Any, expected: int) -> None:
    actual = getattr(component, "dimension", None)
    if isinstance(actual, int) and not isinstance(actual, bool) and actual != expected:
        raise ConfigurationError(
            f"{label} dimension {actual} does not match embedding dimension {expected}"
        )


def _build_qdrant_client(config: QdrantConfig) -> Any:
    """Create the real Qdrant client. This is the composition root's job, so
    QdrantVectorStore never has to create one itself. Called only when the
    caller did not inject a client; the SDK is imported lazily."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=config.url, api_key=config.api_key, timeout=config.timeout)


def _select_vector_store(settings: AppSettings, qdrant_client: Any) -> VectorStore:
    if settings.vector_store == "qdrant":
        client = qdrant_client if qdrant_client is not None else _build_qdrant_client(settings.qdrant)
        return QdrantVectorStore(
            settings.qdrant, dimension=settings.embedding_dimension, client=client
        )
    return InMemoryVectorStore()


def build_research_application(
    settings: AppSettings,
    *,
    embedding_service: Any,
    vector_store: VectorStore | None = None,
    llm_provider: Any = None,
    qdrant_client: Any = None,
) -> ResearchApplicationService:
    """Assemble the full research pipeline from settings and injected pieces.

    ``embedding_service`` is required: build it from the embedding config with
    the existing embedding factory/service and pass it in. ``vector_store`` and
    ``llm_provider`` default to what ``settings`` selects; tests inject fakes.
    """
    if not isinstance(settings, AppSettings):
        raise ConfigurationError("settings must be an AppSettings")
    if not callable(getattr(embedding_service, "embed_text", None)):
        raise ConfigurationError("embedding_service must provide embed_text(text)")
    _check_dimension("embedding_service", embedding_service, settings.embedding_dimension)

    if vector_store is None:
        vector_store = _select_vector_store(settings, qdrant_client)
    elif not isinstance(vector_store, VectorStore):
        raise ConfigurationError("vector_store must implement the VectorStore protocol")
    _check_dimension("vector_store", vector_store, settings.embedding_dimension)

    if llm_provider is None:
        llm_provider = create_llm_provider(settings.llm)
    llm_service = LLMService(llm_provider)

    retrieval = RetrievalService(embedding_service, vector_store)
    quality = RetrievalQualityService(retrieval, settings.retrieval)
    evidence = EvidenceContextBuilder(
        quality, max_total_characters=settings.max_evidence_characters
    )
    answers = ResearchAnswerService(llm_service)
    query_service = ResearchQueryService(evidence, answers)
    return ResearchApplicationService(query_service)


def build_http_app(settings: AppSettings, **dependencies: Any) -> Any:
    """Compose the pipeline and wrap it in the FastAPI app (create_app is unchanged)."""
    from nexora.api.app import create_app  # local import keeps FastAPI optional here

    return create_app(build_research_application(settings, **dependencies))
