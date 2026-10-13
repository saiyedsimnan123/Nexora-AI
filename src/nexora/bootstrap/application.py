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


def build_ingestion_service(
    settings: AppSettings,
    *,
    persistence_service: Any = None,
    session: Any = None,
    vector_store: VectorStore | None = None,
    embedding_service: Any,
    qdrant_client: Any = None,
) -> Any:
    """Assemble DocumentIngestionService using settings and injected dependencies.

    Keeps construction lazy; performs no network calls and creates no database tables.
    """
    if not isinstance(settings, AppSettings):
        raise ConfigurationError("settings must be an AppSettings")
    if not callable(getattr(embedding_service, "embed_text", None)) and not callable(
        getattr(embedding_service, "embed_texts", None)
    ):
        raise ConfigurationError("embedding_service must provide embed_text/embed_texts")
    _check_dimension("embedding_service", embedding_service, settings.embedding_dimension)

    if vector_store is None:
        vector_store = _select_vector_store(settings, qdrant_client)
    elif not isinstance(vector_store, VectorStore):
        raise ConfigurationError("vector_store must implement the VectorStore protocol")
    _check_dimension("vector_store", vector_store, settings.embedding_dimension)

    if persistence_service is None:
        if session is None:
            raise ConfigurationError("persistence_service or session is required")
        from nexora.document.service import DocumentPersistenceService

        persistence_service = DocumentPersistenceService(session)

    from nexora.document.ingestion import DocumentIngestionService

    return DocumentIngestionService(
        persistence_service,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )


def build_http_app(
    settings: AppSettings,
    *,
    embedding_service: Any,
    vector_store: VectorStore | None = None,
    llm_provider: Any = None,
    qdrant_client: Any = None,
    session_factory: Any = None,
    persistence_service: Any = None,
    ingestion_service: Any = None,
    document_pipeline: Any = None,
    max_upload_size_bytes: int = 50 * 1024 * 1024,
    **dependencies: Any,
) -> Any:
    """Compose the pipeline and wrap it in the FastAPI app."""
    from nexora.api.app import create_app  # local import keeps FastAPI optional here

    if not isinstance(settings, AppSettings):
        raise ConfigurationError("settings must be an AppSettings")

    if vector_store is None:
        vector_store = _select_vector_store(settings, qdrant_client)
    elif not isinstance(vector_store, VectorStore):
        raise ConfigurationError("vector_store must implement the VectorStore protocol")
    _check_dimension("vector_store", vector_store, settings.embedding_dimension)

    research_service = build_research_application(
        settings,
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_provider=llm_provider,
        qdrant_client=qdrant_client,
    )

    if ingestion_service is None:
        session = dependencies.get("session")
        if persistence_service is not None or session is not None:
            ingestion_service = build_ingestion_service(
                settings,
                persistence_service=persistence_service,
                session=session,
                vector_store=vector_store,
                embedding_service=embedding_service,
                qdrant_client=qdrant_client,
            )

    return create_app(
        research_service,
        ingestion_service=ingestion_service,
        document_pipeline=document_pipeline,
        session_factory=session_factory,
        max_upload_size_bytes=max_upload_size_bytes,
    )
