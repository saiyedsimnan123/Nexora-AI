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
    verifier: Any = None,
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

    verifier_inst = verifier
    if verifier_inst is None:
        from nexora.research.verification import ClaimToEvidenceVerifier
        verifier_inst = ClaimToEvidenceVerifier(llm_service)

    return ResearchApplicationService(query_service, verifier=verifier_inst)


def build_verification_service(
    settings: AppSettings,
    *,
    llm_provider: Any = None,
    expected_paper_id: str | None = None,
) -> Any:
    """Assemble ClaimToEvidenceVerifier using settings and injected dependencies."""
    from nexora.research.verification import ClaimToEvidenceVerifier

    if not isinstance(settings, AppSettings):
        raise ConfigurationError("settings must be an AppSettings")

    if llm_provider is None:
        llm_provider = create_llm_provider(settings.llm)
    llm_service = LLMService(llm_provider)

    return ClaimToEvidenceVerifier(
        llm_service,
        expected_paper_id=expected_paper_id,
    )


def build_ingestion_service(
    settings: AppSettings,
    *,
    session_factory: Any = None,
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

    if session_factory is not None:
        from nexora.document.ingestion import DocumentIngestionService

        return DocumentIngestionService(
            vector_store=vector_store,
            embedding_service=embedding_service,
            session_factory=session_factory,
        )

    if persistence_service is None:
        if session is None:
            raise ConfigurationError(
                "persistence_service, session, or session_factory is required"
            )
        from nexora.document.service import DocumentPersistenceService

        persistence_service = DocumentPersistenceService(session)

    from nexora.document.ingestion import DocumentIngestionService

    return DocumentIngestionService(
        persistence_service,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )


def build_analysis_service(
    settings: AppSettings,
    *,
    llm_provider: Any = None,
    session_factory: Any = None,
    vector_store: VectorStore | None = None,
    max_evidence_characters: int | None = None,
) -> Any:
    """Assemble PaperAnalysisService using settings and injected dependencies."""
    from nexora.research.analysis_service import PaperAnalysisService

    if not isinstance(settings, AppSettings):
        raise ConfigurationError("settings must be an AppSettings")

    if llm_provider is None:
        llm_provider = create_llm_provider(settings.llm)
    llm_service = LLMService(llm_provider)

    max_chars = (
        max_evidence_characters
        if max_evidence_characters is not None
        else settings.max_evidence_characters
    )

    return PaperAnalysisService(
        llm_service,
        session_factory=session_factory,
        vector_store=vector_store,
        max_evidence_characters=max_chars,
    )


def build_comparison_service(
    settings: AppSettings,
    *,
    analysis_service: Any = None,
    llm_provider: Any = None,
    session_factory: Any = None,
    vector_store: VectorStore | None = None,
    max_papers: int = 5,
) -> Any:
    """Assemble PaperComparisonService using settings and injected dependencies."""
    from nexora.research.analysis_service import PaperComparisonService

    if not isinstance(settings, AppSettings):
        raise ConfigurationError("settings must be an AppSettings")

    if llm_provider is None:
        llm_provider = create_llm_provider(settings.llm)
    llm_service = LLMService(llm_provider)

    if analysis_service is None:
        analysis_service = build_analysis_service(
            settings,
            llm_provider=llm_provider,
            session_factory=session_factory,
            vector_store=vector_store,
        )

    return PaperComparisonService(
        analysis_service,
        llm_service,
        max_papers=max_papers,
    )


def build_gap_service(
    settings: AppSettings,
    *,
    analysis_service: Any = None,
    llm_provider: Any = None,
    session_factory: Any = None,
    vector_store: VectorStore | None = None,
    prompt_builder: Any = None,
    max_evidence_characters: int | None = None,
    verifier: Any = None,
) -> Any:
    """Assemble ResearchGapService using settings and injected dependencies."""
    from nexora.research.gap_service import ResearchGapService

    if not isinstance(settings, AppSettings):
        raise ConfigurationError("settings must be an AppSettings")

    if llm_provider is None:
        llm_provider = create_llm_provider(settings.llm)
    llm_service = LLMService(llm_provider)

    if analysis_service is None:
        analysis_service = build_analysis_service(
            settings,
            llm_provider=llm_provider,
            session_factory=session_factory,
            vector_store=vector_store,
        )

    max_chars = (
        max_evidence_characters
        if max_evidence_characters is not None
        else settings.max_evidence_characters
    )

    return ResearchGapService(
        llm_service,
        paper_analysis_service=analysis_service,
        session_factory=session_factory,
        vector_store=vector_store,
        prompt_builder=prompt_builder,
        max_evidence_characters=max_chars,
        verifier=verifier,
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
    analysis_service: Any = None,
    comparison_service: Any = None,
    gap_service: Any = None,
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

    verifier = dependencies.get("verifier") or dependencies.get("verification_service")
    research_service = build_research_application(
        settings,
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_provider=llm_provider,
        qdrant_client=qdrant_client,
        verifier=verifier,
    )

    if ingestion_service is None:
        session = dependencies.get("session")
        if session_factory is not None or persistence_service is not None or session is not None:
            ingestion_service = build_ingestion_service(
                settings,
                session_factory=session_factory,
                persistence_service=persistence_service,
                session=session,
                vector_store=vector_store,
                embedding_service=embedding_service,
                qdrant_client=qdrant_client,
            )

    if analysis_service is None:
        analysis_service = build_analysis_service(
            settings,
            llm_provider=llm_provider,
            session_factory=session_factory,
            vector_store=vector_store,
        )

    if comparison_service is None:
        comparison_service = build_comparison_service(
            settings,
            analysis_service=analysis_service,
            llm_provider=llm_provider,
            session_factory=session_factory,
            vector_store=vector_store,
        )

    if gap_service is None:
        gap_service = dependencies.get("gap_service")
        if gap_service is None:
            gap_service = build_gap_service(
                settings,
                analysis_service=analysis_service,
                llm_provider=llm_provider,
                session_factory=session_factory,
                vector_store=vector_store,
                verifier=verifier,
            )

    rag_pipeline = dependencies.get("rag_pipeline")
    return create_app(
        research_service,
        rag_pipeline=rag_pipeline,
        ingestion_service=ingestion_service,
        analysis_service=analysis_service,
        comparison_service=comparison_service,
        gap_service=gap_service,
        document_pipeline=document_pipeline,
        session_factory=session_factory,
        max_upload_size_bytes=max_upload_size_bytes,
    )

