"""End-to-end RAG orchestration pipeline for Nexora AI.

Chains retrieval, evidence formatting, answer generation, and citation verification:
1. Accepts a research question and retrieval configurations.
2. Retrieves relevant chunks via BM25, semantic vector, or hybrid RRF search.
3. Converts retrieved candidates into an EvidenceContext.
4. Handles empty/unanswerable retrieval results truthfully (no fabricated answers or citations).
5. Generates grounded answers using configurable providers (offline deterministic or live LLM).
6. Verifies citation labels [E#] against retrieved sources to guarantee reference integrity.
7. Packages traceable sources with paper titles and excerpts into a structured RAGResponse.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from nexora.embedding.hash_provider import HashEmbeddingProvider
from nexora.embedding.service import EmbeddingService
from nexora.rag.generator import (
    AnswerGenerator,
    DeterministicMockGenerator,
    GeneratedAnswer,
    LLMAnswerGenerator,
)
from nexora.rag.models import AttributedSource, RAGRequest, RAGResponse
from nexora.retrieval.bm25 import BM25Index
from nexora.retrieval.dataset import (
    get_pilot_benchmark_dataset,
    load_retrieval_dataset,
)
from nexora.retrieval.evidence import (
    EvidenceContext,
    EvidenceItem,
    verify_citations,
)
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I could not find sufficient evidence in the indexed research papers to answer this question."
)

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"bearer\s+[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"api[_\-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
]


def _sanitize_error_message(error: Any) -> str:
    msg = str(error)
    for pattern in _SECRET_PATTERNS:
        msg = pattern.sub("[REDACTED]", msg)
    return msg



def _truncate_excerpt(text: str, max_chars: int = 300) -> str:
    cleaned = text.strip().replace("\n", " ")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."


class RAGPipeline:
    """Orchestrates research retrieval, evidence context building, generation, and attribution."""

    def __init__(
        self,
        retrieval_service: Any,
        generator: AnswerGenerator | None = None,
        *,
        default_retrieval_mode: str = "hybrid",
        min_score: float | None = None,
    ) -> None:
        if not callable(getattr(retrieval_service, "retrieve", None)):
            raise TypeError("retrieval_service must provide retrieve(query, ...)")
        if default_retrieval_mode not in ("bm25", "vector", "hybrid"):
            raise ValueError(f"Invalid default_retrieval_mode: {default_retrieval_mode!r}")

        self._retrieval = retrieval_service
        self._generator = generator or DeterministicMockGenerator()
        self._default_retrieval_mode = default_retrieval_mode
        self._min_score = min_score

    @property
    def generator(self) -> AnswerGenerator:
        return self._generator

    @property
    def is_mock(self) -> bool:
        return self._generator.is_mock

    def answer(
        self,
        request: RAGRequest | str,
        *,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        paper_id: str | None = None,
        document_id: str | None = None,
    ) -> RAGResponse:
        """Execute end-to-end RAG workflow for a research question.

        Args:
            request: Either a RAGRequest instance or a string query.
            retrieval_mode: Optional override for retrieval mode ('bm25', 'vector', 'hybrid').
            top_k: Optional override for number of retrieved chunks (default: 5).
            min_score: Optional minimum relevance score threshold.
            model: Optional model name override for answer generation.
            temperature: Optional generation temperature.
            max_output_tokens: Optional token generation limit.
            paper_id: Optional paper UUID to filter evidence to a specific paper.
            document_id: Optional document identifier to filter evidence.

        Returns:
            A RAGResponse containing the grounded answer, attributed sources,
            and reference verification status.
        """
        # 1. Normalize input request
        if isinstance(request, str):
            effective_mode = retrieval_mode or self._default_retrieval_mode
            effective_top_k = top_k if top_k is not None else 5
            req = RAGRequest(
                query=request,
                retrieval_mode=effective_mode,
                top_k=effective_top_k,
                min_score=min_score if min_score is not None else self._min_score,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                paper_id=paper_id,
                document_id=document_id,
            )
        elif isinstance(request, RAGRequest):
            req = request
            if (
                retrieval_mode is not None
                or top_k is not None
                or min_score is not None
                or temperature is not None
                or max_output_tokens is not None
                or paper_id is not None
                or document_id is not None
            ):
                req = RAGRequest(
                    query=req.query,
                    retrieval_mode=retrieval_mode if retrieval_mode is not None else req.retrieval_mode,
                    top_k=top_k if top_k is not None else req.top_k,
                    min_score=min_score if min_score is not None else req.min_score,
                    temperature=temperature if temperature is not None else req.temperature,
                    max_output_tokens=max_output_tokens if max_output_tokens is not None else req.max_output_tokens,
                    paper_id=paper_id if paper_id is not None else req.paper_id,
                    document_id=document_id if document_id is not None else req.document_id,
                )
        else:
            raise TypeError(f"request must be a RAGRequest or string, got {type(request).__name__}")

        query = req.query
        mode = req.retrieval_mode
        limit = req.top_k
        score_thresh = req.min_score

        # 2. Retrieve candidates from underlying retrieval service
        try:
            try:
                search_result = self._retrieval.retrieve(
                    query,
                    limit=limit,
                    mode=mode,
                    paper_id=req.paper_id,
                    document_id=req.document_id,
                )
            except TypeError:
                search_result = self._retrieval.retrieve(query, limit=limit, mode=mode)
        except Exception as exc:
            logger.error("Retrieval failed for query %r: %s", query, exc)
            return RAGResponse(
                query=query,
                answer="",
                sources=[],
                cited_sources=[],
                is_grounded=False,
                is_mock=self._generator.is_mock,
                model=model or self._generator.model_name,
                status="error",
                error=f"Retrieval error: {exc}",
                retrieval_mode=mode,
                evidence_count=0,
            )

        raw_chunks: list[RetrievedChunk] = search_result.results if hasattr(search_result, "results") else getattr(search_result, "chunks", [])

        # Filter out chunks with zero relevance or below min_score threshold
        valid_chunks: list[RetrievedChunk] = []
        for c in raw_chunks:
            if c.score <= 0.0:
                continue
            if score_thresh is not None and c.score < score_thresh:
                continue
            valid_chunks.append(c)

        # 3. Handle empty retrieval truthfully
        if not valid_chunks:
            return RAGResponse(
                query=query,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                sources=[],
                cited_sources=[],
                is_grounded=True,
                is_mock=self._generator.is_mock,
                model=model or self._generator.model_name,
                status="insufficient_evidence",
                retrieval_mode=mode,
                evidence_count=0,
            )

        # 4. Construct EvidenceContext and AttributedSource items
        evidence_items: list[EvidenceItem] = []
        sources: list[AttributedSource] = []

        for idx, c in enumerate(valid_chunks):
            meta = c.metadata if isinstance(c.metadata, dict) else {}
            paper_title = str(meta.get("title") or c.document_id or "Unknown Paper")
            paper_id = meta.get("paper_id")
            label = f"[E{idx + 1}]"

            evidence_items.append(
                EvidenceItem(
                    document_id=c.document_id,
                    chunk_id=c.chunk_id,
                    text=c.text,
                    score=c.score,
                    metadata=meta,
                    position=idx,
                )
            )

            sources.append(
                AttributedSource(
                    label=label,
                    document_id=c.document_id,
                    chunk_id=c.chunk_id,
                    paper_id=str(paper_id) if paper_id is not None else None,
                    paper_title=paper_title,
                    score=c.score,
                    text=c.text,
                    text_excerpt=_truncate_excerpt(c.text),
                    metadata=meta,
                )
            )

        context = EvidenceContext(query=query, items=evidence_items, total_candidates=len(valid_chunks))

        # 5. Generate answer using configured generator
        try:
            generated: GeneratedAnswer = self._generator.generate(
                query,
                context,
                model=model,
                temperature=req.temperature,
                max_output_tokens=req.max_output_tokens,
            )
        except Exception as exc:
            clean_error = _sanitize_error_message(exc)
            logger.error("Generation failed for query %r: %s", query, clean_error)
            return RAGResponse(
                query=query,
                answer="",
                sources=sources,
                cited_sources=[],
                is_grounded=False,
                is_mock=self._generator.is_mock,
                model=model or self._generator.model_name,
                status="error",
                error=f"Answer generation error: {clean_error}",
                retrieval_mode=mode,
                evidence_count=len(sources),
            )

        # 6. Verify citations and extract cited sources
        verification = verify_citations(generated.text, context)

        # Map valid citation labels to their respective AttributedSource
        source_by_label = {s.label: s for s in sources}
        cited_sources = [
            source_by_label[lbl]
            for lbl in verification.valid_citations
            if lbl in source_by_label
        ]

        # Check if the generated answer truthfully indicates insufficient evidence
        lower_ans = generated.text.lower()
        is_insufficient = False
        if not verification.valid_citations and any(
            phrase in lower_ans
            for phrase in (
                "insufficient evidence",
                "not enough evidence",
                "could not find sufficient evidence",
                "cannot find sufficient evidence",
                "no sufficient evidence",
                "evidence is insufficient",
                "does not contain sufficient evidence",
                "no evidence was retrieved",
            )
        ):
            is_insufficient = True

        status = "insufficient_evidence" if is_insufficient else "success"
        is_grounded = True if is_insufficient else verification.is_fully_grounded

        return RAGResponse(
            query=query,
            answer=generated.text,
            sources=sources,
            cited_sources=cited_sources,
            is_grounded=is_grounded,
            is_mock=generated.is_mock,
            model=generated.model,
            status=status,
            error=None,
            usage=generated.usage,
            retrieval_mode=mode,
            evidence_count=len(sources),
        )


def _build_benchmark_retrieval_service(
    dataset_path: str | None = None,
) -> RetrievalService:
    """Build and index in-memory BM25 and vector retrieval services from dataset."""
    if dataset_path:
        ds = load_retrieval_dataset(dataset_path)
    else:
        ds = get_pilot_benchmark_dataset()

    chunks = ds.chunks

    # 1. Build and populate BM25 Index
    bm25_index = BM25Index()
    bm25_index.index_chunks(chunks)

    # 2. Build and populate in-memory Vector Store with hash embeddings
    dimension = 64
    provider = HashEmbeddingProvider(dimension=dimension)
    emb_service = EmbeddingService(provider)
    vector_store = InMemoryVectorStore(dimension=dimension)

    records = [
        VectorRecord(
            id=c["chunk_id"],
            vector=emb_service.embed_text(c["text"]),
            payload=dict(c),
        )
        for c in chunks
    ]
    vector_store.upsert(records)

    # 3. Create unified RetrievalService
    return RetrievalService(
        embedding_service=emb_service,
        vector_store=vector_store,
        dimension=dimension,
        bm25_index=bm25_index,
    )


def create_offline_rag_pipeline(
    dataset_path: str | None = None,
    *,
    default_mode: str = "bm25",
) -> RAGPipeline:
    """Create a fully functional, 100% offline RAG pipeline.

    Loads the curated research paper benchmark (Attention, LoRA, FlashAttention),
    indexes it into in-memory BM25 and vector stores, and attaches a deterministic
    offline generator. Requires zero API keys or external services.

    Args:
        dataset_path: Optional path to a research dataset file. Defaults to the
            curated pilot benchmark.
        default_mode: Default retrieval mode ('bm25', 'vector', or 'hybrid').

    Returns:
        An initialized, ready-to-use RAGPipeline instance.
    """
    retrieval_service = _build_benchmark_retrieval_service(dataset_path)
    generator = DeterministicMockGenerator()

    return RAGPipeline(
        retrieval_service=retrieval_service,
        generator=generator,
        default_retrieval_mode=default_mode,
    )


def create_live_rag_pipeline(
    llm_service: Any | None = None,
    *,
    llm_config: Any | None = None,
    dataset_path: str | None = None,
    retrieval_service: Any | None = None,
    default_mode: str = "hybrid",
    min_score: float | None = None,
    model: str | None = None,
    filter_invalid_citations: bool = True,
    environ: Mapping[str, str] | None = None,
) -> RAGPipeline:
    """Create an end-to-end RAG pipeline backed by a real LLM provider (OpenAI).

    Requires valid LLM configuration (NEXORA_LLM_API_KEY or OPENAI_API_KEY).
    Raises ConfigurationError if the provider is unconfigured or credentials are missing.
    """
    from nexora.config.settings import ConfigurationError

    if llm_service is None:
        if llm_config is None:
            from nexora.llm.config import LLMConfig

            if not LLMConfig.has_credentials(environ):
                raise ConfigurationError(
                    "Live LLM generation requested, but no API key configured. "
                    "Set NEXORA_LLM_API_KEY or OPENAI_API_KEY."
                )

            try:
                llm_config = LLMConfig.from_env(environ)
            except Exception as exc:
                raise ConfigurationError(
                    f"Invalid LLM configuration for live pipeline: {exc}"
                ) from exc

        if not getattr(llm_config, "api_key", None):
            raise ConfigurationError(
                "Live LLM generation requested, but no API key configured. "
                "Set NEXORA_LLM_API_KEY or OPENAI_API_KEY."
            )


        from nexora.llm.service import LLMService, create_llm_provider

        provider = create_llm_provider(llm_config)
        llm_service = LLMService(provider)

    effective_model = model or getattr(llm_config, "model", None) or "gpt-4o-mini"
    generator = LLMAnswerGenerator(
        llm_service,
        model_name=effective_model,
        filter_invalid_citations=filter_invalid_citations,
    )

    if retrieval_service is None:
        retrieval_service = _build_benchmark_retrieval_service(dataset_path)

    return RAGPipeline(
        retrieval_service=retrieval_service,
        generator=generator,
        default_retrieval_mode=default_mode,
        min_score=min_score,
    )


def create_configured_rag_pipeline(
    dataset_path: str | None = None,
    *,
    default_mode: str = "hybrid",
    generator_mode: str | None = None,
    model: str | None = None,
    llm_service: Any | None = None,
    llm_config: Any | None = None,
    retrieval_service: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> RAGPipeline:
    """Factory creating either a live or offline RAG pipeline based on configuration.

    Mode selection precedence:
    1. Explicit ``generator_mode`` argument ('live' or 'offline'/'mock').
    2. Environment variable ``NEXORA_RAG_GENERATOR_MODE`` ('live' or 'offline'/'mock').
    3. Auto-detection: If live credentials exist (or llm_service injected), uses live generation;
       otherwise defaults safely to deterministic offline mock mode.

    If live mode is explicitly requested but credentials are missing, raises ConfigurationError
    instead of silently falling back to mock output.
    """
    import os
    from nexora.config.settings import ConfigurationError
    from nexora.llm.config import LLMConfig

    env = os.environ if environ is None else environ

    mode = generator_mode
    if mode is None:
        mode = env.get("NEXORA_RAG_GENERATOR_MODE")
    if mode is not None:
        mode = mode.strip().lower()

    # Explicit live requested
    if mode == "live":
        return create_live_rag_pipeline(
            llm_service=llm_service,
            llm_config=llm_config,
            dataset_path=dataset_path,
            retrieval_service=retrieval_service,
            default_mode=default_mode,
            model=model,
            environ=env,
        )

    # Explicit offline requested
    if mode in ("offline", "mock"):
        retrieval = retrieval_service or _build_benchmark_retrieval_service(dataset_path)
        generator = DeterministicMockGenerator(model_name=model or "offline-deterministic-mock-v1")
        return RAGPipeline(
            retrieval_service=retrieval,
            generator=generator,
            default_retrieval_mode=default_mode,
        )

    # Auto-detection: check if live credentials exist
    has_live_credentials = (
        llm_service is not None
        or (llm_config is not None and getattr(llm_config, "api_key", None))
        or LLMConfig.has_credentials(env)
    )

    if has_live_credentials:
        try:
            return create_live_rag_pipeline(
                llm_service=llm_service,
                llm_config=llm_config,
                dataset_path=dataset_path,
                retrieval_service=retrieval_service,
                default_mode=default_mode,
                model=model,
                environ=env,
            )
        except ConfigurationError:
            pass

    return create_offline_rag_pipeline(dataset_path=dataset_path, default_mode=default_mode)


def create_rag_pipeline(
    retrieval_service: Any,
    generator: AnswerGenerator | None = None,
    *,
    default_mode: str = "hybrid",
    min_score: float | None = None,
) -> RAGPipeline:
    """Create a RAG pipeline with custom retrieval and generation components."""
    return RAGPipeline(
        retrieval_service=retrieval_service,
        generator=generator,
        default_retrieval_mode=default_mode,
        min_score=min_score,
    )
