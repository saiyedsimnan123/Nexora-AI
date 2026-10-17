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
from typing import Any

from nexora.embedding.hash_provider import HashEmbeddingProvider
from nexora.embedding.service import EmbeddingService
from nexora.rag.generator import (
    AnswerGenerator,
    DeterministicMockGenerator,
    GeneratedAnswer,
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
            )
        elif isinstance(request, RAGRequest):
            req = request
            if retrieval_mode is not None:
                req = RAGRequest(
                    query=req.query,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k if top_k is not None else req.top_k,
                    min_score=min_score if min_score is not None else req.min_score,
                    temperature=temperature if temperature is not None else req.temperature,
                    max_output_tokens=max_output_tokens if max_output_tokens is not None else req.max_output_tokens,
                )
        else:
            raise TypeError(f"request must be a RAGRequest or string, got {type(request).__name__}")

        query = req.query
        mode = req.retrieval_mode
        limit = req.top_k
        score_thresh = req.min_score

        # 2. Retrieve candidates from underlying retrieval service
        try:
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
            logger.error("Generation failed for query %r: %s", query, exc)
            return RAGResponse(
                query=query,
                answer="",
                sources=sources,
                cited_sources=[],
                is_grounded=False,
                is_mock=self._generator.is_mock,
                model=model or self._generator.model_name,
                status="error",
                error=f"Answer generation error: {exc}",
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

        return RAGResponse(
            query=query,
            answer=generated.text,
            sources=sources,
            cited_sources=cited_sources,
            is_grounded=verification.is_fully_grounded,
            is_mock=generated.is_mock,
            model=generated.model,
            status="success",
            error=None,
            usage=generated.usage,
            retrieval_mode=mode,
            evidence_count=len(sources),
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
    retrieval_service = RetrievalService(
        embedding_service=emb_service,
        vector_store=vector_store,
        dimension=dimension,
        bm25_index=bm25_index,
    )

    # 4. Attach deterministic mock generator
    generator = DeterministicMockGenerator()

    return RAGPipeline(
        retrieval_service=retrieval_service,
        generator=generator,
        default_retrieval_mode=default_mode,
    )


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
