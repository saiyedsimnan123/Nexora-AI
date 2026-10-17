"""Okapi BM25 lexical retrieval engine for Nexora AI.

Provides deterministic, fast, pure-Python Okapi BM25 scoring and ranking:
- 100% offline: relies only on Python standard library (no external packages or model weights).
- Evaluates real research benchmark papers by exact keyword, phrase, and identifier matching.
- Compatible with Nexora's ``SearchResult`` and ``RetrievedChunk`` models.
- Supports Reciprocal Rank Fusion (RRF) for hybrid retrieval (Vector + BM25).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from nexora.retrieval.models import RetrievedChunk, SearchResult

__all__ = [
    "DEFAULT_STOP_WORDS",
    "tokenize",
    "BM25Config",
    "BM25Document",
    "BM25Index",
    "reciprocal_rank_fusion",
]

DEFAULT_STOP_WORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "using", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves",
})


def tokenize(
    text: str,
    stopwords: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Extract lowercase alphanumeric and hyphenated tokens from ``text``.

    Splits punctuation while preserving compound terms and identifiers
    (e.g., 'gpt-3', '1706.03762', 'flashattention', 'io-aware').
    If ``stopwords`` is provided, matching tokens are filtered out.
    """
    if not isinstance(text, str):
        return []
    raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*", text.lower())
    clean_tokens = [t.strip("-._") for t in raw_tokens]
    valid = [t for t in clean_tokens if t]
    if stopwords:
        filtered = [t for t in valid if t not in stopwords]
        return filtered if filtered else valid
    return valid


@dataclass(frozen=True)
class BM25Config:
    """Hyperparameters for Okapi BM25 scoring."""

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    remove_stopwords: bool = True
    stopwords: frozenset[str] = field(default_factory=lambda: DEFAULT_STOP_WORDS)

    def __post_init__(self) -> None:
        if isinstance(self.k1, bool) or not isinstance(self.k1, (int, float)) or self.k1 < 0:
            raise ValueError(f"k1 must be a non-negative number, got {self.k1}")
        if isinstance(self.b, bool) or not isinstance(self.b, (int, float)) or not (0.0 <= self.b <= 1.0):
            raise ValueError(f"b must be between 0.0 and 1.0, got {self.b}")
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)) or self.epsilon < 0:
            raise ValueError(f"epsilon must be a non-negative number, got {self.epsilon}")


@dataclass
class BM25Document:
    """Internal indexed document representation."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    tokens: list[str]
    length: int
    term_frequencies: Counter[str]


class BM25Index:
    """In-memory Okapi BM25 index for fast lexical retrieval.

    Computes standard Lucene-style non-negative Okapi BM25 scores:
    IDF(q) = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
    Score(D, Q) = sum_{q in Q} IDF(q) * (tf(q, D) * (k1 + 1)) / (tf(q, D) + k1 * (1 - b + b * (|D| / avgdl)))
    """

    def __init__(self, config: BM25Config | None = None) -> None:
        self.config = config if config is not None else BM25Config()
        self._documents: dict[str, BM25Document] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._total_tokens: int = 0

    @property
    def total_documents(self) -> int:
        """Total number of documents in the index."""
        return len(self._documents)

    @property
    def average_document_length(self) -> float:
        """Average document token length across the index."""
        if not self._documents:
            return 0.0
        return self._total_tokens / len(self._documents)

    def count(self) -> int:
        """Return count of indexed documents."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear all indexed documents."""
        self._documents.clear()
        self._document_frequencies.clear()
        self._total_tokens = 0

    def add_document(
        self,
        chunk_id: str,
        text: str,
        document_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index a single text document / chunk.

        If ``chunk_id`` already exists, it is replaced.
        """
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # If replacing an existing document, subtract its frequencies first
        if chunk_id in self._documents:
            old_doc = self._documents[chunk_id]
            for term in old_doc.term_frequencies:
                self._document_frequencies[term] -= 1
                if self._document_frequencies[term] <= 0:
                    del self._document_frequencies[term]
            self._total_tokens -= old_doc.length

        sw = self.config.stopwords if self.config.remove_stopwords else None
        tokens = tokenize(text, stopwords=sw)
        length = len(tokens)
        term_freqs = Counter(tokens)

        # Update global frequencies
        for term in term_freqs:
            self._document_frequencies[term] += 1
        self._total_tokens += length

        doc = BM25Document(
            chunk_id=chunk_id,
            document_id=document_id or f"doc_{chunk_id}",
            text=text,
            metadata=dict(metadata) if metadata else {},
            tokens=tokens,
            length=length,
            term_frequencies=term_freqs,
        )
        self._documents[chunk_id] = doc

    def index_chunks(self, chunks: Sequence[dict[str, Any] | Any]) -> int:
        """Index an iterable of chunk dicts or chunk objects.

        Returns the number of documents indexed.
        """
        count = 0
        for chunk in chunks:
            if isinstance(chunk, dict):
                c_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
                text = str(chunk.get("text") or "")
                doc_id = str(chunk.get("document_id") or "")
                meta = dict(chunk)
            else:
                c_id = str(getattr(chunk, "chunk_id", getattr(chunk, "id", "")))
                text = str(getattr(chunk, "text", ""))
                doc_id = str(getattr(chunk, "document_id", ""))
                meta = getattr(chunk, "payload", getattr(chunk, "metadata", {}))

            if c_id and text:
                self.add_document(c_id, text, document_id=doc_id, metadata=meta)
                count += 1
        return count

    def idf(self, term: str) -> float:
        """Compute non-negative Okapi BM25 inverse document frequency for ``term``."""
        n_docs = len(self._documents)
        if n_docs == 0:
            return 0.0
        df = self._document_frequencies.get(term, 0)
        # Using ln(1 + (N - df + 0.5) / (df + 0.5)) to prevent negative weights for frequent terms
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], doc: BM25Document) -> float:
        """Calculate BM25 relevance score for ``doc`` against ``query_tokens``."""
        if not query_tokens or doc.length == 0:
            return 0.0

        avgdl = self.average_document_length
        if avgdl <= 0.0:
            return 0.0

        k1 = self.config.k1
        b = self.config.b
        len_norm = 1.0 - b + b * (doc.length / avgdl)

        total_score = 0.0
        query_freqs = Counter(query_tokens)

        for term, q_tf in query_freqs.items():
            tf = doc.term_frequencies.get(term, 0)
            if tf <= 0:
                continue

            idf = self.idf(term)
            # Standard saturation numerator & denominator
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * len_norm
            term_score = idf * (numerator / denominator)

            # Query term weight scaling if repeated in query
            if q_tf > 1:
                term_score *= (1.0 + math.log(q_tf))

            total_score += term_score

        return max(0.0, total_score)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        min_score: float | None = None,
        paper_id: str | None = None,
        document_id: str | None = None,
    ) -> SearchResult:
        """Search the index using Okapi BM25 ranking.

        Args:
            query: The search text query.
            limit: Maximum number of results to return.
            min_score: Optional minimum score threshold.
            paper_id: Optional paper identifier to restrict search to a specific paper.
            document_id: Optional document identifier to restrict search to a specific document.

        Returns:
            A ``SearchResult`` with chunks ordered by relevance score descending.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit}")

        sw = self.config.stopwords if self.config.remove_stopwords else None
        query_tokens = tokenize(query, stopwords=sw)
        if not query_tokens or not self._documents:
            return SearchResult(query=query, results=[])

        # Score documents that contain at least one query token and match any filters
        scored_docs: list[tuple[float, BM25Document]] = []
        for doc in self._documents.values():
            if paper_id is not None and str(doc.metadata.get("paper_id")) != str(paper_id):
                continue
            if document_id is not None and doc.document_id != str(document_id):
                continue
            s = self.score(query_tokens, doc)
            if s > 0.0:
                if min_score is None or s >= min_score:
                    scored_docs.append((s, doc))

        # Sort descending by score; break ties deterministically by chunk_id
        scored_docs.sort(key=lambda item: (-item[0], item[1].chunk_id))

        retrieved_chunks = [
            RetrievedChunk(
                document_id=doc.document_id,
                chunk_id=doc.chunk_id,
                text=doc.text,
                score=round(score, 6),
                metadata=doc.metadata,
            )
            for score, doc in scored_docs[:limit]
        ]

        return SearchResult(
            query=query,
            results=retrieved_chunks,
        )


def reciprocal_rank_fusion(
    vector_results: SearchResult,
    bm25_results: SearchResult,
    *,
    rrf_k: int = 60,
    limit: int = 10,
) -> SearchResult:
    """Fuse semantic vector search and BM25 lexical search results using Reciprocal Rank Fusion (RRF).

    Formula for each document d:
    RRF_Score(d) = sum_{m in {vector, bm25}} (1 / (rrf_k + rank_m(d)))

    Args:
        vector_results: SearchResult from vector store.
        bm25_results: SearchResult from BM25Index.
        rrf_k: Smoothing constant (default 60, standard in IR literature).
        limit: Maximum number of fused chunks to return.

    Returns:
        A new SearchResult with fused rankings and reciprocal rank scores.
    """
    if not isinstance(vector_results, SearchResult):
        raise TypeError("vector_results must be a SearchResult")
    if not isinstance(bm25_results, SearchResult):
        raise TypeError("bm25_results must be a SearchResult")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError(f"rrf_k must be a positive integer, got {rrf_k}")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit}")

    fused_scores: dict[str, float] = {}
    chunk_store: dict[str, RetrievedChunk] = {}

    # 1. Accumulate vector ranks
    for rank, chunk in enumerate(vector_results.chunks, start=1):
        fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        chunk_store[chunk.chunk_id] = chunk

    # 2. Accumulate BM25 ranks
    for rank, chunk in enumerate(bm25_results.chunks, start=1):
        fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        if chunk.chunk_id not in chunk_store:
            chunk_store[chunk.chunk_id] = chunk

    # Sort descending by fused score
    sorted_items = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    fused_chunks: list[RetrievedChunk] = []
    for chunk_id, fused_score in sorted_items[:limit]:
        base_chunk = chunk_store[chunk_id]
        meta = dict(base_chunk.metadata)
        meta["rrf_score"] = round(fused_score, 6)
        fused_chunks.append(
            RetrievedChunk(
                document_id=base_chunk.document_id,
                chunk_id=base_chunk.chunk_id,
                text=base_chunk.text,
                score=round(fused_score, 6),
                metadata=meta,
            )
        )

    return SearchResult(
        query=vector_results.query or bm25_results.query,
        results=fused_chunks,
    )
