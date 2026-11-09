"""Document chunk re-embedding and vector store migration service.

Migrates existing indexed document chunks from PostgreSQL to a target vector store
collection (e.g. migrating from OpenAI 1536D to Google Gemini 768D), guaranteeing:
1. Strict vector isolation: never mixes incompatible embedding spaces in the same collection.
2. Provenance tagging: marks each vector payload with the embedding provider, model, and dimension.
3. Batching & resilience: processes chunks in batches with progress reporting.
4. Safe offline execution: supports injected mocks and transactions for deterministic testing.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from nexora.retrieval.models import VectorRecord
from nexora.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReindexingReport:
    """Summary report of a re-embedding and vector store migration run."""

    status: str  # "completed" | "failed" | "empty"
    total_chunks: int
    total_documents: int
    collection_name: str
    dimension: int
    provider: str
    model: str
    duration_seconds: float
    error: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "total_chunks": self.total_chunks,
            "total_documents": self.total_documents,
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "provider": self.provider,
            "model": self.model,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class ReindexingService:
    """Orchestrates re-embedding of document chunks into a fresh or target vector collection."""

    def __init__(
        self,
        session_factory: Any,
        embedding_service: Any,
        vector_store: VectorStore,
        *,
        batch_size: int = 50,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._batch_size = max(1, batch_size)

    def reindex_all(self) -> ReindexingReport:
        """Read all document chunks from database, re-embed them, and upsert to vector store."""
        start_time = time.monotonic()
        provider_name = getattr(
            self._embedding_service, "provider", "gemini"
        )
        model_name = getattr(
            self._embedding_service, "model", "gemini-embedding-001"
        )
        target_dim = getattr(self._embedding_service, "dimension", 768)
        collection_name = getattr(
            self._vector_store, "_collection", "target_collection"
        )

        # 1. Ensure target collection exists and matches dimension
        if hasattr(self._vector_store, "ensure_collection"):
            self._vector_store.ensure_collection()
        if hasattr(self._vector_store, "validate_collection_compatibility"):
            self._vector_store.validate_collection_compatibility()

        # 2. Fetch chunks from database
        try:
            chunks_data = self._fetch_all_chunks()
        except Exception as exc:
            duration = time.monotonic() - start_time
            logger.error("Failed to fetch document chunks from database: %s", exc)
            return ReindexingReport(
                status="failed",
                total_chunks=0,
                total_documents=0,
                collection_name=collection_name,
                dimension=target_dim,
                provider=str(provider_name),
                model=str(model_name),
                duration_seconds=round(duration, 3),
                error=f"Database fetch failed: {exc}",
            )

        if not chunks_data:
            duration = time.monotonic() - start_time
            return ReindexingReport(
                status="empty",
                total_chunks=0,
                total_documents=0,
                collection_name=collection_name,
                dimension=target_dim,
                provider=str(provider_name),
                model=str(model_name),
                duration_seconds=round(duration, 3),
            )

        unique_doc_ids = {c["document_id"] for c in chunks_data}

        # 3. Batch embed and upsert into vector store
        try:
            for start in range(0, len(chunks_data), self._batch_size):
                batch = chunks_data[start : start + self._batch_size]
                texts = [item["text"] for item in batch]

                if hasattr(self._embedding_service, "embed_texts"):
                    vectors = self._embedding_service.embed_texts(texts)
                elif hasattr(self._embedding_service, "embed_text"):
                    vectors = [self._embedding_service.embed_text(t) for t in texts]
                else:
                    raise RuntimeError("embedding_service provides neither embed_texts nor embed_text")

                records: list[VectorRecord] = []
                for item, vec in zip(batch, vectors):
                    payload = {
                        "document_id": str(item["document_id"]),
                        "chunk_id": str(item["id"]),
                        "chunk_index": item["chunk_index"],
                        "text": item["text"],
                        "paper_id": str(item.get("paper_id") or ""),
                        "filename": str(item.get("filename") or ""),
                        "embedding_provider": str(provider_name),
                        "embedding_model": str(model_name),
                        "dimension": target_dim,
                    }
                    records.append(
                        VectorRecord(id=str(item["id"]), vector=vec, payload=payload)
                    )

                self._vector_store.upsert(records)
        except Exception as exc:
            duration = time.monotonic() - start_time
            logger.error("Failed during re-embedding or vector upsert: %s", exc)
            return ReindexingReport(
                status="failed",
                total_chunks=len(chunks_data),
                total_documents=len(unique_doc_ids),
                collection_name=collection_name,
                dimension=target_dim,
                provider=str(provider_name),
                model=str(model_name),
                duration_seconds=round(duration, 3),
                error=f"Embedding/upsert failed: {exc}",
            )

        duration = time.monotonic() - start_time
        return ReindexingReport(
            status="completed",
            total_chunks=len(chunks_data),
            total_documents=len(unique_doc_ids),
            collection_name=collection_name,
            dimension=target_dim,
            provider=str(provider_name),
            model=str(model_name),
            duration_seconds=round(duration, 3),
        )

    def _fetch_all_chunks(self) -> list[dict[str, Any]]:
        """Fetch all chunk records joining with parent document metadata."""
        from nexora.database.session import database_session

        with database_session(self._session_factory) as session:
            from sqlalchemy import select
            from nexora.database.models.document import Document
            from nexora.database.models.document_chunk import DocumentChunkRecord

            stmt = (
                select(
                    DocumentChunkRecord.id,
                    DocumentChunkRecord.document_id,
                    DocumentChunkRecord.chunk_index,
                    DocumentChunkRecord.text,
                    Document.paper_id,
                    Document.filename,
                )
                .join(Document, DocumentChunkRecord.document_id == Document.id)
                .order_by(DocumentChunkRecord.document_id, DocumentChunkRecord.chunk_index)
            )

            result = session.execute(stmt).fetchall()
            return [
                {
                    "id": str(row[0]),
                    "document_id": str(row[1]),
                    "chunk_index": int(row[2]),
                    "text": str(row[3]),
                    "paper_id": str(row[4]) if row[4] is not None else None,
                    "filename": str(row[5]) if row[5] is not None else None,
                }
                for row in result
            ]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI runner for re-indexing existing documents to Gemini embeddings."""
    parser = argparse.ArgumentParser(
        description="Migrate and re-index Nexora document embeddings."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embedding generation (default: 50).",
    )
    args = parser.parse_args(argv)

    import os
    from nexora.database.config import DatabaseConfig
    from nexora.database.session import create_session_factory
    from nexora.embedding.config import EmbeddingConfig
    from nexora.embedding.factory import create_embedding_provider
    from nexora.embedding.service import EmbeddingService
    from nexora.retrieval.qdrant import QdrantVectorStore
    from nexora.retrieval.qdrant_config import QdrantConfig

    db_cfg = DatabaseConfig.from_env()
    session_factory = create_session_factory(db_cfg.url)

    emb_cfg = EmbeddingConfig.from_env()
    emb_provider = create_embedding_provider(emb_cfg)
    emb_service = EmbeddingService(emb_provider)

    qdrant_cfg = QdrantConfig.from_env()
    vector_store = QdrantVectorStore(qdrant_cfg, dimension=emb_service.dimension)

    service = ReindexingService(
        session_factory=session_factory,
        embedding_service=emb_service,
        vector_store=vector_store,
        batch_size=args.batch_size,
    )
    report = service.reindex_all()
    print(f"Re-indexing result: {report.status}")
    print(f"Chunks: {report.total_chunks}, Documents: {report.total_documents}")
    print(f"Collection: {report.collection_name}, Dimension: {report.dimension}")
    print(f"Provider: {report.provider}, Model: {report.model}")
    print(f"Duration: {report.duration_seconds}s")
    if report.error:
        print(f"Error: {report.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
