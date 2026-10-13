"""Durable vector indexing recovery service (Milestone 12C-22).

Provides bounded, idempotent, out-of-band recovery for documents whose relational
metadata was persisted but whose vector-store indexing is pending, failed, or inconsistent
(e.g., following process crashes, network partitions, or vector cluster outages).

Design & Architectural Trade-offs:
- Durable indexing status on `documents` table:
  Every document record durably stores `indexing_status` ('pending', 'completed', 'failed'),
  `indexing_error` (last failure message), `indexed_at` (successful timestamp), and
  `indexing_attempts` (retry attempt counter).
- Trade-off vs Transactional Outbox:
  A separate transactional outbox table adds duplicate state tracking and separate table
  joins for 1-to-1 document-indexing lifecycles. Maintaining status directly on `Document`
  keeps relational integrity atomic, simplifies queries, avoids orphan jobs, and does not
  require Redis or external message brokers.
- Concurrency & Idempotency:
  - Vector record IDs are deterministically derived from relational chunk UUIDs (`str(chunk.id)`).
    Re-indexing a document performs an idempotent upsert that overwrites existing vectors in-place
    without creating duplicate vectors.
  - Per-paper locking (`get_paper_ingestion_lock`) serializes recovery with concurrent live
    ingestion requests for the same paper, preventing race conditions.
  - Recovery does not perform bulk deletions of other documents' vectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import logging
from typing import Any, Sequence
from uuid import UUID

from nexora.database.models.document import IndexingStatus
from nexora.database.repository.base import RecordNotFoundError
from nexora.database.session import database_transaction
from nexora.document.ingestion import get_paper_ingestion_lock
from nexora.document.service import DocumentPersistenceService
from nexora.retrieval.models import VectorRecord
from nexora.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecoveryItemResult:
    """Outcome of an individual document recovery attempt."""

    document_id: UUID
    paper_id: UUID
    status: IndexingStatus
    chunks_count: int
    indexed_chunks_count: int
    error: str | None = None
    attempts: int = 0


@dataclass(frozen=True)
class RecoveryBatchResult:
    """Summary of a bounded recovery batch execution."""

    scanned_count: int
    recovered_count: int
    failed_count: int
    items: list[RecoveryItemResult] = field(default_factory=list)


class DocumentRecoveryService:
    """Bounded, idempotent service for discovering and recovering unindexed documents."""

    def __init__(
        self,
        session_factory: Any,
        vector_store: VectorStore,
        embedding_service: Any,
    ) -> None:
        if session_factory is None or not callable(session_factory):
            raise TypeError("session_factory must be a callable session factory")
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                f"vector_store must implement VectorStore protocol, got {type(vector_store).__name__}"
            )
        if embedding_service is None or not (
            callable(getattr(embedding_service, "embed_texts", None))
            or callable(getattr(embedding_service, "embed_text", None))
        ):
            raise TypeError("embedding_service must provide embed_texts or embed_text")

        self._session_factory = session_factory
        self._vector_store = vector_store
        self._embedding_service = embedding_service

    def _resolve_vectors(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if hasattr(self._embedding_service, "embed_texts"):
            return self._embedding_service.embed_texts(texts)
        return [self._embedding_service.embed_text(t) for t in texts]

    def _recover_single_document(
        self, persistence: DocumentPersistenceService, doc: Any
    ) -> RecoveryItemResult:
        """Process vector reindexing and state transition for a single document."""
        chunk_records = list(persistence.list_chunks_for_document(doc.id))
        chunks_count = len(chunk_records)

        if chunks_count == 0:
            # Document has no chunks, mark as completed directly
            persistence.update_document_indexing_status(
                doc.id,
                IndexingStatus.COMPLETED,
                error=None,
                indexed_at=datetime.datetime.now(datetime.timezone.utc),
            )
            return RecoveryItemResult(
                document_id=doc.id,
                paper_id=doc.paper_id,
                status=IndexingStatus.COMPLETED,
                chunks_count=0,
                indexed_chunks_count=0,
                attempts=doc.indexing_attempts,
            )

        try:
            texts = [c.text for c in chunk_records]
            vectors = self._resolve_vectors(texts)

            records_to_upsert: list[VectorRecord] = []
            for chunk_rec, vector in zip(chunk_records, vectors):
                rec_id = str(chunk_rec.id)
                payload: dict[str, object] = {
                    "document_id": str(doc.id),
                    "chunk_id": rec_id,
                    "paper_id": str(doc.paper_id),
                    "chunk_index": chunk_rec.chunk_index,
                    "text": chunk_rec.text,
                    "filename": doc.filename,
                    "source": doc.storage_key,
                }
                records_to_upsert.append(
                    VectorRecord(id=rec_id, vector=vector, payload=payload)
                )

            # Idempotent upsert in vector store
            self._vector_store.upsert(records_to_upsert)

            # Update durable status to COMPLETED
            updated_doc = persistence.update_document_indexing_status(
                doc.id,
                IndexingStatus.COMPLETED,
                error=None,
                indexed_at=datetime.datetime.now(datetime.timezone.utc),
            )
            return RecoveryItemResult(
                document_id=doc.id,
                paper_id=doc.paper_id,
                status=IndexingStatus.COMPLETED,
                chunks_count=chunks_count,
                indexed_chunks_count=len(records_to_upsert),
                attempts=updated_doc.indexing_attempts,
            )
        except Exception as exc:
            logger.warning(
                "Recovery attempt failed for document %s: %s", doc.id, exc
            )
            # Record durable FAILED state with error message and increment attempt counter
            updated_doc = persistence.update_document_indexing_status(
                doc.id,
                IndexingStatus.FAILED,
                error=str(exc),
                attempts_increment=1,
            )
            return RecoveryItemResult(
                document_id=doc.id,
                paper_id=doc.paper_id,
                status=IndexingStatus.FAILED,
                chunks_count=chunks_count,
                indexed_chunks_count=0,
                error=str(exc),
                attempts=updated_doc.indexing_attempts,
            )

    def recover_document(self, document_id: UUID) -> RecoveryItemResult:
        """Recover vector indexing for a specific document by its ID."""
        if not isinstance(document_id, UUID):
            raise TypeError(f"document_id must be a UUID, got {type(document_id).__name__}")

        with database_transaction(self._session_factory) as session:
            persistence = DocumentPersistenceService(session)
            doc = persistence.get_document(document_id)
            if doc is None:
                raise RecordNotFoundError(f"Document with id {document_id} not found")
            paper_id = doc.paper_id

        # Acquire in-memory paper lock before starting write transaction to prevent lock inversion
        with get_paper_ingestion_lock(paper_id):
            with database_transaction(self._session_factory) as session:
                persistence = DocumentPersistenceService(session)
                doc = persistence.get_document(document_id)
                if doc is None:
                    raise RecordNotFoundError(f"Document with id {document_id} not found")
                return self._recover_single_document(persistence, doc)

    def recover_batch(
        self,
        batch_size: int = 10,
        *,
        max_attempts: int | None = None,
        include_failed: bool = True,
    ) -> RecoveryBatchResult:
        """Find and recover unindexed documents in bounded, deterministic batches."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        statuses = [IndexingStatus.PENDING.value]
        if include_failed:
            statuses.append(IndexingStatus.FAILED.value)

        # 1. Discover target document IDs and paper IDs in a bounded read
        with database_transaction(self._session_factory) as session:
            persistence = DocumentPersistenceService(session)
            unindexed_docs = persistence.list_unindexed_documents(
                status=statuses,
                limit=batch_size,
                max_attempts=max_attempts,
            )
            targets = [(d.id, d.paper_id) for d in unindexed_docs]

        scanned_count = len(targets)
        recovered_count = 0
        failed_count = 0
        results: list[RecoveryItemResult] = []

        # 2. Process each document in its own transaction for failure isolation
        for doc_id, paper_id in targets:
            # Acquire paper lock FIRST before opening write transaction
            with get_paper_ingestion_lock(paper_id):
                with database_transaction(self._session_factory) as session:
                    persistence = DocumentPersistenceService(session)
                    doc = persistence.get_document(doc_id)
                    if doc is None or doc.indexing_status == IndexingStatus.COMPLETED.value:
                        continue

                    item_res = self._recover_single_document(persistence, doc)
                    results.append(item_res)
                    if item_res.status == IndexingStatus.COMPLETED:
                        recovered_count += 1
                    else:
                        failed_count += 1

        return RecoveryBatchResult(
            scanned_count=scanned_count,
            recovered_count=recovered_count,
            failed_count=failed_count,
            items=results,
        )

    def get_indexing_summary(self) -> dict[str, int]:
        """Return counts of documents by indexing status across the database."""
        with database_transaction(self._session_factory) as session:
            persistence = DocumentPersistenceService(session)
            return persistence.count_indexing_status()
