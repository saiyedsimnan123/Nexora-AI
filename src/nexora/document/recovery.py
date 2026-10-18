"""Durable vector indexing recovery service (Milestone 12C-22 & Follow-up).

Provides bounded, idempotent, out-of-band recovery for documents whose relational
metadata was persisted but whose vector-store indexing is pending, failed, or inconsistent
(e.g., following process crashes, network partitions, or vector cluster outages).

Cross-Worker & Multi-Process Coordination Design:
1. PostgreSQL-Backed Atomic Claims:
   Independent recovery workers running across multiple processes coordinate via
   database row locks with `SKIP LOCKED` (`claim_unindexed_documents`).
   In a short atomic claim transaction (< 5ms):
   - Candidate unindexed or expired processing documents are locked using
     `SELECT ... FOR UPDATE SKIP LOCKED`.
   - Their status is atomically transitioned to `processing`, `claimed_at` is set,
     `claimed_by` is tagged with a unique worker token, and `indexing_attempts` is incremented.
   - The transaction commits immediately.
   Concurrent workers in separate processes will skip locked rows and claim distinct
   batches without contention or deadlocks.
2. Zero Transactions Held During Embedding / Vector Work:
   All embedding generation and vector store upserts take place OUTSIDE any database
   transaction. Database connections and locks are released before calling the embedding
   service or vector store, preventing connection pool starvation or lock retention.
3. Safe Expiry & Crash Recovery Semantics:
   If a worker process terminates abruptly mid-flight, documents remain in `processing`
   state with their `claimed_at` timestamp.
   When `claim_timeout_seconds` elapses, the claim is considered expired/stale.
   Subsequent recovery passes automatically re-claim stale documents.
   Retries are bounded by `indexing_attempts < max_attempts`, preventing infinite crash loops.
   Fencing tokens (`claimed_by`) guard finalization so zombie workers cannot overwrite
   state if another worker completed the job.
4. Concurrency & Idempotency:
   - Derived vector record IDs are deterministic (`str(chunk.id)`), ensuring all upserts
     are idempotent and overwrite existing vectors in-place without duplicates.
   - In-process paper locking (`get_paper_ingestion_lock`) remains active as a fast local
     optimization to serialize recovery with live ingestion in the same process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import logging
from typing import Any, Sequence
from uuid import UUID, uuid4

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
        *,
        default_claim_timeout_seconds: float = 300.0,
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
        self._default_claim_timeout = default_claim_timeout_seconds

    def _resolve_vectors(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if hasattr(self._embedding_service, "embed_texts"):
            return self._embedding_service.embed_texts(texts)
        return [self._embedding_service.embed_text(t) for t in texts]

    def _execute_document_indexing(
        self,
        doc_id: UUID,
        paper_id: UUID,
        claim_token: str,
        attempts: int,
    ) -> RecoveryItemResult:
        """Process vector reindexing and state transition for a claimed document.

        Must be called under `get_paper_ingestion_lock(paper_id)` to serialize threads in
        the same process.
        """
        # 1. Read chunks from DB in a short read transaction
        with database_transaction(self._session_factory) as session:
            persistence = DocumentPersistenceService(session)
            doc = persistence.get_document(doc_id)
            if doc is None:
                return RecoveryItemResult(
                    document_id=doc_id,
                    paper_id=paper_id,
                    status=IndexingStatus.FAILED,
                    chunks_count=0,
                    indexed_chunks_count=0,
                    error="Document not found during processing",
                    attempts=attempts,
                )
            filename = doc.filename
            storage_key = doc.storage_key
            paper_repo = getattr(persistence, "_paper_repo", None)
            paper_rec = paper_repo.get_by_id(paper_id) if paper_repo and hasattr(paper_repo, "get_by_id") else None
            recovery_col_id = str(paper_rec.collection_id) if paper_rec and getattr(paper_rec, "collection_id", None) else None
            raw_chunks = list(persistence.list_chunks_for_document(doc_id))
            chunk_data = [
                (str(c.id), c.chunk_index, c.text)
                for c in raw_chunks
            ]

        chunks_count = len(chunk_data)
        if chunks_count == 0:
            with database_transaction(self._session_factory) as session:
                persistence = DocumentPersistenceService(session)
                finalized = persistence.finalize_document_indexing(
                    doc_id,
                    IndexingStatus.COMPLETED,
                    claim_token=claim_token,
                    error=None,
                    indexed_at=datetime.datetime.now(datetime.timezone.utc),
                )
            if finalized is None:
                logger.warning(
                    "Finalization rejected for document %s: claim token %s lost or superseded",
                    doc_id,
                    claim_token,
                )
                return RecoveryItemResult(
                    document_id=doc_id,
                    paper_id=paper_id,
                    status=IndexingStatus.FAILED,
                    chunks_count=0,
                    indexed_chunks_count=0,
                    error="Recovery claim lost or superseded by another worker",
                    attempts=attempts,
                )
            return RecoveryItemResult(
                document_id=doc_id,
                paper_id=paper_id,
                status=IndexingStatus.COMPLETED,
                chunks_count=0,
                indexed_chunks_count=0,
                attempts=attempts,
            )

        # 2. Embedding generation and vector store upsert (OUTSIDE ANY DB TRANSACTION)
        try:
            texts = [c[2] for c in chunk_data]
            vectors = self._resolve_vectors(texts)

            records_to_upsert: list[VectorRecord] = []
            for (rec_id, chunk_index, chunk_text), vector in zip(chunk_data, vectors):
                payload: dict[str, object] = {
                    "document_id": str(doc_id),
                    "chunk_id": rec_id,
                    "paper_id": str(paper_id),
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "filename": filename,
                    "source": storage_key,
                }
                if recovery_col_id:
                    payload["collection_id"] = recovery_col_id
                records_to_upsert.append(
                    VectorRecord(id=rec_id, vector=vector, payload=payload)
                )

            self._vector_store.upsert(records_to_upsert)

            # 3. Finalize in short write transaction
            with database_transaction(self._session_factory) as session:
                persistence = DocumentPersistenceService(session)
                finalized = persistence.finalize_document_indexing(
                    doc_id,
                    IndexingStatus.COMPLETED,
                    claim_token=claim_token,
                    error=None,
                    indexed_at=datetime.datetime.now(datetime.timezone.utc),
                )

            if finalized is None:
                logger.warning(
                    "Finalization rejected for document %s: claim token %s lost or superseded",
                    doc_id,
                    claim_token,
                )
                return RecoveryItemResult(
                    document_id=doc_id,
                    paper_id=paper_id,
                    status=IndexingStatus.FAILED,
                    chunks_count=chunks_count,
                    indexed_chunks_count=0,
                    error="Recovery claim lost or superseded by another worker",
                    attempts=attempts,
                )

            return RecoveryItemResult(
                document_id=doc_id,
                paper_id=paper_id,
                status=IndexingStatus.COMPLETED,
                chunks_count=chunks_count,
                indexed_chunks_count=len(records_to_upsert),
                attempts=attempts,
            )
        except Exception as exc:
            logger.warning("Recovery processing failed for document %s: %s", doc_id, exc)
            with database_transaction(self._session_factory) as session:
                persistence = DocumentPersistenceService(session)
                persistence.finalize_document_indexing(
                    doc_id,
                    IndexingStatus.FAILED,
                    claim_token=claim_token,
                    error=str(exc),
                )

            return RecoveryItemResult(
                document_id=doc_id,
                paper_id=paper_id,
                status=IndexingStatus.FAILED,
                chunks_count=chunks_count,
                indexed_chunks_count=0,
                error=str(exc),
                attempts=attempts,
            )

    def recover_document(
        self,
        document_id: UUID,
        *,
        max_attempts: int | None = None,
        claim_timeout_seconds: float | None = None,
    ) -> RecoveryItemResult:
        """Recover vector indexing for a specific document by its ID."""
        if not isinstance(document_id, UUID):
            raise TypeError(f"document_id must be a UUID, got {type(document_id).__name__}")

        timeout = (
            claim_timeout_seconds
            if claim_timeout_seconds is not None
            else self._default_claim_timeout
        )
        claim_token = str(uuid4())

        # In a short read, find the paper_id for this document
        with database_transaction(self._session_factory) as session:
            persistence = DocumentPersistenceService(session)
            doc = persistence.get_document(document_id)
            if doc is None:
                raise RecordNotFoundError(f"Document with id {document_id} not found")
            paper_id = doc.paper_id

        # Under in-process paper lock, claim and process the document
        with get_paper_ingestion_lock(paper_id):
            with database_transaction(self._session_factory) as session:
                persistence = DocumentPersistenceService(session)
                doc = persistence.get_document(document_id)
                if doc is None:
                    raise RecordNotFoundError(f"Document with id {document_id} not found")
                if doc.indexing_status == IndexingStatus.COMPLETED.value:
                    chunk_records = list(persistence.list_chunks_for_document(doc.id))
                    return RecoveryItemResult(
                        document_id=doc.id,
                        paper_id=doc.paper_id,
                        status=IndexingStatus.COMPLETED,
                        chunks_count=len(chunk_records),
                        indexed_chunks_count=len(chunk_records),
                        attempts=doc.indexing_attempts,
                    )

                claimed = persistence.claim_document_for_recovery(
                    document_id,
                    max_attempts=max_attempts,
                    claim_timeout_seconds=timeout,
                    claim_token=claim_token,
                )
                if claimed is None:
                    refreshed = persistence.get_document(document_id)
                    curr_status = (
                        IndexingStatus(refreshed.indexing_status)
                        if refreshed
                        else IndexingStatus.FAILED
                    )
                    error_msg = (
                        refreshed.indexing_error
                        if refreshed and refreshed.indexing_error
                        else "Document is actively claimed by another worker"
                    )
                    curr_attempts = refreshed.indexing_attempts if refreshed else doc.indexing_attempts
                    return RecoveryItemResult(
                        document_id=doc.id,
                        paper_id=doc.paper_id,
                        status=curr_status,
                        chunks_count=0,
                        indexed_chunks_count=0,
                        error=error_msg,
                        attempts=curr_attempts,
                    )
                attempts = claimed.indexing_attempts

            return self._execute_document_indexing(document_id, paper_id, claim_token, attempts)

    MAX_BATCH_SIZE: int = 100

    def recover_batch(
        self,
        batch_size: int = 10,
        *,
        max_attempts: int | None = None,
        claim_timeout_seconds: float | None = None,
        include_failed: bool = True,
    ) -> RecoveryBatchResult:
        """Find and recover unindexed documents in bounded, deterministic batches.

        Uses PostgreSQL FOR UPDATE SKIP LOCKED to coordinate independent worker processes.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size cannot exceed maximum allowed limit of {self.MAX_BATCH_SIZE}"
            )

        timeout = (
            claim_timeout_seconds
            if claim_timeout_seconds is not None
            else self._default_claim_timeout
        )
        claim_token = str(uuid4())

        # 1. Short atomic claim transaction with row locking (SKIP LOCKED)
        with database_transaction(self._session_factory) as session:
            persistence = DocumentPersistenceService(session)
            claimed_docs = persistence.claim_unindexed_documents(
                limit=batch_size,
                max_attempts=max_attempts,
                claim_timeout_seconds=timeout,
                claim_token=claim_token,
                include_failed=include_failed,
            )
            targets = [(d.id, d.paper_id, d.indexing_attempts) for d in claimed_docs]

        scanned_count = len(targets)
        recovered_count = 0
        failed_count = 0
        results: list[RecoveryItemResult] = []

        # 2. Process each claimed document outside the claim transaction under paper lock
        for doc_id, paper_id, attempts in targets:
            with get_paper_ingestion_lock(paper_id):
                item_res = self._execute_document_indexing(doc_id, paper_id, claim_token, attempts)
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

    def reindex_document(self, doc_id: UUID) -> RecoveryItemResult:
        """Manually trigger reindexing of an existing document to ensure vector payloads have collection_id."""
        with database_transaction(self._session_factory) as session:
            persistence = DocumentPersistenceService(session)
            doc = persistence.get_document(doc_id)
            if doc is None:
                raise RecordNotFoundError(f"Document {doc_id} not found")
            paper_id = doc.paper_id
            attempts = doc.indexing_attempts or 0
        claim_token = f"manual-reindex-{uuid4()}"
        with get_paper_ingestion_lock(paper_id):
            return self._execute_document_indexing(doc_id, paper_id, claim_token, attempts)
