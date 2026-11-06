"""Background worker for durable knowledge graph extraction (Stage 5 Step 4.6.2).

Polls for pending or stale graph extraction jobs, prepares context without holding
database locks, executes LLM extraction with lease heartbeats, and persists graph
entities and relationships under two-gate fencing tokens.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.orm import Session, sessionmaker

from nexora.database.repository.graph_extraction_job import (
    GraphExtractionJobRepository,
)
from nexora.database.session import (
    create_database_session,
    worker_database_transaction,
)
from nexora.graph.models import FencingTokenError, GraphExtractionError
from nexora.graph.service import KnowledgeGraphExtractionService

logger = logging.getLogger(__name__)


class GraphExtractionWorker:
    """Asynchronous background worker for executing durable graph extraction jobs."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        graph_service: KnowledgeGraphExtractionService,
        *,
        worker_token: str | None = None,
        poll_interval_seconds: float = 1.0,
        lease_duration_seconds: int = 300,
        heartbeat_interval_seconds: float = 30.0,
        max_attempts: int = 3,
        batch_size: int = 1,
    ) -> None:
        self._session_factory = session_factory
        self._service = graph_service
        self._worker_token = worker_token or f"worker-{uuid4().hex[:8]}"
        self._poll_interval = poll_interval_seconds
        self._lease_duration = lease_duration_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._max_attempts = max_attempts
        self._batch_size = max(1, batch_size)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def worker_token(self) -> str:
        """Unique fencing identifier for this worker instance."""
        return self._worker_token

    @property
    def is_running(self) -> bool:
        """Whether the background worker thread is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the worker thread if not already running."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"GraphExtractionWorker-{self._worker_token}",
            daemon=True,
        )
        self._thread.start()
        logger.info("GraphExtractionWorker started with token %s", self._worker_token)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for completion."""
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("GraphExtractionWorker stopped: %s", self._worker_token)

    def _run_loop(self) -> None:
        """Internal polling loop."""
        while not self._stop_event.is_set():
            try:
                processed = self.run_once()
                if processed == 0:
                    self._stop_event.wait(self._poll_interval)
            except Exception as exc:
                logger.exception("Unexpected error in graph worker polling loop: %s", exc)
                self._stop_event.wait(self._poll_interval)

    def run_once(self) -> int:
        """Execute a single polling cycle.

        1. Sweeps expired leases to failed or resets attempts.
        2. Claims up to batch_size pending or stale jobs using dialect-aware locking.
        3. Processes each claimed job with zero open transactions during LLM inference.
        4. Persists graph nodes and edges under two-gate fencing tokens.

        Returns:
            Number of jobs processed in this cycle.
        """
        now_utc = datetime.now(timezone.utc)

        # 1. Sweep expired leases in a quick transaction
        try:
            with worker_database_transaction(self._session_factory) as sweep_session:
                repo = GraphExtractionJobRepository(sweep_session)
                repo.sweep_expired_leases(
                    lease_expiry_threshold=now_utc,
                    max_attempts=self._max_attempts,
                )
        except Exception as exc:
            logger.warning("Failed to sweep expired graph extraction leases: %s", exc)

        # 2. Claim next available job
        claimed_job_info: tuple[UUID, UUID, UUID, bool] | None = None
        try:
            with worker_database_transaction(self._session_factory) as claim_session:
                repo = GraphExtractionJobRepository(claim_session)
                jobs = repo.claim_pending_or_stale(
                    limit=self._batch_size,
                    claimed_by=self._worker_token,
                    lease_duration_seconds=self._lease_duration,
                    max_attempts=self._max_attempts,
                )
                if jobs:
                    job = jobs[0]
                    claimed_job_info = (job.id, job.paper_id, job.collection_id, job.metadata_extracted)
        except Exception as exc:
            logger.warning("Failed to claim graph extraction job: %s", exc)
            return 0

        if claimed_job_info is None:
            return 0

        job_id, paper_id, collection_id, metadata_extracted = claimed_job_info
        self._process_claimed_job(
            job_id=job_id,
            paper_id=paper_id,
            collection_id=collection_id,
            metadata_extracted=metadata_extracted,
        )
        return 1

    def _process_claimed_job(
        self,
        *,
        job_id: UUID,
        paper_id: UUID,
        collection_id: UUID,
        metadata_extracted: bool,
    ) -> None:
        """Process a claimed job through read preparation, LLM call, and gated persistence."""
        logger.info("Processing graph extraction job %s for paper %s", job_id, paper_id)

        # Step 1: Read chunk evidence in a short, read-only transaction (zero locks held afterwards)
        read_session = create_database_session(self._session_factory)
        try:
            paper, context, messages = self._service.prepare_llm_evidence_context(
                paper_id,
                collection_id=collection_id,
                session=read_session,
            )
        except Exception as exc:
            logger.exception("Failed reading context for graph job %s: %s", job_id, exc)
            self._handle_job_error(job_id, str(exc))
            return
        finally:
            read_session.close()

        # Step 2: If no chunks or no LLM service configured, only perform deterministic metadata extraction
        if not messages or getattr(self._service, "_llm", None) is None:
            self._persist_deterministic_only(
                job_id=job_id,
                paper_id=paper_id,
                collection_id=collection_id,
                metadata_extracted=metadata_extracted,
            )
            return

        # Step 3: Run LLM extraction with background lease renewal heartbeat
        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, stop_heartbeat),
            name=f"GraphHeartbeat-{job_id}",
            daemon=True,
        )
        heartbeat_thread.start()

        llm_payload = None
        try:
            # LLM network call runs completely OUTSIDE of any database transaction or connection
            response = self._service._llm.generate(messages, temperature=0.0)
            llm_payload = self._service.parse_llm_extraction_response(response.text)
        except Exception as exc:
            logger.warning("LLM extraction failed for graph job %s: %s", job_id, exc)
            self._handle_job_error(job_id, str(exc))
            return
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=2.0)

        # Step 4: Persist graph nodes, edges, and finalize job inside gated transaction
        try:
            with worker_database_transaction(self._session_factory) as persist_session:
                repo = GraphExtractionJobRepository(persist_session)

                # Gate 1: Verify job exists, status == processing, claimed_by == worker_token, lease unexpired
                gate1_now = datetime.now(timezone.utc)
                if not repo.verify_fencing_gate1(
                    job_id=job_id,
                    claimed_by=self._worker_token,
                    unexpired_as_of=gate1_now,
                ):
                    raise FencingTokenError(
                        f"Gate 1 fencing verification failed for job {job_id}: lease expired or token mismatch."
                    )

                # Self-healing Tier 1: If metadata extraction was not completed yet, run it now
                if not metadata_extracted:
                    self._service.extract_metadata_deterministic(
                        paper_id=paper_id,
                        collection_id=collection_id,
                        session=persist_session,
                    )

                # Tier 2: Persist extracted entities and relationships
                self._service.persist_llm_extracted_payload(
                    paper_id=paper_id,
                    collection_id=collection_id,
                    session=persist_session,
                    payload=llm_payload,
                    context=context,
                )

                # Gate 2: Verify fresh unexpired timestamp and update status = completed
                gate2_now = datetime.now(timezone.utc)
                ok = repo.finalize_job_gate2(
                    job_id=job_id,
                    claimed_by=self._worker_token,
                    completion_time=gate2_now,
                )
                if not ok:
                    raise FencingTokenError(
                        f"Gate 2 finalization failed for job {job_id}: lease expired during persistence."
                    )

            logger.info("Successfully completed graph extraction job %s for paper %s", job_id, paper_id)

        except FencingTokenError as fte:
            # Fencing violation: Transaction rolled back automatically.
            # Do NOT reschedule or fail the job because this worker no longer holds a valid lease.
            logger.warning("Graph extraction job %s rolled back due to fencing violation: %s", job_id, fte)

        except Exception as exc:
            logger.exception("Failed persisting graph extraction job %s: %s", job_id, exc)
            self._handle_job_error(job_id, str(exc))

    def _persist_deterministic_only(
        self,
        *,
        job_id: UUID,
        paper_id: UUID,
        collection_id: UUID,
        metadata_extracted: bool,
    ) -> None:
        """Handle jobs that only require deterministic metadata extraction (no chunks or no LLM)."""
        try:
            with worker_database_transaction(self._session_factory) as persist_session:
                repo = GraphExtractionJobRepository(persist_session)
                gate1_now = datetime.now(timezone.utc)
                if not repo.verify_fencing_gate1(
                    job_id=job_id,
                    claimed_by=self._worker_token,
                    unexpired_as_of=gate1_now,
                ):
                    raise FencingTokenError(f"Gate 1 verification failed for job {job_id}.")

                if not metadata_extracted:
                    self._service.extract_metadata_deterministic(
                        paper_id=paper_id,
                        collection_id=collection_id,
                        session=persist_session,
                    )

                gate2_now = datetime.now(timezone.utc)
                ok = repo.finalize_job_gate2(
                    job_id=job_id,
                    claimed_by=self._worker_token,
                    completion_time=gate2_now,
                )
                if not ok:
                    raise FencingTokenError(f"Gate 2 finalization failed for job {job_id}.")

            logger.info("Successfully completed deterministic-only graph job %s", job_id)

        except FencingTokenError as fte:
            logger.warning("Deterministic graph job %s rolled back due to fencing: %s", job_id, fte)
        except Exception as exc:
            logger.exception("Failed completing deterministic graph job %s: %s", job_id, exc)
            self._handle_job_error(job_id, str(exc))

    def _heartbeat_loop(self, job_id: UUID, stop_event: threading.Event) -> None:
        """Background lease renewal loop running while an external LLM call is in-flight."""
        while not stop_event.wait(self._heartbeat_interval):
            try:
                with worker_database_transaction(self._session_factory) as hb_session:
                    repo = GraphExtractionJobRepository(hb_session)
                    renewed = repo.renew_lease(
                        job_id,
                        claimed_by=self._worker_token,
                        extension_seconds=self._lease_duration,
                    )
                    if not renewed:
                        logger.warning("Heartbeat could not renew lease for job %s: lost claim", job_id)
                        break
            except Exception as exc:
                logger.warning("Heartbeat error renewing lease for job %s: %s", job_id, exc)

    def _handle_job_error(self, job_id: UUID, error_message: str) -> None:
        """Reschedule job for retry or transition to failed if attempts exhausted."""
        try:
            with worker_database_transaction(self._session_factory) as err_session:
                repo = GraphExtractionJobRepository(err_session)
                repo.reschedule_or_fail_job(
                    job_id=job_id,
                    error_message=error_message,
                    claimed_by=self._worker_token,
                    max_attempts=self._max_attempts,
                )
        except Exception as exc:
            logger.error("Failed recording error for graph job %s: %s", job_id, exc)


KnowledgeGraphWorker = GraphExtractionWorker

__all__ = [
    "GraphExtractionWorker",
    "KnowledgeGraphWorker",
]
