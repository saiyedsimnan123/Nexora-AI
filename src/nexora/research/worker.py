"""Durable background worker for research RAG query jobs.

Executes asynchronous research queries, literature retrieval, and Gemini answer
synthesis with heartbeat lease renewals, two-gate fencing tokens, and bounded retries.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

from sqlalchemy.orm import Session, sessionmaker

from nexora.database.models.research_job import ResearchJobStatus
from nexora.database.repository.research_job import ResearchJobRepository
from nexora.database.session import (
    create_database_session,
    worker_database_transaction,
)
from nexora.llm.gemini_provider import sanitize_secret_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchRunOutcome:
    """Outcome summary from the most recent worker execution cycle."""

    job_id: UUID
    final_status: str
    will_retry: bool
    backoff_seconds: float = 0.0
    error_message: str | None = None


def _is_permanent_failure(error_text: str) -> bool:
    """Return True when an error is permanent and should fail immediately without retries."""
    lowered = error_text.lower()
    permanent_markers = (
        "must",
        "invalid",
        "api key",
        "unauthorized",
        "forbidden",
        "permission denied",
        "not configured",
        "model not found",
        "http 400",
        "http 401",
        "http 403",
        "http 404",
    )
    return any(marker in lowered for marker in permanent_markers)


class ResearchJobWorker:
    """Asynchronous background worker for executing durable research RAG jobs."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        rag_pipeline: Any | Callable[[], Any],
        *,
        worker_token: str | None = None,
        poll_interval_seconds: float = 1.0,
        lease_duration_seconds: float = 300.0,
        heartbeat_interval_seconds: float = 20.0,
        max_attempts: int = 3,
        batch_size: int = 1,
    ) -> None:
        self._session_factory = session_factory
        self._rag_pipeline_source = rag_pipeline
        self._worker_token = worker_token or f"research-worker-{uuid4().hex[:8]}"
        self._poll_interval = poll_interval_seconds
        self._lease_duration = lease_duration_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._max_attempts = max_attempts
        self._batch_size = max(1, batch_size)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._local_outcome = threading.local()

    @property
    def last_run_outcome(self) -> ResearchRunOutcome | None:
        """Return the outcome of the last job executed on the calling thread."""
        return getattr(self._local_outcome, "value", None)

    @property
    def worker_token(self) -> str:
        """Unique fencing identifier for this worker instance."""
        return self._worker_token

    @property
    def is_running(self) -> bool:
        """Whether the background worker thread is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def _resolve_rag_pipeline(self) -> Any:
        """Resolve or reconstruct the RAG pipeline."""
        if callable(self._rag_pipeline_source):
            return self._rag_pipeline_source()
        return self._rag_pipeline_source

    def start(self) -> None:
        """Start the worker thread if not already running."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"ResearchJobWorker-{self._worker_token}",
            daemon=True,
        )
        self._thread.start()
        logger.info("ResearchJobWorker started with token %s", self._worker_token)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for completion."""
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("ResearchJobWorker stopped: %s", self._worker_token)

    def _run_loop(self) -> None:
        """Internal polling loop."""
        while not self._stop_event.is_set():
            try:
                processed = self.run_once()
                if processed == 0:
                    self._stop_event.wait(self._poll_interval)
            except Exception as exc:
                logger.exception("Unexpected error in research worker polling loop: %s", exc)
                self._stop_event.wait(self._poll_interval)

    def run_once(self, *, target_job_id: UUID | None = None) -> int:
        """Execute a single processing cycle.

        1. Sweeps expired leases to failed or resets for retry.
        2. Claims up to batch_size pending/queued jobs (prioritizing target_job_id if provided).
        3. Executes RAG answer generation with background lease heartbeats.
        4. Persists grounded answer, citations, and metadata.

        Returns:
            Number of jobs processed in this cycle.
        """
        self._local_outcome.value = None
        now_utc = datetime.now(timezone.utc)

        # 1. Sweep expired leases in a short transaction
        try:
            with worker_database_transaction(self._session_factory) as sweep_session:
                repo = ResearchJobRepository(sweep_session)
                repo.sweep_expired_leases(
                    lease_expiry_threshold=now_utc,
                    max_attempts=self._max_attempts,
                )
        except Exception as exc:
            logger.warning("Failed to sweep expired research job leases: %s", exc)

        # 2. Claim next available job
        claimed_job_data: dict[str, Any] | None = None
        try:
            with worker_database_transaction(self._session_factory) as claim_session:
                repo = ResearchJobRepository(claim_session)
                jobs = repo.claim_pending_or_stale(
                    limit=self._batch_size,
                    claimed_by=self._worker_token,
                    lease_duration_seconds=self._lease_duration,
                    max_attempts=self._max_attempts,
                    target_job_id=target_job_id,
                )
                if jobs:
                    j = jobs[0]
                    claimed_job_data = {
                        "id": j.id,
                        "query": j.query,
                        "retrieval_mode": j.retrieval_mode,
                        "top_k": j.top_k,
                        "min_score": j.min_score,
                        "temperature": j.temperature,
                        "max_output_tokens": j.max_output_tokens,
                        "paper_id": j.paper_id,
                        "collection_id": j.collection_id,
                        "attempts": j.attempts,
                        "max_attempts": j.max_attempts,
                    }
        except Exception as exc:
            logger.warning("Failed to claim research job: %s", exc)
            return 0

        if claimed_job_data is None:
            return 0

        self._process_claimed_job(claimed_job_data)
        return 1

    def _heartbeat_loop(self, job_id: UUID, stop_event: threading.Event) -> None:
        """Renew the worker lease periodically while RAG generation is running."""
        while not stop_event.wait(self._heartbeat_interval):
            try:
                with worker_database_transaction(self._session_factory) as hb_session:
                    repo = ResearchJobRepository(hb_session)
                    renewed = repo.renew_lease(
                        job_id=job_id,
                        claimed_by=self._worker_token,
                        lease_duration_seconds=self._lease_duration,
                    )
                    if not renewed:
                        logger.warning("Heartbeat lease renewal failed for research job %s", job_id)
                        break
            except Exception as exc:
                logger.warning("Error during heartbeat renewal for job %s: %s", job_id, exc)

    def _process_claimed_job(self, job_data: dict[str, Any]) -> None:
        """Process a single claimed research job."""
        job_id: UUID = job_data["id"]
        logger.info("Executing research job %s: %s", job_id, job_data["query"][:60])

        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, stop_heartbeat),
            name=f"ResearchHeartbeat-{job_id}",
            daemon=True,
        )
        heartbeat_thread.start()

        pipeline = None
        try:
            pipeline = self._resolve_rag_pipeline()
            if pipeline is None:
                raise RuntimeError("No RAG pipeline available to execute research job.")

            scoped_paper_id = str(job_data["paper_id"]) if job_data.get("paper_id") else None
            scoped_collection_ids = [str(job_data["collection_id"])] if job_data.get("collection_id") else None

            # Execute RAG answer generation completely outside DB transaction
            response = pipeline.answer(
                request=job_data["query"],
                retrieval_mode=job_data.get("retrieval_mode") or "hybrid",
                top_k=job_data.get("top_k") or 5,
                min_score=job_data.get("min_score"),
                temperature=job_data.get("temperature"),
                max_output_tokens=job_data.get("max_output_tokens"),
                paper_id=scoped_paper_id,
                collection_ids=scoped_collection_ids,
            )

            if getattr(response, "status", None) == "error":
                err_msg = getattr(response, "error", None) or "RAG pipeline execution failed."
                raise RuntimeError(err_msg)

            result_dict = response.to_dict()

            # Persist completion
            with worker_database_transaction(self._session_factory) as finish_session:
                repo = ResearchJobRepository(finish_session)
                completed = repo.complete_job(
                    job_id=job_id,
                    claimed_by=self._worker_token,
                    result=result_dict,
                )
                if not completed:
                    logger.warning("Failed to mark research job %s as completed (fencing lost)", job_id)
                else:
                    logger.info("Research job %s successfully completed", job_id)

            self._local_outcome.value = ResearchRunOutcome(
                job_id=job_id,
                final_status=ResearchJobStatus.COMPLETED.value,
                will_retry=False,
            )

        except Exception as exc:
            sanitized_err = sanitize_secret_text(str(exc))
            logger.warning("Research job %s execution failed: %s", job_id, sanitized_err)

            attempts = job_data.get("attempts", 1)
            job_max_attempts = job_data.get("max_attempts", self._max_attempts)
            backoff_seconds = min(30.0, float(2.0 ** attempts))
            retryable = not _is_permanent_failure(sanitized_err)
            will_retry = bool(retryable and attempts < job_max_attempts)

            try:
                with worker_database_transaction(self._session_factory) as err_session:
                    repo = ResearchJobRepository(err_session)
                    repo.fail_job(
                        job_id=job_id,
                        claimed_by=self._worker_token,
                        error_message=sanitized_err,
                        retryable=retryable,
                        backoff_seconds=backoff_seconds,
                    )
            except Exception as inner_exc:
                logger.error("Failed recording failure for job %s: %s", job_id, inner_exc)

            self._local_outcome.value = ResearchRunOutcome(
                job_id=job_id,
                final_status=(
                    ResearchJobStatus.QUEUED.value
                    if will_retry
                    else ResearchJobStatus.FAILED.value
                ),
                will_retry=will_retry,
                backoff_seconds=backoff_seconds if will_retry else 0.0,
                error_message=sanitized_err,
            )

        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=2.0)
