"""Repository for GraphExtractionJob models."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from nexora.database.models.document import Document
from nexora.database.models.graph_extraction_job import (
    GraphExtractionJob,
    GraphExtractionStatus,
)
from nexora.database.models.paper import Paper
from nexora.database.repository.base import (
    BaseRepository,
    ConstraintError,
    DuplicateRecordError,
    RecordNotFoundError,
)

logger = logging.getLogger(__name__)


class GraphExtractionJobRepository(BaseRepository[GraphExtractionJob]):
    """Repository for managing durable Knowledge Graph extraction background jobs."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, GraphExtractionJob)

    def create_job(
        self,
        *,
        collection_id: UUID,
        paper_id: UUID,
        document_id: UUID | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        status: str = GraphExtractionStatus.PENDING.value,
        metadata_extracted: bool = False,
        job_metadata: dict[str, Any] | None = None,
        validate_ownership: bool = True,
    ) -> GraphExtractionJob:
        """Create a new durable extraction job with tenant and ownership validation."""
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
        if not isinstance(paper_id, UUID):
            raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")
        if document_id is not None and not isinstance(document_id, UUID):
            raise TypeError(f"document_id must be a UUID, got {type(document_id).__name__}")

        if validate_ownership:
            paper = self._session.get(Paper, paper_id)
            if paper is None:
                raise RecordNotFoundError(f"Paper with id {paper_id} not found.")
            if paper.collection_id != collection_id:
                raise ConstraintError(
                    f"Tenant boundary violation: Paper {paper_id} belongs to collection "
                    f"{paper.collection_id}, not requested collection {collection_id}."
                )
            if document_id is not None:
                doc = self._session.get(Document, document_id)
                if doc is None:
                    raise RecordNotFoundError(f"Document with id {document_id} not found.")
                if doc.paper_id != paper_id:
                    raise ConstraintError(
                        f"Ownership mismatch: Document {document_id} belongs to paper "
                        f"{doc.paper_id}, not requested paper {paper_id}."
                    )

        return self._create(
            collection_id=collection_id,
            paper_id=paper_id,
            document_id=document_id,
            status=status,
            metadata_extracted=metadata_extracted,
            priority=priority,
            max_attempts=max_attempts,
            job_metadata=job_metadata,
        )

    def get_active_by_paper_id(self, paper_id: UUID) -> GraphExtractionJob | None:
        """Retrieve any currently active (pending or processing) job for a paper."""
        stmt = (
            select(GraphExtractionJob)
            .where(
                GraphExtractionJob.paper_id == paper_id,
                GraphExtractionJob.status.in_(
                    (
                        GraphExtractionStatus.PENDING.value,
                        GraphExtractionStatus.PROCESSING.value,
                    )
                ),
            )
            .order_by(GraphExtractionJob.created_at.desc())
        )
        return self._session.scalars(stmt).first()

    def get_latest_by_paper_id(self, paper_id: UUID) -> GraphExtractionJob | None:
        """Retrieve the most recent job for a paper regardless of status."""
        stmt = (
            select(GraphExtractionJob)
            .where(GraphExtractionJob.paper_id == paper_id)
            .order_by(GraphExtractionJob.created_at.desc())
        )
        return self._session.scalars(stmt).first()

    def create_or_get_active_job(
        self,
        *,
        collection_id: UUID,
        paper_id: UUID,
        document_id: UUID | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        metadata_extracted: bool = False,
        job_metadata: dict[str, Any] | None = None,
    ) -> tuple[GraphExtractionJob, bool]:
        """Atomically create a new job or retrieve the existing active job for a paper.

        Returns:
            tuple[GraphExtractionJob, bool]: (job, created_flag)
        """
        existing = self.get_active_by_paper_id(paper_id)
        if existing is not None:
            return existing, False

        try:
            with self._session.begin_nested():
                job = self.create_job(
                    collection_id=collection_id,
                    paper_id=paper_id,
                    document_id=document_id,
                    priority=priority,
                    max_attempts=max_attempts,
                    status=GraphExtractionStatus.PENDING.value,
                    metadata_extracted=metadata_extracted,
                    job_metadata=job_metadata,
                )
                self._session.flush()
                return job, True
        except (IntegrityError, DuplicateRecordError):
            # Concurrent race condition: another thread inserted the active job
            active = self.get_active_by_paper_id(paper_id)
            if active is not None:
                return active, False
            raise

    def claim_pending_or_stale(
        self,
        limit: int = 1,
        *,
        claim_token: str | None = None,
        claimed_by: str | None = None,
        claim_timeout_seconds: float = 300.0,
        lease_duration_seconds: float | None = None,
        max_attempts: int | None = None,
    ) -> list[GraphExtractionJob]:
        """Atomically claim a batch of pending or expired processing jobs.

        Dialect-specific execution:
        - PostgreSQL: Uses `with_for_update(skip_locked=True)` for non-blocking horizontal concurrency.
        - SQLite: Assumes the transaction caller holds an immediate write reservation (`BEGIN IMMEDIATE`).
        """
        token = claimed_by or claim_token
        if not token or not isinstance(token, str):
            raise ValueError("claim_token must be a non-empty string")

        timeout = lease_duration_seconds if lease_duration_seconds is not None else claim_timeout_seconds

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        now = datetime.datetime.now(datetime.timezone.utc)
        new_expiry = now + datetime.timedelta(seconds=timeout)

        status_filter = or_(
            and_(
                GraphExtractionJob.status == GraphExtractionStatus.PENDING.value,
                or_(
                    GraphExtractionJob.next_retry_at.is_(None),
                    GraphExtractionJob.next_retry_at <= now,
                ),
            ),
            and_(
                GraphExtractionJob.status == GraphExtractionStatus.PROCESSING.value,
                or_(
                    GraphExtractionJob.lease_expires_at.is_(None),
                    GraphExtractionJob.lease_expires_at <= now,
                ),
            ),
        )

        bind = self._session.bind
        is_postgresql = bool(bind and bind.dialect.name == "postgresql")

        attempts_filter = (
            (GraphExtractionJob.attempts < max_attempts)
            if max_attempts is not None
            else (GraphExtractionJob.attempts < GraphExtractionJob.max_attempts)
        )

        stmt = (
            select(GraphExtractionJob)
            .where(
                status_filter,
                attempts_filter,
            )
            .order_by(
                GraphExtractionJob.priority.desc(),
                GraphExtractionJob.created_at.asc(),
            )
            .limit(limit)
        )

        if is_postgresql:
            stmt = stmt.with_for_update(skip_locked=True)

        jobs = list(self._session.scalars(stmt).all())
        claimed_jobs: list[GraphExtractionJob] = []

        for job in jobs:
            job.status = GraphExtractionStatus.PROCESSING.value
            job.claimed_at = now
            job.lease_expires_at = new_expiry
            job.claimed_by = token
            job.attempts += 1
            claimed_jobs.append(job)

        if claimed_jobs:
            self._session.flush()

        return claimed_jobs

    def sweep_expired_leases(
        self,
        *,
        now: datetime.datetime | None = None,
        lease_expiry_threshold: datetime.datetime | None = None,
        max_attempts: int | None = None,
    ) -> int:
        """Detect and sweep expired processing jobs:
        - Fails jobs that reached or exceeded max_attempts.
        - Resets retryable jobs back to pending with cleared lease so another worker can claim.

        Returns the total number of swept jobs.
        """
        eval_now = lease_expiry_threshold or now or datetime.datetime.now(datetime.timezone.utc)

        # 1. Transition exhausted expired claims to FAILED
        exhausted_cond = (
            (GraphExtractionJob.attempts >= max_attempts)
            if max_attempts is not None
            else (GraphExtractionJob.attempts >= GraphExtractionJob.max_attempts)
        )
        stmt_failed = (
            update(GraphExtractionJob)
            .where(
                GraphExtractionJob.status == GraphExtractionStatus.PROCESSING.value,
                GraphExtractionJob.lease_expires_at <= eval_now,
                exhausted_cond,
            )
            .values(
                status=GraphExtractionStatus.FAILED.value,
                last_error="Lease expired after maximum extraction attempts reached.",
                claimed_at=None,
                claimed_by=None,
                lease_expires_at=None,
                updated_at=eval_now,
            )
        )
        res_failed = self._session.execute(stmt_failed)

        # 2. Reset unexhausted expired claims to PENDING
        retryable_cond = (
            (GraphExtractionJob.attempts < max_attempts)
            if max_attempts is not None
            else (GraphExtractionJob.attempts < GraphExtractionJob.max_attempts)
        )
        stmt_reset = (
            update(GraphExtractionJob)
            .where(
                GraphExtractionJob.status == GraphExtractionStatus.PROCESSING.value,
                GraphExtractionJob.lease_expires_at <= eval_now,
                retryable_cond,
            )
            .values(
                status=GraphExtractionStatus.PENDING.value,
                claimed_at=None,
                claimed_by=None,
                lease_expires_at=None,
                updated_at=eval_now,
            )
        )
        res_reset = self._session.execute(stmt_reset)

        total_swept = int(res_failed.rowcount + res_reset.rowcount)
        if total_swept > 0:
            self._session.flush()
        return total_swept

    def renew_lease(
        self,
        job_id: UUID,
        *,
        claim_token: str | None = None,
        claimed_by: str | None = None,
        claim_timeout_seconds: float = 300.0,
        extension_seconds: float | None = None,
    ) -> bool:
        """Renew the lease deadline for a processing job while heartbeat is alive.

        Returns True if the lease was successfully renewed, False if the lease was lost.
        """
        token = claimed_by or claim_token
        if not token:
            raise ValueError("claim_token or claimed_by must be provided")

        timeout = extension_seconds if extension_seconds is not None else claim_timeout_seconds
        now = datetime.datetime.now(datetime.timezone.utc)
        new_expiry = now + datetime.timedelta(seconds=timeout)

        stmt = (
            update(GraphExtractionJob)
            .where(
                GraphExtractionJob.id == job_id,
                GraphExtractionJob.status == GraphExtractionStatus.PROCESSING.value,
                GraphExtractionJob.claimed_by == token,
                GraphExtractionJob.lease_expires_at > now,
            )
            .values(
                lease_expires_at=new_expiry,
                updated_at=now,
            )
        )
        result = self._session.execute(stmt)
        if result.rowcount == 1:
            self._session.flush()
            return True
        return False
        return False

    def verify_fencing_gate1(
        self,
        job_id: UUID,
        *,
        claim_token: str | None = None,
        claimed_by: str | None = None,
        now: datetime.datetime | None = None,
        unexpired_as_of: datetime.datetime | None = None,
    ) -> bool:
        """Gate 1: Verify the job is in processing state, unexpired, and held by claim_token.

        On PostgreSQL, acquires a row lock. On SQLite, performs an atomic touch update.
        """
        token = claimed_by or claim_token
        if not token:
            raise ValueError("claim_token or claimed_by must be provided")

        eval_now = unexpired_as_of or now or datetime.datetime.now(datetime.timezone.utc)
        bind = self._session.bind
        is_postgresql = bool(bind and bind.dialect.name == "postgresql")

        if is_postgresql:
            job = self._session.scalars(
                select(GraphExtractionJob)
                .where(GraphExtractionJob.id == job_id)
                .with_for_update()
            ).one_or_none()

            if (
                job is None
                or job.status != GraphExtractionStatus.PROCESSING.value
                or job.claimed_by != token
                or job.lease_expires_at is None
                or job.lease_expires_at <= eval_now
            ):
                return False
            return True
        else:
            stmt = (
                update(GraphExtractionJob)
                .where(
                    GraphExtractionJob.id == job_id,
                    GraphExtractionJob.status == GraphExtractionStatus.PROCESSING.value,
                    GraphExtractionJob.claimed_by == token,
                    GraphExtractionJob.lease_expires_at > eval_now,
                )
                .values(updated_at=eval_now)
            )
            res = self._session.execute(stmt)
            return res.rowcount == 1

    def finalize_job_gate2(
        self,
        job_id: UUID,
        *,
        claim_token: str | None = None,
        claimed_by: str | None = None,
        now: datetime.datetime | None = None,
        completion_time: datetime.datetime | None = None,
    ) -> bool:
        """Gate 2: Atomically complete the job with a fresh post-write timestamp.

        Enforces lease_expires_at > now. Returns True if completed, False if lease was lost.
        """
        token = claimed_by or claim_token
        if not token:
            raise ValueError("claim_token or claimed_by must be provided")

        eval_now = completion_time or now or datetime.datetime.now(datetime.timezone.utc)

        stmt = (
            update(GraphExtractionJob)
            .where(
                GraphExtractionJob.id == job_id,
                GraphExtractionJob.status == GraphExtractionStatus.PROCESSING.value,
                GraphExtractionJob.claimed_by == token,
                GraphExtractionJob.lease_expires_at > eval_now,
            )
            .values(
                status=GraphExtractionStatus.COMPLETED.value,
                completed_at=eval_now,
                claimed_at=None,
                claimed_by=None,
                lease_expires_at=None,
                last_error=None,
                updated_at=eval_now,
            )
        )
        res = self._session.execute(stmt)
        if res.rowcount == 1:
            self._session.flush()
            return True
        return False

    def reschedule_or_fail_job(
        self,
        job_id: UUID,
        *,
        claim_token: str | None = None,
        claimed_by: str | None = None,
        error_message: str,
        backoff_seconds: float = 10.0,
        max_attempts: int | None = None,
    ) -> GraphExtractionStatus:
        """Transition an errored job to failed (if attempts >= max_attempts) or reschedule it."""
        token = claimed_by or claim_token
        now = datetime.datetime.now(datetime.timezone.utc)
        job = self.get_by_id(job_id)
        if job is None or (token is not None and job.claimed_by != token and job.status == GraphExtractionStatus.PROCESSING.value):
            return GraphExtractionStatus.FAILED

        effective_max = max_attempts if max_attempts is not None else job.max_attempts
        if job.attempts >= effective_max:
            job.status = GraphExtractionStatus.FAILED.value
            job.last_error = error_message
            job.claimed_at = None
            job.claimed_by = None
            job.lease_expires_at = None
            self._session.flush()
            return GraphExtractionStatus.FAILED
        else:
            job.status = GraphExtractionStatus.PENDING.value
            job.last_error = error_message
            job.claimed_at = None
            job.claimed_by = None
            job.lease_expires_at = None
            job.next_retry_at = now + datetime.timedelta(seconds=backoff_seconds)
            self._session.flush()
            return GraphExtractionStatus.PENDING

    def reset_job_for_retry(self, job_id: UUID) -> GraphExtractionJob:
        """Manually reset a failed job back to pending state with reset attempt counter."""
        job = self.get_by_id_or_raise(job_id)
        if job.status != GraphExtractionStatus.FAILED.value:
            raise ConstraintError(
                f"Only failed jobs can be manually retried, current status is '{job.status}'"
            )

        job.status = GraphExtractionStatus.PENDING.value
        job.attempts = 0
        job.last_error = None
        job.claimed_at = None
        job.claimed_by = None
        job.lease_expires_at = None
        job.next_retry_at = None
        job.completed_at = None
        self._session.flush()
        return job

    def count_by_status(self) -> dict[str, int]:
        """Return counts of extraction jobs grouped by status."""
        stmt = (
            select(GraphExtractionJob.status, func.count(GraphExtractionJob.id))
            .group_by(GraphExtractionJob.status)
        )
        counts = {s.value: 0 for s in GraphExtractionStatus}
        for status_val, count in self._session.execute(stmt):
            counts[str(status_val)] = int(count)
        return counts
