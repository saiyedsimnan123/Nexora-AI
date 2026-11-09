"""Repository for ResearchJob models."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session

from nexora.database.models.collection import Collection
from nexora.database.models.paper import Paper
from nexora.database.models.research_job import (
    ResearchJob,
    ResearchJobStatus,
)
from nexora.database.repository.base import (
    BaseRepository,
    ConstraintError,
    DuplicateRecordError,
    RecordNotFoundError,
)
from nexora.llm.gemini_provider import sanitize_secret_text

logger = logging.getLogger(__name__)


class ResearchJobRepository(BaseRepository[ResearchJob]):
    """Repository for managing durable research RAG background jobs."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, ResearchJob)

    def create_job(
        self,
        *,
        query: str,
        user_id: UUID | None = None,
        guest_session_id: str | None = None,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        min_score: float | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        collection_id: UUID | None = None,
        paper_id: UUID | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        status: str = ResearchJobStatus.QUEUED.value,
        validate_ownership: bool = True,
    ) -> ResearchJob:
        """Create a new durable research RAG job with tenant and ownership validation."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if user_id is None and not guest_session_id:
            raise ValueError("Either user_id or guest_session_id must be provided")

        if user_id is not None and not isinstance(user_id, UUID):
            raise TypeError(f"user_id must be a UUID, got {type(user_id).__name__}")
        if collection_id is not None and not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")
        if paper_id is not None and not isinstance(paper_id, UUID):
            raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")

        if validate_ownership:
            if paper_id is not None:
                paper = self._session.get(Paper, paper_id)
                if paper is None:
                    raise RecordNotFoundError(f"Paper with id {paper_id} not found.")
                if collection_id is not None and paper.collection_id != collection_id:
                    raise ConstraintError(
                        f"Tenant boundary violation: Paper {paper_id} belongs to collection "
                        f"{paper.collection_id}, not requested collection {collection_id}."
                    )
            if collection_id is not None:
                col = self._session.get(Collection, collection_id)
                if col is None:
                    raise RecordNotFoundError(f"Collection with id {collection_id} not found.")
                if user_id is not None and getattr(col, "workspace_id", None) is not None:
                    from nexora.database.models.workspace import Workspace

                    ws = self._session.get(Workspace, col.workspace_id)
                    if ws is not None and getattr(ws, "user_id", None) is not None and ws.user_id != user_id:
                        raise ConstraintError(
                            f"Access denied: Collection {collection_id} does not belong to user {user_id}."
                        )

        return self._create(
            user_id=user_id,
            guest_session_id=guest_session_id.strip() if guest_session_id else None,
            collection_id=collection_id,
            paper_id=paper_id,
            query=query.strip(),
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            min_score=min_score,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            status=status,
            priority=priority,
            max_attempts=max_attempts,
        )

    def get_by_id(self, job_id: UUID) -> ResearchJob | None:
        """Retrieve a job by primary key UUID."""
        return super().get_by_id(job_id)

    def get_for_user_or_guest(
        self,
        job_id: UUID,
        *,
        user_id: UUID | None = None,
        guest_session_id: str | None = None,
    ) -> ResearchJob | None:
        """Fetch a job ensuring the requesting user or guest owns it."""
        job = self.get_by_id(job_id)
        if job is None:
            return None

        # Check user ownership
        if user_id is not None:
            from nexora.auth.guest import GUEST_USER_ID

            if job.user_id == user_id:
                return job
            if user_id == GUEST_USER_ID and guest_session_id and job.guest_session_id == guest_session_id:
                return job
            return None

        # Check guest ownership
        if guest_session_id and job.guest_session_id == guest_session_id:
            return job

        return None

    def claim_pending_or_stale(
        self,
        limit: int = 1,
        *,
        claimed_by: str,
        lease_duration_seconds: float = 300.0,
        max_attempts: int | None = None,
    ) -> list[ResearchJob]:
        """Atomically claim a batch of queued or expired processing jobs.

        Dialect-specific execution:
        - PostgreSQL: Uses `with_for_update(skip_locked=True)` for non-blocking horizontal concurrency.
        - SQLite: Assumes the transaction caller holds an immediate write reservation.
        """
        if not claimed_by or not isinstance(claimed_by, str):
            raise ValueError("claimed_by must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        now = datetime.datetime.now(datetime.timezone.utc)
        new_expiry = now + datetime.timedelta(seconds=lease_duration_seconds)

        status_filter = or_(
            and_(
                ResearchJob.status == ResearchJobStatus.QUEUED.value,
                or_(
                    ResearchJob.next_retry_at.is_(None),
                    ResearchJob.next_retry_at <= now,
                ),
            ),
            and_(
                ResearchJob.status == ResearchJobStatus.PROCESSING.value,
                or_(
                    ResearchJob.lease_expires_at.is_(None),
                    ResearchJob.lease_expires_at <= now,
                ),
            ),
        )

        attempts_filter = (
            (ResearchJob.attempts < max_attempts)
            if max_attempts is not None
            else (ResearchJob.attempts < ResearchJob.max_attempts)
        )

        bind = self._session.bind
        is_postgresql = bool(bind and bind.dialect.name == "postgresql")

        stmt = (
            select(ResearchJob)
            .where(status_filter, attempts_filter)
            .order_by(
                ResearchJob.priority.desc(),
                ResearchJob.created_at.asc(),
            )
            .limit(limit)
        )

        if is_postgresql:
            stmt = stmt.with_for_update(skip_locked=True)

        jobs = list(self._session.scalars(stmt).all())
        claimed_jobs: list[ResearchJob] = []

        for job in jobs:
            job.status = ResearchJobStatus.PROCESSING.value
            job.claimed_at = now
            job.lease_expires_at = new_expiry
            job.claimed_by = claimed_by
            job.attempts += 1
            claimed_jobs.append(job)

        if claimed_jobs:
            self._session.flush()

        return claimed_jobs

    def renew_lease(
        self,
        job_id: UUID,
        claimed_by: str,
        lease_duration_seconds: float = 300.0,
    ) -> bool:
        """Extend the lease expiration for an actively processing job."""
        now = datetime.datetime.now(datetime.timezone.utc)
        new_expiry = now + datetime.timedelta(seconds=lease_duration_seconds)

        stmt = (
            update(ResearchJob)
            .where(
                ResearchJob.id == job_id,
                ResearchJob.claimed_by == claimed_by,
                ResearchJob.status == ResearchJobStatus.PROCESSING.value,
            )
            .values(lease_expires_at=new_expiry)
        )
        result = self._session.execute(stmt)
        return result.rowcount > 0

    def complete_job(
        self,
        job_id: UUID,
        claimed_by: str,
        result: dict[str, Any],
    ) -> bool:
        """Mark a job as completed and store its synthesized answer and citations."""
        now = datetime.datetime.now(datetime.timezone.utc)
        stmt = (
            update(ResearchJob)
            .where(
                ResearchJob.id == job_id,
                ResearchJob.claimed_by == claimed_by,
                ResearchJob.status == ResearchJobStatus.PROCESSING.value,
            )
            .values(
                status=ResearchJobStatus.COMPLETED.value,
                result=result,
                completed_at=now,
                lease_expires_at=None,
                error_message=None,
            )
        )
        res = self._session.execute(stmt)
        return res.rowcount > 0

    def fail_job(
        self,
        job_id: UUID,
        claimed_by: str,
        error_message: str,
        *,
        retryable: bool = True,
        backoff_seconds: float = 5.0,
    ) -> bool:
        """Record job failure with secret sanitization and bounded retries."""
        sanitized_error = sanitize_secret_text(error_message)[:2000]
        job = self.get_by_id(job_id)
        if job is None or job.claimed_by != claimed_by:
            return False

        now = datetime.datetime.now(datetime.timezone.utc)
        if retryable and job.attempts < job.max_attempts:
            next_retry = now + datetime.timedelta(seconds=backoff_seconds)
            job.status = ResearchJobStatus.QUEUED.value
            job.claimed_by = None
            job.lease_expires_at = None
            job.next_retry_at = next_retry
            job.error_message = sanitized_error
        else:
            job.status = ResearchJobStatus.FAILED.value
            job.completed_at = now
            job.lease_expires_at = None
            job.error_message = sanitized_error

        self._session.flush()
        return True

    def sweep_expired_leases(
        self,
        *,
        lease_expiry_threshold: datetime.datetime | None = None,
        max_attempts: int | None = None,
    ) -> int:
        """Reset stale processing jobs or transition exhausted attempts to failed."""
        now = lease_expiry_threshold or datetime.datetime.now(datetime.timezone.utc)

        # 1. Exhausted jobs -> failed
        exhausted_stmt = (
            select(ResearchJob)
            .where(
                ResearchJob.status == ResearchJobStatus.PROCESSING.value,
                ResearchJob.lease_expires_at.is_not(None),
                ResearchJob.lease_expires_at <= now,
                (ResearchJob.attempts >= (max_attempts if max_attempts is not None else ResearchJob.max_attempts)),
            )
        )
        exhausted = list(self._session.scalars(exhausted_stmt).all())
        for job in exhausted:
            job.status = ResearchJobStatus.FAILED.value
            job.completed_at = now
            job.lease_expires_at = None
            job.error_message = "Worker lease expired and max retry attempts exhausted."

        # 2. Retryable jobs -> queued
        retryable_stmt = (
            select(ResearchJob)
            .where(
                ResearchJob.status == ResearchJobStatus.PROCESSING.value,
                ResearchJob.lease_expires_at.is_not(None),
                ResearchJob.lease_expires_at <= now,
                (ResearchJob.attempts < (max_attempts if max_attempts is not None else ResearchJob.max_attempts)),
            )
        )
        retryable = list(self._session.scalars(retryable_stmt).all())
        for job in retryable:
            job.status = ResearchJobStatus.QUEUED.value
            job.claimed_by = None
            job.lease_expires_at = None
            job.next_retry_at = now

        count = len(exhausted) + len(retryable)
        if count > 0:
            self._session.flush()
        return count

    def reset_job_for_retry(self, job_id: UUID) -> ResearchJob | None:
        """Manually reset a failed job back to queued."""
        job = self.get_by_id(job_id)
        if job is None:
            return None
        job.status = ResearchJobStatus.QUEUED.value
        job.attempts = 0
        job.claimed_by = None
        job.lease_expires_at = None
        job.next_retry_at = None
        job.completed_at = None
        job.error_message = None
        self._session.flush()
        return job

    def list_jobs(
        self,
        *,
        user_id: UUID | None = None,
        guest_session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ResearchJob]:
        """List jobs filtered by owner and optional status."""
        stmt = select(ResearchJob)
        if user_id is not None:
            stmt = stmt.where(ResearchJob.user_id == user_id)
        elif guest_session_id:
            stmt = stmt.where(ResearchJob.guest_session_id == guest_session_id)
        if status:
            stmt = stmt.where(ResearchJob.status == status)

        stmt = stmt.order_by(ResearchJob.created_at.desc()).offset(offset).limit(limit)
        return list(self._session.scalars(stmt).all())
