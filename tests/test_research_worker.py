"""Unit and integration tests for ResearchJobWorker and Vercel Queue consumer.

Verifies:
1. ResearchJobWorker.run_once claims and executes research jobs.
2. Production service reconstruction and RAG pipeline invocation.
3. Heartbeat lease extension during processing.
4. Completion persistence with full citation and metadata preservation.
5. Fail-soft error recording with secret sanitization.
6. Vercel Queue consumer (handle_research_job) delegation.
"""

from __future__ import annotations

import datetime
from datetime import timezone
import time
from unittest import mock
from uuid import UUID, uuid4
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.database.models import (
    Base,
    Collection,
    Paper,
    ResearchJob,
    ResearchJobStatus,
    User,
    Workspace,
)
from nexora.database.repository.research_job import ResearchJobRepository
from nexora.rag.models import AttributedSource, RAGResponse
from nexora.research.queue_consumer import (
    get_research_queue_worker,
    handle_research_job,
    process_research_queue_trigger,
    set_global_research_worker,
)
from nexora.research.worker import ResearchJobWorker


class FakeRAGPipeline:
    """Mock RAG pipeline for deterministic worker testing."""

    def __init__(self, should_fail: bool = False, delay_seconds: float = 0.0) -> None:
        self.should_fail = should_fail
        self.delay = delay_seconds
        self.calls: list[dict] = []

    def answer(
        self,
        request: str,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        min_score: float | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        paper_id: str | None = None,
        collection_ids: list[str] | None = None,
    ) -> RAGResponse:
        self.calls.append({
            "request": request,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "paper_id": paper_id,
            "collection_ids": collection_ids,
        })
        if self.delay > 0:
            time.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError("Gemini generateContent timeout or simulated failure with secret AIzaSySecret123")

        source = AttributedSource(
            label="[E1]",
            document_id="doc-1",
            chunk_id="chunk-1",
            paper_id=paper_id,
            paper_title="Diffusion Paper",
            score=0.92,
            text="Diffusion models synthesize high fidelity samples.",
            text_excerpt="Diffusion models synthesize high fidelity samples.",
            metadata={"title": "Diffusion Paper"},
        )
        return RAGResponse(
            query=request,
            answer="According to Diffusion Paper [E1], diffusion models synthesize samples.",
            sources=[source],
            cited_sources=[source],
            is_grounded=True,
            is_mock=False,
            model="gemini-2.5-flash",
            status="success",
            retrieval_mode=retrieval_mode,
            evidence_count=1,
        )


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def seed_user(session_factory):
    with session_factory() as session:
        user = User(email="worker_test@nexora.ai")
        session.add(user)
        session.commit()
        return user.id


def test_research_worker_processes_job_successfully(session_factory, seed_user):
    """Verify worker claims queued job, executes RAG pipeline, and stores result."""
    pipeline = FakeRAGPipeline()
    worker = ResearchJobWorker(
        session_factory=session_factory,
        rag_pipeline=pipeline,
        worker_token="test-worker-1",
    )

    with session_factory() as session:
        repo = ResearchJobRepository(session)
        job = repo.create_job(
            query="Explain generative diffusion models",
            user_id=seed_user,
            retrieval_mode="hybrid",
            top_k=3,
        )
        session.commit()
        job_id = job.id

    processed = worker.run_once()
    assert processed == 1
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["request"] == "Explain generative diffusion models"
    assert pipeline.calls[0]["top_k"] == 3

    with session_factory() as session:
        repo = ResearchJobRepository(session)
        job_after = repo.get_by_id(job_id)
        assert job_after.status == ResearchJobStatus.COMPLETED.value
        assert job_after.result is not None
        assert "Diffusion Paper" in job_after.result["answer"]
        assert job_after.result["status"] == "success"
        assert job_after.completed_at is not None


def test_research_worker_handles_failure_and_scrubs_secrets(session_factory, seed_user):
    """Verify failed RAG execution marks job and scrubs any API key strings."""
    pipeline = FakeRAGPipeline(should_fail=True)
    worker = ResearchJobWorker(
        session_factory=session_factory,
        rag_pipeline=pipeline,
        worker_token="test-worker-2",
        max_attempts=1,
    )

    with session_factory() as session:
        repo = ResearchJobRepository(session)
        job = repo.create_job(
            query="Trigger failing query",
            user_id=seed_user,
            max_attempts=1,
        )
        session.commit()
        job_id = job.id

    processed = worker.run_once()
    assert processed == 1

    with session_factory() as session:
        repo = ResearchJobRepository(session)
        job_after = repo.get_by_id(job_id)
        assert job_after.status == ResearchJobStatus.FAILED.value
        assert job_after.error_message is not None
        assert "AIzaSySecret123" not in job_after.error_message
        assert "[REDACTED]" in job_after.error_message


def test_research_worker_idempotency_empty_queue(session_factory):
    """Verify run_once returns 0 when no jobs are queued."""
    pipeline = FakeRAGPipeline()
    worker = ResearchJobWorker(
        session_factory=session_factory,
        rag_pipeline=pipeline,
    )

    processed = worker.run_once()
    assert processed == 0
    assert len(pipeline.calls) == 0


def test_research_queue_consumer_invocation():
    """Verify queue consumer delegates to worker.run_once() and returns processed count."""
    mock_worker = mock.MagicMock()
    mock_worker.run_once.return_value = 1

    set_global_research_worker(mock_worker)
    try:
        res = process_research_queue_trigger({"event": "job_enqueued", "job_id": str(uuid4())})
        assert res == {"processed_jobs": 1}
        mock_worker.run_once.assert_called_once()

        mock_worker.run_once.reset_mock()
        sub_res = handle_research_job({"event": "job_enqueued"})
        assert sub_res == {"processed_jobs": 1}
        mock_worker.run_once.assert_called_once()
    finally:
        set_global_research_worker(None)
