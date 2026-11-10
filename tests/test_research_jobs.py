"""Unit and integration tests for durable background research jobs.

Verifies:
1. ResearchJob model schema and constraints (queued, processing, completed, failed).
2. ResearchJobRepository creation, tenant boundary enforcement, and ownership checks.
3. Atomic job claiming (claim_pending_or_stale) with lease duration.
4. Heartbeat lease renewal (renew_lease).
5. Job completion (complete_job) and result persistence.
6. Bounded retries with exponential backoff and error sanitization (fail_job).
7. Expired lease sweeping (sweep_expired_leases).
8. API endpoints:
   - POST /api/v1/research/jobs returns HTTP 202 with job ID and poll URL.
   - GET /api/v1/research/jobs/{job_id} returns HTTP 200 with job status and result.
9. Strict tenant isolation across user and guest session boundaries.
10. Input validation (empty query, out-of-range temperature).
"""

from __future__ import annotations

import datetime
from datetime import timezone
from uuid import UUID, uuid4
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from nexora.api.app import create_app
from nexora.auth.guest import GUEST_USER_ID, DEFAULT_GUEST_TOKEN
from nexora.auth.service import AuthenticationService
from nexora.database.models import (
    ApiKey,
    Base,
    Collection,
    Paper,
    ResearchJob,
    ResearchJobStatus,
    User,
    Workspace,
)
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.database.repository.research_job import ResearchJobRepository


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
def seed_data(session_factory):
    with session_factory() as session:
        user_a = User(email="user_a@nexora.ai")
        user_b = User(email="user_b@nexora.ai")
        session.add_all([user_a, user_b])
        session.flush()

        ws_a = Workspace(user_id=user_a.id, name="Workspace A")
        ws_b = Workspace(user_id=user_b.id, name="Workspace B")
        session.add_all([ws_a, ws_b])
        session.flush()

        col_a = Collection(workspace_id=ws_a.id, name="Collection A")
        col_b = Collection(workspace_id=ws_b.id, name="Collection B")
        session.add_all([col_a, col_b])
        session.flush()

        paper_a = Paper(collection_id=col_a.id, title="Attention Mechanisms in Transformers")
        paper_b = Paper(collection_id=col_b.id, title="Low-Rank Adaptation Analysis")
        session.add_all([paper_a, paper_b])
        session.flush()

        key_repo = ApiKeyRepository(session)
        _, token_a = key_repo.generate_key(user_id=user_a.id, name="Key A")
        _, token_b = key_repo.generate_key(user_id=user_b.id, name="Key B")

        session.commit()

        return {
            "user_a_id": user_a.id,
            "user_b_id": user_b.id,
            "token_a": token_a,
            "token_b": token_b,
            "col_a_id": col_a.id,
            "col_b_id": col_b.id,
            "paper_a_id": paper_a.id,
            "paper_b_id": paper_b.id,
        }


# =========================================================================
# 1. Repository & Model Tests
# =========================================================================

def test_research_job_create_and_query(session_factory, seed_data):
    """Verify research job creation with valid ownership and defaults."""
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        job = repo.create_job(
            query="What are the efficiency tradeoffs of LoRA?",
            user_id=seed_data["user_a_id"],
            collection_id=seed_data["col_a_id"],
            paper_id=seed_data["paper_a_id"],
            retrieval_mode="hybrid",
            top_k=5,
        )
        session.commit()
        job_id = job.id

    with session_factory() as session:
        repo = ResearchJobRepository(session)
        fetched = repo.get_by_id(job_id)
        assert fetched is not None
        assert fetched.query == "What are the efficiency tradeoffs of LoRA?"
        assert fetched.status == ResearchJobStatus.QUEUED.value
        assert fetched.attempts == 0
        assert fetched.max_attempts == 3
        assert fetched.retrieval_mode == "hybrid"
        assert fetched.result is None


def test_research_job_rejects_empty_query_or_no_owner(session_factory, seed_data):
    """Verify validation triggers before writing to DB."""
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            repo.create_job(query="", user_id=seed_data["user_a_id"])

        with pytest.raises(ValueError, match="Either user_id or guest_session_id must be provided"):
            repo.create_job(query="Valid query")


def test_research_job_claim_and_lease(session_factory, seed_data):
    """Verify atomic claiming sets processing status, worker token, and lease expiry."""
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        job = repo.create_job(
            query="Compare LoRA and prefix tuning",
            user_id=seed_data["user_a_id"],
        )
        session.commit()
        job_id = job.id

    worker_token = "worker-test-1"
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        claimed = repo.claim_pending_or_stale(
            limit=1,
            claimed_by=worker_token,
            lease_duration_seconds=120.0,
        )
        session.commit()

        assert len(claimed) == 1
        assert claimed[0].id == job_id
        assert claimed[0].status == ResearchJobStatus.PROCESSING.value
        assert claimed[0].claimed_by == worker_token
        assert claimed[0].attempts == 1
        assert claimed[0].lease_expires_at is not None


def test_research_job_lease_renewal_and_completion(session_factory, seed_data):
    """Verify lease renewal extends expiry and completion stores result."""
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        job = repo.create_job(
            query="Analyze transformer scaling laws",
            user_id=seed_data["user_a_id"],
        )
        session.commit()
        job_id = job.id

    worker_token = "worker-test-2"
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        claimed = repo.claim_pending_or_stale(limit=1, claimed_by=worker_token, lease_duration_seconds=60.0)
        session.commit()
        original_expiry = claimed[0].lease_expires_at

    # Renew lease
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        renewed = repo.renew_lease(job_id, claimed_by=worker_token, lease_duration_seconds=300.0)
        session.commit()
        assert renewed is True

        job_renewed = repo.get_by_id(job_id)
        exp1 = job_renewed.lease_expires_at
        exp2 = original_expiry
        if exp1.tzinfo is None and exp2.tzinfo is not None:
            exp1 = exp1.replace(tzinfo=timezone.utc)
        elif exp1.tzinfo is not None and exp2.tzinfo is None:
            exp2 = exp2.replace(tzinfo=timezone.utc)
        assert exp1 > exp2

    # Complete job
    fake_result = {
        "query": "Analyze transformer scaling laws",
        "answer": "Compute scales with model parameters quadratically [E1].",
        "sources": [],
        "cited_sources": [],
        "is_grounded": True,
        "is_mock": False,
        "model": "gemini-2.5-flash",
        "status": "success",
        "retrieval_mode": "hybrid",
        "evidence_count": 1,
    }
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        completed = repo.complete_job(job_id, claimed_by=worker_token, result=fake_result)
        session.commit()
        assert completed is True

        job_done = repo.get_by_id(job_id)
        assert job_done.status == ResearchJobStatus.COMPLETED.value
        assert job_done.result == fake_result
        assert job_done.lease_expires_at is None
        assert job_done.completed_at is not None


def test_research_job_bounded_retries_and_secret_sanitization(session_factory, seed_data):
    """Verify failed job retries with backoff and scrubs sensitive API keys from errors."""
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        job = repo.create_job(
            query="Query with secret leak simulation",
            user_id=seed_data["user_a_id"],
            max_attempts=2,
        )
        session.commit()
        job_id = job.id

    worker_token = "worker-test-3"
    secret_error = "Gemini API rejected key AIzaSyFakeSecretKey998877665544332211: quota exceeded"

    # Attempt 1: retryable
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        repo.claim_pending_or_stale(limit=1, claimed_by=worker_token)
        repo.fail_job(job_id, claimed_by=worker_token, error_message=secret_error, retryable=True, backoff_seconds=0.0)
        session.commit()

        j1 = repo.get_by_id(job_id)
        assert j1.status == ResearchJobStatus.QUEUED.value
        assert j1.attempts == 1
        assert "AIzaSyFakeSecretKey" not in j1.error_message
        assert "[REDACTED]" in j1.error_message

    # Attempt 2: exhausted max_attempts -> failed
    with session_factory() as session:
        repo = ResearchJobRepository(session)
        repo.claim_pending_or_stale(limit=1, claimed_by=worker_token)
        repo.fail_job(job_id, claimed_by=worker_token, error_message="Final attempt failed", retryable=True)
        session.commit()

        j2 = repo.get_by_id(job_id)
        assert j2.status == ResearchJobStatus.FAILED.value
        assert j2.attempts == 2


def test_research_job_sweep_expired_leases(session_factory, seed_data):
    """Verify expired leases reset to queued or transition to failed."""
    past_time = datetime.datetime.now(timezone.utc) - datetime.timedelta(minutes=10)
    with session_factory() as session:
        j_stale_retryable = ResearchJob(
            query="Stale job 1",
            user_id=seed_data["user_a_id"],
            status=ResearchJobStatus.PROCESSING.value,
            attempts=1,
            max_attempts=3,
            lease_expires_at=past_time,
            claimed_by="dead-worker-1",
        )
        j_stale_exhausted = ResearchJob(
            query="Stale job 2",
            user_id=seed_data["user_a_id"],
            status=ResearchJobStatus.PROCESSING.value,
            attempts=3,
            max_attempts=3,
            lease_expires_at=past_time,
            claimed_by="dead-worker-2",
        )
        session.add_all([j_stale_retryable, j_stale_exhausted])
        session.commit()
        id_retryable = j_stale_retryable.id
        id_exhausted = j_stale_exhausted.id

    with session_factory() as session:
        repo = ResearchJobRepository(session)
        swept = repo.sweep_expired_leases()
        session.commit()
        assert swept == 2

        r1 = repo.get_by_id(id_retryable)
        assert r1.status == ResearchJobStatus.QUEUED.value
        assert r1.claimed_by is None

        r2 = repo.get_by_id(id_exhausted)
        assert r2.status == ResearchJobStatus.FAILED.value
        assert "lease expired" in r2.error_message


# =========================================================================
# 2. REST API Endpoints & Tenant Isolation Tests
# =========================================================================

class DummyResearchApp:
    def ask(self, req):
        return None


def test_api_submit_and_poll_research_job(session_factory, seed_data):
    """Verify POST /api/v1/research/jobs returns 202 and GET returns 200."""
    auth_service = AuthenticationService(session_factory=session_factory)
    app = create_app(
        DummyResearchApp(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {seed_data['token_a']}"}

    # 1. Submit job
    payload = {
        "query": "What are the core findings on multimodal learning?",
        "retrieval_mode": "hybrid",
        "top_k": 5,
    }
    submit_res = client.post("/api/v1/research/jobs", json=payload, headers=headers)
    assert submit_res.status_code == 202
    submit_data = submit_res.json()
    assert "job_id" in submit_data
    assert submit_data["status"] == "queued"
    assert submit_data["poll_url"] == f"/api/v1/research/jobs/{submit_data['job_id']}"
    job_id = submit_data["job_id"]

    # 2. Poll job status
    poll_res = client.get(f"/api/v1/research/jobs/{job_id}", headers=headers)
    assert poll_res.status_code == 200
    poll_data = poll_res.json()
    assert poll_data["job_id"] == job_id
    assert poll_data["status"] == "queued"
    assert poll_data["query"] == payload["query"]
    assert poll_data["result"] is None


def test_api_tenant_isolation_cross_user_forbidden(session_factory, seed_data):
    """Verify User B cannot view or poll User A's research job."""
    auth_service = AuthenticationService(session_factory=session_factory)
    app = create_app(
        DummyResearchApp(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
    )
    client = TestClient(app)
    headers_a = {"Authorization": f"Bearer {seed_data['token_a']}"}
    headers_b = {"Authorization": f"Bearer {seed_data['token_b']}"}

    # User A submits job
    payload = {"query": "Confidential research inquiry", "retrieval_mode": "vector", "top_k": 3}
    submit_res = client.post("/api/v1/research/jobs", json=payload, headers=headers_a)
    assert submit_res.status_code == 202
    job_id = submit_res.json()["job_id"]

    # User B attempts to access User A's job -> 403 Forbidden
    res_b = client.get(f"/api/v1/research/jobs/{job_id}", headers=headers_b)
    assert res_b.status_code == 403


def test_api_guest_access_preview_session(session_factory):
    """Verify preview guest session can submit and poll research jobs without sign-up."""
    auth_service = AuthenticationService(session_factory=session_factory)
    app = create_app(
        DummyResearchApp(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
        environ={"VERCEL_ENV": "preview"},
    )
    client = TestClient(app)
    guest_headers = {"Authorization": f"Bearer {DEFAULT_GUEST_TOKEN}"}

    payload = {"query": "Guest research question", "retrieval_mode": "hybrid", "top_k": 3}
    submit_res = client.post("/api/v1/research/jobs", json=payload, headers=guest_headers)
    assert submit_res.status_code == 202
    job_id = submit_res.json()["job_id"]

    poll_res = client.get(f"/api/v1/research/jobs/{job_id}", headers=guest_headers)
    assert poll_res.status_code == 200
    assert poll_res.json()["job_id"] == job_id


def test_api_submit_validation_errors(session_factory, seed_data):
    """Verify input parameter validation on job submission."""
    auth_service = AuthenticationService(session_factory=session_factory)
    app = create_app(
        DummyResearchApp(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {seed_data['token_a']}"}

    # Empty query
    res_empty = client.post("/api/v1/research/jobs", json={"query": "   "}, headers=headers)
    assert res_empty.status_code == 400

    # Invalid retrieval mode
    res_mode = client.post("/api/v1/research/jobs", json={"query": "valid", "retrieval_mode": "invalid"}, headers=headers)
    assert res_mode.status_code == 400

    # Non-existent job
    fake_uuid = str(uuid4())
    res_404 = client.get(f"/api/v1/research/jobs/{fake_uuid}", headers=headers)
    assert res_404.status_code == 404


def test_api_submit_awaits_vercel_queue_send_coroutine_without_warning(
    session_factory,
    seed_data,
    monkeypatch,
):
    """Verify POST /api/v1/research/jobs awaits async vercel.queue.send with zero RuntimeWarning."""
    import warnings

    sent_calls: list[dict] = []

    async def fake_async_send(topic, payload, *, idempotency_key=None, **kwargs):
        sent_calls.append({
            "topic": topic,
            "payload": payload,
            "idempotency_key": idempotency_key,
            "awaited": True,
        })
        return "msg_awaited_001"

    monkeypatch.setattr("vercel.queue.send", fake_async_send)

    auth_service = AuthenticationService(session_factory=session_factory)
    app = create_app(
        DummyResearchApp(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
        environ={"VERCEL": "1", "VERCEL_ENV": "preview"},
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {seed_data['token_a']}"}

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = client.post(
            "/api/v1/research/jobs",
            json={"query": "Does flash attention await coroutines?"},
            headers=headers,
        )

    assert res.status_code == 202
    job_id = res.json()["job_id"]
    assert len(sent_calls) == 1
    assert sent_calls[0]["awaited"] is True
    assert sent_calls[0]["topic"] == "research-query-jobs"
    assert sent_calls[0]["payload"] == {"job_id": job_id}
    assert sent_calls[0]["idempotency_key"] == job_id


def test_api_queue_dispatch_failure_marks_job_failed_instead_of_queued(
    session_factory,
    seed_data,
    monkeypatch,
):
    """Queue send failure must mark job failed in DB with scrubbed error and fail fast."""
    async def failing_send(topic, payload, *, idempotency_key=None, **kwargs):
        raise RuntimeError("Queue service unavailable with secret AIzaSyFakeDispatchKey123456")

    monkeypatch.setattr("vercel.queue.send", failing_send)

    auth_service = AuthenticationService(session_factory=session_factory)
    app = create_app(
        DummyResearchApp(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
        environ={"VERCEL": "1", "VERCEL_ENV": "preview"},
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {seed_data['token_a']}"}

    res = client.post(
        "/api/v1/research/jobs",
        json={"query": "Trigger dispatch failure"},
        headers=headers,
    )
    assert res.status_code == 500

    # Verify the DB job record did NOT stay stuck in queued state
    with session_factory() as session:
        jobs = ResearchJobRepository(session).list_jobs(user_id=seed_data["user_a_id"])
        assert len(jobs) == 1
        failed_job = jobs[0]
        assert failed_job.status == ResearchJobStatus.FAILED.value
        assert failed_job.completed_at is not None
        assert "Queue dispatch failed" in failed_job.error_message or "Failed to dispatch" in failed_job.error_message
        assert "AIzaSyFakeDispatchKey" not in failed_job.error_message
        assert "[REDACTED]" in failed_job.error_message


def test_api_guest_session_isolation_across_sessions(session_factory, seed_data):
    """Verify distinct guest sessions cannot read each other's jobs and prod rejects unauthenticated polling."""
    auth_service = AuthenticationService(session_factory=session_factory)
    preview_app = create_app(
        DummyResearchApp(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
        environ={"VERCEL_ENV": "preview"},
    )
    preview_client = TestClient(preview_app)

    guest1_headers = {
        "Authorization": f"Bearer {DEFAULT_GUEST_TOKEN}",
        "X-Guest-Session-ID": "guest-alpha",
    }
    guest2_headers = {
        "Authorization": f"Bearer {DEFAULT_GUEST_TOKEN}",
        "X-Guest-Session-ID": "guest-beta",
    }

    res = preview_client.post(
        "/api/v1/research/jobs",
        json={"query": "Guest alpha private inquiry"},
        headers=guest1_headers,
    )
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    # Same guest session can poll
    assert preview_client.get(f"/api/v1/research/jobs/{job_id}", headers=guest1_headers).status_code == 200
    # Different guest session is forbidden
    assert preview_client.get(f"/api/v1/research/jobs/{job_id}", headers=guest2_headers).status_code == 403
    # Guest omitting session header defaults to guest_preview != guest-alpha -> forbidden
    assert preview_client.get(
        f"/api/v1/research/jobs/{job_id}",
        headers={"Authorization": f"Bearer {DEFAULT_GUEST_TOKEN}"},
    ).status_code == 403

    # Guest cannot access a regular user's collection
    assert preview_client.post(
        "/api/v1/research/jobs",
        json={"query": "Cross tenant probe", "collection_id": str(seed_data["col_a_id"])},
        headers=guest1_headers,
    ).status_code == 403

    # Production environment rejects unauthenticated polling
    prod_app = create_app(
        DummyResearchApp(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
        environ={"VERCEL_ENV": "production"},
    )
    prod_client = TestClient(prod_app)
    assert prod_client.get(f"/api/v1/research/jobs/{job_id}").status_code == 401


def test_end_to_end_research_job_lifecycle_with_queue_consumer_and_gemini_failure(
    session_factory,
    seed_data,
    monkeypatch,
):
    """Full lifecycle: submit -> publish queue msg -> consumer runs worker -> poll result & failure cases."""
    from nexora.rag.models import AttributedSource, RAGResponse
    from nexora.research.queue_consumer import handle_research_job, set_global_research_worker
    from nexora.research.worker import ResearchJobWorker

    published_messages: list[tuple[str, dict]] = []

    async def capture_send(topic, payload, *, idempotency_key=None, **kwargs):
        published_messages.append((topic, payload))
        return "msg_e2e_1"

    monkeypatch.setattr("vercel.queue.send", capture_send)

    auth_service = AuthenticationService(session_factory=session_factory)
    app = create_app(
        DummyResearchApp(),
        session_factory=session_factory,
        auth_service=auth_service,
        enforce_auth=True,
        environ={"VERCEL": "1", "VERCEL_ENV": "preview"},
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {seed_data['token_a']}"}

    # --- Case 1: Successful RAG synthesis lifecycle ---
    class SuccessRAGPipeline:
        def answer(self, request, **kwargs):
            src = AttributedSource(
                label="[E1]",
                document_id="doc-e2e",
                chunk_id="chk-e2e",
                paper_id=str(seed_data["paper_a_id"]),
                paper_title="Attention Paper",
                score=0.95,
                text="Self-attention computes dot-product weights.",
                text_excerpt="Self-attention computes dot-product weights.",
                metadata={},
            )
            return RAGResponse(
                query=request,
                answer="Transformers rely on self-attention [E1].",
                sources=[src],
                cited_sources=[src],
                is_grounded=True,
                is_mock=False,
                model="gemini-2.5-flash",
                status="success",
                retrieval_mode="hybrid",
                evidence_count=1,
            )

    submit_res = client.post(
        "/api/v1/research/jobs",
        json={"query": "How does self-attention work?"},
        headers=headers,
    )
    assert submit_res.status_code == 202
    job_id = submit_res.json()["job_id"]
    assert len(published_messages) == 1
    assert published_messages[0] == ("research-query-jobs", {"job_id": job_id})

    # Poll initial status: queued
    assert client.get(f"/api/v1/research/jobs/{job_id}", headers=headers).json()["status"] == "queued"

    # Worker consumes queued message via subscriber entrypoint
    worker = ResearchJobWorker(session_factory=session_factory, rag_pipeline=SuccessRAGPipeline())
    set_global_research_worker(worker)
    try:
        consumer_res = handle_research_job(published_messages[0][1])
        assert consumer_res == {"processed_jobs": 1}
    finally:
        set_global_research_worker(None)

    # Poll completed result
    poll_done = client.get(f"/api/v1/research/jobs/{job_id}", headers=headers)
    assert poll_done.status_code == 200
    done_data = poll_done.json()
    assert done_data["status"] == "completed"
    assert "self-attention [E1]" in done_data["result"]["answer"]
    assert done_data["result"]["model"] == "gemini-2.5-flash"

    # --- Case 2: Gemini permanent failure surfaces useful error without endless polling ---
    class FailingGeminiPipeline:
        def answer(self, request, **kwargs):
            raise RuntimeError("Gemini HTTP 401 unauthorized: invalid API key AIzaSyBadSecretKey999")

    published_messages.clear()
    submit_fail = client.post(
        "/api/v1/research/jobs",
        json={"query": "Trigger Gemini error"},
        headers=headers,
    )
    assert submit_fail.status_code == 202
    fail_job_id = submit_fail.json()["job_id"]
    assert len(published_messages) == 1

    fail_worker = ResearchJobWorker(session_factory=session_factory, rag_pipeline=FailingGeminiPipeline())
    set_global_research_worker(fail_worker)
    try:
        consumer_res2 = handle_research_job(published_messages[0][1])
        assert consumer_res2 == {"processed_jobs": 1}
    finally:
        set_global_research_worker(None)

    poll_fail = client.get(f"/api/v1/research/jobs/{fail_job_id}", headers=headers)
    assert poll_fail.status_code == 200
    fail_data = poll_fail.json()
    assert fail_data["status"] == "failed"
    assert fail_data["error"] is not None
    assert "Gemini HTTP 401" in fail_data["error"]
    assert "AIzaSyBadSecretKey999" not in fail_data["error"]
    assert "[REDACTED]" in fail_data["error"]
