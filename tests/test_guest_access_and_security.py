"""Automated tests for preview guest access, security restrictions, and protected endpoints."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from tests.test_api_auth_and_authorization import auth_test_db, FakeResearchService

from nexora.api.app import create_app
from nexora.auth.guest import (
    DEFAULT_GUEST_TOKEN,
    GUEST_COLLECTION_ID,
    GUEST_EMAIL,
    GUEST_USER_ID,
    GUEST_WORKSPACE_ID,
    ensure_guest_tenant,
    is_guest_access_allowed,
    is_guest_token,
)
from nexora.auth.service import AuthenticationService
from nexora.database.models.collection import Collection
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.paper import PaperRepository
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository

VALID_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


def test_is_guest_token():
    assert is_guest_token(DEFAULT_GUEST_TOKEN) is True
    assert is_guest_token("nx_guest_preview_abc123") is True
    assert is_guest_token("guest_preview") is True
    assert is_guest_token("guest_session") is True
    assert is_guest_token("nx_live_regular_api_key") is False
    assert is_guest_token("") is False
    assert is_guest_token(None) is False


def test_is_guest_access_allowed_production_guard():
    # Production is ALWAYS forbidden
    assert is_guest_access_allowed({"VERCEL_ENV": "production"}) is False
    assert (
        is_guest_access_allowed(
            {"VERCEL_ENV": "production", "NEXORA_GUEST_ACCESS_ENABLED": "true"}
        )
        is False
    )

    # Preview is allowed
    assert is_guest_access_allowed({"VERCEL_ENV": "preview"}) is True
    assert (
        is_guest_access_allowed(
            {"VERCEL_ENV": "preview", "NEXORA_GUEST_ACCESS_ENABLED": "true"}
        )
        is True
    )

    # Explicit flag enabled (non-production)
    assert is_guest_access_allowed({"NEXORA_GUEST_ACCESS_ENABLED": "true"}) is True
    assert is_guest_access_allowed({"NEXORA_GUEST_ACCESS_ENABLED": "1"}) is True

    # Default / unconfigured
    assert is_guest_access_allowed({}) is False
    assert is_guest_access_allowed({"VERCEL_ENV": "development"}) is False


def test_guest_session_endpoint_preview_and_production(auth_test_db):
    session_factory = auth_test_db["session_factory"]

    # 1. Preview app -> returns 200 with guest session details
    preview_app = create_app(
        FakeResearchService(),
        session_factory=session_factory,
        enforce_auth=True,
        environ={"VERCEL_ENV": "preview"},
    )
    client = TestClient(preview_app)
    resp = client.post("/api/v1/auth/guest-session")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["token"] == DEFAULT_GUEST_TOKEN
    assert data["user_id"] == str(GUEST_USER_ID)
    assert data["workspace_id"] == str(GUEST_WORKSPACE_ID)
    assert data["collection_id"] == str(GUEST_COLLECTION_ID)
    assert data["email"] == GUEST_EMAIL
    assert data["guest_access"] is True

    # 2. Production app -> returns 403 Forbidden
    prod_app = create_app(
        FakeResearchService(),
        session_factory=session_factory,
        enforce_auth=True,
        environ={"VERCEL_ENV": "production"},
    )
    prod_client = TestClient(prod_app)
    prod_resp = prod_client.post("/api/v1/auth/guest-session")
    assert prod_resp.status_code == 403
    assert prod_resp.json()["error"]["code"] == "forbidden"


def test_production_security_guards(auth_test_db):
    """Verify that in production, guest tokens and missing credentials are strictly rejected."""
    session_factory = auth_test_db["session_factory"]
    token_a = auth_test_db["token_a"]

    prod_app = create_app(
        FakeResearchService(),
        session_factory=session_factory,
        enforce_auth=True,
        environ={"VERCEL_ENV": "production"},
    )
    client = TestClient(prod_app)

    # 1. Missing credentials in production -> 401 Unauthorized
    resp_no_auth = client.post(
        "/api/v1/documents/ingest",
        data={"paper_id": str(uuid4())},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert resp_no_auth.status_code == 401
    assert resp_no_auth.json()["error"]["code"] == "unauthorized"
    assert "not provided" in resp_no_auth.json()["error"]["message"]

    # 2. Guest token presented in production -> 401 Unauthorized
    resp_guest = client.post(
        "/api/v1/documents/ingest",
        headers={"Authorization": f"Bearer {DEFAULT_GUEST_TOKEN}"},
        data={"paper_id": str(uuid4())},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert resp_guest.status_code == 401
    assert "Guest access is not permitted in this environment" in resp_guest.json()["error"]["message"]

    # 3. Invalid API key presented in production -> 401 Unauthorized
    resp_invalid = client.post(
        "/api/v1/documents/ingest",
        headers={"Authorization": "Bearer nx_live_bogus_token_12345"},
        data={"paper_id": str(uuid4())},
        files={"file": ("paper.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert resp_invalid.status_code == 401
    assert "Invalid or expired API credential" in resp_invalid.json()["error"]["message"]


def test_preview_rejects_invalid_tokens_but_accepts_guest(auth_test_db):
    """Verify that in preview, invalid credentials still fail with 401, while guest tokens succeed."""
    session_factory = auth_test_db["session_factory"]

    preview_app = create_app(
        FakeResearchService(),
        session_factory=session_factory,
        enforce_auth=True,
        environ={"VERCEL_ENV": "preview"},
    )
    client = TestClient(preview_app)

    # 1. Invalid token -> 401 Unauthorized
    resp_invalid = client.get(
        f"/api/v1/documents/{uuid4()}",
        headers={"Authorization": "Bearer nx_live_definitely_invalid_key"},
    )
    assert resp_invalid.status_code == 401
    assert "Invalid or expired API credential" in resp_invalid.json()["error"]["message"]

    # 2. Malformed scheme -> 401 Unauthorized
    resp_malformed = client.get(
        f"/api/v1/documents/{uuid4()}",
        headers={"Authorization": "Basic something"},
    )
    assert resp_malformed.status_code == 401
    assert "Invalid authorization scheme" in resp_malformed.json()["error"]["message"]

    # 3. Valid guest token -> 404 (because doc doesn't exist, not 401 unauthorized!)
    resp_guest = client.get(
        f"/api/v1/documents/{uuid4()}",
        headers={"Authorization": f"Bearer {DEFAULT_GUEST_TOKEN}"},
    )
    assert resp_guest.status_code == 404
    assert "not found" in resp_guest.json()["error"]["message"].lower()


def test_preview_pdf_upload_auto_creates_paper_for_guest(auth_test_db):
    """Verify PDF upload in preview mode auto-creates the paper and ingests it."""
    session_factory = auth_test_db["session_factory"]
    paper_uuid = uuid4()

    mock_ingestion = MagicMock()
    fake_result = MagicMock()
    fake_result.document_id = uuid4()
    fake_result.paper_id = paper_uuid
    fake_result.chunks_count = 5
    fake_result.indexed_chunks_count = 5
    mock_ingestion.process_and_ingest.return_value = fake_result

    preview_app = create_app(
        FakeResearchService(),
        session_factory=session_factory,
        ingestion_service=mock_ingestion,
        enforce_auth=True,
        environ={"VERCEL_ENV": "preview"},
    )
    client = TestClient(preview_app)

    # Upload with guest bearer token
    resp = client.post(
        "/api/v1/documents/ingest",
        headers={"Authorization": f"Bearer {DEFAULT_GUEST_TOKEN}"},
        data={"paper_id": str(paper_uuid)},
        files={"file": ("my_experiment_results.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["paper_id"] == str(paper_uuid)
    assert data["chunks_count"] == 5

    # Verify that the paper was created in PostgreSQL under the guest collection
    with session_factory() as session:
        paper_repo = PaperRepository(session)
        created_paper = paper_repo.get_by_id(paper_uuid)
        assert created_paper is not None
        assert created_paper.collection_id == GUEST_COLLECTION_ID
        assert created_paper.title == "My Experiment Results"


def test_guest_tenant_isolation_boundaries(auth_test_db):
    """Verify guest user cannot access private papers owned by registered User A, and vice-versa."""
    session_factory = auth_test_db["session_factory"]
    token_a = auth_test_db["token_a"]
    paper_a_id = auth_test_db["paper_a1_id"]

    # 1. Create a paper owned by the guest user
    guest_paper_id = uuid4()
    with session_factory() as session:
        ensure_guest_tenant(session)
        paper_repo = PaperRepository(session)
        paper_repo.create(
            collection_id=GUEST_COLLECTION_ID,
            title="Guest Only Paper",
        )
        paper = paper_repo.get_by_id(paper_repo._session.scalars(
            __import__("sqlalchemy").select(Paper).where(Paper.collection_id == GUEST_COLLECTION_ID)
        ).first().id)
        guest_paper_id = paper.id

    preview_app = create_app(
        FakeResearchService(),
        session_factory=session_factory,
        enforce_auth=True,
        environ={"VERCEL_ENV": "preview"},
    )
    client = TestClient(preview_app)

    # 2. Guest user attempts to access User A's private paper -> 403 Forbidden
    auth_service = AuthenticationService(session_factory)
    with pytest.raises(Exception) as excinfo:
        auth_service.verify_paper_ownership(paper_a_id, GUEST_USER_ID)
    assert "forbidden" in str(excinfo.value).lower()

    # 3. User A attempts to access Guest user's paper -> 403 Forbidden
    with pytest.raises(Exception) as excinfo2:
        auth_service.verify_paper_ownership(guest_paper_id, auth_test_db["user_a_id"])
    assert "forbidden" in str(excinfo2.value).lower()

    # 4. Guest user accesses their own paper -> succeeds
    paper, col, ws = auth_service.verify_paper_ownership(guest_paper_id, GUEST_USER_ID)
    assert paper.id == guest_paper_id
    assert col.id == GUEST_COLLECTION_ID
    assert ws.user_id == GUEST_USER_ID


def test_guest_research_features(auth_test_db):
    """Verify research endpoints (query, compare, analyze, gaps) work with guest credentials."""
    from tests.test_api_auth_and_authorization import (
        FakeAnalysisService,
        FakeComparisonService,
        FakeGapService,
        FakeRAGPipeline,
    )
    from nexora.research.comparison_table import BENCHMARK_PAPERS

    session_factory = auth_test_db["session_factory"]

    # Create a paper owned by guest
    guest_paper_uuid = uuid4()
    with session_factory() as session:
        ensure_guest_tenant(session)
        paper_repo = PaperRepository(session)
        paper_repo.create(
            collection_id=GUEST_COLLECTION_ID,
            title="Guest Ingested Study",
        )
        p = paper_repo._session.scalars(
            __import__("sqlalchemy").select(Paper).where(Paper.collection_id == GUEST_COLLECTION_ID)
        ).first()
        guest_paper_uuid = p.id

    preview_app = create_app(
        FakeResearchService(),
        session_factory=session_factory,
        rag_pipeline=FakeRAGPipeline(),
        analysis_service=FakeAnalysisService(),
        comparison_service=FakeComparisonService(),
        gap_service=FakeGapService(),
        enforce_auth=True,
        environ={"VERCEL_ENV": "preview"},
    )
    client = TestClient(preview_app)
    headers = {"Authorization": f"Bearer {DEFAULT_GUEST_TOKEN}"}

    # 1. Benchmark paper comparison succeeds with guest token
    p1 = BENCHMARK_PAPERS["arxiv_1706.03762"]["uuid"]
    p2 = BENCHMARK_PAPERS["arxiv_2106.09685"]["uuid"]
    resp_cmp = client.post(
        "/api/v1/research/compare",
        headers=headers,
        json={"paper_ids": [p1, p2]},
    )
    assert resp_cmp.status_code == 200
    assert "matrix" in resp_cmp.json()

    # 2. Paper analysis on benchmark paper succeeds with guest token
    resp_ana_bm = client.post(
        "/api/v1/papers/analyze",
        headers=headers,
        json={"paper_id": p1},
    )
    assert resp_ana_bm.status_code == 200

    # 3. Paper analysis on guest's own paper succeeds
    resp_ana_guest = client.post(
        "/api/v1/papers/analyze",
        headers=headers,
        json={"paper_id": str(guest_paper_uuid)},
    )
    assert resp_ana_guest.status_code == 200

    # 4. Research gaps on benchmark papers succeeds
    resp_gaps = client.post(
        "/api/v1/research/gaps",
        headers=headers,
        json={"paper_ids": [p1, p2]},
    )
    assert resp_gaps.status_code == 200

    # 5. RAG query with guest token succeeds
    resp_rag = client.post(
        "/api/v1/research/query",
        headers=headers,
        json={"query": "What are transformer attention layers?"},
    )
    assert resp_rag.status_code == 200
    assert "answer" in resp_rag.json()


def test_diagnostics_guest_access_status(auth_test_db):
    """Verify diagnostics reports guest access status appropriately."""
    session_factory = auth_test_db["session_factory"]

    # Preview environment
    preview_app = create_app(
        FakeResearchService(),
        session_factory=session_factory,
        enforce_auth=True,
        environ={"VERCEL_ENV": "preview"},
    )
    client = TestClient(preview_app)

    ready_resp = client.get("/ready")
    assert ready_resp.json()["checks"]["auth"]["guest_access"] is True

    diag_resp = client.get("/api/diagnostics/environment")
    assert diag_resp.status_code == 200
    assert diag_resp.json()["active_configuration"]["guest_access_enabled"] is True

    # Production environment
    prod_app = create_app(
        FakeResearchService(),
        session_factory=session_factory,
        enforce_auth=True,
        environ={"VERCEL_ENV": "production"},
    )
    prod_client = TestClient(prod_app)

    prod_ready = prod_client.get("/ready")
    assert prod_ready.json()["checks"]["auth"]["guest_access"] is False

    prod_diag = prod_client.get("/api/diagnostics/environment")
    assert prod_diag.status_code == 200
    assert prod_diag.json()["active_configuration"]["guest_access_enabled"] is False

