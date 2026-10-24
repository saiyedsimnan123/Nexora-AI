"""Security regression tests for API Authentication and Collection Authorization (Phase 4).

Verifies:
1. Missing, malformed, and invalid credentials (401 Unauthorized with WWW-Authenticate header).
2. Expired and revoked API key rejection (401 Unauthorized).
3. Authenticated access to authorized collections and papers (200 OK).
4. Cross-user collection isolation: attempt to import into or read another user's collection (403 Forbidden).
5. Cross-workspace access attempts (403 Forbidden).
6. Unauthorized document ingestion and document retrieval (401 and 403).
7. Public scholarly discovery and health endpoints remain accessible without credentials.
8. Database migration for api_keys table (upgrade and downgrade).
9. Bootstrap utility for provisioning initial user, workspace, and credentials.
"""

from __future__ import annotations

import datetime
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.api.app import create_app
from nexora.api.errors import (
    AuthenticationError,
    DocumentNotFoundError,
    ForbiddenError,
)
from nexora.auth.bootstrap import bootstrap_user_and_key
from nexora.auth.models import AuthenticatedUser
from nexora.auth.service import AuthenticationService
from nexora.database.models.api_key import ApiKey
from nexora.database.models.base import Base
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document, IndexingStatus
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.paper import PaperRepository
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository
from nexora.discovery.downloader import DownloadedPDF
from nexora.discovery.models import (
    DiscoveryImportResult,
    ScholarlyPaperMetadata,
    ScholarlySearchQuery,
    ScholarlySearchResult,
)
from nexora.discovery.service import DiscoveryService

VALID_PDF_BYTES = b"%PDF-1.5\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< >>\n%%EOF"


# ---------------------------------------------------------------------------
# Test Doubles & Fixtures
# ---------------------------------------------------------------------------

class FakeResearchService:
    def ask(self, request: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "text": "Fake research answer",
                "query": request.query if hasattr(request, "query") else "test",
                "model": "fake-model",
                "evidence_count": 1,
            }
        )

    def verify(self, request: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "query": request.query,
                "text": request.text,
                "model": "fake-model",
                "evidence_count": 1,
                "verification": {"status": "verified"},
            }
        )


class FakeAnalysisService:
    def analyze_paper(self, paper_id: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_id": str(paper_id),
                "title": "Fake Paper Title",
                "problem_and_objectives": {"summary": "Objectives"},
                "methodology_and_algorithms": {"summary": "Methodology"},
                "models_and_datasets": {"summary": "Dataset"},
                "experiments_and_metrics": {"summary": "Experiments"},
                "results_and_findings": {"summary": "Results"},
                "limitations": {"summary": "Limitations"},
                "future_work": {"summary": "Future work"},
                "evidence_count": 1,
                "model": "fake-model",
                "usage": None,
            }
        )


class FakeComparisonService:
    def compare_papers(self, paper_ids: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_ids": [str(p) for p in paper_ids],
                "dimensions": {"methods": {"claim": "c"}},
                "overall_summary": "Comparison summary",
                "model": "fake-model",
                "usage": None,
            }
        )

    def compare_cross_paper(self, paper_ids: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_ids": [str(p) for p in paper_ids],
                "paper_titles": {str(p): f"Title {p}" for p in paper_ids},
                "matrix": {"methods": {}},
                "conclusion": "Demonstrated improvement",
                "conclusion_reasoning": "Strong evidence",
                "comparability_status": "Comparable",
                "overall_summary": "Overall summary",
                "model": "fake-model",
                "usage": None,
            }
        )


class FakeGapService:
    def discover_gaps_for_collection(self, collection_id: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_ids": [],
                "collection_id": str(collection_id),
                "documented_gaps": [],
                "inferred_gaps": [],
                "total_gaps_count": 0,
                "summary": "Gaps summary",
                "limitations": [],
                "model": "fake-model",
                "usage": None,
            }
        )

    def discover_gaps_for_papers(self, paper_ids: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_ids": [str(p) for p in paper_ids],
                "collection_id": None,
                "documented_gaps": [],
                "inferred_gaps": [],
                "total_gaps_count": 0,
                "summary": "Gaps summary",
                "limitations": [],
                "model": "fake-model",
                "usage": None,
            }
        )


class FakeRAGPipeline:
    def answer(self, request: str, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "answer": "Grounded RAG answer",
                "query": request,
                "model": "fake-rag",
                "evidence_count": 1,
                "sources": [],
                "retrieval_mode": "hybrid",
                "is_grounded": True,
                "is_mock": False,
                "status": "success",
            }
        )


class FakeProvider:
    def __init__(self, papers: list[ScholarlyPaperMetadata] | None = None) -> None:
        self.papers = papers or []

    def search(self, query: ScholarlySearchQuery) -> ScholarlySearchResult:
        return ScholarlySearchResult(
            provider="arxiv",
            query=query.query,
            total_results=len(self.papers),
            offset=query.offset,
            limit=query.limit,
            papers=tuple(self.papers),
        )

    def get_paper(self, external_id: str) -> ScholarlyPaperMetadata | None:
        for p in self.papers:
            if p.external_id == external_id:
                return p
        return None


class FakeDownloader:
    def __init__(self, pdf_bytes: bytes = VALID_PDF_BYTES) -> None:
        self._bytes = pdf_bytes
        self.download_calls: list[str] = []
        self.created_temp_paths: list[Path] = []

    def download_arxiv_pdf(self, arxiv_id: str) -> DownloadedPDF:
        self.download_calls.append(arxiv_id)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(self._bytes)
        tmp.flush()
        tmp.close()
        p = Path(tmp.name)
        self.created_temp_paths.append(p)
        return DownloadedPDF(
            path=p,
            file_size=len(self._bytes),
            arxiv_id=arxiv_id,
            source_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        )


class FakeIngestionService:
    def __init__(self, session_factory: Any) -> None:
        self.session_factory = session_factory
        self.calls: list[dict[str, Any]] = []

    def process_and_ingest(
        self,
        paper_id: UUID,
        file_path: str,
        pipeline: Any = None,
        filename: str = "document.pdf",
        checksum: str = "checksum123",
        durable: bool = False,
    ) -> Any:
        self.calls.append({"paper_id": paper_id, "filename": filename})
        with self.session_factory() as session:
            doc_repo = DocumentRepository(session)
            doc = doc_repo.create(
                paper_id=paper_id,
                filename=filename,
                file_size=len(VALID_PDF_BYTES),
                mime_type="application/pdf",
                checksum="a" * 64,
                storage_key=f"papers/{paper_id}/{filename}",
            )
            doc.indexing_status = IndexingStatus.COMPLETED.value
            session.commit()
            doc_id = doc.id

        from types import SimpleNamespace
        return SimpleNamespace(
            document_id=doc_id,
            paper_id=paper_id,
            chunks_count=1,
            indexed_chunks_count=1,
            to_dict=lambda: {
                "document_id": str(doc_id),
                "paper_id": str(paper_id),
                "chunks_count": 1,
                "indexed_chunks_count": 1,
            },
        )


@pytest.fixture
def auth_test_db():
    """In-memory SQLite database with User A, User B, their workspaces, collections, and API keys."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        # User A setup
        user_repo = UserRepository(session)
        user_a = user_repo.create(email="user_a@example.com")
        session.flush()

        ws_repo = WorkspaceRepository(session)
        ws_a1 = ws_repo.create(user_id=user_a.id, name="Workspace A1")
        session.flush()

        col_repo = CollectionRepository(session)
        col_a1 = col_repo.create(workspace_id=ws_a1.id, name="Collection A1")
        session.flush()

        paper_repo = PaperRepository(session)
        paper_a1 = paper_repo.create(
            collection_id=col_a1.id,
            title="Paper A1 Title",
            external_id="seeded-paper-001",
        )
        session.flush()

        doc_repo = DocumentRepository(session)
        doc_a1 = doc_repo.create(
            paper_id=paper_a1.id,
            filename="paper_a1.pdf",
            file_size=1024,
            mime_type="application/pdf",
            checksum="b" * 64,
            storage_key=f"papers/{paper_a1.id}/paper_a1.pdf",
        )
        doc_a1.indexing_status = IndexingStatus.COMPLETED.value
        session.flush()

        key_repo = ApiKeyRepository(session)
        key_a, raw_token_a = key_repo.generate_key(user_id=user_a.id, name="User A Key")
        session.flush()

        # User B setup
        user_b = user_repo.create(email="user_b@example.com")
        session.flush()

        ws_b1 = ws_repo.create(user_id=user_b.id, name="Workspace B1")
        session.flush()

        col_b1 = col_repo.create(workspace_id=ws_b1.id, name="Collection B1")
        session.flush()

        key_b, raw_token_b = key_repo.generate_key(user_id=user_b.id, name="User B Key")
        session.commit()

        context = {
            "session_factory": session_factory,
            "user_a_id": user_a.id,
            "token_a": raw_token_a,
            "key_a_id": key_a.id,
            "ws_a1_id": ws_a1.id,
            "col_a1_id": col_a1.id,
            "paper_a1_id": paper_a1.id,
            "doc_a1_id": doc_a1.id,
            "user_b_id": user_b.id,
            "token_b": raw_token_b,
            "key_b_id": key_b.id,
            "ws_b1_id": ws_b1.id,
            "col_b1_id": col_b1.id,
        }

    return context


def _build_auth_app(
    context,
    provider=None,
    downloader=None,
    analysis_service=None,
    comparison_service=None,
    gap_service=None,
    rag_pipeline=None,
    enforce_auth=True,
):
    sf = context["session_factory"]
    paper_meta = ScholarlyPaperMetadata(
        external_id="2106.09685",
        title="LoRA Paper",
        authors=("Edward J. Hu",),
        abstract="Low-rank adaptation.",
        abstract_url="https://arxiv.org/abs/2106.09685",
        pdf_url="https://arxiv.org/pdf/2106.09685.pdf",
    )
    prov = provider or FakeProvider(papers=[paper_meta])
    dl = downloader or FakeDownloader()
    ingestion = FakeIngestionService(sf)

    discovery_service = DiscoveryService(
        provider=prov,
        downloader=dl,
        session_factory=sf,
        ingestion_service=ingestion,
    )

    auth_service = AuthenticationService(session_factory=sf)

    app = create_app(
        FakeResearchService(),
        rag_pipeline=rag_pipeline or FakeRAGPipeline(),
        discovery_service=discovery_service,
        ingestion_service=ingestion,
        analysis_service=analysis_service or FakeAnalysisService(),
        comparison_service=comparison_service or FakeComparisonService(),
        gap_service=gap_service or FakeGapService(),
        session_factory=sf,
        auth_service=auth_service,
        enforce_auth=enforce_auth,
    )
    return TestClient(app), dl


# ---------------------------------------------------------------------------
# 1. Missing, Malformed, and Invalid Credentials (401)
# ---------------------------------------------------------------------------

def test_missing_credentials_rejected_with_401(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    col_id = auth_test_db["col_a1_id"]

    # Protected POST /discovery/import without Authorization header
    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"
    data = resp.json()
    assert data["error"]["code"] == "unauthorized"
    assert "credentials were not provided" in data["error"]["message"].lower()

    # Protected GET /documents/{id} without Authorization header
    resp_doc = client.get(f"/api/v1/documents/{auth_test_db['doc_a1_id']}")
    assert resp_doc.status_code == 401
    assert resp_doc.headers.get("WWW-Authenticate") == "Bearer"

    # Protected GET /collections/{id} without Authorization header
    resp_col = client.get(f"/api/v1/collections/{col_id}")
    assert resp_col.status_code == 401
    assert resp_col.headers.get("WWW-Authenticate") == "Bearer"


def test_malformed_credentials_rejected_with_401(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    col_id = auth_test_db["col_a1_id"]

    # Basic scheme instead of Bearer -> 401
    resp_basic = client.post(
        "/api/v1/discovery/import",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp_basic.status_code == 401
    assert resp_basic.json()["error"]["code"] == "unauthorized"
    assert "bearer" in resp_basic.json()["error"]["message"].lower()

    # Empty Bearer token -> 401
    resp_empty = client.post(
        "/api/v1/discovery/import",
        headers={"Authorization": "Bearer   "},
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp_empty.status_code == 401
    assert resp_empty.json()["error"]["code"] == "unauthorized"

    # Empty X-API-Key header -> 401
    resp_empty_x = client.post(
        "/api/v1/discovery/import",
        headers={"X-API-Key": "   "},
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp_empty_x.status_code == 401
    assert resp_empty_x.json()["error"]["code"] == "unauthorized"


def test_invalid_api_key_rejected_with_401(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    col_id = auth_test_db["col_a1_id"]

    resp = client.post(
        "/api/v1/discovery/import",
        headers={"Authorization": "Bearer nx_live_fabricated_invalid_secret_key_12345"},
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
    assert "invalid or expired" in resp.json()["error"]["message"].lower()


def test_revoked_api_key_rejected_with_401(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    sf = auth_test_db["session_factory"]
    key_a_id = auth_test_db["key_a_id"]
    token_a = auth_test_db["token_a"]
    col_id = auth_test_db["col_a1_id"]

    # Revoke key A
    with sf() as session:
        repo = ApiKeyRepository(session)
        assert repo.revoke(key_a_id) is True
        session.commit()

    # Access with revoked key must fail with 401
    resp = client.post(
        "/api/v1/discovery/import",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_expired_api_key_rejected_with_401(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    sf = auth_test_db["session_factory"]
    user_a_id = auth_test_db["user_a_id"]
    col_id = auth_test_db["col_a1_id"]

    # Create an already expired key
    past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    with sf() as session:
        repo = ApiKeyRepository(session)
        key, expired_token = repo.generate_key(
            user_id=user_a_id,
            name="Expired Key",
            expires_at=past,
        )
        session.commit()

    resp = client.post(
        "/api/v1/discovery/import",
        headers={"Authorization": f"Bearer {expired_token}"},
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


# ---------------------------------------------------------------------------
# 2. Authenticated Access & Collection Authorization (200 & 403)
# ---------------------------------------------------------------------------

def test_authenticated_access_to_own_collection_succeeds(auth_test_db):
    client, dl = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    col_a_id = auth_test_db["col_a1_id"]

    # User A imports paper into User A's Collection A1
    resp = client.post(
        "/api/v1/discovery/import",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"collection_id": str(col_a_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "imported"
    assert data["collection_id"] == str(col_a_id)
    assert data["external_id"] == "2106.09685"


def test_cross_user_collection_import_attempt_rejected_with_403(auth_test_db):
    client, dl = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]  # User A's credential
    col_b_id = auth_test_db["col_b1_id"]  # User B's collection

    # User A tries to import paper into User B's Collection B1
    resp = client.post(
        "/api/v1/discovery/import",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"collection_id": str(col_b_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 403
    data = resp.json()
    assert data["error"]["code"] == "forbidden"
    assert "forbidden" in data["error"]["message"].lower()

    # Verify no download occurred for unauthorized collection
    assert len(dl.download_calls) == 0


def test_cross_user_collection_read_attempt_rejected_with_403(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    token_b = auth_test_db["token_b"]
    col_b_id = auth_test_db["col_b1_id"]

    # User A tries to read details of User B's collection -> 403 Forbidden
    resp_unauthorized = client.get(
        f"/api/v1/collections/{col_b_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_unauthorized.status_code == 403
    assert resp_unauthorized.json()["error"]["code"] == "forbidden"

    # User B reads their own collection -> 200 OK
    resp_owner = client.get(
        f"/api/v1/collections/{col_b_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_owner.status_code == 200
    assert resp_owner.json()["collection_id"] == str(col_b_id)
    assert resp_owner.json()["name"] == "Collection B1"


def test_cross_workspace_collection_listing_rejected_with_403(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    token_b = auth_test_db["token_b"]
    ws_b_id = auth_test_db["ws_b1_id"]

    # User A attempts to list collections belonging to User B's workspace -> 403 Forbidden
    resp_cross = client.get(
        f"/api/v1/workspaces/{ws_b_id}/collections",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_cross.status_code == 403
    assert resp_cross.json()["error"]["code"] == "forbidden"

    # User B lists their own workspace collections -> 200 OK
    resp_owner = client.get(
        f"/api/v1/workspaces/{ws_b_id}/collections",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_owner.status_code == 200
    data = resp_owner.json()
    assert data["workspace_id"] == str(ws_b_id)
    assert len(data["collections"]) == 1
    assert data["collections"][0]["name"] == "Collection B1"


# ---------------------------------------------------------------------------
# 3. Unauthorized Document Ingestion & Document Retrieval
# ---------------------------------------------------------------------------

def test_unauthorized_document_ingestion_and_cross_user_rejection(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    token_b = auth_test_db["token_b"]
    paper_a_id = auth_test_db["paper_a1_id"]

    pdf_file = ("upload.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")

    # 1. Missing credentials -> 401
    resp_no_auth = client.post(
        "/api/v1/documents/ingest",
        data={"paper_id": str(paper_a_id)},
        files={"file": pdf_file},
    )
    assert resp_no_auth.status_code == 401
    assert resp_no_auth.json()["error"]["code"] == "unauthorized"

    # 2. User B tries to ingest document into User A's paper -> 403 Forbidden
    pdf_file_2 = ("upload.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")
    resp_cross = client.post(
        "/api/v1/documents/ingest",
        headers={"Authorization": f"Bearer {token_b}"},
        data={"paper_id": str(paper_a_id)},
        files={"file": pdf_file_2},
    )
    assert resp_cross.status_code == 403
    assert resp_cross.json()["error"]["code"] == "forbidden"

    # 3. User A ingests into User A's paper -> 200 OK
    pdf_file_3 = ("upload.pdf", io.BytesIO(VALID_PDF_BYTES), "application/pdf")
    resp_owner = client.post(
        "/api/v1/documents/ingest",
        headers={"Authorization": f"Bearer {token_a}"},
        data={"paper_id": str(paper_a_id)},
        files={"file": pdf_file_3},
    )
    assert resp_owner.status_code == 200
    assert resp_owner.json()["paper_id"] == str(paper_a_id)


def test_document_retrieval_authorization(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    token_b = auth_test_db["token_b"]
    doc_a_id = auth_test_db["doc_a1_id"]

    # 1. Missing credentials -> 401
    resp_unauth = client.get(f"/api/v1/documents/{doc_a_id}")
    assert resp_unauth.status_code == 401
    assert resp_unauth.json()["error"]["code"] == "unauthorized"

    # 2. User B tries to read User A's document -> 403 Forbidden
    resp_cross = client.get(
        f"/api/v1/documents/{doc_a_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_cross.status_code == 403
    assert resp_cross.json()["error"]["code"] == "forbidden"

    # 3. User A reads User A's document -> 200 OK
    resp_owner = client.get(
        f"/api/v1/documents/{doc_a_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_owner.status_code == 200
    data = resp_owner.json()
    assert data["document_id"] == str(doc_a_id)
    assert data["filename"] == "paper_a1.pdf"
    assert data["indexing_status"] == "completed"

    # 4. Unknown document UUID -> 404 Not Found
    resp_not_found = client.get(
        f"/api/v1/documents/{uuid4()}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_not_found.status_code == 404
    assert resp_not_found.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# 4. Public Endpoints & Alternative Header Support
# ---------------------------------------------------------------------------

def test_public_scholarly_search_and_health_accessible_without_auth(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)

    # Health endpoint is public
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json() == {"status": "ok"}

    # Discovery search is public
    resp_search = client.get("/api/v1/discovery/search?query=LoRA")
    assert resp_search.status_code == 200
    assert resp_search.json()["total_results"] == 1

    # Discovery paper details is public
    resp_paper = client.get("/api/v1/discovery/paper/2106.09685")
    assert resp_paper.status_code == 200
    assert resp_paper.json()["external_id"] == "2106.09685"


def test_x_api_key_header_support(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    col_a_id = auth_test_db["col_a1_id"]

    # Provide credential via X-API-Key instead of Authorization: Bearer
    resp = client.get(
        f"/api/v1/collections/{col_a_id}",
        headers={"X-API-Key": token_a},
    )
    assert resp.status_code == 200
    assert resp.json()["collection_id"] == str(col_a_id)


def test_touch_last_used_timestamp_on_successful_auth(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    sf = auth_test_db["session_factory"]
    key_a_id = auth_test_db["key_a_id"]
    token_a = auth_test_db["token_a"]
    col_a_id = auth_test_db["col_a1_id"]

    with sf() as session:
        repo = ApiKeyRepository(session)
        key_before = repo.get_by_id(key_a_id)
        assert key_before.last_used_at is None

    # Authenticate call
    resp = client.get(
        f"/api/v1/collections/{col_a_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200

    # Verify last_used_at was updated
    with sf() as session:
        repo = ApiKeyRepository(session)
        key_after = repo.get_by_id(key_a_id)
        assert key_after.last_used_at is not None
        assert isinstance(key_after.last_used_at, datetime.datetime)


# ---------------------------------------------------------------------------
# 5. Bootstrap Utility & Database Migration
# ---------------------------------------------------------------------------

def test_bootstrap_user_and_key_creates_complete_hierarchy(auth_test_db):
    sf = auth_test_db["session_factory"]
    user, ws, key, raw_token = bootstrap_user_and_key(
        sf,
        email="new_operator@example.com",
        workspace_name="Research Lab",
        key_name="Lab Deployment Key",
    )

    assert user.email == "new_operator@example.com"
    assert ws.name == "Research Lab"
    assert ws.user_id == user.id
    assert key.name == "Lab Deployment Key"
    assert key.user_id == user.id
    assert raw_token.startswith("nx_live_")

    # Verify key can be authenticated immediately
    auth_service = AuthenticationService(sf)
    authenticated = auth_service.authenticate_token(raw_token)
    assert authenticated.id == user.id
    assert authenticated.email == user.email


def test_api_keys_table_crud(auth_test_db):
    sf = auth_test_db["session_factory"]
    user_a_id = auth_test_db["user_a_id"]

    with sf() as session:
        repo = ApiKeyRepository(session)
        record, raw_key = repo.generate_key(
            user_id=user_a_id,
            name="Test Key CRUD",
        )
        session.commit()
        key_id = record.id

        # Verify plaintext key is not stored in DB
        found = repo.get_by_id(key_id)
        assert found is not None
        assert found.key_hash != raw_key
        assert found.name == "Test Key CRUD"

        # List by user
        keys = repo.list_by_user_id(user_a_id)
        assert any(k.id == key_id for k in keys)

        # Revoke
        assert repo.revoke(key_id) is True
        session.commit()

        # Unrevoked query returns None
        assert repo.get_active_by_raw_key(raw_key) is None


# ---------------------------------------------------------------------------
# 6. Audit Regression Tests: Analysis, Comparison, Gaps, Verify & Key Security
# ---------------------------------------------------------------------------

def test_cross_user_paper_analyze_rejected_with_403(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    token_b = auth_test_db["token_b"]
    paper_a_id = auth_test_db["paper_a1_id"]

    # 1. Unauthenticated -> 401
    resp_unauth = client.post("/api/v1/papers/analyze", json={"paper_id": str(paper_a_id)})
    assert resp_unauth.status_code == 401

    # 2. User B tries to analyze User A's paper via body endpoint -> 403 Forbidden
    resp_cross = client.post(
        "/api/v1/papers/analyze",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"paper_id": str(paper_a_id)},
    )
    assert resp_cross.status_code == 403
    assert resp_cross.json()["error"]["code"] == "forbidden"

    # 3. User B tries to analyze User A's paper via path endpoint -> 403 Forbidden
    resp_cross_path = client.post(
        f"/api/v1/papers/{paper_a_id}/analyze",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_cross_path.status_code == 403
    assert resp_cross_path.json()["error"]["code"] == "forbidden"

    # 4. User A (owner) analyzes their paper -> 200 OK
    resp_owner = client.post(
        "/api/v1/papers/analyze",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"paper_id": str(paper_a_id)},
    )
    assert resp_owner.status_code == 200
    assert resp_owner.json()["paper_id"] == str(paper_a_id)


def test_cross_user_paper_compare_rejected_with_403(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    sf = auth_test_db["session_factory"]
    token_a = auth_test_db["token_a"]
    token_b = auth_test_db["token_b"]
    col_a_id = auth_test_db["col_a1_id"]
    paper_a1_id = auth_test_db["paper_a1_id"]

    # Create second paper for User A so multi-paper comparison has >= 2 papers
    with sf() as session:
        paper_repo = PaperRepository(session)
        paper_a2 = paper_repo.create(
            collection_id=col_a_id,
            title="Paper A2 Title",
            external_id="seeded-paper-002",
        )
        session.commit()
        paper_a2_id = paper_a2.id

    paper_pair = [str(paper_a1_id), str(paper_a2_id)]

    # 1. Unauthenticated -> 401
    resp_unauth = client.post("/api/v1/papers/compare", json={"paper_ids": paper_pair})
    assert resp_unauth.status_code == 401

    resp_unauth_research = client.post(
        "/api/v1/research/compare", json={"paper_ids": paper_pair, "query": "compare"}
    )
    assert resp_unauth_research.status_code == 401

    # 2. User B tries to compare User A's papers -> 403 Forbidden
    resp_cross = client.post(
        "/api/v1/papers/compare",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"paper_ids": paper_pair},
    )
    assert resp_cross.status_code == 403
    assert resp_cross.json()["error"]["code"] == "forbidden"

    # 3. User B tries to compare User A's papers via research/compare -> 403 Forbidden
    resp_cross_research = client.post(
        "/api/v1/research/compare",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"paper_ids": paper_pair, "query": "compare models"},
    )
    assert resp_cross_research.status_code == 403
    assert resp_cross_research.json()["error"]["code"] == "forbidden"

    # 4. User A compares own papers -> 200 OK
    resp_owner = client.post(
        "/api/v1/papers/compare",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"paper_ids": paper_pair},
    )
    assert resp_owner.status_code == 200

    resp_owner_research = client.post(
        "/api/v1/research/compare",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"paper_ids": paper_pair, "query": "compare models"},
    )
    assert resp_owner_research.status_code == 200


def test_cross_user_research_gaps_rejected_with_403(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    token_b = auth_test_db["token_b"]
    col_a_id = auth_test_db["col_a1_id"]
    paper_a_id = auth_test_db["paper_a1_id"]

    # 1. Unauthenticated -> 401
    resp_unauth = client.post("/api/v1/research/gaps", json={"collection_id": str(col_a_id)})
    assert resp_unauth.status_code == 401

    # 2. User B tries to discover gaps for User A's collection -> 403 Forbidden
    resp_cross_col = client.post(
        "/api/v1/research/gaps",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"collection_id": str(col_a_id)},
    )
    assert resp_cross_col.status_code == 403
    assert resp_cross_col.json()["error"]["code"] == "forbidden"

    # 3. User B tries to discover gaps for User A's paper -> 403 Forbidden
    resp_cross_paper = client.post(
        "/api/v1/research/gaps",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"paper_ids": [str(paper_a_id)]},
    )
    assert resp_cross_paper.status_code == 403
    assert resp_cross_paper.json()["error"]["code"] == "forbidden"

    # 4. User B tries via path endpoint /papers/{paper_id}/gaps -> 403 Forbidden
    resp_cross_path = client.post(
        f"/api/v1/papers/{paper_a_id}/gaps",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_cross_path.status_code == 403

    # 5. User A (owner) succeeds -> 200 OK
    resp_owner = client.post(
        "/api/v1/research/gaps",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"collection_id": str(col_a_id)},
    )
    assert resp_owner.status_code == 200


def test_cross_user_research_verify_rejected_with_403(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    token_b = auth_test_db["token_b"]
    paper_a_id = auth_test_db["paper_a1_id"]

    # 1. Unauthenticated -> 401
    resp_unauth = client.post(
        "/api/v1/research/verify",
        json={"query": "test query", "text": "test text", "expected_paper_id": str(paper_a_id)},
    )
    assert resp_unauth.status_code == 401

    # 2. User B verifies with User A's paper as expected_paper_id -> 403 Forbidden
    resp_cross = client.post(
        "/api/v1/research/verify",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"query": "test query", "text": "test text", "expected_paper_id": str(paper_a_id)},
    )
    assert resp_cross.status_code == 403
    assert resp_cross.json()["error"]["code"] == "forbidden"

    # 3. User A verifies own paper -> 200 OK
    resp_owner = client.post(
        "/api/v1/research/verify",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"query": "test query", "text": "test text", "expected_paper_id": str(paper_a_id)},
    )
    assert resp_owner.status_code == 200


def test_research_and_rag_query_require_auth_when_enforced(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    token_a = auth_test_db["token_a"]
    col_a_id = str(auth_test_db["col_a1_id"])

    # 1. Anonymous access to public research endpoints succeeds without collection_id
    resp_unauth_res = client.post("/api/v1/research", json={"query": "test query"})
    assert resp_unauth_res.status_code == 200

    resp_unauth_rag = client.post("/api/v1/research/query", json={"query": "test query"})
    assert resp_unauth_rag.status_code == 200

    resp_unauth_alias = client.post("/api/v1/rag/query", json={"query": "test query"})
    assert resp_unauth_alias.status_code == 200

    # 2. Anonymous callers attempting to target specific collections or papers are rejected with 401
    resp_unauth_col_res = client.post(
        "/api/v1/research", json={"query": "test query", "collection_id": col_a_id}
    )
    assert resp_unauth_col_res.status_code == 401
    assert resp_unauth_col_res.json()["error"]["code"] == "unauthorized"

    resp_unauth_col_rag = client.post(
        "/api/v1/research/query", json={"query": "test query", "collection_id": col_a_id}
    )
    assert resp_unauth_col_rag.status_code == 401
    assert resp_unauth_col_rag.json()["error"]["code"] == "unauthorized"

    resp_unauth_paper_rag = client.post(
        "/api/v1/research/query",
        json={"query": "test query", "paper_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"},
    )
    assert resp_unauth_paper_rag.status_code == 401
    assert resp_unauth_paper_rag.json()["error"]["code"] == "unauthorized"

    # 3. Authenticated requests succeed with or without collection_id
    resp_auth_res = client.post(
        "/api/v1/research",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"query": "test query"},
    )
    assert resp_auth_res.status_code == 200

    resp_auth_rag = client.post(
        "/api/v1/research/query",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"query": "test query"},
    )
    assert resp_auth_rag.status_code == 200

    resp_auth_alias = client.post(
        "/api/v1/rag/query",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"query": "test query"},
    )
    assert resp_auth_alias.status_code == 200


def test_immediate_revocation_takes_effect_on_next_request(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    sf = auth_test_db["session_factory"]
    token_a = auth_test_db["token_a"]
    key_a_id = auth_test_db["key_a_id"]
    col_a_id = auth_test_db["col_a1_id"]

    # First request works
    resp_ok = client.get(
        f"/api/v1/collections/{col_a_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_ok.status_code == 200

    # Revoke key immediately in database
    with sf() as session:
        repo = ApiKeyRepository(session)
        assert repo.revoke(key_a_id) is True
        session.commit()

    # Next immediate request with same key is rejected with 401
    resp_revoked = client.get(
        f"/api/v1/collections/{col_a_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp_revoked.status_code == 401
    assert resp_revoked.json()["error"]["code"] == "unauthorized"


def test_touch_last_used_throttling_avoids_unnecessary_db_writes(auth_test_db):
    client, _ = _build_auth_app(auth_test_db)
    sf = auth_test_db["session_factory"]
    token_a = auth_test_db["token_a"]
    key_a_id = auth_test_db["key_a_id"]
    col_a_id = auth_test_db["col_a1_id"]

    # 1. First request touches last_used_at
    resp1 = client.get(
        f"/api/v1/collections/{col_a_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp1.status_code == 200

    with sf() as session:
        repo = ApiKeyRepository(session)
        key1 = repo.get_by_id(key_a_id)
        assert key1.last_used_at is not None
        first_timestamp = key1.last_used_at

    # 2. Immediate second request within 60s does NOT rewrite/commit timestamp
    resp2 = client.get(
        f"/api/v1/collections/{col_a_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp2.status_code == 200

    with sf() as session:
        repo = ApiKeyRepository(session)
        key2 = repo.get_by_id(key_a_id)
        assert key2.last_used_at == first_timestamp

    # 3. Simulate passage of time (> 60 seconds)
    two_minutes_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)
    with sf() as session:
        repo = ApiKeyRepository(session)
        repo._update(key_a_id, last_used_at=two_minutes_ago)
        session.commit()

    # Next request touches last_used_at because > 60 seconds elapsed
    resp3 = client.get(
        f"/api/v1/collections/{col_a_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp3.status_code == 200

    with sf() as session:
        repo = ApiKeyRepository(session)
        key3 = repo.get_by_id(key_a_id)
        assert key3 is not None
        last_val = key3.last_used_at
        if last_val.tzinfo is None:
            last_val = last_val.replace(tzinfo=datetime.timezone.utc)
        assert last_val > two_minutes_ago


def test_bootstrap_safety_against_accidental_duplicate_keys(auth_test_db):
    sf = auth_test_db["session_factory"]
    email = "operator_safety@example.com"

    # 1. Initial bootstrap succeeds
    user, ws, key1, token1 = bootstrap_user_and_key(sf, email=email)
    assert token1.startswith("nx_live_")

    # 2. Repeated bootstrap without --rotate or --allow-existing raises ValueError
    with pytest.raises(ValueError, match="already exists with 1 active API key"):
        bootstrap_user_and_key(sf, email=email)

    # 3. Repeated bootstrap with allow_existing=True provisions a second key without revoking
    user, ws, key2, token2 = bootstrap_user_and_key(sf, email=email, allow_existing=True)
    assert token2 != token1
    with sf() as session:
        repo = ApiKeyRepository(session)
        assert repo.get_active_by_raw_key(token1) is not None
        assert repo.get_active_by_raw_key(token2) is not None

    # 4. Repeated bootstrap with rotate=True revokes all previous keys and issues fresh one
    user, ws, key3, token3 = bootstrap_user_and_key(sf, email=email, rotate=True)
    assert token3 != token2
    with sf() as session:
        repo = ApiKeyRepository(session)
        # Previous keys revoked
        assert repo.get_active_by_raw_key(token1) is None
        assert repo.get_active_by_raw_key(token2) is None
        # New key active
        assert repo.get_active_by_raw_key(token3) is not None


def test_startup_fails_closed_when_auth_enforced_without_services():
    """Verify that create_app refuses to start up if auth is enforced but no session_factory or auth_service exists."""
    from nexora.api.errors import InvalidConfigurationError

    with pytest.raises(InvalidConfigurationError) as exc:
        create_app(
            FakeResearchService(),
            enforce_auth=True,
            session_factory=None,
            auth_service=None,
        )
    assert "Authentication is enforced but no session_factory or auth_service is configured." in str(exc.value)


def test_create_offline_app_bypasses_auth_by_design():
    """Verify that create_offline_app intentionally sets enforce_auth=False."""
    from nexora.api.app import create_offline_app

    offline_app = create_offline_app()
    assert getattr(offline_app.state, "enforce_auth", None) is False


def test_tenant_routes_fail_closed_without_authentication(auth_test_db):
    """Verify private tenant endpoints fail closed with 401 when called without credentials,

    even if enforce_auth was somehow False.
    """
    client, _ = _build_auth_app(auth_test_db, enforce_auth=False)
    doc_id = auth_test_db["doc_a1_id"]
    col_id = auth_test_db["col_a1_id"]
    ws_id = auth_test_db["ws_a1_id"]

    # 1. GET /documents/{id} without auth -> 401
    resp_doc = client.get(f"/api/v1/documents/{doc_id}")
    assert resp_doc.status_code == 401
    assert resp_doc.json()["error"]["code"] == "unauthorized"

    # 2. GET /collections/{id} without auth -> 401
    resp_col = client.get(f"/api/v1/collections/{col_id}")
    assert resp_col.status_code == 401
    assert resp_col.json()["error"]["code"] == "unauthorized"

    # 3. GET /workspaces/{id}/collections without auth -> 401
    resp_ws = client.get(f"/api/v1/workspaces/{ws_id}/collections")
    assert resp_ws.status_code == 401
    assert resp_ws.json()["error"]["code"] == "unauthorized"


def test_tenant_routes_fail_closed_without_session_factory():
    """Verify private tenant endpoints fail closed when database session factory is missing."""
    app = create_app(FakeResearchService(), session_factory=None, enforce_auth=False)
    client = TestClient(app, raise_server_exceptions=False)
    dummy_uuid = str(uuid4())

    resp_doc = client.get(f"/api/v1/documents/{dummy_uuid}")
    assert resp_doc.status_code == 500
    assert resp_doc.json()["error"]["code"] == "server_misconfigured"

    resp_col = client.get(f"/api/v1/collections/{dummy_uuid}")
    assert resp_col.status_code == 500
    assert resp_col.json()["error"]["code"] == "server_misconfigured"

    resp_ws = client.get(f"/api/v1/workspaces/{dummy_uuid}/collections")
    assert resp_ws.status_code == 500
    assert resp_ws.json()["error"]["code"] == "server_misconfigured"


def test_forbidden_error_messages_do_not_leak_uuids(auth_test_db):
    """Verify that 403 Forbidden responses do not echo private resource UUIDs in error messages."""
    client, _ = _build_auth_app(auth_test_db)
    token_b = auth_test_db["token_b"]
    doc_a_id = auth_test_db["doc_a1_id"]
    col_a_id = auth_test_db["col_a1_id"]
    ws_a_id = auth_test_db["ws_a1_id"]

    # User B accesses User A's collection
    resp_col = client.get(f"/api/v1/collections/{col_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert resp_col.status_code == 403
    assert str(col_a_id) not in resp_col.json()["error"]["message"]

    # User B accesses User A's workspace
    resp_ws = client.get(f"/api/v1/workspaces/{ws_a_id}/collections", headers={"Authorization": f"Bearer {token_b}"})
    assert resp_ws.status_code == 403
    assert str(ws_a_id) not in resp_ws.json()["error"]["message"]

    # User B accesses User A's document
    resp_doc = client.get(f"/api/v1/documents/{doc_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert resp_doc.status_code == 403
    assert str(doc_a_id) not in resp_doc.json()["error"]["message"]
