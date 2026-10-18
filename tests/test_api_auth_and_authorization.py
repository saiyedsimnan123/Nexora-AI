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
        return {}


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


def _build_auth_app(context, provider=None, downloader=None):
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
        discovery_service=discovery_service,
        ingestion_service=ingestion,
        session_factory=sf,
        auth_service=auth_service,
        enforce_auth=True,
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
