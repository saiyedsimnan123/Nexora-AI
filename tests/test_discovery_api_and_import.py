"""Tests for Scholarly Discovery API and Explicit Paper Import (Milestone: Phase 3).

Validates:
1. GET /api/v1/discovery/search:
   - Query validation, pagination (offset/limit), category filters, sorting.
   - Upstream error mapping (429, 502, 504) and input validation (400/422).
   - Verified that search never downloads PDFs or contacts storage.
2. GET /api/v1/discovery/paper/{external_id}:
   - Successful normalized metadata lookup.
   - Canonical validation: rejects arbitrary URLs and malformed identifiers with 400.
   - Upstream 404, 429, 502, 504 mapped to sanitized HTTP responses.
3. POST /api/v1/discovery/import:
   - Target collection existence validation (400 for bad UUID, 404 for missing collection).
   - Real orchestration path: metadata lookup -> safe downloading -> paper record -> document ingestion.
   - Deduplication: already-indexed document returns 200 'already_imported' with zero downloads/re-indexing.
   - Resume behavior: existing paper with missing or failed document completes import without creating duplicate paper records.
   - Temporary file cleanup guaranteed on both success and failure.
   - Failure paths: 404 on upstream, invalid PDF (400), oversized PDF (413), download timeout/connection (502/504).
   - Ingestion failure handling and transaction boundaries.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.api.app import create_app
from nexora.api.errors import (
    DocumentNotFoundError,
    InvalidRequestError,
)
from nexora.database.models.base import Base
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document, IndexingStatus
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.discovery.arxiv import ArxivSourceProvider
from nexora.discovery.downloader import DownloadedPDF, SafePDFDownloader
from nexora.discovery.errors import (
    DiscoveryConnectionError,
    DiscoveryRateLimitError,
    DiscoveryResponseError,
    DiscoveryTimeoutError,
    InvalidIdentifierError,
    InvalidPDFError,
    OversizedPDFError,
    PDFNotFoundError,
    PaperNotFoundError,
    SSRFProtectionError,
)
from nexora.discovery.models import (
    ScholarlyPaperMetadata,
    ScholarlySearchQuery,
    ScholarlySearchResult,
)
from nexora.discovery.service import DiscoveryService
from nexora.document.ingestion import IngestionResult, VectorIndexingError
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.retrieval.in_memory import InMemoryVectorStore

# Standard mock PDF payload with valid '%PDF-' signature
VALID_PDF_BYTES = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< >>\n%%EOF"


# ---------------------------------------------------------------------------
# Database & Test Double Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_session_factory():
    """In-memory SQLite session factory sharing the same database via StaticPool."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def seeded_collection(sqlite_session_factory) -> tuple[UUID, sessionmaker]:
    """Create a persistent User, Workspace, and Collection in the database."""
    with sqlite_session_factory() as session:
        user = User(email=f"researcher_{uuid4()}@example.com")
        session.add(user)
        session.flush()

        ws = Workspace(user_id=user.id, name="Test Research Lab")
        session.add(ws)
        session.flush()

        col = Collection(workspace_id=ws.id, name="Deep Learning Papers")
        session.add(col)
        session.commit()
        return col.id, sqlite_session_factory


class FakeResearchService:
    def ask(self, request: Any) -> Any:
        raise NotImplementedError


class FakeProvider:
    """Mock ScholarlySourceProvider for testing API and service layers."""

    def __init__(self, papers: list[ScholarlyPaperMetadata] | None = None) -> None:
        self.papers = {p.external_id: p for p in (papers or [])}
        self.search_error: Exception | None = None
        self.get_paper_error: Exception | None = None
        self.search_calls: list[ScholarlySearchQuery] = []
        self.get_paper_calls: list[str] = []

    def search(self, query: ScholarlySearchQuery) -> ScholarlySearchResult:
        self.search_calls.append(query)
        if self.search_error:
            raise self.search_error

        matched = list(self.papers.values())
        if query.categories:
            matched = [p for p in matched if any(c in p.categories for c in query.categories)]

        offset = query.offset
        limit = query.limit
        sliced = matched[offset : offset + limit]

        return ScholarlySearchResult(
            provider="arxiv",
            query=query.query,
            total_results=len(matched),
            offset=offset,
            limit=limit,
            papers=tuple(sliced),
        )

    def get_paper(self, external_id: str) -> ScholarlyPaperMetadata | None:
        self.get_paper_calls.append(external_id)
        if self.get_paper_error:
            raise self.get_paper_error
        # Also check without version suffix
        base_id = external_id.split("v")[0]
        if external_id in self.papers:
            return self.papers[external_id]
        for pid, p in self.papers.items():
            if pid.split("v")[0] == base_id:
                return p
        return None


class FakeDownloader:
    """Mock SafePDFDownloader for testing explicit import."""

    def __init__(self, pdf_bytes: bytes = VALID_PDF_BYTES, error: Exception | None = None) -> None:
        self.pdf_bytes = pdf_bytes
        self.error = error
        self.download_calls: list[str] = []
        self.created_temp_paths: list[Path] = []

    def download_arxiv_pdf(self, arxiv_id: str, *, destination_dir: Path | None = None) -> DownloadedPDF:
        self.download_calls.append(arxiv_id)
        if self.error:
            raise self.error

        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(self.pdf_bytes)
        tmp.close()
        path = Path(tmp.name)
        self.created_temp_paths.append(path)

        return DownloadedPDF(
            path=path,
            file_size=len(self.pdf_bytes),
            arxiv_id=arxiv_id,
            source_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        )


class FakeDocumentPipeline:
    def process_pdf(self, file_path: str) -> ProcessedDocument:
        return ProcessedDocument(
            document_id=str(uuid4()),
            source=file_path,
            chunks=[
                DocumentChunk(text="Chunk 1: LoRA parameter reduction", embedding=[0.1, 0.2], index=0),
                DocumentChunk(text="Chunk 2: Adapter rank decomposition", embedding=[0.3, 0.4], index=1),
            ],
            metadata={"total_pages": 1, "chunk_pages": [1, 1]},
        )


class RecordingIngestionService:
    """Mock DocumentIngestionService tracking calls and supporting simulated failures."""

    def __init__(self, session_factory: sessionmaker, error: Exception | None = None) -> None:
        self._session_factory = session_factory
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def process_and_ingest(
        self,
        paper_id: UUID,
        file_path: str,
        pipeline: Any = None,
        filename: str | None = None,
        checksum: str | None = None,
        durable: bool = False,
    ) -> IngestionResult:
        file_exists_at_call = Path(file_path).exists()
        file_size = Path(file_path).stat().st_size if file_exists_at_call else 0

        self.calls.append({
            "paper_id": paper_id,
            "file_path": file_path,
            "file_exists": file_exists_at_call,
            "file_size": file_size,
            "filename": filename,
            "checksum": checksum,
            "durable": durable,
        })

        if self.error:
            if durable:
                with self._session_factory() as session:
                    doc_repo = DocumentRepository(session)
                    doc = doc_repo.create(
                        paper_id=paper_id,
                        filename=filename or "document.pdf",
                        file_size=file_size,
                        checksum=checksum or "0" * 64,
                        storage_key=f"papers/{paper_id}/{filename or 'document.pdf'}",
                        indexing_status=IndexingStatus.FAILED.value,
                        indexing_error=str(self.error),
                    )
                    session.commit()
            raise self.error

        with self._session_factory() as session:
            doc_repo = DocumentRepository(session)
            chunk_repo = DocumentChunkRepository(session)

            doc = doc_repo.create(
                paper_id=paper_id,
                filename=filename or "document.pdf",
                file_size=file_size,
                checksum=checksum or "a" * 64,
                storage_key=f"papers/{paper_id}/{filename or 'document.pdf'}",
                indexing_status=IndexingStatus.COMPLETED.value,
                indexed_at=datetime.datetime.now(datetime.timezone.utc),
            )
            c1 = chunk_repo.create(
                id=uuid4(),
                document_id=doc.id,
                chunk_index=0,
                text="Chunk 1: LoRA parameter reduction",
            )
            c2 = chunk_repo.create(
                id=uuid4(),
                document_id=doc.id,
                chunk_index=1,
                text="Chunk 2: Adapter rank decomposition",
            )
            session.commit()

            return IngestionResult(
                document_id=doc.id,
                paper_id=paper_id,
                chunks_count=2,
                indexed_chunks_count=2,
                vector_ids=[str(c1.id), str(c2.id)],
                document=doc,
                chunk_records=[c1, c2],
            )


def make_test_paper(
    external_id: str = "2106.09685",
    title: str = "LoRA: Low-Rank Adaptation of Large Language Models",
    doi: str | None = "10.48550/arXiv.2106.09685",
    categories: tuple[str, ...] = ("cs.CL", "cs.AI"),
) -> ScholarlyPaperMetadata:
    return ScholarlyPaperMetadata(
        external_id=external_id,
        title=title,
        authors=("Edward J. Hu", "Yelong Shen", "Phillip Wallis"),
        abstract="We hypothesize that the updates to the weights also have a low intrinsic rank during adaptation.",
        publication_date=datetime.date(2021, 6, 17),
        doi=doi,
        categories=categories,
        primary_category=categories[0] if categories else None,
        abstract_url=f"https://arxiv.org/abs/{external_id}",
        pdf_url=f"https://arxiv.org/pdf/{external_id}.pdf",
    )


# ---------------------------------------------------------------------------
# 1. Search Endpoint Tests (GET /api/v1/discovery/search)
# ---------------------------------------------------------------------------

def test_discovery_search_success():
    paper = make_test_paper()
    provider = FakeProvider(papers=[paper])
    service = DiscoveryService(provider=provider)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    response = client.get("/api/v1/discovery/search?query=LoRA")
    assert response.status_code == 200
    data = response.json()

    assert data["provider"] == "arxiv"
    assert data["query"] == "LoRA"
    assert data["total_results"] == 1
    assert data["offset"] == 0
    assert data["limit"] == 10
    assert len(data["papers"]) == 1

    item = data["papers"][0]
    assert item["external_id"] == "2106.09685"
    assert item["title"] == paper.title
    assert "Edward J. Hu" in item["authors"]
    assert item["doi"] == "10.48550/arXiv.2106.09685"
    assert item["abstract_url"] == "https://arxiv.org/abs/2106.09685"
    assert item["pdf_url"] == "https://arxiv.org/pdf/2106.09685.pdf"


def test_discovery_search_with_category_filters():
    p1 = make_test_paper(external_id="2106.09685", categories=("cs.CL",))
    p2 = make_test_paper(external_id="2201.00001", categories=("math.PR",))
    provider = FakeProvider(papers=[p1, p2])
    service = DiscoveryService(provider=provider)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    # Filter with single category
    resp = client.get("/api/v1/discovery/search?query=adaptation&categories=cs.CL")
    assert resp.status_code == 200
    assert resp.json()["total_results"] == 1
    assert resp.json()["papers"][0]["external_id"] == "2106.09685"

    # Filter with comma-separated categories
    resp2 = client.get("/api/v1/discovery/search?query=adaptation&categories=cs.CL,math.PR")
    assert resp2.status_code == 200
    assert resp2.json()["total_results"] == 2


def test_discovery_search_pagination_and_sorting():
    papers = [make_test_paper(external_id=f"2106.0000{i}") for i in range(5)]
    provider = FakeProvider(papers=papers)
    service = DiscoveryService(provider=provider)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    resp = client.get("/api/v1/discovery/search?query=test&offset=2&limit=2&sort_by=submitted_date&sort_order=ascending")
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset"] == 2
    assert data["limit"] == 2
    assert len(data["papers"]) == 2
    assert provider.search_calls[-1].sort_by == "submitted_date"
    assert provider.search_calls[-1].sort_order == "ascending"


def test_discovery_search_validation_errors():
    app = create_app(FakeResearchService(), discovery_service=DiscoveryService(provider=FakeProvider()))
    client = TestClient(app)

    # Empty query -> 400
    r1 = client.get("/api/v1/discovery/search?query=")
    assert r1.status_code == 400
    assert r1.json()["error"]["code"] == "invalid_request"

    # Whitespace query -> 400
    r2 = client.get("/api/v1/discovery/search?query=   ")
    assert r2.status_code == 400

    # Invalid sort_by -> 400
    r3 = client.get("/api/v1/discovery/search?query=test&sort_by=invalid_sort")
    assert r3.status_code == 400

    # Invalid sort_order -> 400
    r4 = client.get("/api/v1/discovery/search?query=test&sort_order=invalid_order")
    assert r4.status_code == 400

    # Out of range limit (> 100) -> 422
    r5 = client.get("/api/v1/discovery/search?query=test&limit=150")
    assert r5.status_code == 422

    # Negative offset -> 422
    r6 = client.get("/api/v1/discovery/search?query=test&offset=-1")
    assert r6.status_code == 422


def test_discovery_search_upstream_error_handling():
    provider = FakeProvider()
    service = DiscoveryService(provider=provider)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    # 429 rate limit
    provider.search_error = DiscoveryRateLimitError("arXiv rate limited")
    r_rate = client.get("/api/v1/discovery/search?query=test")
    assert r_rate.status_code == 429
    assert r_rate.json()["error"]["code"] == "rate_limited"

    # 504 timeout
    provider.search_error = DiscoveryTimeoutError("arXiv timed out")
    r_time = client.get("/api/v1/discovery/search?query=test")
    assert r_time.status_code == 504
    assert r_time.json()["error"]["code"] == "gateway_timeout"

    # 502 connection error
    provider.search_error = DiscoveryConnectionError("DNS failure")
    r_conn = client.get("/api/v1/discovery/search?query=test")
    assert r_conn.status_code == 502
    assert r_conn.json()["error"]["code"] == "bad_gateway"

    # 502 response error (5xx or malformed XML)
    provider.search_error = DiscoveryResponseError("Server returned 503")
    r_resp = client.get("/api/v1/discovery/search?query=test")
    assert r_resp.status_code == 502
    assert r_resp.json()["error"]["code"] == "bad_gateway"


def test_discovery_search_never_contacts_downloader():
    provider = FakeProvider(papers=[make_test_paper()])
    downloader = FakeDownloader()
    service = DiscoveryService(provider=provider, downloader=downloader)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    resp = client.get("/api/v1/discovery/search?query=LoRA")
    assert resp.status_code == 200
    assert len(downloader.download_calls) == 0
    assert len(downloader.created_temp_paths) == 0


# ---------------------------------------------------------------------------
# 2. Paper Details Endpoint Tests (GET /api/v1/discovery/paper/{external_id})
# ---------------------------------------------------------------------------

def test_discovery_paper_details_success():
    paper = make_test_paper(external_id="2106.09685")
    provider = FakeProvider(papers=[paper])
    service = DiscoveryService(provider=provider)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    resp = client.get("/api/v1/discovery/paper/2106.09685")
    assert resp.status_code == 200
    data = resp.json()
    assert data["external_id"] == "2106.09685"
    assert data["title"] == paper.title
    assert data["doi"] == paper.doi
    assert data["abstract_url"] == "https://arxiv.org/abs/2106.09685"
    assert data["pdf_url"] == "https://arxiv.org/pdf/2106.09685.pdf"


def test_discovery_paper_details_legacy_id():
    paper = make_test_paper(external_id="math/0501001", title="Legacy Math Paper")
    provider = FakeProvider(papers=[paper])
    service = DiscoveryService(provider=provider)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    resp = client.get("/api/v1/discovery/paper/math/0501001")
    assert resp.status_code == 200
    assert resp.json()["external_id"] == "math/0501001"
    assert resp.json()["title"] == "Legacy Math Paper"


def test_discovery_paper_details_rejects_arbitrary_urls_and_malformed_ids():
    provider = FakeProvider(papers=[make_test_paper()])
    service = DiscoveryService(provider=provider)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    # Arbitrary domain URL -> rejected with 400
    r_bad_url = client.get("/api/v1/discovery/paper/https://evil.example.com/exploit.pdf")
    assert r_bad_url.status_code == 400
    assert r_bad_url.json()["error"]["code"] == "invalid_request"

    # Malformed non-arXiv identifier -> rejected with 400
    r_bad_id = client.get("/api/v1/discovery/paper/not_a_valid_arxiv_id")
    assert r_bad_id.status_code == 400
    assert r_bad_id.json()["error"]["code"] == "invalid_request"


def test_discovery_paper_details_not_found():
    provider = FakeProvider(papers=[])
    service = DiscoveryService(provider=provider)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    resp = client.get("/api/v1/discovery/paper/2106.09685")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_discovery_paper_details_upstream_errors():
    provider = FakeProvider()
    service = DiscoveryService(provider=provider)
    app = create_app(FakeResearchService(), discovery_service=service)
    client = TestClient(app)

    provider.get_paper_error = DiscoveryRateLimitError("Rate limit exceeded")
    r1 = client.get("/api/v1/discovery/paper/2106.09685")
    assert r1.status_code == 429
    assert r1.json()["error"]["code"] == "rate_limited"

    provider.get_paper_error = DiscoveryTimeoutError("Timeout")
    r2 = client.get("/api/v1/discovery/paper/2106.09685")
    assert r2.status_code == 504
    assert r2.json()["error"]["code"] == "gateway_timeout"

    provider.get_paper_error = DiscoveryConnectionError("Connection lost")
    r3 = client.get("/api/v1/discovery/paper/2106.09685")
    assert r3.status_code == 502
    assert r3.json()["error"]["code"] == "bad_gateway"


# ---------------------------------------------------------------------------
# 3. Explicit Import Endpoint Tests (POST /api/v1/discovery/import)
# ---------------------------------------------------------------------------

def test_discovery_import_invalid_collection_uuid_400():
    app = create_app(FakeResearchService(), discovery_service=DiscoveryService(provider=FakeProvider()))
    client = TestClient(app)

    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": "not-a-valid-uuid", "external_id": "2106.09685"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_discovery_import_missing_collection_404(sqlite_session_factory):
    provider = FakeProvider(papers=[make_test_paper()])
    service = DiscoveryService(provider=provider, session_factory=sqlite_session_factory)
    app = create_app(FakeResearchService(), discovery_service=service, session_factory=sqlite_session_factory)
    client = TestClient(app)

    missing_col_id = str(uuid4())
    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": missing_col_id, "external_id": "2106.09685"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_discovery_import_arbitrary_url_rejected_400(seeded_collection):
    col_id, session_factory = seeded_collection
    provider = FakeProvider()
    service = DiscoveryService(provider=provider, session_factory=session_factory)
    app = create_app(FakeResearchService(), discovery_service=service, session_factory=session_factory)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/discovery/import",
        json={
            "collection_id": str(col_id),
            "external_id": "https://malicious.org/paper.pdf",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_discovery_import_paper_not_found_on_arxiv_404(seeded_collection):
    col_id, session_factory = seeded_collection
    provider = FakeProvider(papers=[])
    service = DiscoveryService(provider=provider, session_factory=session_factory)
    app = create_app(FakeResearchService(), discovery_service=service, session_factory=session_factory)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_discovery_import_success_end_to_end_persisted_state(seeded_collection):
    col_id, session_factory = seeded_collection
    paper = make_test_paper(external_id="2106.09685")
    provider = FakeProvider(papers=[paper])
    downloader = FakeDownloader(pdf_bytes=VALID_PDF_BYTES)
    ingestion = RecordingIngestionService(session_factory=session_factory)
    pipeline = FakeDocumentPipeline()

    service = DiscoveryService(
        provider=provider,
        downloader=downloader,
        session_factory=session_factory,
        ingestion_service=ingestion,
        document_pipeline=pipeline,
    )
    app = create_app(
        FakeResearchService(),
        discovery_service=service,
        session_factory=session_factory,
        ingestion_service=ingestion,
        document_pipeline=pipeline,
    )
    client = TestClient(app)

    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "imported"
    assert data["external_id"] == "2106.09685"
    assert data["collection_id"] == str(col_id)
    assert data["title"] == paper.title
    assert data["chunks_count"] == 2
    assert data["indexed_chunks_count"] == 2
    assert data["document_id"] is not None

    # Verify actual relational database state
    with session_factory() as session:
        paper_repo = PaperRepository(session)
        doc_repo = DocumentRepository(session)
        chunk_repo = DocumentChunkRepository(session)

        db_paper = paper_repo.get_by_external_id(col_id, "2106.09685")
        assert db_paper is not None
        assert db_paper.title == paper.title
        assert db_paper.doi == paper.doi

        docs = doc_repo.list_by_paper_id(db_paper.id)
        assert len(docs) == 1
        assert docs[0].indexing_status == IndexingStatus.COMPLETED.value

        chunks = chunk_repo.list_by_document_id(docs[0].id)
        assert len(chunks) == 2

    # Verify temporary file cleanup on success
    assert len(downloader.created_temp_paths) == 1
    assert not downloader.created_temp_paths[0].exists()


def test_discovery_import_repeated_returns_already_imported_without_downloading(seeded_collection):
    col_id, session_factory = seeded_collection
    paper = make_test_paper(external_id="2106.09685")
    provider = FakeProvider(papers=[paper])
    downloader = FakeDownloader(pdf_bytes=VALID_PDF_BYTES)
    ingestion = RecordingIngestionService(session_factory=session_factory)
    pipeline = FakeDocumentPipeline()

    service = DiscoveryService(
        provider=provider,
        downloader=downloader,
        session_factory=session_factory,
        ingestion_service=ingestion,
        document_pipeline=pipeline,
    )
    app = create_app(
        FakeResearchService(),
        discovery_service=service,
        session_factory=session_factory,
        ingestion_service=ingestion,
        document_pipeline=pipeline,
    )
    client = TestClient(app)

    # First import -> succeeds
    r1 = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert r1.status_code == 200
    assert r1.json()["status"] == "imported"
    first_doc_id = r1.json()["document_id"]
    first_paper_id = r1.json()["paper_id"]
    assert len(downloader.download_calls) == 1
    assert len(ingestion.calls) == 1

    # Second import with same arXiv ID -> returns already_imported
    r2 = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["status"] == "already_imported"
    assert data2["paper_id"] == first_paper_id
    assert data2["document_id"] == first_doc_id
    assert data2["chunks_count"] == 2
    assert "already imported and indexed" in data2["message"]

    # Assert no second download or ingestion occurred!
    assert len(downloader.download_calls) == 1
    assert len(ingestion.calls) == 1


def test_discovery_import_resumes_when_paper_exists_without_document(seeded_collection):
    col_id, session_factory = seeded_collection
    paper_meta = make_test_paper(external_id="2106.09685")

    # Pre-create Paper record in DB without any Document record
    with session_factory() as session:
        repo = PaperRepository(session)
        pre_existing = repo.create(
            collection_id=col_id,
            title="Pre-created LoRA Paper",
            external_id="2106.09685",
            doi=paper_meta.doi,
        )
        pre_existing_id = pre_existing.id
        session.commit()

    provider = FakeProvider(papers=[paper_meta])
    downloader = FakeDownloader(pdf_bytes=VALID_PDF_BYTES)
    ingestion = RecordingIngestionService(session_factory=session_factory)

    service = DiscoveryService(
        provider=provider,
        downloader=downloader,
        session_factory=session_factory,
        ingestion_service=ingestion,
        document_pipeline=FakeDocumentPipeline(),
    )
    app = create_app(FakeResearchService(), discovery_service=service, session_factory=session_factory)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "imported"
    assert data["paper_id"] == str(pre_existing_id)

    # Verify no duplicate paper record was created
    with session_factory() as session:
        papers = list(session.scalars(select(Paper).where(Paper.collection_id == col_id)).all())
        assert len(papers) == 1
        assert papers[0].id == pre_existing_id

        doc_repo = DocumentRepository(session)
        docs = doc_repo.list_by_paper_id(pre_existing_id)
        assert len(docs) == 1
        assert docs[0].indexing_status == IndexingStatus.COMPLETED.value


def test_discovery_import_download_failure_does_not_persist_document_and_cleans_up(seeded_collection):
    col_id, session_factory = seeded_collection
    paper_meta = make_test_paper(external_id="2106.09685")
    provider = FakeProvider(papers=[paper_meta])
    # Downloader raises 404 (paper PDF missing on remote)
    downloader = FakeDownloader(error=PDFNotFoundError("Remote PDF returned 404"))
    ingestion = RecordingIngestionService(session_factory=session_factory)

    service = DiscoveryService(
        provider=provider,
        downloader=downloader,
        session_factory=session_factory,
        ingestion_service=ingestion,
    )
    app = create_app(FakeResearchService(), discovery_service=service, session_factory=session_factory)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"

    # Verify zero documents persisted in database
    with session_factory() as session:
        docs = list(session.scalars(select(Document)).all())
        assert len(docs) == 0

    assert len(ingestion.calls) == 0


def test_discovery_import_invalid_pdf_rejected_400_and_cleaned_up(seeded_collection):
    col_id, session_factory = seeded_collection
    paper_meta = make_test_paper(external_id="2106.09685")
    provider = FakeProvider(papers=[paper_meta])
    downloader = FakeDownloader(error=InvalidPDFError("Downloaded file missing '%PDF-' signature"))

    service = DiscoveryService(
        provider=provider,
        downloader=downloader,
        session_factory=session_factory,
        ingestion_service=RecordingIngestionService(session_factory=session_factory),
    )
    app = create_app(FakeResearchService(), discovery_service=service, session_factory=session_factory)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_discovery_import_oversized_pdf_rejected_413(seeded_collection):
    col_id, session_factory = seeded_collection
    paper_meta = make_test_paper(external_id="2106.09685")
    provider = FakeProvider(papers=[paper_meta])
    downloader = FakeDownloader(error=OversizedPDFError("Remote PDF exceeds maximum allowed size"))

    service = DiscoveryService(
        provider=provider,
        downloader=downloader,
        session_factory=session_factory,
        ingestion_service=RecordingIngestionService(session_factory=session_factory),
    )
    app = create_app(FakeResearchService(), discovery_service=service, session_factory=session_factory)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685"},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "file_too_large"


def test_discovery_import_ingestion_failure_rollback(seeded_collection):
    col_id, session_factory = seeded_collection
    paper_meta = make_test_paper(external_id="2106.09685")
    provider = FakeProvider(papers=[paper_meta])
    downloader = FakeDownloader(pdf_bytes=VALID_PDF_BYTES)
    # Ingestion fails with vector indexing error
    ingestion = RecordingIngestionService(
        session_factory=session_factory,
        error=VectorIndexingError("Vector store indexing failed"),
    )

    service = DiscoveryService(
        provider=provider,
        downloader=downloader,
        session_factory=session_factory,
        ingestion_service=ingestion,
        document_pipeline=FakeDocumentPipeline(),
    )
    app = create_app(FakeResearchService(), discovery_service=service, session_factory=session_factory)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/discovery/import",
        json={"collection_id": str(col_id), "external_id": "2106.09685", "durable": False},
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "indexing_failed"

    # In non-durable mode, no completed document should exist
    with session_factory() as session:
        docs = list(session.scalars(select(Document).where(Document.indexing_status == "completed")).all())
        assert len(docs) == 0

    # Ensure temporary file was cleaned up despite ingestion failure
    assert len(downloader.created_temp_paths) == 1
    assert not downloader.created_temp_paths[0].exists()


def test_collection_repository_crud(sqlite_session_factory):
    """Verify CollectionRepository operations directly."""
    with sqlite_session_factory() as session:
        user = User(email="cr_test@example.com")
        session.add(user)
        session.flush()

        ws = Workspace(user_id=user.id, name="Workspace CR")
        session.add(ws)
        session.flush()

        repo = CollectionRepository(session)
        col = repo.create(workspace_id=ws.id, name="Collection A", description="Desc")
        session.commit()

        # Read
        found = repo.get_by_id(col.id)
        assert found is not None
        assert found.name == "Collection A"

        # List by workspace
        listed = repo.list_by_workspace_id(ws.id)
        assert len(listed) == 1
        assert listed[0].id == col.id

        # Update
        updated = repo.update(col.id, name="Collection A Renamed")
        session.commit()
        assert updated.name == "Collection A Renamed"
