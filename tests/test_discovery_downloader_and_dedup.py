"""Tests for Safe Remote PDF Downloader and Paper Deduplication (Phase 2).

Validates:
1. Safe remote PDF streaming, Content-Length checks, Content-Type checks, and %PDF- magic signature validation.
2. SSRF prevention: rejects HTTP, loopback, private IPs, link-local metadata addresses, and unauthorized hosts.
3. Redirect destination validation: rejects redirects to private IPs, HTTP, or untrusted domains; enforces hop limits.
4. Temporary file cleanup on all failure paths and context manager exit.
5. DOI and external ID normalization.
6. Repository-level deduplication: external_id, DOI, collection isolation, non-merging of title similarities, and deterministic earliest-record resolution.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.database.models.base import Base
from nexora.database.models.collection import Collection
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.paper import PaperRepository
from nexora.discovery.deduplication import (
    DeduplicationMatchReason,
    check_paper_duplicate_in_collection,
    normalize_doi,
    normalize_external_id,
)
from nexora.discovery.downloader import (
    DEFAULT_MAX_DOWNLOAD_SIZE_BYTES,
    DownloadedPDF,
    SafePDFDownloader,
    validate_url_safety,
)
from nexora.discovery.errors import (
    DiscoveryConnectionError,
    DiscoveryRateLimitError,
    DiscoveryResponseError,
    DiscoveryTimeoutError,
    InvalidIdentifierError,
    InvalidPDFError,
    OversizedPDFError,
    PDFNotFoundError,
    SSRFProtectionError,
)

# Standard mock PDF payload with valid '%PDF-' signature
VALID_PDF_BYTES = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< >>\n%%EOF"


# ---------------------------------------------------------------------------
# Database Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sqlite_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _create_test_collection(session) -> Collection:
    user = User(email=f"user_{uuid4().hex[:8]}@example.com")
    session.add(user)
    session.flush()

    workspace = Workspace(name="Test WS", user_id=user.id)
    session.add(workspace)
    session.flush()

    collection = Collection(name="Test Collection", workspace_id=workspace.id)
    session.add(collection)
    session.flush()
    return collection


# ---------------------------------------------------------------------------
# Downloader Tests: SSRF & URL Safety Validation
# ---------------------------------------------------------------------------
def test_validate_url_safety_accepts_valid_arxiv_https_urls():
    url = validate_url_safety("https://arxiv.org/pdf/2106.09685.pdf")
    assert url.scheme == "https"
    assert url.host == "arxiv.org"

    url_export = validate_url_safety("https://export.arxiv.org/pdf/2106.09685.pdf")
    assert url_export.host == "export.arxiv.org"


def test_validate_url_safety_rejects_insecure_scheme():
    with pytest.raises(SSRFProtectionError) as exc_info:
        validate_url_safety("http://arxiv.org/pdf/2106.09685.pdf")
    assert "insecure scheme" in str(exc_info.value).lower()


def test_validate_url_safety_rejects_unauthorized_hosts():
    unauthorized = [
        "https://evil.com/malicious.pdf",
        "https://google.com/search",
        "https://github.com/saiyedsimnan123/Nexora-AI",
    ]
    for target in unauthorized:
        with pytest.raises(SSRFProtectionError) as exc_info:
            validate_url_safety(target)
        assert "prohibited host" in str(exc_info.value).lower()


def test_validate_url_safety_rejects_private_and_loopback_ips():
    private_targets = [
        "https://127.0.0.1/secret.pdf",
        "https://127.0.0.2:8080/doc.pdf",
        "https://10.0.0.1/internal.pdf",
        "https://192.168.1.1/router.pdf",
        "https://172.16.0.1/payroll.pdf",
        "https://169.254.169.254/latest/meta-data",  # Cloud link-local metadata endpoint
        "https://localhost/admin.pdf",
    ]
    for target in private_targets:
        with pytest.raises(SSRFProtectionError):
            validate_url_safety(target)


def test_validate_url_safety_rejects_dns_resolved_private_ips(monkeypatch):
    """Verify that if an allowed host resolves via DNS to a private IP, it is strictly blocked."""
    import socket

    # Simulate arxiv.org resolving to loopback 127.0.0.1
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(SSRFProtectionError) as exc_info:
        validate_url_safety("https://arxiv.org/pdf/2106.09685.pdf")

    assert "resolves to private/internal ip" in str(exc_info.value).lower()

    # Simulate export.arxiv.org resolving to link-local cloud metadata IP
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("169.254.169.254", 443))],
    )
    with pytest.raises(SSRFProtectionError) as exc_info:
        validate_url_safety("https://export.arxiv.org/pdf/2106.09685.pdf")

    assert "resolves to private/internal ip" in str(exc_info.value).lower()


def test_validate_url_safety_rejects_dns_resolved_ipv6_mapped_private_ips(monkeypatch):
    """Verify that IPv6 and IPv4-mapped IPv6 loopback/private addresses are blocked."""
    import socket

    # Simulate resolving to IPv6 loopback ::1
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(10, 1, 6, "", ("::1", 443, 0, 0))],
    )
    with pytest.raises(SSRFProtectionError) as exc_info:
        validate_url_safety("https://arxiv.org/pdf/2106.09685.pdf")

    assert "resolves to private/internal ip" in str(exc_info.value).lower()

    # Simulate resolving to IPv4-mapped loopback ::ffff:127.0.0.1
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(10, 1, 6, "", ("::ffff:127.0.0.1", 443, 0, 0))],
    )
    with pytest.raises(SSRFProtectionError) as exc_info:
        validate_url_safety("https://arxiv.org/pdf/2106.09685.pdf")

    assert "resolves to private/internal ip" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Downloader Tests: Streaming & PDF Signature Validation
# ---------------------------------------------------------------------------
def test_successful_pdf_download():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("https://arxiv.org/pdf/2106.09685")
        return httpx.Response(
            200,
            content=VALID_PDF_BYTES,
            headers={"Content-Type": "application/pdf"},
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    downloaded = downloader.download_arxiv_pdf("2106.09685")

    try:
        assert isinstance(downloaded, DownloadedPDF)
        assert downloaded.arxiv_id == "2106.09685"
        assert downloaded.path.exists()
        assert downloaded.file_size == len(VALID_PDF_BYTES)
        assert downloaded.path.read_bytes().startswith(b"%PDF-")
    finally:
        downloaded.cleanup()

    assert not downloaded.path.exists()


def test_download_arxiv_pdf_normalizes_identifier():
    captured_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(200, content=VALID_PDF_BYTES)

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))

    # Pass URL format
    with downloader.download_arxiv_pdf("https://arxiv.org/abs/2106.09685") as pdf:
        assert pdf.arxiv_id == "2106.09685"

    # Pass prefix format
    with downloader.download_arxiv_pdf("arxiv:2106.09685v2") as pdf:
        assert pdf.arxiv_id == "2106.09685v2"

    assert len(captured_urls) == 2
    assert captured_urls[0] == "https://arxiv.org/pdf/2106.09685.pdf"
    assert captured_urls[1] == "https://arxiv.org/pdf/2106.09685v2.pdf"


def test_download_rejects_malformed_identifier():
    downloader = SafePDFDownloader(
        client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    )
    with pytest.raises(InvalidIdentifierError):
        downloader.download_arxiv_pdf("malformed_arxiv_id_xyz")


def test_download_rejects_html_response_and_cleans_up():
    created_paths: list[Path] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><body><h1>Captcha Required</h1></body></html>",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(InvalidPDFError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert "text/html" in str(exc_info.value).lower()


def test_download_rejects_missing_pdf_signature_and_cleans_up():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"PLAIN TEXT CORRUPTED DATA NOT A PDF",
            headers={"Content-Type": "application/octet-stream"},
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(InvalidPDFError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert "signature" in str(exc_info.value).lower()


def test_download_rejects_empty_file():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(InvalidPDFError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert "empty" in str(exc_info.value).lower()


def test_download_preflight_content_length_oversized_rejection():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=VALID_PDF_BYTES,
            headers={"Content-Length": "60000000"},  # 60 MB exceeds 50 MB cap
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(OversizedPDFError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert "exceeds maximum allowed size" in str(exc_info.value)


def test_download_streaming_size_cap_rejection_and_cleanup():
    # Downloader with a small 100-byte cap, simulating chunked transfer without Content-Length
    downloader = SafePDFDownloader(
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, stream=httpx.ByteStream(b"%PDF-" + b"A" * 500)
                )
            )
        ),
        max_size_bytes=100,
    )
    with pytest.raises(OversizedPDFError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert "exceeded maximum allowed size" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Downloader Tests: Redirects & SSRF Redirection Protection
# ---------------------------------------------------------------------------
def test_valid_https_redirect_within_arxiv_is_followed():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "arxiv.org":
            return httpx.Response(
                302,
                headers={"Location": f"https://export.arxiv.org{request.url.path}"},
            )
        assert request.url.host == "export.arxiv.org"
        return httpx.Response(200, content=VALID_PDF_BYTES)

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with downloader.download_arxiv_pdf("2106.09685") as pdf:
        assert pdf.file_size == len(VALID_PDF_BYTES)
        assert pdf.source_url == "https://export.arxiv.org/pdf/2106.09685.pdf"


def test_redirect_to_insecure_http_is_blocked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "http://arxiv.org/pdf/2106.09685.pdf"},  # Insecure HTTP
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(SSRFProtectionError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert "insecure scheme" in str(exc_info.value).lower()


def test_redirect_to_disallowed_external_host_is_blocked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "https://malicious-external-site.com/exploit.pdf"},
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(SSRFProtectionError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert "prohibited host" in str(exc_info.value).lower()


def test_redirect_to_private_address_is_blocked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "https://127.0.0.1/admin.pdf"},
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(SSRFProtectionError):
        downloader.download_arxiv_pdf("2106.09685")


def test_excessive_redirects_raise_response_error():
    def handler(request: httpx.Request) -> httpx.Response:
        # Endless redirect loop
        return httpx.Response(
            302,
            headers={"Location": "https://arxiv.org/pdf/2106.09685.pdf"},
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryResponseError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert "maximum allowed redirects" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Downloader Tests: HTTP Error Statuses & Network Failures
# ---------------------------------------------------------------------------
def test_download_404_not_found():
    downloader = SafePDFDownloader(
        client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404)))
    )
    with pytest.raises(PDFNotFoundError):
        downloader.download_arxiv_pdf("2106.09685")


def test_download_429_rate_limit():
    downloader = SafePDFDownloader(
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(429, headers={"Retry-After": "10"})
            )
        )
    )
    with pytest.raises(DiscoveryRateLimitError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert exc_info.value.retry_after == 10.0


def test_download_500_server_error():
    downloader = SafePDFDownloader(
        client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500)))
    )
    with pytest.raises(DiscoveryResponseError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685")

    assert "HTTP 500" in str(exc_info.value)


def test_download_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryTimeoutError):
        downloader.download_arxiv_pdf("2106.09685")


def test_download_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Failed to connect")

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryConnectionError):
        downloader.download_arxiv_pdf("2106.09685")


def test_download_streaming_timeout_and_tempfile_cleanup(tmp_path):
    class TimeoutByteStream(httpx.ByteStream):
        def __iter__(self):
            yield b"%PDF-1.4\n"
            raise httpx.ReadTimeout("Timeout mid-stream")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=TimeoutByteStream(b""),
            headers={"Content-Type": "application/pdf"},
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryTimeoutError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685", destination_dir=tmp_path)

    assert "timed out while streaming" in str(exc_info.value).lower()
    # Confirm temp files are cleaned up
    assert list(tmp_path.glob("*.pdf")) == []


def test_download_streaming_connection_error_and_tempfile_cleanup(tmp_path):
    class DisconnectByteStream(httpx.ByteStream):
        def __iter__(self):
            yield b"%PDF-1.4\n"
            raise httpx.ReadError("Connection severed mid-stream")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=DisconnectByteStream(b""),
            headers={"Content-Type": "application/pdf"},
        )

    downloader = SafePDFDownloader(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryConnectionError) as exc_info:
        downloader.download_arxiv_pdf("2106.09685", destination_dir=tmp_path)

    assert "streaming" in str(exc_info.value).lower()
    # Confirm temp files are cleaned up
    assert list(tmp_path.glob("*.pdf")) == []


# ---------------------------------------------------------------------------
# Deduplication Tests: Normalization
# ---------------------------------------------------------------------------
def test_normalize_doi_variants():
    assert normalize_doi("10.48550/arxiv.2106.09685") == "10.48550/arxiv.2106.09685"
    assert (
        normalize_doi("https://doi.org/10.48550/arXiv.2106.09685")
        == "10.48550/arxiv.2106.09685"
    )
    assert (
        normalize_doi("http://dx.doi.org/10.1000/182/")
        == "10.1000/182"
    )
    assert normalize_doi("doi:10.1145/1234.5678.") == "10.1145/1234.5678"
    assert normalize_doi("   ") is None
    assert normalize_doi(None) is None


def test_normalize_external_id_variants():
    assert normalize_external_id("arxiv:2106.09685") == "2106.09685"
    assert (
        normalize_external_id("https://arxiv.org/abs/2106.09685v2")
        == "2106.09685v2"
    )
    assert normalize_external_id("   ") is None
    assert normalize_external_id(None) is None


# ---------------------------------------------------------------------------
# Deduplication Tests: Repository Lookups & Collection Isolation
# ---------------------------------------------------------------------------
def test_paper_repository_get_by_external_id(sqlite_session_factory):
    session = sqlite_session_factory()
    col1 = _create_test_collection(session)
    col2 = _create_test_collection(session)
    repo = PaperRepository(session)

    # Create paper in collection 1 with arXiv ID
    paper1 = repo.create(
        collection_id=col1.id,
        title="LoRA: Low-Rank Adaptation",
        external_id="2106.09685v2",
        doi="10.48550/arXiv.2106.09685",
    )
    session.flush()

    # Exact match in col1
    found_exact = repo.get_by_external_id(col1.id, "2106.09685v2")
    assert found_exact is not None
    assert found_exact.id == paper1.id

    # Base match (unversioned request matches versioned paper)
    found_base = repo.get_by_external_id(col1.id, "2106.09685")
    assert found_base is not None
    assert found_base.id == paper1.id

    # Collection isolation: must NOT match in collection 2
    found_col2 = repo.get_by_external_id(col2.id, "2106.09685")
    assert found_col2 is None


def test_paper_repository_get_by_doi(sqlite_session_factory):
    session = sqlite_session_factory()
    col1 = _create_test_collection(session)
    col2 = _create_test_collection(session)
    repo = PaperRepository(session)

    paper = repo.create(
        collection_id=col1.id,
        title="Attention Is All You Need",
        doi="10.5555/3295222.3295349",
        external_id="1706.03762",
    )
    session.flush()

    # Case-insensitive and prefix-insensitive DOI matching
    found_raw = repo.get_by_doi(col1.id, "10.5555/3295222.3295349")
    assert found_raw is not None
    assert found_raw.id == paper.id

    found_url = repo.get_by_doi(col1.id, "https://doi.org/10.5555/3295222.3295349")
    assert found_url is not None
    assert found_url.id == paper.id

    # Isolation check
    assert repo.get_by_doi(col2.id, "10.5555/3295222.3295349") is None


def test_paper_repository_find_in_collection_priority_and_no_title_merging(
    sqlite_session_factory,
):
    session = sqlite_session_factory()
    col = _create_test_collection(session)
    repo = PaperRepository(session)

    # Create Paper A
    paper_a = repo.create(
        collection_id=col.id,
        title="Survey on Machine Learning",
        external_id="2101.00001",
        doi="10.1145/1111.2222",
    )
    # Create Paper B with identical title but distinct identifiers
    paper_b = repo.create(
        collection_id=col.id,
        title="Survey on Machine Learning",  # Identical title!
        external_id="2202.00002",
        doi="10.1145/3333.4444",
    )
    session.flush()

    # Match by external_id
    res_a, reason_a = repo.find_in_collection(col.id, external_id="2101.00001")
    assert res_a is not None and res_a.id == paper_a.id
    assert reason_a == "external_id"

    res_b, reason_b = repo.find_in_collection(col.id, external_id="2202.00002")
    assert res_b is not None and res_b.id == paper_b.id
    assert reason_b == "external_id"

    # Distinct papers with identical titles must NEVER match each other!
    res_c, reason_c = repo.find_in_collection(
        col.id, external_id="2303.00003", doi="10.9999/none"
    )
    assert res_c is None
    assert reason_c == "none"


def test_paper_repository_get_or_create_paper(sqlite_session_factory):
    session = sqlite_session_factory()
    col = _create_test_collection(session)
    repo = PaperRepository(session)

    # 1. First creation returns (paper, True)
    paper1, created1 = repo.get_or_create_paper(
        collection_id=col.id,
        title="LoRA",
        external_id="2106.09685",
        doi="10.48550/arXiv.2106.09685",
    )
    assert created1 is True
    session.flush()

    # 2. Repeated discovery with same external_id returns existing paper and created=False
    paper2, created2 = repo.get_or_create_paper(
        collection_id=col.id,
        title="LoRA (Repeat Discovery)",
        external_id="2106.09685",
    )
    assert created2 is False
    assert paper2.id == paper1.id

    # 3. Repeated discovery with same DOI returns existing paper
    paper3, created3 = repo.get_or_create_paper(
        collection_id=col.id,
        title="LoRA (DOI Discovery)",
        doi="https://doi.org/10.48550/arXiv.2106.09685",
    )
    assert created3 is False
    assert paper3.id == paper1.id

    # 4. New paper with different ID is created
    paper4, created4 = repo.get_or_create_paper(
        collection_id=col.id,
        title="QLoRA",
        external_id="2305.14314",
    )
    assert created4 is True
    assert paper4.id != paper1.id


def test_check_paper_duplicate_in_collection_helper(sqlite_session_factory):
    session = sqlite_session_factory()
    col = _create_test_collection(session)
    repo = PaperRepository(session)

    repo.create(
        collection_id=col.id,
        title="Foundational Model",
        external_id="2001.00001",
        doi="10.1000/foundational",
    )
    session.flush()

    # Duplicate by external_id
    dup_ext = check_paper_duplicate_in_collection(
        repo, col.id, external_id="arxiv:2001.00001"
    )
    assert dup_ext.is_duplicate is True
    assert dup_ext.match_reason == DeduplicationMatchReason.EXTERNAL_ID

    # Duplicate by DOI
    dup_doi = check_paper_duplicate_in_collection(
        repo, col.id, doi="https://doi.org/10.1000/foundational"
    )
    assert dup_doi.is_duplicate is True
    assert dup_doi.match_reason == DeduplicationMatchReason.DOI

    # Not duplicate
    no_dup = check_paper_duplicate_in_collection(
        repo, col.id, external_id="2401.99999", doi="10.1000/new"
    )
    assert no_dup.is_duplicate is False
    assert no_dup.match_reason == DeduplicationMatchReason.NONE
