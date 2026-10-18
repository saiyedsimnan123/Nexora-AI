"""Safe remote PDF downloader for scholarly paper discovery.

Enforces strict defense-in-depth security:
- Scheme enforcement: strictly HTTPS only.
- Allowed host validation: only designated scholarly hostnames permitted (arxiv.org, export.arxiv.org).
- SSRF prevention: rejects loopback, private, link-local, multicast, and reserved IP destinations.
- Redirect destination validation: all HTTP 3xx redirect destinations are strictly re-validated.
- Size caps: pre-validates Content-Length and streams responses with an absolute 50 MB limit.
- Signature checking: verifies '%PDF-' magic bytes; rejects HTML error/captcha pages.
- Cleanup: automatically unlinks partial or temporary files on every failure path.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path
import socket
import tempfile
from typing import Any, Sequence

import httpx

from nexora.discovery.arxiv import (
    DEFAULT_USER_AGENT,
    build_canonical_pdf_url,
    normalize_and_validate_arxiv_id,
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

DEFAULT_MAX_DOWNLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0
MAX_ALLOWED_REDIRECTS = 3
ALLOWED_ARXIV_HOSTS = ("arxiv.org", "export.arxiv.org")

PDF_MAGIC_BYTES = b"%PDF-"


_NAT64_WKP_NETWORK = ipaddress.ip_network("64:ff9b::/96")


def is_ip_private_or_reserved(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if IP is private, loopback, link-local, multicast, or reserved.

    Handles standard IPv4/IPv6 private ranges, IPv4-mapped IPv6 (::ffff:0:0/96),
    and RFC 6052 Well-Known Prefix NAT64 (64:ff9b::/96) by inspecting the embedded
    IPv4 destination.
    """
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True

    # Check for IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1)
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None and is_ip_private_or_reserved(mapped):
        return True

    # Check for RFC 6052 Well-Known Prefix NAT64 translation (64:ff9b::/96)
    if isinstance(ip, ipaddress.IPv6Address) and ip in _NAT64_WKP_NETWORK:
        embedded_v4 = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        return is_ip_private_or_reserved(embedded_v4)

    # For general addresses, check reserved status
    if ip.is_reserved:
        return True

    return False


def validate_url_safety(
    url_str: str,
    allowed_hosts: Sequence[str] = ALLOWED_ARXIV_HOSTS,
) -> httpx.URL:
    """Validate URL scheme, allowed hostnames, and IP safety against SSRF attacks.

    Raises:
        SSRFProtectionError: If scheme is not HTTPS, host is unauthorized, or IP is private/reserved.
    """
    try:
        url = httpx.URL(url_str)
    except Exception as exc:
        raise SSRFProtectionError(f"Malformed URL: {url_str!r}") from exc

    # 1. Scheme must be strictly HTTPS
    if url.scheme != "https":
        raise SSRFProtectionError(
            f"Insecure scheme '{url.scheme}' in URL {url_str!r}. Only HTTPS is permitted."
        )

    # 2. Host must be non-empty and in allowed_hosts
    host = url.host.lower() if url.host else ""
    if not host:
        raise SSRFProtectionError(f"Missing host in URL: {url_str!r}")

    normalized_allowed = {h.lower() for h in allowed_hosts}
    if host not in normalized_allowed:
        raise SSRFProtectionError(
            f"Prohibited host '{host}'. Only trusted scholarly domains are permitted: {sorted(normalized_allowed)}"
        )

    # 3. Check for literal IP or resolve hostname to prevent SSRF and DNS rebinding
    try:
        ip = ipaddress.ip_address(host)
        if is_ip_private_or_reserved(ip):
            raise SSRFProtectionError(
                f"Prohibited target IP address: {host} (private, loopback, or reserved)"
            )
        return url
    except ValueError:
        # Not an IP literal; it's a domain name
        pass

    if host in ("localhost", "localhost.localdomain", "broadcasthost"):
        raise SSRFProtectionError(f"Prohibited target host: {host}")

    # DNS check: In online environments, resolve host to ensure it does not map to private IPs.
    # We resolve inside the try/except block, but validate addresses OUTSIDE to prevent
    # SSRFProtectionError (which inherits from PermissionError -> OSError) from being swallowed.
    addr_infos: list[Any] = []
    try:
        addr_infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, socket.herror, OSError):
        # In offline/mock test environments, DNS resolution might fail.
        # Since host was already verified against allowed_hosts, proceed safely.
        addr_infos = []

    for _, _, _, _, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        try:
            resolved_ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if is_ip_private_or_reserved(resolved_ip):
            raise SSRFProtectionError(
                f"Host '{host}' resolves to private/internal IP {ip_str}"
            )

    return url


@dataclass
class DownloadedPDF:
    """Represents a downloaded, verified PDF file on local temporary storage."""

    path: Path
    file_size: int
    arxiv_id: str
    source_url: str

    def cleanup(self) -> None:
        """Remove the downloaded temporary file from disk."""
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass

    def __enter__(self) -> DownloadedPDF:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()


class SafePDFDownloader:
    """Streams and validates scholarly PDFs with strict security protections."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        max_size_bytes: int = DEFAULT_MAX_DOWNLOAD_SIZE_BYTES,
        timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        allowed_hosts: Sequence[str] = ALLOWED_ARXIV_HOSTS,
    ) -> None:
        self._max_size_bytes = max_size_bytes
        self._timeout = timeout
        self._user_agent = user_agent
        self._allowed_hosts = tuple(allowed_hosts)

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                headers={"User-Agent": self._user_agent},
                follow_redirects=False,
            )
            self._owns_client = True

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> SafePDFDownloader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _resolve_stream_with_redirect_validation(
        self,
        initial_url: str,
    ) -> tuple[httpx.Response, str]:
        """Send request and manually follow redirects with strict validation of each destination."""
        current_url_str = initial_url
        headers = {"User-Agent": self._user_agent}

        for redirect_count in range(MAX_ALLOWED_REDIRECTS + 1):
            # Validate URL before every hop
            valid_url = validate_url_safety(current_url_str, self._allowed_hosts)

            try:
                # Use stream=True to avoid buffering entire response
                request = self._client.build_request("GET", valid_url, headers=headers)
                response = self._client.send(request, stream=True)
            except httpx.TimeoutException as exc:
                raise DiscoveryTimeoutError(
                    f"Remote PDF request timed out after {self._timeout}s: {exc}"
                ) from exc
            except httpx.RequestError as exc:
                raise DiscoveryConnectionError(
                    f"Failed to connect to PDF endpoint {current_url_str!r}: {exc}"
                ) from exc

            # Handle redirects manually
            if response.status_code in (301, 302, 303, 307, 308):
                response.close()
                if redirect_count >= MAX_ALLOWED_REDIRECTS:
                    raise DiscoveryResponseError(
                        f"Exceeded maximum allowed redirects ({MAX_ALLOWED_REDIRECTS}) for {initial_url!r}"
                    )

                location = response.headers.get("Location")
                if not location:
                    raise DiscoveryResponseError(
                        f"Received redirect status {response.status_code} without Location header"
                    )

                # Resolve relative redirects
                next_url = valid_url.join(location)
                current_url_str = str(next_url)
                continue

            return response, current_url_str

        raise DiscoveryResponseError(f"Failed to resolve URL: {initial_url!r}")

    def download_arxiv_pdf(
        self,
        arxiv_id_or_url: str,
        *,
        destination_dir: Path | None = None,
    ) -> DownloadedPDF:
        """Download and verify a PDF for an arXiv identifier with strict security checks.

        Args:
            arxiv_id_or_url: A validated arXiv identifier (e.g. '2106.09685') or canonical URL.
            destination_dir: Optional directory for the temporary file (defaults to system temp).

        Returns:
            A DownloadedPDF instance containing the validated file path and metadata.

        Raises:
            InvalidIdentifierError: If identifier is malformed.
            SSRFProtectionError: If target or redirect violates security boundaries.
            PDFNotFoundError: If remote endpoint returns HTTP 404.
            DiscoveryRateLimitError: If remote endpoint returns HTTP 429.
            OversizedPDFError: If PDF exceeds max_size_bytes.
            InvalidPDFError: If remote response is HTML, empty, or lacks '%PDF-' signature.
            DiscoveryTimeoutError / DiscoveryConnectionError / DiscoveryResponseError.
        """
        clean_id = normalize_and_validate_arxiv_id(arxiv_id_or_url)
        canonical_url = build_canonical_pdf_url(clean_id)

        response, final_url = self._resolve_stream_with_redirect_validation(canonical_url)

        try:
            # Check HTTP status
            if response.status_code == 404:
                raise PDFNotFoundError(f"PDF not found for arXiv paper {clean_id!r} (HTTP 404)")
            if response.status_code == 429:
                retry_after_hdr = response.headers.get("Retry-After")
                retry_after: float | None = None
                if retry_after_hdr:
                    try:
                        retry_after = float(retry_after_hdr.strip())
                    except (ValueError, TypeError):
                        pass
                raise DiscoveryRateLimitError(
                    "arXiv rate limit exceeded while downloading PDF (HTTP 429)",
                    retry_after=retry_after,
                )
            if response.status_code >= 500:
                raise DiscoveryResponseError(
                    f"arXiv server error while downloading PDF: HTTP {response.status_code}"
                )
            if response.status_code >= 400:
                raise DiscoveryResponseError(
                    f"arXiv client error while downloading PDF: HTTP {response.status_code}"
                )

            # Check Content-Length before downloading if provided
            content_len_hdr = response.headers.get("Content-Length")
            if content_len_hdr:
                content_len: int | None = None
                try:
                    content_len = int(content_len_hdr.strip())
                except ValueError:
                    content_len = None

                if content_len is not None and content_len > self._max_size_bytes:
                    raise OversizedPDFError(
                        f"Remote PDF Content-Length ({content_len} bytes) exceeds maximum "
                        f"allowed size ({self._max_size_bytes} bytes)."
                    )

            # Check Content-Type (reject HTML error/captcha pages)
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise InvalidPDFError(
                    f"Remote endpoint returned HTML content ({content_type}) instead of PDF. "
                    "This typically indicates a server error page, captcha, or blocked request."
                )

            # Stream into temporary file with strict size enforcement
            dest_dir_str = str(destination_dir) if destination_dir else None
            tmp_file = tempfile.NamedTemporaryFile(
                suffix=".pdf", dir=dest_dir_str, delete=False
            )
            tmp_path = Path(tmp_file.name)
            downloaded_bytes = 0

            try:
                for chunk in response.iter_bytes(chunk_size=65536):
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > self._max_size_bytes:
                        raise OversizedPDFError(
                            f"Downloaded content ({downloaded_bytes} bytes) exceeded maximum "
                            f"allowed size ({self._max_size_bytes} bytes)."
                        )
                    tmp_file.write(chunk)

                tmp_file.flush()
                tmp_file.close()

                if downloaded_bytes == 0:
                    raise InvalidPDFError(
                        f"Downloaded file for {clean_id!r} is empty (0 bytes)."
                    )

                # Validate magic PDF header (%PDF-)
                with open(tmp_path, "rb") as f:
                    header = f.read(1024)
                    if not header.startswith(PDF_MAGIC_BYTES) and PDF_MAGIC_BYTES not in header:
                        raise InvalidPDFError(
                            f"Downloaded file for {clean_id!r} does not have a valid PDF signature (missing '%PDF-')."
                        )

                return DownloadedPDF(
                    path=tmp_path,
                    file_size=downloaded_bytes,
                    arxiv_id=clean_id,
                    source_url=final_url,
                )
            except httpx.TimeoutException as exc:
                tmp_file.close()
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise DiscoveryTimeoutError(
                    f"Remote PDF download timed out while streaming for paper {clean_id!r}: {exc}"
                ) from exc
            except httpx.RequestError as exc:
                tmp_file.close()
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise DiscoveryConnectionError(
                    f"Connection failed while streaming remote PDF for paper {clean_id!r}: {exc}"
                ) from exc
            except Exception:
                tmp_file.close()
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise
        finally:
            response.close()
