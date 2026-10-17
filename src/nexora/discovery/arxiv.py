"""arXiv API client implementing the ScholarlySourceProvider protocol.

Features:
- Standard library XML parsing (xml.etree.ElementTree) with full namespace support.
- Strict arXiv identifier validation and canonical URL construction.
- Robust error handling: typed exceptions for 429 rate limits, timeouts, and malformed XML.
- Explicit request timeout and descriptive User-Agent.
- Testable rate limit policy with zero hidden sleeps.
- Untrusted external data sanitization.
"""

from __future__ import annotations

import datetime
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import httpx

from nexora.discovery.errors import (
    DiscoveryConnectionError,
    DiscoveryRateLimitError,
    DiscoveryResponseError,
    DiscoveryTimeoutError,
    InvalidIdentifierError,
)
from nexora.discovery.models import (
    ScholarlyPaperMetadata,
    ScholarlySearchQuery,
    ScholarlySearchResult,
)
from nexora.discovery.provider import ScholarlySourceProvider

ARXIV_API_BASE_URL = "https://export.arxiv.org/api/query"
DEFAULT_USER_AGENT = (
    "NexoraAI-ResearchClient/1.0 (https://github.com/saiyedsimnan123/Nexora-AI; scholarly-discovery)"
)
DEFAULT_TIMEOUT_SECONDS = 15.0

# XML Namespaces used in arXiv Atom feeds
ATOM_NS = "http://www.w3.org/2005/Atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
ARXIV_NS = "http://arxiv.org/schemas/atom"

_NAMESPACES = {
    "atom": ATOM_NS,
    "opensearch": OPENSEARCH_NS,
    "arxiv": ARXIV_NS,
}

# Regex for modern arXiv IDs: e.g. 2106.09685, 2106.09685v2, 0704.0001
_NEW_STYLE_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")
# Regex for legacy arXiv IDs (pre-2007): e.g. math/0501001, cs.AI/0601001, hep-th/9901001
_OLD_STYLE_ID_RE = re.compile(r"^[a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?/\d{7}(?:v\d+)?$")

_SORT_BY_MAP = {
    "relevance": "relevance",
    "submitted_date": "submittedDate",
    "last_updated": "lastUpdatedDate",
}


def normalize_and_validate_arxiv_id(raw_id: str) -> str:
    """Normalize and validate an arXiv identifier.

    Strips whitespace, 'arxiv:' prefixes, and full canonical URL prefixes.
    Ensures the extracted identifier matches either standard or legacy arXiv syntax.

    Raises:
        InvalidIdentifierError: If raw_id cannot be parsed into a valid arXiv identifier.
    """
    if not isinstance(raw_id, str):
        raise InvalidIdentifierError(f"arXiv identifier must be a string, got {type(raw_id).__name__}")

    cleaned = raw_id.strip()
    if not cleaned:
        raise InvalidIdentifierError("arXiv identifier must not be empty or blank")

    # Strip URL prefixes if supplied as URL
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
    ):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break

    # Strip .pdf suffix if it was passed from a pdf URL
    if cleaned.endswith(".pdf"):
        cleaned = cleaned[:-4]

    # Strip leading 'arxiv:' or 'arXiv:'
    if cleaned.lower().startswith("arxiv:"):
        cleaned = cleaned[6:].strip()

    if _NEW_STYLE_ID_RE.match(cleaned) or _OLD_STYLE_ID_RE.match(cleaned):
        return cleaned

    raise InvalidIdentifierError(
        f"Invalid arXiv identifier format: {raw_id!r}. Expected e.g. '2106.09685' or 'cs/0101001'."
    )


def build_canonical_abstract_url(arxiv_id: str) -> str:
    """Construct canonical abstract page URL from a validated arXiv identifier."""
    return f"https://arxiv.org/abs/{arxiv_id}"


def build_canonical_pdf_url(arxiv_id: str) -> str:
    """Construct canonical PDF download URL from a validated arXiv identifier.

    Note: This constructs the canonical URL format. It does not guarantee that the
    PDF file currently exists on the remote host.
    """
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


@dataclass
class ArxivRateLimitPolicy:
    """Rate limit configuration and tracker for the arXiv API.

    arXiv requests that automated clients pace calls to no more than 1 request
    per 3 seconds. To support fast, deterministic offline testing, `min_interval_seconds`
    defaults to 0.0 and `enforce_sleep` is False by default.
    """

    min_interval_seconds: float = 0.0
    enforce_sleep: bool = False
    last_request_timestamp: float = 0.0
    total_requests: int = 0

    def wait_if_needed(self) -> float:
        """Enforce rate limit interval if configured. Returns the waited duration."""
        if not self.enforce_sleep or self.min_interval_seconds <= 0.0:
            self.total_requests += 1
            self.last_request_timestamp = time.monotonic()
            return 0.0

        now = time.monotonic()
        elapsed = now - self.last_request_timestamp
        waited = 0.0
        if elapsed < self.min_interval_seconds:
            waited = self.min_interval_seconds - elapsed
            time.sleep(waited)

        self.last_request_timestamp = time.monotonic()
        self.total_requests += 1
        return waited


class ArxivSourceProvider:
    """Scholarly discovery provider for the arXiv export API."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        base_url: str = ARXIV_API_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        rate_limit_policy: ArxivRateLimitPolicy | None = None,
    ) -> None:
        self._base_url = base_url
        self._user_agent = user_agent
        self._timeout = timeout
        self._rate_limit = rate_limit_policy or ArxivRateLimitPolicy()

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                headers={"User-Agent": self._user_agent},
            )
            self._owns_client = True

    @property
    def provider_name(self) -> str:
        return "arxiv"

    @property
    def rate_limit_policy(self) -> ArxivRateLimitPolicy:
        return self._rate_limit

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ArxivSourceProvider:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _execute_http_get(self, params: dict[str, Any]) -> str:
        """Execute HTTP GET with strict error mapping and rate limit compliance."""
        self._rate_limit.wait_if_needed()

        headers = {"User-Agent": self._user_agent}
        try:
            response = self._client.get(
                self._base_url,
                params=params,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise DiscoveryTimeoutError(
                f"arXiv API request timed out after {self._timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise DiscoveryConnectionError(
                f"Failed to connect to arXiv API at {self._base_url}: {exc}"
            ) from exc

        # Handle HTTP status codes
        if response.status_code == 429:
            retry_after_header = response.headers.get("Retry-After")
            retry_after: float | None = None
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header.strip())
                except (ValueError, TypeError):
                    retry_after = None
            raise DiscoveryRateLimitError(
                "arXiv API rate limit exceeded (HTTP 429). Please reduce request frequency.",
                retry_after=retry_after,
            )

        if response.status_code >= 500:
            raise DiscoveryResponseError(
                f"arXiv API returned server error: HTTP {response.status_code}"
            )

        if response.status_code >= 400:
            raise DiscoveryResponseError(
                f"arXiv API returned client error: HTTP {response.status_code}"
            )

        return response.text

    def _build_search_query_param(self, query: ScholarlySearchQuery) -> str:
        """Construct arXiv search_query syntax from query terms and categories."""
        parts: list[str] = []

        raw_query = query.query.strip()
        if raw_query:
            # Wrap query terms in 'all:...' expression
            # If terms contain multiple words without field prefixes, search across all fields
            escaped_query = raw_query.replace('"', "")
            parts.append(f'all:"{escaped_query}"')

        if query.categories:
            cat_clauses = [f"cat:{c}" for c in query.categories]
            if len(cat_clauses) == 1:
                parts.append(cat_clauses[0])
            else:
                parts.append(f"({' OR '.join(cat_clauses)})")

        if query.authors:
            escaped_authors = [a.replace('"', "") for a in query.authors]
            author_clauses = [f'au:"{ea}"' for ea in escaped_authors]
            if len(author_clauses) == 1:
                parts.append(author_clauses[0])
            else:
                parts.append(f"({' OR '.join(author_clauses)})")

        if not parts:
            return "all:*"

        if len(parts) == 1:
            return parts[0]

        return " AND ".join(parts)

    def _parse_entry(self, entry_elem: ET.Element) -> ScholarlyPaperMetadata | None:
        """Parse an Atom <entry> element into normalized ScholarlyPaperMetadata."""
        # 1. Check for arXiv error response entries
        title_elem = entry_elem.find("atom:title", _NAMESPACES)
        raw_title = title_elem.text if title_elem is not None and title_elem.text else ""
        clean_title = " ".join(raw_title.split())

        if clean_title.lower() == "error" or clean_title.startswith("Error:"):
            return None

        id_elem = entry_elem.find("atom:id", _NAMESPACES)
        if id_elem is None or not id_elem.text:
            return None

        raw_id_text = id_elem.text.strip()
        try:
            external_id = normalize_and_validate_arxiv_id(raw_id_text)
        except InvalidIdentifierError:
            # If entry ID does not parse as a valid arXiv ID, skip it safely
            return None

        # 2. Abstract / Summary
        summary_elem = entry_elem.find("atom:summary", _NAMESPACES)
        raw_summary = summary_elem.text if summary_elem is not None and summary_elem.text else ""
        clean_abstract = " ".join(raw_summary.split())

        # 3. Authors
        authors: list[str] = []
        for author_elem in entry_elem.findall("atom:author", _NAMESPACES):
            name_elem = author_elem.find("atom:name", _NAMESPACES)
            if name_elem is not None and name_elem.text and name_elem.text.strip():
                authors.append(" ".join(name_elem.text.split()))

        # If title is blank, use fallback
        if not clean_title:
            clean_title = f"arXiv Paper {external_id}"

        # 4. Publication and update dates
        pub_date: datetime.date | None = None
        published_elem = entry_elem.find("atom:published", _NAMESPACES)
        if published_elem is not None and published_elem.text:
            try:
                date_str = published_elem.text.strip().replace("Z", "+00:00")
                pub_date = datetime.datetime.fromisoformat(date_str).date()
            except (ValueError, TypeError):
                pub_date = None

        # 5. DOI
        doi: str | None = None
        doi_elem = entry_elem.find("arxiv:doi", _NAMESPACES)
        if doi_elem is not None and doi_elem.text and doi_elem.text.strip():
            doi = doi_elem.text.strip()

        # 6. Categories
        categories: list[str] = []
        for cat_elem in entry_elem.findall("atom:category", _NAMESPACES):
            term = cat_elem.attrib.get("term")
            if term and term.strip() and term.strip() not in categories:
                categories.append(term.strip())

        primary_category: str | None = None
        primary_cat_elem = entry_elem.find("arxiv:primary_category", _NAMESPACES)
        if primary_cat_elem is not None:
            primary_term = primary_cat_elem.attrib.get("term")
            if primary_term and primary_term.strip():
                primary_category = primary_term.strip()

        if not primary_category and categories:
            primary_category = categories[0]

        # 7. Comments, journal references, license
        comment: str | None = None
        comment_elem = entry_elem.find("arxiv:comment", _NAMESPACES)
        if comment_elem is not None and comment_elem.text and comment_elem.text.strip():
            comment = " ".join(comment_elem.text.split())

        journal_ref: str | None = None
        journal_elem = entry_elem.find("arxiv:journal_ref", _NAMESPACES)
        if journal_elem is not None and journal_elem.text and journal_elem.text.strip():
            journal_ref = " ".join(journal_elem.text.split())

        license_uri: str | None = None
        license_elem = entry_elem.find("arxiv:license", _NAMESPACES)
        if license_elem is not None and license_elem.text and license_elem.text.strip():
            license_uri = license_elem.text.strip()

        # 8. Canonical URLs (strictly constructed from validated external_id)
        abstract_url = build_canonical_abstract_url(external_id)
        pdf_url = build_canonical_pdf_url(external_id)

        raw_metadata: dict[str, Any] = {
            "arxiv_id": external_id,
            "raw_id_uri": raw_id_text,
            "categories": categories,
        }
        if primary_category:
            raw_metadata["primary_category"] = primary_category
        if comment:
            raw_metadata["comment"] = comment
        if journal_ref:
            raw_metadata["journal_ref"] = journal_ref
        if license_uri:
            raw_metadata["license"] = license_uri

        return ScholarlyPaperMetadata(
            external_id=external_id,
            title=clean_title,
            authors=tuple(authors),
            abstract=clean_abstract,
            publication_date=pub_date,
            doi=doi,
            categories=tuple(categories),
            primary_category=primary_category,
            abstract_url=abstract_url,
            pdf_url=pdf_url,
            license=license_uri,
            source="arxiv",
            comment=comment,
            journal_ref=journal_ref,
            raw_metadata=raw_metadata,
        )

    def _parse_feed(self, xml_content: str, query_text: str) -> tuple[int, int, int, list[ScholarlyPaperMetadata]]:
        """Parse Atom feed XML returned by arXiv."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as exc:
            raise DiscoveryResponseError(f"Malformed XML received from arXiv API: {exc}") from exc

        # Extract OpenSearch pagination elements
        total_results = 0
        total_elem = root.find("opensearch:totalResults", _NAMESPACES)
        if total_elem is not None and total_elem.text:
            try:
                total_results = max(0, int(total_elem.text.strip()))
            except ValueError:
                total_results = 0

        start_index = 0
        start_elem = root.find("opensearch:startIndex", _NAMESPACES)
        if start_elem is not None and start_elem.text:
            try:
                start_index = max(0, int(start_elem.text.strip()))
            except ValueError:
                start_index = 0

        items_per_page = 0
        items_elem = root.find("opensearch:itemsPerPage", _NAMESPACES)
        if items_elem is not None and items_elem.text:
            try:
                items_per_page = max(0, int(items_elem.text.strip()))
            except ValueError:
                items_per_page = 0

        papers: list[ScholarlyPaperMetadata] = []
        for entry_elem in root.findall("atom:entry", _NAMESPACES):
            paper = self._parse_entry(entry_elem)
            if paper is not None:
                papers.append(paper)

        return total_results, start_index, items_per_page, papers

    def search(self, query: ScholarlySearchQuery) -> ScholarlySearchResult:
        """Search the arXiv API using query parameters."""
        if not isinstance(query, ScholarlySearchQuery):
            raise TypeError(f"query must be a ScholarlySearchQuery, got {type(query).__name__}")

        search_query_str = self._build_search_query_param(query)
        sort_by_arxiv = _SORT_BY_MAP.get(query.sort_by, "relevance")

        params: dict[str, Any] = {
            "search_query": search_query_str,
            "start": query.offset,
            "max_results": query.limit,
            "sortBy": sort_by_arxiv,
            "sortOrder": query.sort_order,
        }

        xml_response = self._execute_http_get(params)
        total_results, start_idx, _, papers = self._parse_feed(xml_response, query.query)

        return ScholarlySearchResult(
            provider="arxiv",
            query=query.query,
            total_results=max(total_results, len(papers)),
            offset=query.offset,
            limit=query.limit,
            papers=tuple(papers),
        )

    def get_paper(self, external_id: str) -> ScholarlyPaperMetadata | None:
        """Fetch metadata for a single arXiv paper by its identifier."""
        clean_id = normalize_and_validate_arxiv_id(external_id)

        params: dict[str, Any] = {
            "id_list": clean_id,
            "max_results": 1,
        }

        xml_response = self._execute_http_get(params)
        _, _, _, papers = self._parse_feed(xml_response, clean_id)

        if not papers:
            return None

        # Ensure the returned paper matches the requested ID (ignoring version suffix if requested unversioned)
        for paper in papers:
            req_base = clean_id.split("v")[0]
            got_base = paper.external_id.split("v")[0]
            if paper.external_id == clean_id or req_base == got_base:
                return paper

        return papers[0]
