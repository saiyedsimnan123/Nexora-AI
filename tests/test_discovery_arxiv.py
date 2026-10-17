"""Deterministic offline tests for Scholarly Discovery Models, Provider Protocol, and arXiv Client."""

from __future__ import annotations

import datetime
from typing import Any

import httpx
import pytest

from nexora.discovery.arxiv import (
    ArxivRateLimitPolicy,
    ArxivSourceProvider,
    build_canonical_abstract_url,
    build_canonical_pdf_url,
    normalize_and_validate_arxiv_id,
)
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


# ---------------------------------------------------------------------------
# Test Fixtures & Sample Feeds
# ---------------------------------------------------------------------------
SAMPLE_ARXIV_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <link href="http://arxiv.org/api/query?search_query=all:lora&amp;start=0&amp;max_results=2" rel="self" type="application/atom+xml"/>
  <title type="html">ArXiv Query: search_query=all:lora</title>
  <updated>2026-10-18T00:00:00Z</updated>
  <opensearch:totalResults>142</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>2</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2106.09685v2</id>
    <updated>2021-10-16T04:19:42Z</updated>
    <published>2021-06-17T17:59:16Z</published>
    <title>  LoRA: Low-Rank Adaptation of
    Large Language Models  </title>
    <summary>  An important paradigm of NLP consists of large-scale pre-training.
    We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained weights.  </summary>
    <author>
      <name>  Edward J. Hu  </name>
    </author>
    <author>
      <name>Yelong Shen</name>
    </author>
    <arxiv:doi>10.48550/arXiv.2106.09685</arxiv:doi>
    <arxiv:comment>14 pages, 7 figures</arxiv:comment>
    <arxiv:journal_ref>ICLR 2022</arxiv:journal_ref>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:license>http://creativecommons.org/licenses/by/4.0/</arxiv:license>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2305.14314v1</id>
    <updated>2023-05-23T17:59:00Z</updated>
    <published>2023-05-23T17:59:00Z</published>
    <title>QLoRA: Efficient Finetuning of Quantized LLMs</title>
    <summary>We present QLoRA, an efficient finetuning approach.</summary>
    <author>
      <name>Tim Dettmers</name>
    </author>
    <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

EMPTY_ARXIV_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>0</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>10</opensearch:itemsPerPage>
</feed>
"""

ERROR_ENTRY_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/api/errors#invalid_id</id>
    <title>Error</title>
    <summary>incorrect id format for 9999.invalid</summary>
  </entry>
</feed>
"""


# ---------------------------------------------------------------------------
# Identifier Normalization & Validation Tests
# ---------------------------------------------------------------------------
def test_normalize_and_validate_modern_arxiv_ids():
    assert normalize_and_validate_arxiv_id("2106.09685") == "2106.09685"
    assert normalize_and_validate_arxiv_id("2106.09685v2") == "2106.09685v2"
    assert normalize_and_validate_arxiv_id("0704.0001") == "0704.0001"
    assert normalize_and_validate_arxiv_id("2403.12345") == "2403.12345"


def test_normalize_and_validate_legacy_arxiv_ids():
    assert normalize_and_validate_arxiv_id("math/0501001") == "math/0501001"
    assert normalize_and_validate_arxiv_id("math/0501001v3") == "math/0501001v3"
    assert normalize_and_validate_arxiv_id("cs.AI/0601001") == "cs.AI/0601001"
    assert normalize_and_validate_arxiv_id("hep-th/9901001") == "hep-th/9901001"


def test_normalize_and_validate_prefixes_and_urls():
    assert normalize_and_validate_arxiv_id("arxiv:2106.09685") == "2106.09685"
    assert normalize_and_validate_arxiv_id("arXiv:2106.09685v1") == "2106.09685v1"
    assert normalize_and_validate_arxiv_id("https://arxiv.org/abs/2106.09685") == "2106.09685"
    assert normalize_and_validate_arxiv_id("http://arxiv.org/abs/2106.09685v2") == "2106.09685v2"
    assert normalize_and_validate_arxiv_id("https://arxiv.org/pdf/2106.09685.pdf") == "2106.09685"
    assert normalize_and_validate_arxiv_id("  2106.09685 \n ") == "2106.09685"


def test_invalid_arxiv_ids_raise_typed_error():
    invalid_examples = [
        "",
        "   ",
        "invalid_id",
        "1234",
        "2106.09",
        "https://evil.com/malicious.pdf",
        "99999.999999",
    ]
    for inv in invalid_examples:
        with pytest.raises(InvalidIdentifierError):
            normalize_and_validate_arxiv_id(inv)


def test_canonical_urls_construction():
    aid = "2106.09685"
    assert build_canonical_abstract_url(aid) == "https://arxiv.org/abs/2106.09685"
    assert build_canonical_pdf_url(aid) == "https://arxiv.org/pdf/2106.09685.pdf"


# ---------------------------------------------------------------------------
# Models & Validation Tests
# ---------------------------------------------------------------------------
def test_scholarly_paper_metadata_validation_and_serialization():
    meta = ScholarlyPaperMetadata(
        external_id="2106.09685",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        authors=("Edward J. Hu", "Yelong Shen"),
        abstract="Freezes pre-trained weights and injects trainable matrices.",
        publication_date=datetime.date(2021, 6, 17),
        doi="10.48550/arXiv.2106.09685",
        categories=("cs.CL", "cs.AI"),
        primary_category="cs.CL",
        abstract_url="https://arxiv.org/abs/2106.09685",
        pdf_url="https://arxiv.org/pdf/2106.09685.pdf",
        license="http://creativecommons.org/licenses/by/4.0/",
        source="arxiv",
        comment="14 pages",
        journal_ref="ICLR 2022",
    )
    d = meta.to_dict()
    assert d["external_id"] == "2106.09685"
    assert d["publication_date"] == "2021-06-17"
    assert d["authors"] == ["Edward J. Hu", "Yelong Shen"]
    assert d["source"] == "arxiv"

    # Post-init rejects empty external_id or title
    with pytest.raises(TypeError):
        ScholarlyPaperMetadata(
            external_id="",
            title="Title",
            authors=("A",),
            abstract="Abs",
        )


def test_scholarly_search_query_validation():
    # Valid query
    q = ScholarlySearchQuery(query="LoRA", categories=("cs.CL",), limit=25, offset=5)
    assert q.query == "LoRA"
    assert q.limit == 25
    assert q.offset == 5

    # Invalid limits
    with pytest.raises(ValueError):
        ScholarlySearchQuery(query="LoRA", limit=0)
    with pytest.raises(ValueError):
        ScholarlySearchQuery(query="LoRA", limit=150)

    # Empty query without categories or authors
    with pytest.raises(ValueError):
        ScholarlySearchQuery(query="   ")

    # Valid empty query if category supplied
    q_cat = ScholarlySearchQuery(query="", categories=("cs.AI",))
    assert q_cat.categories == ("cs.AI",)


def test_scholarly_search_result_serialization():
    res = ScholarlySearchResult(
        provider="arxiv",
        query="LoRA",
        total_results=42,
        offset=0,
        limit=10,
        papers=(),
    )
    data = res.to_dict()
    assert data["provider"] == "arxiv"
    assert data["total_results"] == 42
    assert data["papers"] == []


# ---------------------------------------------------------------------------
# arXiv Client: Search & Pagination Tests (MockTransport)
# ---------------------------------------------------------------------------
def test_arxiv_provider_implements_protocol():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=EMPTY_ARXIV_FEED_XML))
    with httpx.Client(transport=transport) as client:
        provider = ArxivSourceProvider(client=client)
        assert isinstance(provider, ScholarlySourceProvider)
        assert provider.provider_name == "arxiv"


def test_arxiv_search_parses_feed_with_namespaces():
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, text=SAMPLE_ARXIV_FEED_XML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ArxivSourceProvider(client=client)

    query = ScholarlySearchQuery(
        query="lora",
        categories=("cs.CL",),
        limit=2,
        offset=0,
    )
    result = provider.search(query)

    assert result.provider == "arxiv"
    assert result.total_results == 142
    assert len(result.papers) == 2

    # Check paper 1
    p1 = result.papers[0]
    assert p1.external_id == "2106.09685v2"
    assert "LoRA: Low-Rank Adaptation of Large Language Models" in p1.title
    assert "Edward J. Hu" in p1.authors
    assert "Yelong Shen" in p1.authors
    assert p1.publication_date == datetime.date(2021, 6, 17)
    assert p1.doi == "10.48550/arXiv.2106.09685"
    assert p1.primary_category == "cs.CL"
    assert "cs.AI" in p1.categories
    assert p1.abstract_url == "https://arxiv.org/abs/2106.09685v2"
    assert p1.pdf_url == "https://arxiv.org/pdf/2106.09685v2.pdf"
    assert p1.license == "http://creativecommons.org/licenses/by/4.0/"
    assert p1.journal_ref == "ICLR 2022"

    # Check paper 2
    p2 = result.papers[1]
    assert p2.external_id == "2305.14314v1"
    assert p2.title == "QLoRA: Efficient Finetuning of Quantized LLMs"
    assert p2.authors == ("Tim Dettmers",)
    assert p2.doi is None  # Handled missing optional DOI cleanly
    assert p2.primary_category == "cs.LG"

    # Verify query params passed to arXiv
    assert len(captured_requests) == 1
    req_params = dict(captured_requests[0].url.params)
    assert "all:\"lora\"" in req_params["search_query"]
    assert "cat:cs.CL" in req_params["search_query"]
    assert req_params["max_results"] == "2"
    assert req_params["start"] == "0"


def test_arxiv_search_empty_feed():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=EMPTY_ARXIV_FEED_XML))
    provider = ArxivSourceProvider(client=httpx.Client(transport=transport))

    result = provider.search(ScholarlySearchQuery(query="nonexistent_concept_xyz_123"))
    assert result.total_results == 0
    assert len(result.papers) == 0


def test_arxiv_search_category_and_author_filtering():
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, text=EMPTY_ARXIV_FEED_XML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ArxivSourceProvider(client=client)

    query = ScholarlySearchQuery(
        query="transformers",
        categories=("cs.AI", "cs.LG"),
        authors=("Vaswani",),
        limit=10,
        sort_by="submitted_date",
        sort_order="descending",
    )
    provider.search(query)

    params = dict(captured_requests[0].url.params)
    sq = params["search_query"]
    assert 'all:"transformers"' in sq
    assert "(cat:cs.AI OR cat:cs.LG)" in sq
    assert 'au:"Vaswani"' in sq
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"


# ---------------------------------------------------------------------------
# arXiv Client: Get Paper by Identifier Tests
# ---------------------------------------------------------------------------
def test_arxiv_get_paper_success():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=SAMPLE_ARXIV_FEED_XML))
    provider = ArxivSourceProvider(client=httpx.Client(transport=transport))

    paper = provider.get_paper("2106.09685")
    assert paper is not None
    assert paper.external_id.startswith("2106.09685")
    assert "LoRA" in paper.title


def test_arxiv_get_paper_not_found():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=EMPTY_ARXIV_FEED_XML))
    provider = ArxivSourceProvider(client=httpx.Client(transport=transport))

    paper = provider.get_paper("0000.00000")
    assert paper is None


def test_arxiv_get_paper_error_entry():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=ERROR_ENTRY_FEED_XML))
    provider = ArxivSourceProvider(client=httpx.Client(transport=transport))

    paper = provider.get_paper("9999.9999")
    assert paper is None


def test_arxiv_get_paper_rejects_malformed_identifier():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=SAMPLE_ARXIV_FEED_XML))
    provider = ArxivSourceProvider(client=httpx.Client(transport=transport))

    with pytest.raises(InvalidIdentifierError):
        provider.get_paper("malformed_arxiv_id_abc")


# ---------------------------------------------------------------------------
# Error Handling, Rate Limiting & Timeout Tests
# ---------------------------------------------------------------------------
def test_arxiv_rate_limit_429_handling():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            text="Too Many Requests",
            headers={"Retry-After": "5.5"},
        )

    provider = ArxivSourceProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryRateLimitError) as exc_info:
        provider.search(ScholarlySearchQuery(query="lora"))

    assert exc_info.value.retry_after == 5.5
    assert "rate limit exceeded" in str(exc_info.value).lower()


def test_arxiv_server_error_500_handling():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    provider = ArxivSourceProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryResponseError) as exc_info:
        provider.search(ScholarlySearchQuery(query="lora"))

    assert "HTTP 500" in str(exc_info.value)


def test_arxiv_client_error_400_handling():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    provider = ArxivSourceProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryResponseError) as exc_info:
        provider.search(ScholarlySearchQuery(query="lora"))

    assert "HTTP 400" in str(exc_info.value)


def test_arxiv_timeout_handling():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    provider = ArxivSourceProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryTimeoutError):
        provider.search(ScholarlySearchQuery(query="lora"))


def test_arxiv_connection_error_handling():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    provider = ArxivSourceProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryConnectionError):
        provider.search(ScholarlySearchQuery(query="lora"))


def test_arxiv_malformed_xml_handling():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<feed>Truncated or malformed XML...")

    provider = ArxivSourceProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(DiscoveryResponseError) as exc_info:
        provider.search(ScholarlySearchQuery(query="lora"))

    assert "Malformed XML" in str(exc_info.value)


def test_arxiv_rate_limit_policy_tracking():
    policy = ArxivRateLimitPolicy(min_interval_seconds=0.0, enforce_sleep=False)
    assert policy.total_requests == 0

    provider = ArxivSourceProvider(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, text=EMPTY_ARXIV_FEED_XML))
        ),
        rate_limit_policy=policy,
    )
    provider.search(ScholarlySearchQuery(query="test1"))
    provider.search(ScholarlySearchQuery(query="test2"))

    assert policy.total_requests == 2
    assert policy.last_request_timestamp > 0.0
