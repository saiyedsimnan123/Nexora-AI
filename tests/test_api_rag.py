"""Tests for RAG REST API endpoints (/api/v1/research/query and /health).

Covers:
- Successful retrieval-to-grounded-answer HTTP requests with full evidence attribution.
- Support for BM25, vector, and hybrid retrieval modes.
- OpenAPI specification and route metadata.
- Validation of invalid requests (400 for semantic errors, 422 for schema violations).
- Truthful insufficient evidence handling (honest 200 with zero fabricated citations).
- Provider and internal error handling (500 without leaking secrets or tracebacks).
- 100% offline operation without API keys or internet access.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nexora.api.app import create_app, create_offline_app
from nexora.api.models import AttributedSource, RAGRequest, RAGResponse
from nexora.rag.models import AttributedSource as RagAttributedSource, RAGResponse as RagPipelineResponse
from nexora.rag.pipeline import create_offline_rag_pipeline

RAG_URL = "/api/v1/research/query"
RAG_ALIAS_URL = "/api/v1/rag/query"
HEALTH_URL = "/health"
SECRET_KEY = "sk-mock-secret-that-must-not-leak"


class MockResearchService:
    """Minimal research service to satisfy create_app contract."""

    def ask(self, request):
        return {"text": "mock"}


class FailingGeneratorPipeline:
    """Pipeline that simulates an answer-generation provider error."""

    def __init__(self, error_message: str = "Provider rate limit exceeded"):
        self.error_message = error_message
        self.is_mock = True

    def answer(self, request, **kwargs):
        return RagPipelineResponse(
            query=request if isinstance(request, str) else request.query,
            answer="",
            sources=[],
            cited_sources=[],
            is_grounded=False,
            is_mock=True,
            model="failing-provider",
            status="error",
            error=self.error_message,
        )


class ExplodingPipeline:
    """Pipeline that raises an unhandled exception with sensitive text."""

    def answer(self, request, **kwargs):
        raise RuntimeError(f"Unexpected connection failure with token {SECRET_KEY}")


@pytest.fixture(scope="module")
def shared_pipeline():
    """Create offline pipeline once for faster read-only test cases."""
    return create_offline_rag_pipeline()


@pytest.fixture
def offline_client(shared_pipeline):
    """TestClient wired with real offline RAG pipeline backed by pilot benchmark."""
    app = create_app(MockResearchService(), rag_pipeline=shared_pipeline)
    return TestClient(app, raise_server_exceptions=False)


def test_health_check_returns_ok_and_fast(offline_client):
    """GET /health returns 200 with status ok."""
    response = offline_client.get(HEALTH_URL)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rag_query_successful_hybrid_flow(offline_client):
    """POST /api/v1/research/query returns grounded answer with traceable evidence."""
    payload = {
        "query": "How does multi-head attention work in the Transformer?",
        "retrieval_mode": "hybrid",
        "top_k": 3,
    }
    response = offline_client.post(RAG_URL, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == payload["query"]
    assert "multi-head" in data["answer"].lower() or "attention" in data["answer"].lower()
    assert data["status"] == "success"
    assert data["is_grounded"] is True
    assert data["is_mock"] is True
    assert data["retrieval_mode"] == "hybrid"
    assert data["evidence_count"] >= 1
    assert data["error"] is None

    # Verify sources structure
    assert len(data["sources"]) > 0
    first_source = data["sources"][0]
    assert first_source["label"].startswith("[E")
    assert first_source["paper_title"]
    assert first_source["chunk_id"]
    assert first_source["document_id"]
    assert isinstance(first_source["score"], (int, float))
    assert first_source["text_excerpt"]

    # Verify citations are traceable
    assert len(data["cited_sources"]) > 0
    assert data["cited_sources"][0]["label"] in data["answer"]


def test_rag_query_bm25_mode(offline_client):
    """BM25 lexical retrieval mode works via API."""
    payload = {
        "query": "What is Low-Rank Adaptation and what are rank decomposition matrices?",
        "retrieval_mode": "bm25",
        "top_k": 2,
    }
    response = offline_client.post(RAG_URL, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["retrieval_mode"] == "bm25"
    assert len(data["sources"]) <= 2
    assert any("lora" in s["paper_title"].lower() for s in data["sources"])


def test_rag_query_vector_mode(offline_client):
    """Vector semantic retrieval mode works via API."""
    payload = {
        "query": "What IO-aware techniques optimize GPU SRAM access in FlashAttention?",
        "retrieval_mode": "vector",
        "top_k": 2,
    }
    response = offline_client.post(RAG_URL, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["retrieval_mode"] == "vector"
    assert any("flashattention" in s["paper_title"].lower() for s in data["sources"])


def test_rag_query_alias_endpoint(offline_client):
    """POST /api/v1/rag/query serves as a valid alias for /api/v1/research/query."""
    payload = {
        "query": "Explain self-attention mechanism",
        "retrieval_mode": "hybrid",
        "top_k": 1,
    }
    response = offline_client.post(RAG_ALIAS_URL, json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_rag_query_with_generation_parameters(offline_client):
    """Optional parameters (temperature, max_output_tokens, min_score) are accepted."""
    payload = {
        "query": "What is the Transformer architecture?",
        "retrieval_mode": "hybrid",
        "top_k": 3,
        "min_score": 0.001,
        "temperature": 0.5,
        "max_output_tokens": 128,
    }
    response = offline_client.post(RAG_URL, json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_rag_query_insufficient_evidence_unanswerable_query(offline_client):
    """Truthful reporting when no evidence matches: honest 200 with insufficient_evidence status."""
    payload = {
        "query": "What is the breeding cycle of emperor penguins in Antarctica during winter?",
        "retrieval_mode": "bm25",
    }
    response = offline_client.post(RAG_URL, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "insufficient_evidence"
    assert "could not find sufficient evidence" in data["answer"].lower()
    assert data["sources"] == []
    assert data["cited_sources"] == []
    assert data["is_grounded"] is True
    assert data["evidence_count"] == 0


def test_rag_query_insufficient_evidence_high_min_score(offline_client):
    """Unreachable min_score cutoff yields truthful insufficient evidence."""
    payload = {
        "query": "Transformer attention mechanism",
        "min_score": 999.0,
    }
    response = offline_client.post(RAG_URL, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "insufficient_evidence"
    assert data["evidence_count"] == 0
    assert data["sources"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   \n\t  "},
        {"query": "What is LoRA?", "retrieval_mode": "keyword"},  # invalid mode
        {"query": "What is LoRA?", "retrieval_mode": "magic"},
        {"query": "What is LoRA?", "temperature": -0.1},
        {"query": "What is LoRA?", "temperature": 2.5},
        {"query": "What is LoRA?", "max_output_tokens": 0},
        {"query": "What is LoRA?", "max_output_tokens": -10},
    ],
)
def test_semantically_invalid_requests_return_400(offline_client, payload):
    """Semantically invalid queries return HTTP 400 invalid_request."""
    response = offline_client.post(RAG_URL, json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing query
        {"query": 12345},  # wrong query type
        {"query": None},
        {"query": ["valid question as list"]},
        {"query": "Valid query", "top_k": 0},  # ge=1 violated
        {"query": "Valid query", "top_k": -5},
        {"query": "Valid query", "top_k": "three"},
        {"query": "Valid query", "unexpected_unknown_key": "val"},  # extra forbidden
    ],
)
def test_schema_invalid_requests_return_422(offline_client, payload):
    """Schema violations return HTTP 422 validation_error."""
    response = offline_client.post(RAG_URL, json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert "details" in body["error"]


def test_provider_failure_returns_500_without_leaking_secrets():
    """Provider failure returns safe 500 without leaking secrets or tracebacks."""
    app = create_app(MockResearchService(), rag_pipeline=FailingGeneratorPipeline(error_message=f"API error: {SECRET_KEY}"))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(RAG_URL, json={"query": "Test query"})
    assert response.status_code == 500

    body = response.json()
    assert body["error"]["code"] == "research_failed"
    assert SECRET_KEY not in response.text
    assert "traceback" not in response.text.lower()


def test_unhandled_pipeline_exception_returns_500_without_leaking_secrets():
    """Unhandled internal exception returns generic 500 without leaking secrets."""
    app = create_app(MockResearchService(), rag_pipeline=ExplodingPipeline())
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(RAG_URL, json={"query": "Test query"})
    assert response.status_code == 500

    body = response.json()
    assert body["error"]["code"] in ("research_failed", "internal_error")
    assert SECRET_KEY not in response.text
    assert "traceback" not in response.text.lower()


def test_create_offline_app_factory():
    """create_offline_app factory returns a functional out-of-the-box offline app."""
    app = create_offline_app()
    client = TestClient(app, raise_server_exceptions=False)

    health_resp = client.get(HEALTH_URL)
    assert health_resp.status_code == 200

    query_resp = client.post(RAG_URL, json={"query": "What is LoRA?", "top_k": 2})
    assert query_resp.status_code == 200
    assert query_resp.json()["status"] == "success"


def test_openapi_schema_documents_rag_endpoint():
    """OpenAPI schema correctly exposes /api/v1/research/query with request and response specs."""
    app = create_offline_app()
    openapi_paths = app.openapi()["paths"]

    assert HEALTH_URL in openapi_paths
    assert RAG_URL in openapi_paths

    operation = openapi_paths[RAG_URL]["post"]
    assert "summary" in operation
    assert "requestBody" in operation
    assert "200" in operation["responses"]
