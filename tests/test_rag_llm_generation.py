"""Tests for Real LLM Answer Generation in the RAG Pipeline.

Verifies:
1. Live-provider integration using a mocked LLM provider/service.
2. Evidence context passed truthfully into prompt and generation messages.
3. Citation validation: valid citations attributed, invalid citations filtered or flagged.
4. Insufficient evidence handling: empty retrieval and model stating insufficient evidence.
5. Missing provider configuration: explicit ConfigurationError when live mode requested without credentials.
6. Offline preservation: default or explicit offline mode stays 100% deterministic mock.
7. Provider errors: safe error response with sanitized secrets (never leaking API keys).
8. Accurate metadata: strict is_mock distinction (False for live, True for mock).
9. FastAPI HTTP API compatibility with live RAG pipeline.
10. CLI execution with --live and --offline flags.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nexora.api.app import create_app
from nexora.config.settings import ConfigurationError
from nexora.llm.base import LLMProvider, LLMProviderError
from nexora.llm.config import LLMConfig
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.prompt import ResearchPromptBuilder
from nexora.llm.service import LLMService
from nexora.rag.cli import run_rag_cli
from nexora.rag.generator import LLMAnswerGenerator
from nexora.rag.models import RAGRequest, RAGResponse
from nexora.rag.pipeline import (
    RAGPipeline,
    create_configured_rag_pipeline,
    create_live_rag_pipeline,
    create_offline_rag_pipeline,
)
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem, verify_citations
from nexora.retrieval.models import RetrievedChunk, SearchResult

SECRET_API_KEY = "sk-live-secret-test-key-999888777"


# ---------------------------------------------------------------------------
# Test Fakes
# ---------------------------------------------------------------------------
class FakeLLMProvider:
    """Mock LLM provider implementing the LLMProvider protocol."""

    def __init__(
        self,
        canned_text: str = "According to LoRA [E1], pre-trained weights are frozen and low-rank matrices are injected.",
        model_name: str = "gpt-4o-mini",
        usage: dict[str, int] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.canned_text = canned_text
        self.model_name = model_name
        self.usage = usage or {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165}
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        })
        if self.error:
            raise self.error
        return LLMResponse(
            text=self.canned_text,
            model=model or self.model_name,
            usage=self.usage,
            raw={"response_id": "resp-test-123"},
        )


class MockResearchService:
    """Minimal research service to satisfy create_app contract."""

    def ask(self, request: Any) -> Any:
        return SimpleNamespace(text="research", to_dict=lambda: {"text": "research"})


# ---------------------------------------------------------------------------
# Tests: Live Provider Integration & Evidence Passing
# ---------------------------------------------------------------------------
def test_live_rag_pipeline_with_mocked_provider():
    """Live RAG pipeline generates grounded answer using LLMAnswerGenerator."""
    provider = FakeLLMProvider()
    service = LLMService(provider)

    pipeline = create_live_rag_pipeline(
        llm_service=service,
        model="gpt-4o-mini",
        default_mode="bm25",
    )

    assert pipeline.is_mock is False
    assert pipeline.generator.is_mock is False
    assert pipeline.generator.model_name == "gpt-4o-mini"

    response = pipeline.answer("How does LoRA freeze pre-trained weights?", top_k=3)

    assert response.status == "success"
    assert response.is_mock is False
    assert response.model == "gpt-4o-mini"
    assert response.is_grounded is True
    assert "[E1]" in response.answer
    assert len(response.cited_sources) >= 1
    assert response.cited_sources[0].label == "[E1]"
    assert response.usage is not None
    assert response.usage["total_tokens"] == 165


def test_evidence_passed_to_llm_messages():
    """Retrieved evidence items are passed as [E#] blocks into generation context."""
    provider = FakeLLMProvider()
    service = LLMService(provider)

    pipeline = create_live_rag_pipeline(
        llm_service=service,
        model="gpt-4o-mini",
        default_mode="bm25",
    )

    query = "How does Low-Rank Adaptation reduce trainable parameters?"
    response = pipeline.answer(query, top_k=2)

    assert len(provider.calls) == 1
    call = provider.calls[0]
    messages: list[LLMMessage] = call["messages"]
    assert len(messages) == 2

    # System instruction enforces grounding & citation rules
    sys_msg = messages[0]
    assert sys_msg.role == "system"
    assert "Cite evidence by its [E#] label" in sys_msg.content
    assert "never invent citations or evidence" in sys_msg.content
    assert "If the evidence is insufficient, say so" in sys_msg.content

    # User message contains question and structured evidence blocks
    user_msg = messages[1]
    assert user_msg.role == "user"
    assert "RESEARCH QUESTION:" in user_msg.content
    assert query in user_msg.content
    assert "RETRIEVED EVIDENCE:" in user_msg.content
    assert "[E1]" in user_msg.content


# ---------------------------------------------------------------------------
# Tests: Valid and Invalid Citations
# ---------------------------------------------------------------------------
def test_valid_citations_fully_attributed():
    """Answer with valid [E1] citation is verified and attributed."""
    provider = FakeLLMProvider(
        canned_text="LoRA freezes pre-trained weights [E1] and injects low-rank matrices.",
    )
    service = LLMService(provider)
    pipeline = create_live_rag_pipeline(llm_service=service, default_mode="bm25")

    response = pipeline.answer("LoRA parameters", top_k=2)

    assert response.is_grounded is True
    assert len(response.cited_sources) == 1
    assert response.cited_sources[0].label == "[E1]"


def test_invalid_citations_filtered_and_unsupplied_flagged():
    """Hallucinated citation [E99] is filtered out to preserve evidence integrity."""
    # LLM hallucinates an unsupplied citation [E99]
    provider = FakeLLMProvider(
        canned_text="LoRA was proposed in [E1], while older methods used [E99].",
    )
    service = LLMService(provider)
    pipeline = create_live_rag_pipeline(
        llm_service=service,
        filter_invalid_citations=True,
        default_mode="bm25",
    )

    response = pipeline.answer("LoRA parameters", top_k=2)

    # [E99] should have been filtered out of the final answer text
    assert "[E99]" not in response.answer
    assert "[E1]" in response.answer
    assert response.is_grounded is True


def test_unfiltered_invalid_citation_marks_ungrounded():
    """When invalid citations are not filtered, is_grounded is truthfully False."""
    provider = FakeLLMProvider(
        canned_text="Claim citing non-existent source [E99].",
    )
    service = LLMService(provider)
    pipeline = create_live_rag_pipeline(
        llm_service=service,
        filter_invalid_citations=False,
        default_mode="bm25",
    )

    response = pipeline.answer("LoRA parameters", top_k=2)

    assert response.is_grounded is False
    assert "[E99]" in response.answer


# ---------------------------------------------------------------------------
# Tests: Insufficient Evidence Handling
# ---------------------------------------------------------------------------
def test_insufficient_evidence_when_retrieval_empty():
    """Pipeline reports honest insufficient_evidence status when no chunks match."""
    provider = FakeLLMProvider()
    service = LLMService(provider)
    pipeline = create_live_rag_pipeline(llm_service=service, default_mode="bm25")

    # High min_score threshold causes 0 chunks to qualify
    response = pipeline.answer("LoRA", min_score=999.0)

    assert response.status == "insufficient_evidence"
    assert response.is_grounded is True
    assert response.evidence_count == 0
    assert len(response.sources) == 0
    assert len(response.cited_sources) == 0
    assert "sufficient evidence" in response.answer.lower()
    # LLM was never called unnecessarily when no evidence exists
    assert len(provider.calls) == 0



def test_insufficient_evidence_when_llm_reports_insufficient():
    """When LLM explicitly reports insufficient evidence, status is insufficient_evidence."""
    provider = FakeLLMProvider(
        canned_text="I could not find sufficient evidence in the indexed research papers to answer this question.",
    )
    service = LLMService(provider)
    pipeline = create_live_rag_pipeline(llm_service=service, default_mode="bm25")

    response = pipeline.answer("What is quantum chromodynamics in protein folding?", top_k=2)

    assert response.status == "insufficient_evidence"
    assert response.is_grounded is True
    assert len(response.cited_sources) == 0


# ---------------------------------------------------------------------------
# Tests: Configuration and Missing Credentials
# ---------------------------------------------------------------------------
def test_create_live_pipeline_raises_on_missing_api_key():
    """create_live_rag_pipeline raises ConfigurationError when no API key is provided."""
    empty_env: dict[str, str] = {}
    with pytest.raises(ConfigurationError) as exc_info:
        create_live_rag_pipeline(environ=empty_env)
    assert "no API key configured" in str(exc_info.value)


def test_create_configured_pipeline_explicit_live_raises_without_key():
    """create_configured_rag_pipeline with generator_mode='live' raises ConfigurationError without key."""
    empty_env: dict[str, str] = {}
    with pytest.raises(ConfigurationError) as exc_info:
        create_configured_rag_pipeline(generator_mode="live", environ=empty_env)
    assert "no API key configured" in str(exc_info.value)


def test_create_configured_pipeline_auto_mode_defaults_to_offline():
    """create_configured_rag_pipeline without credentials safely defaults to offline mock."""
    empty_env: dict[str, str] = {}
    pipeline = create_configured_rag_pipeline(environ=empty_env)

    assert pipeline.is_mock is True
    assert pipeline.generator.is_mock is True
    response = pipeline.answer("How does LoRA work?")
    assert response.is_mock is True
    assert response.status == "success"


def test_create_configured_pipeline_explicit_offline_mode():
    """Explicit offline mode uses DeterministicMockGenerator even if API key exists."""
    env_with_key = {"NEXORA_LLM_API_KEY": SECRET_API_KEY}
    pipeline = create_configured_rag_pipeline(generator_mode="offline", environ=env_with_key)

    assert pipeline.is_mock is True
    assert pipeline.generator.is_mock is True


def test_create_configured_pipeline_auto_detects_credentials():
    """create_configured_rag_pipeline auto-selects live mode when credentials and service are provided."""
    provider = FakeLLMProvider()
    service = LLMService(provider)
    pipeline = create_configured_rag_pipeline(llm_service=service)

    assert pipeline.is_mock is False
    assert pipeline.generator.is_mock is False


# ---------------------------------------------------------------------------
# Tests: Provider Errors & Secret Masking
# ---------------------------------------------------------------------------
def test_provider_error_handling_sanitizes_secrets():
    """LLM provider failure returns status='error' and masks API keys."""
    error = LLMProviderError(f"OpenAI request failed: connection error with token {SECRET_API_KEY}")
    provider = FakeLLMProvider(error=error)
    service = LLMService(provider)

    pipeline = create_live_rag_pipeline(llm_service=service, default_mode="bm25")
    response = pipeline.answer("How does LoRA work?", top_k=2)

    assert response.status == "error"
    assert response.is_grounded is False
    assert response.is_mock is False
    assert response.error is not None
    assert SECRET_API_KEY not in response.error
    assert "[REDACTED]" in response.error


# ---------------------------------------------------------------------------
# Tests: Metadata Integrity (Live vs Mock)
# ---------------------------------------------------------------------------
def test_live_versus_mock_metadata_integrity():
    """Verify is_mock is strictly False for live generator and True for mock generator."""
    # 1. Live generator
    live_service = LLMService(FakeLLMProvider())
    live_pipeline = create_live_rag_pipeline(llm_service=live_service, model="gpt-4o")
    live_resp = live_pipeline.answer("LoRA question")

    assert live_resp.is_mock is False
    assert live_resp.model == "gpt-4o"

    # 2. Offline mock generator
    offline_pipeline = create_offline_rag_pipeline()
    offline_resp = offline_pipeline.answer("LoRA question")

    assert offline_resp.is_mock is True
    assert offline_resp.model == "offline-deterministic-mock-v1"


# ---------------------------------------------------------------------------
# Tests: HTTP API Integration with Live Pipeline
# ---------------------------------------------------------------------------
def test_api_research_query_with_live_rag_pipeline():
    """POST /api/v1/research/query returns is_mock=False when backed by live pipeline."""
    provider = FakeLLMProvider(
        canned_text="LoRA freezes pre-trained weights [E1] and reduces memory requirement.",
        model_name="gpt-4o-mini",
    )
    service = LLMService(provider)
    live_pipeline = create_live_rag_pipeline(llm_service=service, default_mode="bm25")

    app = create_app(MockResearchService(), rag_pipeline=live_pipeline)
    client = TestClient(app, raise_server_exceptions=False)

    payload = {
        "query": "How does LoRA reduce trainable parameters?",
        "retrieval_mode": "bm25",
        "top_k": 2,
    }
    response = client.post("/api/v1/research/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["is_mock"] is False
    assert data["model"] == "gpt-4o-mini"
    assert data["is_grounded"] is True
    assert len(data["cited_sources"]) >= 1
    assert data["cited_sources"][0]["label"] == "[E1]"


# ---------------------------------------------------------------------------
# Tests: CLI Live & Offline Execution
# ---------------------------------------------------------------------------
def test_cli_live_mode_without_key_exits_with_error(capsys):
    """Running CLI with --live when unconfigured prints error and exits with code 1."""
    with patch.dict("os.environ", {}, clear=True):
        code = run_rag_cli(["--live", "How does LoRA work?"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Configuration Error" in captured.err or "API key" in captured.err


def test_cli_offline_flag_succeeds(capsys):
    """Running CLI with --offline flag forces deterministic mock."""
    code = run_rag_cli(["--offline", "--query", "How does LoRA work?", "--retriever", "bm25"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Mock Answer:    Yes (Offline Deterministic Mock)" in captured.out
    assert "Status:         SUCCESS" in captured.out

