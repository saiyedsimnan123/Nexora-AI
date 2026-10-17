"""Tests for the opt-in live LLM smoke test runner.

Verifies:
1. Safe configuration inspection and API key masking (no secrets in logs or output).
2. Unconfigured state detection and actionable guidance when no credentials exist.
3. Successful end-to-end live smoke execution using an injected fake provider.
4. Detection and rejection of fraudulent mock fallbacks during live requests.
5. Error handling and secret redaction when provider throws sensitive exceptions.
6. CLI integration with --smoke and --json flags.
7. Opt-in real live provider test (skipped by default, only executed when NEXORA_RUN_LIVE_TESTS=1).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from nexora.llm.base import LLMProviderError
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.service import LLMService
from nexora.rag.cli import run_rag_cli
from nexora.rag.smoke import (
    SmokeTestResult,
    check_live_configuration,
    mask_api_key,
    run_live_smoke_test,
)

FAKE_SECRET_KEY = "sk-proj-real-looking-secret-key-1234567890abcdef"


class FakeProvider:
    """Mock LLM provider for safe unit testing of smoke test workflows."""

    def __init__(
        self,
        canned_text: str = "According to LoRA [E1], pre-trained model weights are frozen.",
        model_name: str = "gpt-4o-mini",
        error: Exception | None = None,
    ) -> None:
        self.canned_text = canned_text
        self.model_name = model_name
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
        self.calls.append({"messages": messages, "model": model})
        if self.error:
            raise self.error
        return LLMResponse(
            text=self.canned_text,
            model=model or self.model_name,
            usage={"input_tokens": 150, "output_tokens": 30, "total_tokens": 180},
            raw={"id": "resp-smoke-test-1"},
        )


# ---------------------------------------------------------------------------
# Unit Tests (Safe, 100% Offline, Zero Network Calls)
# ---------------------------------------------------------------------------
def test_mask_api_key_safety():
    """mask_api_key conceals secret characters and never outputs full key."""
    assert mask_api_key(None) == "[NOT CONFIGURED]"
    assert mask_api_key("") == "[NOT CONFIGURED]"
    assert mask_api_key("   ") == "[NOT CONFIGURED]"
    assert mask_api_key("short") == "[CONFIGURED - HIDDEN]"

    masked = mask_api_key(FAKE_SECRET_KEY)
    assert FAKE_SECRET_KEY not in masked
    assert masked.startswith("sk-proj")
    assert masked.endswith("cdef")
    assert "..." in masked


def test_check_live_configuration_when_unconfigured():
    """check_live_configuration accurately reports unconfigured state without credentials."""
    empty_env: dict[str, str] = {}
    config = check_live_configuration(environ=empty_env)

    assert config["has_credentials"] is False
    assert config["masked_key"] == "[NOT CONFIGURED]"
    assert config["model"] == "gpt-4o-mini"
    assert config["provider"] == "openai"


def test_check_live_configuration_with_mock_env():
    """check_live_configuration detects credentials and masks key safely."""
    mock_env = {
        "NEXORA_LLM_API_KEY": FAKE_SECRET_KEY,
        "NEXORA_LLM_MODEL": "gpt-4o",
        "NEXORA_LLM_BASE_URL": "https://custom.endpoint.com/v1",
    }
    config = check_live_configuration(environ=mock_env)

    assert config["has_credentials"] is True
    assert FAKE_SECRET_KEY not in config["masked_key"]
    assert config["model"] == "gpt-4o"
    assert config["base_url"] == "https://custom.endpoint.com/v1"


def test_run_live_smoke_test_unconfigured_flow(capsys):
    """When unconfigured, smoke test returns informative unconfigured status."""
    empty_env: dict[str, str] = {}
    result = run_live_smoke_test(environ=empty_env, verbose=True)

    assert result.status == "unconfigured"
    assert result.has_credentials is False
    assert result.is_mock is True
    assert "No API key found" in result.message
    assert result.response is None

    captured = capsys.readouterr()
    assert "[STATUS] UNCONFIGURED" in captured.out
    assert "NEXORA_LLM_API_KEY=sk-..." in captured.out
    assert "offline mode remains 100% operational" in captured.out


def test_run_live_smoke_test_with_injected_service(capsys):
    """Smoke test succeeds and verifies evidence/citations when provider answers."""
    provider = FakeProvider(
        canned_text="LoRA freezes pre-trained weights [E1] and reduces memory footprint.",
        model_name="gpt-4o-mini",
    )
    service = LLMService(provider)

    result = run_live_smoke_test(
        query="How does LoRA freeze pre-trained weights?",
        llm_service=service,
        model="gpt-4o-mini",
        verbose=True,
    )

    assert result.status == "success"
    assert result.is_mock is False
    assert result.model == "gpt-4o-mini"
    assert result.latency_seconds is not None
    assert result.latency_seconds >= 0.0
    assert result.response is not None
    assert result.response.is_grounded is True
    assert len(result.response.cited_sources) == 1
    assert result.response.cited_sources[0].label == "[E1]"

    captured = capsys.readouterr()
    assert "SMOKE TEST RESULT: SUCCESS" in captured.out
    assert "Live Generation:    Yes (is_mock = False)" in captured.out
    assert "YES (All citations point to retrieved evidence)" in captured.out



def test_run_live_smoke_test_detects_mock_fallback_fraud(capsys):
    """Smoke test detects and rejects if a pipeline returns a mock generator."""
    from nexora.rag.generator import DeterministicMockGenerator
    from nexora.rag.pipeline import RAGPipeline, _build_benchmark_retrieval_service

    # Build a pipeline that incorrectly claims to be live but uses mock generator
    mock_pipeline = RAGPipeline(
        retrieval_service=_build_benchmark_retrieval_service(),
        generator=DeterministicMockGenerator(),
    )

    with patch("nexora.rag.smoke.create_live_rag_pipeline", return_value=mock_pipeline):
        env_with_key = {"NEXORA_LLM_API_KEY": FAKE_SECRET_KEY}
        result = run_live_smoke_test(environ=env_with_key, verbose=True)

    assert result.status == "mock_detected"
    assert result.is_mock is True
    captured = capsys.readouterr()
    assert "mock generator when live provider was requested" in captured.out


def test_run_live_smoke_test_handles_provider_error_safely(capsys):
    """Provider errors are caught, reported, and secrets in tracebacks are sanitized."""
    sensitive_error = LLMProviderError(f"Connection timeout with bearer {FAKE_SECRET_KEY}")
    provider = FakeProvider(error=sensitive_error)
    service = LLMService(provider)

    result = run_live_smoke_test(
        llm_service=service,
        model="gpt-4o-mini",
        verbose=True,
    )

    assert result.status == "error"
    assert result.error is not None
    assert FAKE_SECRET_KEY not in result.error
    assert "[REDACTED]" in result.error

    captured = capsys.readouterr()
    assert FAKE_SECRET_KEY not in captured.out
    assert "[REDACTED]" in captured.out


def test_cli_smoke_flag_unconfigured(capsys):
    """CLI --smoke flag exits with code 2 when unconfigured, giving clear setup instructions."""
    with patch.dict("os.environ", {}, clear=True):
        code = run_rag_cli(["--smoke"])
    assert code == 2
    captured = capsys.readouterr()
    assert "[STATUS] UNCONFIGURED" in captured.out
    assert "NEXORA_LLM_API_KEY" in captured.out


def test_cli_smoke_flag_json_output(capsys):
    """CLI --smoke --json flag produces parseable JSON."""
    with patch.dict("os.environ", {}, clear=True):
        code = run_rag_cli(["--smoke", "--json"])
    assert code == 2
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "unconfigured"
    assert data["has_credentials"] is False
    assert data["is_mock"] is True


# ---------------------------------------------------------------------------
# Opt-In Real Live Provider Smoke Test (Requires explicit invocation)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("NEXORA_RUN_LIVE_TESTS"),
    reason="Opt-in live smoke test skipped. Set NEXORA_RUN_LIVE_TESTS=1 to run against configured real provider.",
)
def test_live_llm_provider_real_smoke_call():
    """Opt-in real provider test.

    Runs ONLY when explicitly requested with NEXORA_RUN_LIVE_TESTS=1.
    If no credentials exist, confirms unconfigured report without failing.
    If credentials exist, sends exactly one small query and validates real generation.
    """
    config = check_live_configuration()
    if not config["has_credentials"]:
        pytest.skip("NEXORA_RUN_LIVE_TESTS=1 was set, but no API key is configured in .env or environment.")

    result = run_live_smoke_test(
        query="How does Low-Rank Adaptation (LoRA) freeze pre-trained weights?",
        verbose=True,
    )

    # Real call verification:
    assert result.status == "success", f"Live smoke test failed: {result.error}"
    assert result.is_mock is False, "Expected live AI model generation, but received mock output!"
    assert result.response is not None
    assert result.response.status == "success"
    assert result.response.is_grounded is True
    assert len(result.response.cited_sources) >= 1
    assert result.latency_seconds is not None and result.latency_seconds > 0.0
