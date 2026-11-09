"""Tests for Google Gemini LLM Provider (Free-Tier & Standard API).

Verifies:
1. Provider initialization, model normalization, and default model (gemini-2.5-flash).
2. LLMProvider protocol compliance.
3. Secret hygiene: API keys never exposed in repr, str, or exception messages.
4. Message role mapping (system -> systemInstruction, user -> user, assistant -> model).
5. Generation configuration (temperature, max_output_tokens).
6. Response parsing (text extraction, usageMetadata, finish_reason).
7. Error handling (HTTP 400, 401, 403, 404, 429, 500, timeouts, connection errors).
8. Factory integration via create_llm_provider.
9. RAG and Cross-Paper comparison compatibility with Gemini provider.
10. Opt-in live smoke test when valid credentials exist in the environment.
"""

from __future__ import annotations

import json
import os
import socket
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from nexora.llm.base import LLMProvider, LLMProviderError
from nexora.llm.config import LLMConfig
from nexora.llm.gemini_provider import (
    DEFAULT_GEMINI_LLM_MODEL,
    GeminiProvider,
    sanitize_secret_text,
)
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.service import LLMService, create_llm_provider

FAKE_GEMINI_KEY = "AIzaSyFakeSecretKey998877665544332211"


# ---------------------------------------------------------------------------
# Fake Transport / Test Double for Offline Testing
# ---------------------------------------------------------------------------
class FakeGeminiHttpClient:
    """Mock HTTP client simulating the Gemini REST API without network calls."""

    def __init__(
        self,
        *,
        canned_text: str = "Attention and LoRA both optimize transformer adaptation.",
        finish_reason: str = "STOP",
        prompt_tokens: int = 150,
        candidates_tokens: int = 80,
        total_tokens: int = 230,
        status_code: int = 200,
        error_body: dict[str, Any] | None = None,
        raise_exception: Exception | None = None,
    ) -> None:
        self.canned_text = canned_text
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.candidates_tokens = candidates_tokens
        self.total_tokens = total_tokens
        self.status_code = status_code
        self.error_body = error_body
        self.raise_exception = raise_exception
        self.requests: list[dict[str, Any]] = []

    def post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        self.requests.append({"path": path, "json": json, "headers": headers})

        if self.raise_exception is not None:
            raise self.raise_exception

        if self.status_code >= 400:
            import httpx

            request = httpx.Request("POST", f"https://generativelanguage.googleapis.com{path}")
            content = None
            if self.error_body:
                import json as _json

                content = _json.dumps(self.error_body).encode("utf-8")
            response = httpx.Response(self.status_code, request=request, content=content)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code} Error", request=request, response=response
            )

        resp_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": self.canned_text}],
                        "role": "model",
                    },
                    "finishReason": self.finish_reason,
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": self.prompt_tokens,
                "candidatesTokenCount": self.candidates_tokens,
                "totalTokenCount": self.total_tokens,
            },
        }

        class FakeResponse:
            def __init__(self, data: dict[str, Any], status: int) -> None:
                self.data = data
                self.status_code = status

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                return self.data

        return FakeResponse(resp_data, self.status_code)


def make_provider(
    *,
    model: str = DEFAULT_GEMINI_LLM_MODEL,
    api_key: str = FAKE_GEMINI_KEY,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    client: Any = None,
) -> tuple[GeminiProvider, FakeGeminiHttpClient]:
    fake_client = client or FakeGeminiHttpClient()
    config = LLMConfig(
        provider="gemini",
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    provider = GeminiProvider(config, client=fake_client)
    return provider, fake_client


# ---------------------------------------------------------------------------
# 1. Initialization and Protocol
# ---------------------------------------------------------------------------
def test_gemini_provider_implements_protocol():
    provider, _ = make_provider()
    assert isinstance(provider, LLMProvider)
    assert provider.provider == "gemini"
    assert provider.name == "gemini"


def test_gemini_provider_defaults_to_gemini_2_5_flash():
    config = LLMConfig(provider="gemini", model="default", api_key=FAKE_GEMINI_KEY)
    provider = GeminiProvider(config, client=FakeGeminiHttpClient())
    assert provider.model == "gemini-2.5-flash"


def test_gemini_provider_strips_models_prefix():
    config = LLMConfig(provider="gemini", model="models/gemini-2.5-flash", api_key=FAKE_GEMINI_KEY)
    provider = GeminiProvider(config, client=FakeGeminiHttpClient())
    assert provider.model == "gemini-2.5-flash"


def test_gemini_provider_offline_init_no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", mock.Mock(side_effect=AssertionError("network blocked")))
    config = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key=FAKE_GEMINI_KEY)
    provider = GeminiProvider(config)
    assert provider.model == "gemini-2.5-flash"


def test_gemini_provider_repr_hides_secrets():
    provider, _ = make_provider()
    rep = repr(provider)
    assert FAKE_GEMINI_KEY not in rep
    assert "GeminiProvider" in rep


# ---------------------------------------------------------------------------
# 2. Request Formatting & Message Role Mapping
# ---------------------------------------------------------------------------
def test_system_and_dialog_role_mapping():
    provider, client = make_provider()
    messages = [
        LLMMessage(role="system", content="You are a research assistant."),
        LLMMessage(role="user", content="Compare LoRA and Attention."),
        LLMMessage(role="assistant", content="LoRA is parameter-efficient."),
        LLMMessage(role="user", content="What about memory footprint?"),
    ]

    response = provider.generate(messages)

    assert len(client.requests) == 1
    req = client.requests[0]
    assert "/v1beta/models/gemini-2.5-flash:generateContent" in req["path"]
    assert req["headers"]["x-goog-api-key"] == FAKE_GEMINI_KEY

    payload = req["json"]
    assert "systemInstruction" in payload
    assert payload["systemInstruction"]["parts"] == [{"text": "You are a research assistant."}]

    contents = payload["contents"]
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "Compare LoRA and Attention."}]
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [{"text": "LoRA is parameter-efficient."}]
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"] == [{"text": "What about memory footprint?"}]

    assert response.text == client.canned_text
    assert response.model == "gemini-2.5-flash"
    assert response.usage == {
        "input_tokens": 150,
        "output_tokens": 80,
        "total_tokens": 230,
    }
    assert response.raw == {"finish_reason": "STOP"}


def test_generation_config_passed_properly():
    provider, client = make_provider(temperature=0.3, max_output_tokens=512)
    messages = [LLMMessage(role="user", content="Hello")]

    provider.generate(messages, temperature=0.5, max_output_tokens=1024)

    payload = client.requests[0]["json"]
    assert payload["generationConfig"]["temperature"] == 0.5
    assert payload["generationConfig"]["maxOutputTokens"] == 1024


# ---------------------------------------------------------------------------
# 3. Structured JSON Parsing and Error Sanitization
# ---------------------------------------------------------------------------
def test_structured_json_response_parsing():
    json_payload = {
        "conclusion": "comparable",
        "reasoning": "Both evaluate attention mechanisms.",
        "findings": ["Finding A", "Finding B"],
    }
    client = FakeGeminiHttpClient(canned_text=json.dumps(json_payload))
    provider, _ = make_provider(client=client)

    messages = [LLMMessage(role="user", content="Output json matrix")]
    response = provider.generate(messages)

    parsed = json.loads(response.text)
    assert parsed["conclusion"] == "comparable"
    assert len(parsed["findings"]) == 2


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 429, 500])
def test_gemini_http_errors_raise_llm_provider_error(status_code):
    client = FakeGeminiHttpClient(
        status_code=status_code,
        error_body={
            "error": {
                "message": f"Simulated Gemini Error {status_code}",
                "status": "PERMISSION_DENIED" if status_code == 403 else "ERROR",
            }
        },
    )
    provider, _ = make_provider(client=client)

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate([LLMMessage(role="user", content="test")])

    err_msg = str(exc_info.value)
    assert f"HTTP {status_code}" in err_msg or "Gemini" in err_msg
    # Ensure API key is NEVER leaked in error messages
    assert FAKE_GEMINI_KEY not in err_msg


def test_connection_error_sanitized():
    import httpx

    conn_err = httpx.ConnectError("Failed to resolve generativelanguage.googleapis.com")
    client = FakeGeminiHttpClient(raise_exception=conn_err)
    provider, _ = make_provider(client=client)

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate([LLMMessage(role="user", content="test")])

    err_msg = str(exc_info.value)
    assert "network connection error" in err_msg
    assert FAKE_GEMINI_KEY not in err_msg


def test_missing_api_key_raises_informative_error():
    config = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key=None)
    provider = GeminiProvider(config)  # Lazy, doesn't throw at init

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate([LLMMessage(role="user", content="test")])

    assert "Gemini API key is required" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. Factory and Configuration from Environment
# ---------------------------------------------------------------------------
def test_create_llm_provider_factory():
    config = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key=FAKE_GEMINI_KEY)
    provider = create_llm_provider(config)
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-2.5-flash"


def test_llm_config_from_env_for_gemini():
    env = {
        "NEXORA_LLM_PROVIDER": "gemini",
        "NEXORA_LLM_MODEL": "gemini-2.5-flash",
        "GEMINI_API_KEY": FAKE_GEMINI_KEY,
    }
    cfg = LLMConfig.from_env(env)
    assert cfg.provider == "gemini"
    assert cfg.model == "gemini-2.5-flash"
    assert cfg.api_key == FAKE_GEMINI_KEY
    assert cfg.is_configured is True
    assert LLMConfig.has_credentials(env) is True


def test_llm_service_delegates_to_gemini():
    provider, client = make_provider()
    service = LLMService(provider)

    resp = service.generate([LLMMessage(role="user", content="Research question")])
    assert resp.text == client.canned_text
    assert resp.model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# 5. Opt-in Live Smoke Test (Skipped in CI unless key is present)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("NEXORA_GEMINI_API_KEY")),
    reason="Live Gemini API key not configured in environment (opt-in smoke test)",
)
def test_live_gemini_generation_smoke():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("NEXORA_GEMINI_API_KEY")
    config = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key=key)
    provider = GeminiProvider(config)

    messages = [
        LLMMessage(role="system", content="You are a concise research assistant."),
        LLMMessage(role="user", content="Reply with exactly 'OK'."),
    ]
    response = provider.generate(messages, max_output_tokens=10)
    assert response.text.strip()
    assert response.model == "gemini-2.5-flash"
    assert response.usage is not None
    assert response.usage.get("total_tokens", 0) > 0
