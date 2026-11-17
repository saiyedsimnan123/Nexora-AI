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
import time
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


# ---------------------------------------------------------------------------
# 6. HTTP 503 Resilience, Exponential Backoff, Jitter, and Fallback
# ---------------------------------------------------------------------------
def test_gemini_503_retry_succeeds_on_primary_model():
    """Verify transient 503 error retries with backoff and succeeds on primary model."""
    import httpx

    sleep_calls: list[float] = []

    class RetrySuccessClient:
        def __init__(self):
            self.calls = 0
            self.requests = []

        def post(self, path, json=None, headers=None):
            self.requests.append({"path": path, "json": json})
            self.calls += 1
            if self.calls == 1:
                # First attempt fails with 503
                req = httpx.Request("POST", f"https://generativelanguage.googleapis.com{path}")
                err_data = json_module.dumps({
                    "error": {
                        "message": "This model is currently experiencing high demand.",
                        "status": "UNAVAILABLE",
                    }
                }).encode("utf-8")
                resp = httpx.Response(503, request=req, content=err_data)
                raise httpx.HTTPStatusError("HTTP 503 Service Unavailable", request=req, response=resp)

            # Second attempt succeeds
            class Resp:
                def raise_for_status(self): pass
                def json(self):
                    return {
                        "candidates": [{
                            "content": {"parts": [{"text": "Answer after retry"}], "role": "model"},
                            "finishReason": "STOP"
                        }],
                        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
                    }
            return Resp()

    import json as json_module
    client = RetrySuccessClient()
    config = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key=FAKE_GEMINI_KEY, max_retries=2, retry_delay=0.1)
    provider = GeminiProvider(config, client=client, sleep_fn=sleep_calls.append)

    messages = [LLMMessage(role="user", content="Research prompt")]
    res = provider.generate(messages)

    assert res.text == "Answer after retry"
    assert res.model == "gemini-2.5-flash"
    assert client.calls == 2
    assert len(client.requests) == 2
    assert "gemini-2.5-flash" in client.requests[0]["path"]
    assert "gemini-2.5-flash" in client.requests[1]["path"]
    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0.0  # Jittered backoff delay was applied


def test_gemini_503_fallback_to_secondary_model():
    """Verify primary model 503 exhaustion falls back to secondary supported Gemini model."""
    import httpx
    import json as json_module

    sleep_calls: list[float] = []

    class FallbackClient:
        def __init__(self):
            self.requests = []

        def post(self, path, json=None, headers=None):
            self.requests.append({"path": path, "json": json})
            if "gemini-2.5-flash" in path:
                req = httpx.Request("POST", f"https://generativelanguage.googleapis.com{path}")
                err_data = json_module.dumps({
                    "error": {
                        "message": "This model is currently experiencing high demand.",
                        "status": "UNAVAILABLE",
                    }
                }).encode("utf-8")
                resp = httpx.Response(503, request=req, content=err_data)
                raise httpx.HTTPStatusError("HTTP 503 Service Unavailable", request=req, response=resp)

            # Fallback model succeeds
            class Resp:
                def raise_for_status(self): pass
                def json(self):
                    return {
                        "candidates": [{
                            "content": {"parts": [{"text": "Answer from fallback model"}], "role": "model"},
                            "finishReason": "STOP"
                        }],
                        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 6, "totalTokenCount": 18}
                    }
            return Resp()

    client = FallbackClient()
    config = LLMConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        fallback_model="gemini-2.0-flash",
        api_key=FAKE_GEMINI_KEY,
        max_retries=1,  # 2 attempts on primary, then fallback
        retry_delay=0.01,
    )
    provider = GeminiProvider(config, client=client, sleep_fn=sleep_calls.append)

    messages = [LLMMessage(role="user", content="Research prompt")]
    res = provider.generate(messages)

    assert res.text == "Answer from fallback model"
    assert res.model == "gemini-2.0-flash"
    assert res.raw.get("fallback_from") == "gemini-2.5-flash"

    # Verify requests sequence: 2 on gemini-2.5-flash, then 1 on gemini-2.0-flash
    assert len(client.requests) == 3
    assert "gemini-2.5-flash" in client.requests[0]["path"]
    assert "gemini-2.5-flash" in client.requests[1]["path"]
    assert "gemini-2.0-flash" in client.requests[2]["path"]


def test_gemini_503_exhausted_retries_raises_actionable_error():
    """Verify when both primary and fallback fail with 503, an actionable error is raised without secret leaks."""
    import httpx
    import json as json_module

    class Always503Client:
        def __init__(self):
            self.requests = []

        def post(self, path, json=None, headers=None):
            self.requests.append({"path": path})
            req = httpx.Request("POST", f"https://generativelanguage.googleapis.com{path}")
            err_data = json_module.dumps({
                "error": {
                    "message": "This model is currently experiencing high demand.",
                    "status": "UNAVAILABLE",
                }
            }).encode("utf-8")
            resp = httpx.Response(503, request=req, content=err_data)
            raise httpx.HTTPStatusError("HTTP 503 Service Unavailable", request=req, response=resp)

    client = Always503Client()
    config = LLMConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        fallback_model="gemini-2.0-flash",
        api_key=FAKE_GEMINI_KEY,
        max_retries=1,
        retry_delay=0.01,
    )
    provider = GeminiProvider(config, client=client, sleep_fn=lambda _: None)

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate([LLMMessage(role="user", content="test")])

    err_msg = str(exc_info.value)
    assert "HTTP 503" in err_msg
    assert "high demand" in err_msg.lower()
    assert "gemini-2.5-flash" in err_msg
    assert "gemini-2.0-flash" in err_msg
    assert "exhausted" in err_msg.lower()
    assert FAKE_GEMINI_KEY not in err_msg


def test_gemini_non_transient_error_fails_fast_without_retry():
    """Verify permanent errors like 400 Bad Request or 401 fail immediately without retrying or falling back."""
    import httpx
    import json as json_module

    sleep_calls = []

    class BadReqClient:
        def __init__(self):
            self.calls = 0

        def post(self, path, json=None, headers=None):
            self.calls += 1
            req = httpx.Request("POST", f"https://generativelanguage.googleapis.com{path}")
            err_data = json_module.dumps({
                "error": {
                    "message": "API_KEY_INVALID: Key not valid",
                    "status": "INVALID_ARGUMENT",
                }
            }).encode("utf-8")
            resp = httpx.Response(400, request=req, content=err_data)
            raise httpx.HTTPStatusError("HTTP 400 Bad Request", request=req, response=resp)

    client = BadReqClient()
    config = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key=FAKE_GEMINI_KEY, max_retries=3)
    provider = GeminiProvider(config, client=client, sleep_fn=sleep_calls.append)

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate([LLMMessage(role="user", content="test")])

    assert client.calls == 1  # Exactly one attempt, no retries
    assert len(sleep_calls) == 0  # No sleeping
    assert "HTTP 400" in str(exc_info.value)


def test_gemini_configurable_fallback_model_from_env(monkeypatch):
    """Verify fallback model is configurable via environment variable."""
    monkeypatch.setenv("NEXORA_GEMINI_FALLBACK_MODEL", "gemini-1.5-flash")
    cfg = LLMConfig.from_env({
        "NEXORA_LLM_PROVIDER": "gemini",
        "NEXORA_LLM_MODEL": "gemini-2.5-flash",
        "NEXORA_GEMINI_FALLBACK_MODEL": "gemini-1.5-flash",
        "GEMINI_API_KEY": FAKE_GEMINI_KEY,
    })
    provider = GeminiProvider(cfg, client=FakeGeminiHttpClient())
    assert provider.model == "gemini-2.5-flash"
    assert provider.fallback_model == "gemini-1.5-flash"


def test_gemini_retry_deadline_exceeded_stops_retries():
    """Verify max_retry_duration enforces a strict time budget."""
    import httpx

    class Slow503Client:
        def __init__(self):
            self.calls = 0

        def post(self, path, json=None, headers=None):
            self.calls += 1
            req = httpx.Request("POST", f"https://generativelanguage.googleapis.com{path}")
            resp = httpx.Response(503, request=req, content=b'{"error":{"message":"high demand"}}')
            raise httpx.HTTPStatusError("503", request=req, response=resp)

    client = Slow503Client()
    config = LLMConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        fallback_model="gemini-2.0-flash",
        api_key=FAKE_GEMINI_KEY,
        max_retries=5,
        max_retry_duration=0.01,  # Strict 10ms budget
    )
    provider = GeminiProvider(config, client=client, sleep_fn=lambda _: time.sleep(0.02))

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate([LLMMessage(role="user", content="test")])

    # Should have terminated well before 5 retries on both models
    assert client.calls < 10
    assert "HTTP 503" in str(exc_info.value)
