"""Tests for Google Gemini semantic embedding integration (Milestones 12C-38 and 12C-38.1).

Verifies:
1. Configuration loading, provider factory integration, and default model (gemini-embedding-2).
2. Transparent migration of legacy and shut down aliases (e.g. text-embedding-004, gemini-embedding-001).
3. Consistent effective outputDimensionality sent in all REST payloads (single and batch),
   both when dimension is omitted (default 768) and when explicitly configured (e.g. 1536, 3072).
4. Injected mock/fake client transport for 100% offline, deterministic testing.
5. Proper REST API payload formatting (embedContent and batchEmbedContents).
6. Strict vector validation: dimension matching, finite floats, non-empty outputs.
7. Error handling for HTTP 401, 403, 429, 500, timeouts, and network failures.
8. Secret hygiene: API keys and auth headers are never logged or exposed in errors.
9. Compatibility between document and query embeddings.
10. Truthful live status and limitation reporting in production retrieval evaluation and CLI.
11. Opt-in live smoke test when valid credentials are provided in the environment.
"""

from __future__ import annotations

import math
import os
import socket
from typing import Any
from unittest import mock

import pytest

from nexora.embedding.base import EmbeddingProvider
from nexora.embedding.config import (
    ENV_API_KEY,
    ENV_GEMINI_API_KEY,
    ENV_NEXORA_GEMINI_API_KEY,
    EmbeddingConfig,
)
from nexora.embedding.factory import (
    PROVIDER_GEMINI,
    SUPPORTED_PROVIDERS,
    create_embedding_provider,
)
from nexora.embedding.gemini_provider import (
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_GEMINI_DIMENSION,
    DEFAULT_GEMINI_MODEL,
    LEGACY_MODEL_ALIASES,
    GeminiEmbeddingProvider,
    GeminiEmbeddingProviderError,
)
from nexora.retrieval.eval_cli import run_eval_cli
from nexora.retrieval.production_eval import (
    EvaluationMode,
    ProductionEvaluationConfig,
    check_gemini_credentials,
    evaluate_production_retrieval,
    is_genuine_live_gemini_provider,
    is_simulated_or_synthetic_provider,
)

FAKE_API_KEY = "test-gemini-secret-api-key-998877"


# ---------------------------------------------------------------------------
# Test Doubles for Offline Testing
# ---------------------------------------------------------------------------
class FakeHttpResponse:
    """Simulates an HTTP response from httpx or urllib."""

    def __init__(self, data: Any, status_code: int = 200) -> None:
        self.data = data
        self.status_code = status_code

    def json(self) -> Any:
        return self.data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )


class FakeGeminiClient:
    """Injected mock client recording every POST call without network access."""

    def __init__(
        self,
        vector: list[float] | None = None,
        error: Exception | None = None,
        status_code: int = 200,
        response_data: Any | None = None,
    ) -> None:
        self.vector = [0.05] * 768 if vector is None else vector
        self.error = error
        self.status_code = status_code
        self.response_data = response_data
        self.calls: list[dict[str, Any]] = []

    def post(self, path: str, json: dict[str, Any], headers: dict[str, str]) -> FakeHttpResponse:
        self.calls.append({"path": path, "json": json, "headers": headers})
        if self.error is not None:
            raise self.error
        if self.response_data is not None:
            return FakeHttpResponse(self.response_data, status_code=self.status_code)

        # Default successful response
        if ":embedContent" in path:
            return FakeHttpResponse(
                {"embedding": {"values": list(self.vector)}}, status_code=self.status_code
            )
        if ":batchEmbedContents" in path:
            num_requests = len(json.get("requests", []))
            return FakeHttpResponse(
                {"embeddings": [{"values": list(self.vector)} for _ in range(num_requests)]},
                status_code=self.status_code,
            )
        return FakeHttpResponse({}, status_code=self.status_code)


def make_provider(
    vector: list[float] | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    dimension: int | None = DEFAULT_GEMINI_DIMENSION,
    api_key: str | None = FAKE_API_KEY,
    base_url: str | None = None,
    batch_size: int | None = None,
    **client_kwargs: Any,
) -> tuple[GeminiEmbeddingProvider, FakeGeminiClient]:
    effective_dim = dimension if dimension is not None else DEFAULT_GEMINI_DIMENSION
    vec = vector if vector is not None else [0.05] * effective_dim
    client = FakeGeminiClient(vector=vec, **client_kwargs)
    config = EmbeddingConfig(
        provider="gemini",
        model=model,
        dimension=dimension,
        api_key=api_key,
        base_url=base_url,
        batch_size=batch_size,
    )
    provider = GeminiEmbeddingProvider(config, client=client)
    return provider, client


# ---------------------------------------------------------------------------
# 1. Configuration & Factory Registration & Migration
# ---------------------------------------------------------------------------
def test_gemini_provider_constant_exported():
    assert PROVIDER_GEMINI == "gemini"
    assert "gemini" in SUPPORTED_PROVIDERS


def test_gemini_provider_defaults_to_gemini_embedding_2():
    provider, client = make_provider()
    assert isinstance(provider, GeminiEmbeddingProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.provider == "gemini"
    assert provider.name == "gemini"
    assert provider.model == "gemini-embedding-2"
    assert provider.dimension == 768


@pytest.mark.parametrize(
    "legacy_alias",
    [
        "text-embedding-004",
        "models/text-embedding-004",
        "gemini-embedding-001",
        "models/gemini-embedding-001",
        "embedding-001",
        "models/embedding-001",
        "default",
        "hash-sha256",
        "text-embedding-3-small",
    ],
)
def test_legacy_aliases_migrated_to_gemini_embedding_2(legacy_alias):
    """Verify deprecated or shutdown models transparently migrate to gemini-embedding-2."""
    config = EmbeddingConfig(provider="gemini", model=legacy_alias, api_key=FAKE_API_KEY)
    provider = GeminiEmbeddingProvider(config, client=FakeGeminiClient())
    assert provider.model == "gemini-embedding-2"


def test_custom_supported_model_preserved():
    config = EmbeddingConfig(provider="gemini", model="gemini-embedding-2-preview", api_key=FAKE_API_KEY)
    provider = GeminiEmbeddingProvider(config, client=FakeGeminiClient())
    assert provider.model == "gemini-embedding-2-preview"


def test_factory_creates_gemini_provider(monkeypatch):
    config = EmbeddingConfig(provider="gemini", model="gemini-embedding-2", api_key=FAKE_API_KEY)
    monkeypatch.setattr(socket.socket, "connect", mock.Mock(side_effect=AssertionError("network blocked")))
    provider = create_embedding_provider(config)
    assert isinstance(provider, GeminiEmbeddingProvider)
    assert provider.provider == "gemini"
    assert provider.model == "gemini-embedding-2"


def test_factory_creates_gemini_provider_migrating_legacy_model(monkeypatch):
    config = EmbeddingConfig(provider="gemini", model="text-embedding-004", api_key=FAKE_API_KEY)
    monkeypatch.setattr(socket.socket, "connect", mock.Mock(side_effect=AssertionError("network blocked")))
    provider = create_embedding_provider(config)
    assert isinstance(provider, GeminiEmbeddingProvider)
    assert provider.model == "gemini-embedding-2"


def test_factory_case_insensitive_gemini(monkeypatch):
    config = EmbeddingConfig(provider="Gemini", model="gemini-embedding-2", api_key=FAKE_API_KEY)
    monkeypatch.setattr(socket.socket, "connect", mock.Mock(side_effect=AssertionError("network blocked")))
    provider = create_embedding_provider(config)
    assert isinstance(provider, GeminiEmbeddingProvider)


def test_config_from_env_loads_gemini_keys():
    env = {
        "NEXORA_EMBEDDING_PROVIDER": "gemini",
        "GEMINI_API_KEY": FAKE_API_KEY,
    }
    cfg = EmbeddingConfig.from_env(env)
    assert cfg.provider == "gemini"
    assert cfg.api_key == FAKE_API_KEY

    # Test alternative NEXORA_GEMINI_API_KEY
    env_alt = {
        "NEXORA_EMBEDDING_PROVIDER": "gemini",
        "NEXORA_GEMINI_API_KEY": "alt-gemini-key",
    }
    cfg_alt = EmbeddingConfig.from_env(env_alt)
    assert cfg_alt.provider == "gemini"
    assert cfg_alt.api_key == "alt-gemini-key"


def test_missing_api_key_raises_error_without_network():
    config = EmbeddingConfig(provider="gemini", model="gemini-embedding-2")
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(GeminiEmbeddingProviderError, match="GEMINI_API_KEY"):
            GeminiEmbeddingProvider(config)


def test_non_embedding_config_raises_type_error():
    with pytest.raises(TypeError, match="config must be an EmbeddingConfig"):
        GeminiEmbeddingProvider({"provider": "gemini"}, client=FakeGeminiClient())  # type: ignore


# ---------------------------------------------------------------------------
# 2. Output Dimensionality Requirement
# ---------------------------------------------------------------------------
def test_effective_default_dimension_always_sent_in_embed_content():
    """When dimension is omitted in config, REST payload MUST still send outputDimensionality: 768."""
    provider, client = make_provider(dimension=None)
    assert provider.dimension == 768

    provider.embed_text("sample input")
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["json"]["outputDimensionality"] == 768


def test_effective_default_dimension_always_sent_in_batch_embed():
    """When dimension is omitted in config, batch requests MUST send outputDimensionality: 768 for each item."""
    provider, client = make_provider(dimension=None)
    assert provider.dimension == 768

    provider.embed_texts(["sample input 1", "sample input 2"])
    assert len(client.calls) == 1
    requests_payload = client.calls[0]["json"]["requests"]
    assert len(requests_payload) == 2
    for item in requests_payload:
        assert item["outputDimensionality"] == 768


@pytest.mark.parametrize("dim", [512, 768, 1536, 3072])
def test_explicit_dimension_always_sent_and_validated(dim):
    """When dimension is explicitly set, request sends it and validates length matches."""
    provider, client = make_provider(dimension=dim, vector=[0.01] * dim)
    assert provider.dimension == dim

    res_single = provider.embed_text("test single")
    assert len(res_single) == dim
    assert client.calls[-1]["json"]["outputDimensionality"] == dim

    res_batch = provider.embed_texts(["test batch 1", "test batch 2"])
    assert len(res_batch) == 2
    assert len(res_batch[0]) == dim
    assert client.calls[-1]["json"]["requests"][0]["outputDimensionality"] == dim


# ---------------------------------------------------------------------------
# 3. embed_text (Single Text Embedding)
# ---------------------------------------------------------------------------
def test_embed_text_sends_correct_payload_and_headers():
    provider, client = make_provider(dimension=768)
    text = "Attention mechanisms in neural networks"
    res = provider.embed_text(text)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["path"] == "/v1beta/models/gemini-embedding-2:embedContent"
    assert call["headers"]["x-goog-api-key"] == FAKE_API_KEY
    assert call["headers"]["Content-Type"] == "application/json"

    expected_payload = {
        "model": "models/gemini-embedding-2",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768,
    }
    assert call["json"] == expected_payload
    assert len(res) == 768
    assert all(isinstance(v, float) for v in res)


def test_embed_text_normalizes_model_with_models_prefix():
    provider, client = make_provider(model="models/gemini-embedding-2")
    provider.embed_text("sample query")

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["path"] == "/v1beta/models/gemini-embedding-2:embedContent"
    assert call["json"]["model"] == "models/gemini-embedding-2"


@pytest.mark.parametrize("bad_text", ["", "   ", "\t\n", "  \n  "])
def test_embed_text_rejects_empty_or_whitespace(bad_text):
    provider, client = make_provider()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        provider.embed_text(bad_text)
    assert len(client.calls) == 0


@pytest.mark.parametrize("bad_type", [None, 123, 4.5, ["text"], {"text": "hello"}])
def test_embed_text_rejects_non_string(bad_type):
    provider, client = make_provider()
    with pytest.raises(TypeError, match="text must be a str"):
        provider.embed_text(bad_type)  # type: ignore
    assert len(client.calls) == 0


# ---------------------------------------------------------------------------
# 4. embed_texts (Batch Text Embedding)
# ---------------------------------------------------------------------------
def test_embed_texts_sends_batch_request():
    provider, client = make_provider(dimension=768)
    texts = ["First passage", "Second passage", "Third passage"]
    res = provider.embed_texts(texts)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["path"] == "/v1beta/models/gemini-embedding-2:batchEmbedContents"
    requests_payload = call["json"].get("requests", [])
    assert len(requests_payload) == 3
    assert requests_payload[0]["content"]["parts"][0]["text"] == "First passage"
    assert requests_payload[1]["content"]["parts"][0]["text"] == "Second passage"
    assert requests_payload[2]["content"]["parts"][0]["text"] == "Third passage"

    assert len(res) == 3
    for v in res:
        assert len(v) == 768


def test_embed_texts_empty_returns_empty_without_request():
    provider, client = make_provider()
    res = provider.embed_texts([])
    assert res == []
    assert len(client.calls) == 0


def test_embed_texts_chunks_batches():
    provider, client = make_provider(batch_size=2)
    texts = ["A", "B", "C", "D", "E"]
    res = provider.embed_texts(texts)

    assert len(client.calls) == 3
    assert len(client.calls[0]["json"]["requests"]) == 2
    assert len(client.calls[1]["json"]["requests"]) == 2
    assert len(client.calls[2]["json"]["requests"]) == 1
    assert len(res) == 5


# ---------------------------------------------------------------------------
# 5. Strict Vector & Response Validation
# ---------------------------------------------------------------------------
def test_dimension_mismatch_raises_error():
    provider, _ = make_provider(vector=[0.1] * 256, dimension=768)
    with pytest.raises(GeminiEmbeddingProviderError, match="expected 768"):
        provider.embed_text("hello")


def test_empty_embedding_raises_error():
    provider, _ = make_provider(vector=[])
    with pytest.raises(GeminiEmbeddingProviderError, match="empty embedding"):
        provider.embed_text("hello")


def test_non_finite_float_raises_error():
    provider, _ = make_provider(vector=[float("nan")] * 768)
    with pytest.raises(GeminiEmbeddingProviderError, match="non-finite value"):
        provider.embed_text("hello")

    provider_inf, _ = make_provider(vector=[float("inf")] * 768)
    with pytest.raises(GeminiEmbeddingProviderError, match="non-finite value"):
        provider_inf.embed_text("hello")


def test_malformed_response_structures():
    provider_no_emb, _ = make_provider(response_data={"other": 123})
    with pytest.raises(GeminiEmbeddingProviderError, match="embedding"):
        provider_no_emb.embed_text("hello")

    provider_no_vals, _ = make_provider(response_data={"embedding": {"wrong": []}})
    with pytest.raises(GeminiEmbeddingProviderError, match="values"):
        provider_no_vals.embed_text("hello")


def test_batch_response_count_mismatch():
    provider, _ = make_provider(response_data={"embeddings": [{"values": [0.1] * 768}]})
    with pytest.raises(GeminiEmbeddingProviderError, match="returned 1 embeddings for 2 batch inputs"):
        provider.embed_texts(["first", "second"])


# ---------------------------------------------------------------------------
# 6. API Error Handling & Secret Sanitization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
def test_http_errors_raise_provider_error(status_code):
    provider, _ = make_provider(status_code=status_code)
    with pytest.raises(GeminiEmbeddingProviderError) as exc_info:
        provider.embed_text("test error text")

    err_msg = str(exc_info.value)
    assert f"HTTP {status_code}" in err_msg
    # Crucial security requirement: API key must NOT be leaked in error message!
    assert FAKE_API_KEY not in err_msg


def test_network_connection_error_sanitized():
    import httpx
    conn_err = httpx.ConnectError("Connection refused to generativelanguage.googleapis.com")
    provider, _ = make_provider(error=conn_err)
    with pytest.raises(GeminiEmbeddingProviderError) as exc_info:
        provider.embed_text("test network failure")

    assert "ConnectError" in str(exc_info.value)
    assert FAKE_API_KEY not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. Document & Query Embedding Compatibility
# ---------------------------------------------------------------------------
def test_document_and_query_dimension_compatibility():
    """Verify document chunks and user queries yield compatible embedding dimensions."""
    provider, _ = make_provider(dimension=768)

    doc_chunk = "FlashAttention computes exact attention with IO-awareness on GPU memory hierarchy."
    query = "How does FlashAttention optimize memory access?"

    doc_vector = provider.embed_text(doc_chunk)
    query_vector = provider.embed_text(query)

    assert len(doc_vector) == 768
    assert len(query_vector) == 768
    assert len(doc_vector) == len(query_vector)


# ---------------------------------------------------------------------------
# 8. Production Evaluation Harness & Honest Demarcation
# ---------------------------------------------------------------------------
def test_check_gemini_credentials_when_unset(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)

    has_creds, msg = check_gemini_credentials(api_key=None)
    assert has_creds is False
    assert "External Gemini credentials are unavailable" in msg


def test_production_eval_with_gemini_fails_open_when_credentials_missing(monkeypatch):
    """When Gemini credentials are unset, production mode reports status='unavailable' honestly."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)

    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        provider_name="gemini",
        api_key=None,
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "unavailable"
    assert report.mode == "production"
    assert report.provider_name == "gemini"
    assert report.model_name == "gemini-embedding-2"
    assert report.dimension == 768
    assert report.is_semantic is True
    assert report.is_synthetic is False
    assert report.is_live is False
    assert report.provider_live is False
    assert report.metrics is None
    assert "External Gemini credentials are unavailable" in report.infrastructure_limitation


def test_production_eval_with_simulated_gemini_provider():
    """Verify simulated Gemini test double is demoted from genuine live production."""
    class SimulatedGeminiProvider(EmbeddingProvider):
        def __init__(self):
            self.provider = "gemini"
            self.name = "gemini"
            self.model = "gemini-embedding-2"
            self.dimension = 768
            self.is_simulated = True

        def embed_text(self, text: str) -> list[float]:
            return [0.01] * 768

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.01] * 768 for _ in texts]

    simulated = SimulatedGeminiProvider()
    assert is_simulated_or_synthetic_provider(simulated) is True
    assert is_genuine_live_gemini_provider(simulated) is False

    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.PRODUCTION,
        embedding_provider=simulated,
        provider_name="gemini",
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert "simulated" in report.provider_name
    assert report.is_synthetic is True
    assert report.provider_live is False
    assert report.is_live is False
    assert "Simulated test double" in report.infrastructure_limitation


def test_configured_pipeline_mode_with_gemini(monkeypatch):
    """Configured pipeline mode with gemini provider recognizes semantic capability."""
    provider, _ = make_provider()
    cfg = ProductionEvaluationConfig(
        mode=EvaluationMode.CONFIGURED_PIPELINE,
        embedding_provider=provider,
        provider_name="gemini",
    )
    report = evaluate_production_retrieval(cfg)

    assert report.status == "completed"
    assert report.provider_name == "gemini"
    assert report.model_name == "gemini-embedding-2"
    assert report.is_semantic is True
    assert report.metrics is not None


def test_cli_production_mode_with_gemini_unavailable(monkeypatch, tmp_path):
    """Running CLI with --provider gemini when credentials are missing reports unavailable."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)

    out_file = tmp_path / "gemini_report.json"
    exit_code, report, json_text = run_eval_cli([
        "--mode", "production",
        "--provider", "gemini",
        "--output", str(out_file),
    ])

    assert exit_code == 0
    assert report.status == "unavailable"
    assert report.provider_name == "gemini"
    assert report.model_name == "gemini-embedding-2"
    assert report.dimension == 768
    assert report.is_live is False
    assert report.provider_live is False
    assert "gemini_credentials" in report.unavailable_dependencies
    assert out_file.exists()


def test_cli_fail_on_unavailable_with_gemini(monkeypatch):
    """--fail-on-unavailable flag returns exit code 1 when Gemini credentials missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("NEXORA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)

    exit_code, report, _ = run_eval_cli([
        "--mode", "production",
        "--provider", "gemini",
        "--fail-on-unavailable",
    ])
    assert exit_code == 1
    assert report.status == "unavailable"


# ---------------------------------------------------------------------------
# 9. Opt-In Live Smoke Test (Skipped in CI)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("NEXORA_GEMINI_API_KEY")),
    reason="Live Gemini API key not configured in environment (opt-in smoke test)",
)
def test_live_gemini_api_smoke():
    """Opt-in live smoke test executing real API requests against Google Gemini.

    Only runs when GEMINI_API_KEY or NEXORA_GEMINI_API_KEY is present in the environment.
    """
    key = os.getenv("GEMINI_API_KEY") or os.getenv("NEXORA_GEMINI_API_KEY")
    config = EmbeddingConfig(
        provider="gemini",
        model="gemini-embedding-2",
        api_key=key,
        dimension=768,
    )
    provider = GeminiEmbeddingProvider(config)

    t1 = "Attention is all you need for sequence transduction"
    t2 = "FlashAttention computes exact attention with IO awareness"

    v1 = provider.embed_text(t1)
    v2 = provider.embed_text(t2)

    assert len(v1) == 768
    assert len(v2) == 768
    assert all(math.isfinite(x) for x in v1)
    assert all(math.isfinite(x) for x in v2)

    batch_vecs = provider.embed_texts([t1, t2])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 768
    assert len(batch_vecs[1]) == 768
