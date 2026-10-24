"""Tests for the production ASGI entrypoint and application factory (Milestone 31)."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexora.api.app import _LazyASGIApp, create_offline_app
from nexora.api.errors import InvalidConfigurationError
from nexora.config.settings import ConfigurationError
from nexora.database.models import Base
from nexora.database.session import create_session_factory
from nexora.embedding.config import EmbeddingConfig
from nexora.embedding.factory import create_embedding_provider
from nexora.embedding.hash_provider import DEFAULT_DIMENSION as DEFAULT_HASH_DIMENSION
from nexora.retrieval.models import RetrievedChunk, SearchResult
from nexora.server import (
    _LazyProductionASGIApp,
    _resolve_dimension_for_config,
    _resolve_embedding_dimension,
    create_production_app,
)
from sqlalchemy import create_engine

SECRET_KEY = "test-sk-supersecret-llm-key-999"
SECRET_DB_PASSWORD = "supersecret-password-xyz"
TEST_DB_URL = f"postgresql://nexora_user:{SECRET_DB_PASSWORD}@localhost:5432/nexora_prod"


class FakeEmbeddingService:
    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        return [0.1] * self.dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]


class FakeVectorStore:
    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension
        self.chunks = [
            RetrievedChunk(
                document_id="doc-1",
                chunk_id="chk-1",
                text="sample research content",
                score=0.95,
                metadata={},
            )
        ]

    def upsert(self, records: Any) -> None:
        pass

    def delete(self, ids: Any) -> None:
        pass

    def count(self) -> int:
        return len(self.chunks)

    def search(self, vector: Any, *, limit: int = 10, collection_id: Any = None) -> SearchResult:
        return SearchResult(query="", results=list(self.chunks))


class FakeLLMProvider:
    def generate(self, messages: Any, *, model: Any = None, temperature: Any = None, max_output_tokens: Any = None) -> Any:
        from nexora.llm.models import LLMResponse
        return LLMResponse(text="Production verified answer", model="gpt-4o-mini", usage={"total_tokens": 10})


@pytest.fixture
def base_prod_env() -> dict[str, str]:
    return {
        "NEXORA_DATABASE_URL": TEST_DB_URL,
        "NEXORA_LLM_API_KEY": SECRET_KEY,
        "NEXORA_LLM_MODEL": "gpt-4o-mini",
        "NEXORA_EMBEDDING_PROVIDER": "hash",
        "NEXORA_VECTOR_STORE": "memory",
    }


def test_uvicorn_factory_zero_argument_signature() -> None:
    """Verify that create_production_app can be invoked with zero arguments as required by Uvicorn --factory."""
    sig = inspect.signature(create_production_app)
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            assert param.default is not inspect.Parameter.empty, f"Parameter {param_name} must have a default"


def test_production_factory_success_with_injected_dependencies(base_prod_env: dict[str, str]) -> None:
    mock_session_factory = MagicMock()
    mock_auth_service = MagicMock()
    emb_service = FakeEmbeddingService(dimension=128)
    store = FakeVectorStore(dimension=128)
    provider = FakeLLMProvider()

    app = create_production_app(
        base_prod_env,
        session_factory=mock_session_factory,
        auth_service=mock_auth_service,
        embedding_service=emb_service,
        vector_store=store,
        llm_provider=provider,
    )

    assert isinstance(app, FastAPI)
    assert app.state.enforce_auth is True
    assert app.state.session_factory is mock_session_factory
    assert app.state.auth_service is mock_auth_service
    assert app.state.research_service is not None

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_production_factory_success_from_env_with_mocked_db_engine(
    base_prod_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_engine = SimpleNamespace()
    mock_factory = MagicMock()

    monkeypatch.setattr(
        "nexora.database.connection.create_database_engine",
        lambda cfg: mock_engine,
    )
    monkeypatch.setattr(
        "nexora.database.session.create_session_factory",
        lambda eng: mock_factory,
    )

    app = create_production_app(base_prod_env)

    assert isinstance(app, FastAPI)
    assert app.state.enforce_auth is True
    assert app.state.session_factory is mock_factory
    assert app.state.auth_service is not None

    client = TestClient(app)
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}


def test_zero_argument_factory_invocation(
    base_prod_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for k, v in base_prod_env.items():
        monkeypatch.setenv(k, v)

    mock_engine = SimpleNamespace()
    mock_factory = MagicMock()

    monkeypatch.setattr(
        "nexora.database.connection.create_database_engine",
        lambda cfg: mock_engine,
    )
    monkeypatch.setattr(
        "nexora.database.session.create_session_factory",
        lambda eng: mock_factory,
    )

    # Calling with zero arguments matches uvicorn --factory invocation
    app = create_production_app()
    assert isinstance(app, FastAPI)
    assert app.state.enforce_auth is True
    assert app.state.session_factory is mock_factory


def test_lazy_production_asgi_app_callable(
    base_prod_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for k, v in base_prod_env.items():
        monkeypatch.setenv(k, v)

    mock_engine = SimpleNamespace()
    mock_factory = MagicMock()

    monkeypatch.setattr(
        "nexora.database.connection.create_database_engine",
        lambda cfg: mock_engine,
    )
    monkeypatch.setattr(
        "nexora.database.session.create_session_factory",
        lambda eng: mock_factory,
    )

    lazy_app = _LazyProductionASGIApp()
    client = TestClient(lazy_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert lazy_app.state.enforce_auth is True


def test_asgi_lifespan_startup_and_shutdown_forwarding(
    base_prod_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that _LazyProductionASGIApp correctly forwards lifespan events to FastAPI."""
    for k, v in base_prod_env.items():
        monkeypatch.setenv(k, v)

    mock_engine = SimpleNamespace()
    mock_factory = MagicMock()

    monkeypatch.setattr(
        "nexora.database.connection.create_database_engine",
        lambda cfg: mock_engine,
    )
    monkeypatch.setattr(
        "nexora.database.session.create_session_factory",
        lambda eng: mock_factory,
    )

    lazy_app = _LazyProductionASGIApp()

    incoming_messages: list[dict[str, Any]] = [
        {"type": "lifespan.startup"},
        {"type": "lifespan.shutdown"},
    ]
    sent_messages: list[dict[str, Any]] = []

    async def mock_receive() -> dict[str, Any]:
        return incoming_messages.pop(0)

    async def mock_send(msg: dict[str, Any]) -> None:
        sent_messages.append(msg)

    scope = {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}}
    asyncio.run(lazy_app(scope, mock_receive, mock_send))

    sent_types = [m["type"] for m in sent_messages]
    assert sent_types == ["lifespan.startup.complete", "lifespan.shutdown.complete"]


def test_asgi_lifespan_startup_failure_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that startup failures during lifespan send lifespan.startup.failed and raise."""
    for var in ("NEXORA_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)

    lazy_app = _LazyProductionASGIApp()

    incoming_messages: list[dict[str, Any]] = [{"type": "lifespan.startup"}]
    sent_messages: list[dict[str, Any]] = []

    async def mock_receive() -> dict[str, Any]:
        return incoming_messages.pop(0)

    async def mock_send(msg: dict[str, Any]) -> None:
        sent_messages.append(msg)

    scope = {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}}

    with pytest.raises(ConfigurationError):
        asyncio.run(lazy_app(scope, mock_receive, mock_send))

    assert len(sent_messages) == 1
    assert sent_messages[0]["type"] == "lifespan.startup.failed"
    assert "PostgreSQL database" in sent_messages[0]["message"]


def test_actual_authentication_enforced_on_endpoints(base_prod_env: dict[str, str]) -> None:
    """Verify that actual protected endpoints return HTTP 401 when called without valid credentials."""
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        from nexora.database.repository.api_key import ApiKeyRepository
        from nexora.database.repository.user import UserRepository

        user = UserRepository(session).create(email="test@nexora.ai")
        _, raw_key = ApiKeyRepository(session).generate_key(user_id=user.id, name="Test Key")
        session.commit()

    from nexora.auth.service import AuthenticationService
    auth_service = AuthenticationService(session_factory)

    app = create_production_app(
        base_prod_env,
        session_factory=session_factory,
        auth_service=auth_service,
        embedding_service=FakeEmbeddingService(dimension=128),
        vector_store=FakeVectorStore(dimension=128),
        llm_provider=FakeLLMProvider(),
    )

    client = TestClient(app)

    import uuid
    dummy_col_id = str(uuid.uuid4())

    # 1. Anonymous research targeting a private collection -> 401
    resp_research_col_unauth = client.post(
        "/api/v1/research", json={"query": "What is attention?", "collection_id": dummy_col_id}
    )
    assert resp_research_col_unauth.status_code == 401
    assert resp_research_col_unauth.json()["error"]["code"] == "unauthorized"

    # 1b. Anonymous public research without collection_id -> 200 (isolated benchmark corpus)
    resp_research_unauth = client.post("/api/v1/research", json={"query": "What is attention?"})
    assert resp_research_unauth.status_code == 200

    # 2. Collection details endpoint without credentials -> 401
    resp_col_unauth = client.get(f"/api/v1/collections/{dummy_col_id}")
    assert resp_col_unauth.status_code == 401
    assert resp_col_unauth.json()["error"]["code"] == "unauthorized"

    # 3. Protected endpoint with invalid credentials -> 401
    resp_bad = client.post(
        "/api/v1/research",
        json={"query": "What is attention?"},
        headers={"Authorization": "Bearer invalid_secret_token"},
    )
    assert resp_bad.status_code == 401
    assert resp_bad.json()["error"]["code"] == "unauthorized"

    # 4. Protected endpoint with valid credentials -> 200
    resp_valid = client.post(
        "/api/v1/research",
        json={"query": "What is attention?"},
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert resp_valid.status_code == 200
    assert "text" in resp_valid.json()


def test_missing_database_url_raises_configuration_error() -> None:
    env = {
        "NEXORA_LLM_API_KEY": SECRET_KEY,
        "NEXORA_LLM_MODEL": "gpt-4o-mini",
    }
    with pytest.raises(ConfigurationError) as exc_info:
        create_production_app(env)

    msg = str(exc_info.value)
    assert "PostgreSQL database" in msg or "database configuration" in msg
    assert SECRET_KEY not in msg


def test_database_backed_startup_succeeds_without_llm_key_in_offline_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that database-backed production mode boots cleanly without an LLM key when offline generation is allowed."""
    mock_engine = SimpleNamespace()
    mock_factory = MagicMock()

    monkeypatch.setattr(
        "nexora.database.connection.create_database_engine",
        lambda cfg: mock_engine,
    )
    monkeypatch.setattr(
        "nexora.database.session.create_session_factory",
        lambda eng: mock_factory,
    )

    env = {
        "NEXORA_DATABASE_URL": TEST_DB_URL,
        "NEXORA_LLM_PROVIDER": "openai",
        "NEXORA_LLM_MODEL": "gpt-4o-mini",
        "NEXORA_EMBEDDING_PROVIDER": "hash",
        # No LLM API key
    }
    app = create_production_app(env)
    assert isinstance(app, FastAPI)
    assert app.state.enforce_auth is True
    assert app.state.session_factory is mock_factory

    client = TestClient(app)
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}


def test_missing_llm_key_raises_when_live_generation_requested() -> None:
    """Verify that explicitly requested live generation fails clearly when LLM credentials are missing."""
    env = {
        "NEXORA_DATABASE_URL": TEST_DB_URL,
        "NEXORA_LLM_PROVIDER": "openai",
        "NEXORA_LLM_MODEL": "gpt-4o-mini",
        "NEXORA_RAG_GENERATOR_MODE": "live",
        # No LLM API key
    }
    with pytest.raises(ConfigurationError) as exc_info:
        create_production_app(env)

    msg = str(exc_info.value)
    assert "api_key" in msg.lower() or "llm" in msg.lower()
    assert SECRET_DB_PASSWORD not in msg


def test_missing_llm_key_raises_when_require_llm_key_flag_enabled() -> None:
    """Verify that NEXORA_REQUIRE_LLM_KEY=true enforces LLM credentials on startup."""
    env = {
        "NEXORA_DATABASE_URL": TEST_DB_URL,
        "NEXORA_LLM_PROVIDER": "openai",
        "NEXORA_LLM_MODEL": "gpt-4o-mini",
        "NEXORA_REQUIRE_LLM_KEY": "true",
        # No LLM API key
    }
    with pytest.raises(ConfigurationError) as exc_info:
        create_production_app(env)

    msg = str(exc_info.value)
    assert "api_key" in msg.lower() or "llm" in msg.lower()
    assert SECRET_DB_PASSWORD not in msg


def test_offline_demo_mode_starts_without_api_key_or_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that offline demo mode starts without database credentials or LLM key."""
    monkeypatch.delenv("APP_ENV", raising=False)
    app = create_offline_app()
    assert isinstance(app, FastAPI)
    assert app.state.enforce_auth is False
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_invalid_vector_store_backend_raises_configuration_error(base_prod_env: dict[str, str]) -> None:
    env = dict(base_prod_env, NEXORA_VECTOR_STORE="unsupported_store")
    with pytest.raises(ConfigurationError) as exc_info:
        create_production_app(env)

    assert "unsupported_store" in str(exc_info.value).lower()


def test_qdrant_vector_store_requires_url(base_prod_env: dict[str, str]) -> None:
    env = dict(base_prod_env, NEXORA_VECTOR_STORE="qdrant")
    with pytest.raises(ConfigurationError) as exc_info:
        create_production_app(env)

    assert "NEXORA_QDRANT_URL" in str(exc_info.value)


def test_qdrant_vector_store_valid_configuration(
    base_prod_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = dict(
        base_prod_env,
        NEXORA_VECTOR_STORE="qdrant",
        NEXORA_QDRANT_URL="http://localhost:6333",
        NEXORA_QDRANT_COLLECTION="custom_papers",
    )
    mock_client = MagicMock()
    mock_engine = SimpleNamespace()
    mock_factory = MagicMock()

    monkeypatch.setattr(
        "nexora.database.connection.create_database_engine",
        lambda cfg: mock_engine,
    )
    monkeypatch.setattr(
        "nexora.database.session.create_session_factory",
        lambda eng: mock_factory,
    )

    app = create_production_app(env, qdrant_client=mock_client)
    assert isinstance(app, FastAPI)
    assert app.state.enforce_auth is True


def test_independent_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that only credentials for the selected provider are required."""
    mock_engine = SimpleNamespace()
    mock_factory = MagicMock()

    monkeypatch.setattr(
        "nexora.database.connection.create_database_engine",
        lambda cfg: mock_engine,
    )
    monkeypatch.setattr(
        "nexora.database.session.create_session_factory",
        lambda eng: mock_factory,
    )

    # 1. Hash embedding + OpenAI LLM: no embedding key needed
    env1 = {
        "NEXORA_DATABASE_URL": TEST_DB_URL,
        "NEXORA_LLM_API_KEY": SECRET_KEY,
        "NEXORA_LLM_MODEL": "gpt-4o-mini",
        "NEXORA_EMBEDDING_PROVIDER": "hash",
    }
    app1 = create_production_app(env1)
    assert isinstance(app1, FastAPI)

    # 2. Gemini embedding: requires GEMINI_API_KEY, does not require OpenAI embedding key
    env2 = {
        "NEXORA_DATABASE_URL": TEST_DB_URL,
        "NEXORA_LLM_API_KEY": SECRET_KEY,
        "NEXORA_LLM_MODEL": "gpt-4o-mini",
        "NEXORA_EMBEDDING_PROVIDER": "gemini",
        "GEMINI_API_KEY": "fake-gemini-key",
    }
    app2 = create_production_app(env2)
    assert isinstance(app2, FastAPI)


def test_secrets_hygiene_in_exception_messages() -> None:
    bad_db_url = f"postgresql://admin:{SECRET_DB_PASSWORD}@bad-host:invalid-port/db"
    env = {
        "NEXORA_DATABASE_URL": bad_db_url,
        "NEXORA_LLM_API_KEY": SECRET_KEY,
        "NEXORA_LLM_MODEL": "gpt-4o-mini",
    }
    with pytest.raises(ConfigurationError) as exc_info:
        create_production_app(env)

    err_str = str(exc_info.value)
    assert SECRET_DB_PASSWORD not in err_str
    assert SECRET_KEY not in err_str


@pytest.mark.parametrize(
    ("env_var", "env_val"),
    [
        ("APP_ENV", "production"),
        ("APP_ENV", "prod"),
        ("APP_ENV", "PRODUCTION"),
        ("ENVIRONMENT", "production"),
        ("ENVIRONMENT", "prod"),
        ("NEXORA_APP_MODE", "production"),
        ("NEXORA_APP_MODE", "prod"),
    ],
)
def test_create_offline_app_safeguard_blocks_production_mode(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    env_val: str,
) -> None:
    monkeypatch.setenv(env_var, env_val)
    with pytest.raises(InvalidConfigurationError) as exc_info:
        create_offline_app()

    msg = str(exc_info.value)
    assert "Attempted to launch offline demo application in production mode" in msg
    assert "nexora.server:create_production_app" in msg


def test_lazy_asgi_app_safeguard_blocks_production_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    lazy_demo_app = _LazyASGIApp()
    with pytest.raises(InvalidConfigurationError) as exc_info:
        _ = lazy_demo_app.state

    assert "production mode" in str(exc_info.value)


@pytest.mark.parametrize("env_val", ["development", "dev", "test", "staging", "local", ""])
def test_create_offline_app_allowed_in_non_production(
    monkeypatch: pytest.MonkeyPatch,
    env_val: str,
) -> None:
    for var in ("APP_ENV", "ENVIRONMENT", "NEXORA_APP_MODE"):
        monkeypatch.delenv(var, raising=False)
    if env_val:
        monkeypatch.setenv("APP_ENV", env_val)

    app = create_offline_app()
    assert isinstance(app, FastAPI)
    assert app.state.enforce_auth is False


def test_resolve_embedding_dimension_logic() -> None:
    # 1. Explicit dimension
    cfg1 = EmbeddingConfig(provider="hash", dimension=64)
    assert _resolve_embedding_dimension(cfg1) == 64

    # 2. Hash default (must be 128 matching DEFAULT_HASH_DIMENSION and hash-sha256)
    cfg2 = EmbeddingConfig(provider="hash", model="hash-sha256")
    assert _resolve_embedding_dimension(cfg2) == DEFAULT_HASH_DIMENSION
    assert _resolve_embedding_dimension(cfg2) == 128

    # 3. Gemini default (768)
    cfg3 = EmbeddingConfig(provider="gemini")
    assert _resolve_embedding_dimension(cfg3) == 768

    # 4. OpenAI standard (1536)
    cfg4 = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
    assert _resolve_embedding_dimension(cfg4) == 1536

    # 5. OpenAI large (3072)
    cfg5 = EmbeddingConfig(provider="openai", model="text-embedding-3-large")
    assert _resolve_embedding_dimension(cfg5) == 3072


def test_actual_provider_dimensions_match_resolution() -> None:
    """Verify that every supported provider's concrete output matches the resolved dimension."""
    # 1. Hash provider with default model hash-sha256 -> 128
    cfg_hash = EmbeddingConfig(provider="hash", model="hash-sha256")
    dim_hash = _resolve_embedding_dimension(cfg_hash)
    assert dim_hash == 128
    prov_hash = create_embedding_provider(dataclasses.replace(cfg_hash, dimension=dim_hash))
    assert prov_hash.dimension == 128
    vec_hash = prov_hash.embed_text("test sentence")
    assert len(vec_hash) == 128

    # 2. Hash provider with explicit dimension 64 -> 64
    cfg_hash_64 = EmbeddingConfig(provider="hash", dimension=64)
    dim_hash_64 = _resolve_embedding_dimension(cfg_hash_64)
    assert dim_hash_64 == 64
    prov_hash_64 = create_embedding_provider(cfg_hash_64)
    assert prov_hash_64.dimension == 64
    assert len(prov_hash_64.embed_text("test sentence")) == 64

    # 3. Gemini provider with default model -> 768
    cfg_gemini = EmbeddingConfig(provider="gemini", api_key="fake-key")
    dim_gemini = _resolve_embedding_dimension(cfg_gemini)
    assert dim_gemini == 768
    prov_gemini = create_embedding_provider(dataclasses.replace(cfg_gemini, dimension=dim_gemini))
    assert prov_gemini.dimension == 768

    # 4. OpenAI provider with text-embedding-3-small -> 1536
    cfg_openai_sm = EmbeddingConfig(provider="openai", api_key="fake-key", model="text-embedding-3-small")
    dim_openai_sm = _resolve_embedding_dimension(cfg_openai_sm)
    assert dim_openai_sm == 1536
    prov_openai_sm = create_embedding_provider(dataclasses.replace(cfg_openai_sm, dimension=dim_openai_sm))
    assert prov_openai_sm.dimension == 1536

    # 5. OpenAI provider with text-embedding-3-large -> 3072
    cfg_openai_lg = EmbeddingConfig(provider="openai", api_key="fake-key", model="text-embedding-3-large")
    dim_openai_lg = _resolve_embedding_dimension(cfg_openai_lg)
    assert dim_openai_lg == 3072
    prov_openai_lg = create_embedding_provider(dataclasses.replace(cfg_openai_lg, dimension=dim_openai_lg))
    assert prov_openai_lg.dimension == 3072

    # 6. OpenAI provider with explicit dimension 512
    cfg_openai_512 = EmbeddingConfig(provider="openai", api_key="fake-key", model="text-embedding-3-small", dimension=512)
    dim_openai_512 = _resolve_embedding_dimension(cfg_openai_512)
    assert dim_openai_512 == 512
    prov_openai_512 = create_embedding_provider(cfg_openai_512)
    assert prov_openai_512.dimension == 512


def test_qdrant_vector_store_validate_collection_compatibility() -> None:
    from nexora.retrieval.qdrant import QdrantStoreError, QdrantVectorStore
    from nexora.retrieval.qdrant_config import QdrantConfig

    cfg = QdrantConfig(url="http://localhost:6333", collection_name="papers")

    # Mock client with mismatched collection size
    mock_client = MagicMock()
    mock_info = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=768)
            )
        )
    )
    mock_client.get_collection.return_value = mock_info

    # Store configured for dimension 1536
    store = QdrantVectorStore(cfg, dimension=1536, client=mock_client)

    # Validate compatibility should raise QdrantStoreError
    with pytest.raises(QdrantStoreError) as exc_info:
        store.validate_collection_compatibility()

    err = str(exc_info.value)
    assert "vector dimension 768 does not match configured dimension 1536" in err

    # Now update mock to matching dimension 1536
    mock_info.config.params.vectors.size = 1536
    # Should succeed without error
    store.validate_collection_compatibility()


def test_readiness_fails_closed_on_qdrant_collection_dimension_mismatch(
    base_prod_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexora.retrieval.qdrant import QdrantVectorStore
    from nexora.retrieval.qdrant_config import QdrantConfig

    env = dict(
        base_prod_env,
        NEXORA_VECTOR_STORE="qdrant",
        NEXORA_QDRANT_URL="http://localhost:6333",
        NEXORA_QDRANT_COLLECTION="prod_papers",
    )

    mock_client = MagicMock()
    # Mock collection with dimension 768 (incompatible with OpenAI 1536)
    mock_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=768)
            )
        )
    )
    mock_client.count.return_value = SimpleNamespace(count=10)

    mock_engine = SimpleNamespace()
    mock_factory = MagicMock()

    monkeypatch.setattr(
        "nexora.database.connection.create_database_engine",
        lambda cfg: mock_engine,
    )
    monkeypatch.setattr(
        "nexora.database.session.create_session_factory",
        lambda eng: mock_factory,
    )

    app = create_production_app(env, qdrant_client=mock_client)
    client = TestClient(app)

    # 1. Readiness probe must FAIL (HTTP 503) because collection dimension 768 != 128
    resp_unavailable = client.get("/ready")
    assert resp_unavailable.status_code == 503
    body = resp_unavailable.json()
    assert body["status"] == "unavailable"
    assert body["checks"]["vector_store"]["status"] == "unavailable"
    assert body["checks"]["vector_store"]["error"] == "QdrantStoreError"

    # 2. Update collection dimension to match configured dimension (128 for hash)
    mock_client.get_collection.return_value.config.params.vectors.size = 128
    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 200
    assert resp_ready.json()["status"] == "ready"
    assert resp_ready.json()["checks"]["vector_store"]["status"] == "ready"
