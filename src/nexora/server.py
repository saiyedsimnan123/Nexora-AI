"""Production ASGI entrypoint and application factory for Nexora AI.

This module provides the production application factory for Nexora AI.
Unlike ``nexora.api.app:create_offline_app`` (which boots an unauthenticated,
in-memory demo application), ``create_production_app`` reads configuration
from the environment, enforces authentication, initializes persistent
database and vector store connections, and wires the production composition root.

Startup usage:
    uvicorn nexora.server:create_production_app --factory --host 0.0.0.0 --port 8000
    or
    uvicorn nexora.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Ensure 'src' is on sys.path so absolute 'nexora' imports work when loaded via 'src.nexora.server'
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fastapi import FastAPI

from nexora.api.errors import InvalidConfigurationError
from nexora.bootstrap.application import build_http_app
from nexora.config.settings import AppSettings, ConfigurationError
from nexora.embedding.config import EmbeddingConfig
from nexora.embedding.factory import create_embedding_provider
from nexora.embedding.hash_provider import DEFAULT_DIMENSION as DEFAULT_HASH_DIMENSION
from nexora.embedding.service import EmbeddingService
from nexora.retrieval.qdrant_config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_TIMEOUT,
    ENV_QDRANT_API_KEY,
    ENV_QDRANT_COLLECTION,
    ENV_QDRANT_TIMEOUT,
    ENV_QDRANT_URL,
    QdrantConfig,
)


def _resolve_dimension_for_config(config: EmbeddingConfig) -> int:
    """Resolve the output vector dimension for an EmbeddingConfig."""
    if config.dimension is not None:
        return config.dimension

    provider_name = config.provider.strip().lower() if isinstance(config.provider, str) else ""

    if provider_name in ("hash", "hash-sha256"):
        return DEFAULT_HASH_DIMENSION  # 128
    if provider_name == "gemini":
        from nexora.embedding.gemini_provider import DEFAULT_GEMINI_DIMENSION
        return DEFAULT_GEMINI_DIMENSION  # 768
    if provider_name == "openai":
        model_name = config.model.strip().lower() if isinstance(config.model, str) else ""
        if "large" in model_name:
            return 3072
        return 1536

    raise ConfigurationError(
        f"Unable to determine embedding dimension for provider {config.provider!r}. "
        "Please specify NEXORA_EMBEDDING_DIMENSION in your environment."
    )


def _resolve_embedding_dimension(config: EmbeddingConfig, provider: Any = None) -> int:
    """Resolve the positive integer dimension for the configured provider."""
    if config.dimension is not None:
        return config.dimension
    if provider is not None:
        provider_dim = getattr(provider, "dimension", None)
        if isinstance(provider_dim, int) and not isinstance(provider_dim, bool) and provider_dim > 0:
            return provider_dim
    return _resolve_dimension_for_config(config)


def _load_qdrant_config(environ: Mapping[str, str]) -> QdrantConfig:
    """Load QdrantConfig safely from environment mapping."""
    url = (environ.get(ENV_QDRANT_URL) or environ.get("QDRANT_URL") or "").strip()
    if not url:
        raise ConfigurationError(
            f"{ENV_QDRANT_URL} (or QDRANT_URL) environment variable is required and must not be empty when vector_store is 'qdrant'"
        )
    api_key = (
        environ.get(ENV_QDRANT_API_KEY)
        or environ.get("QDRANT_API_KEY")
        or ""
    ).strip() or None
    collection = (
        environ.get(ENV_QDRANT_COLLECTION)
        or environ.get("QDRANT_COLLECTION")
        or ""
    ).strip() or DEFAULT_COLLECTION_NAME
    raw_timeout = (
        environ.get(ENV_QDRANT_TIMEOUT)
        or environ.get("QDRANT_TIMEOUT")
        or ""
    ).strip()
    timeout = DEFAULT_TIMEOUT
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError:
            raise ConfigurationError(
                f"{ENV_QDRANT_TIMEOUT} must be a number of seconds, got {raw_timeout!r}"
            ) from None
    try:
        return QdrantConfig(
            url=url, api_key=api_key, collection_name=collection, timeout=timeout
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid Qdrant configuration: {exc}") from exc


logger = logging.getLogger(__name__)


def _should_auto_migrate(env: Mapping[str, str]) -> bool:
    default_auto = "false" if "PYTEST_CURRENT_TEST" in os.environ else "true"
    return (env.get("NEXORA_AUTO_MIGRATE") or default_auto).strip().lower() in ("true", "1", "yes")


def _run_startup_migrations_once(env: Mapping[str, str]) -> bool:
    """Execute advisory-locked migrations once during app startup if enabled."""
    if not _should_auto_migrate(env):
        return False

    try:
        from nexora.database.config import DatabaseConfig
        from nexora.database.migrations.runner import apply_pending_migrations

        norm_db_config = DatabaseConfig.from_env(env)
        apply_pending_migrations(norm_db_config.url)
        return True
    except Exception as exc:
        logger.warning("Database auto-migration note: %s", exc)
        return False


def create_production_app(
    environ: Mapping[str, str] | None = None,
    *,
    run_migrations: bool | None = None,
    **overrides: Any,
) -> FastAPI:
    """Zero-argument production ASGI application factory.

    Loads validated settings from the environment, wires the real embedding provider,
    vector store, database session factory, authentication service, and composition root.

    Args:
        environ: Optional environment mapping. Defaults to ``os.environ``.
        run_migrations: Optional override controlling whether auto-migrations execute during startup.
        **overrides: Optional component overrides (for testing and DI).

    Returns:
        Configured FastAPI application instance ready for production traffic.

    Raises:
        ConfigurationError: If any required production configuration is missing or invalid.
        InvalidConfigurationError: If service assembly fails dependency validation.
    """
    if environ is None:
        try:
            import dotenv
            dotenv.load_dotenv()
            for env_p in (
                Path(".env"),
                Path(".env.local"),
                Path(".env.production"),
                Path(".env.preview"),
                Path("frontend/.env"),
                Path("frontend/.env.local"),
            ):
                if env_p.is_file():
                    dotenv.load_dotenv(env_p)
        except Exception:
            pass

    env = os.environ if environ is None else environ

    generator_mode = (
        env.get("NEXORA_RAG_GENERATOR_MODE")
        or env.get("RAG_GENERATOR_MODE")
        or ""
    ).strip().lower()

    # 1. Vector store configuration
    backend_env = env.get("NEXORA_VECTOR_STORE") or env.get("VECTOR_STORE")
    if backend_env and backend_env.strip():
        backend = backend_env.strip().lower()
    elif env.get("NEXORA_QDRANT_URL") or env.get("QDRANT_URL"):
        backend = "qdrant"
    else:
        backend = "memory"

    qdrant_config: QdrantConfig | None = None
    if backend == "qdrant":
        qdrant_config = _load_qdrant_config(env)
    elif backend != "memory":
        raise ConfigurationError(
            f"Unsupported vector_store: {backend!r}. Supported backends: 'memory', 'qdrant'"
        )

    # 2. Database requirement check for production auth/persistence
    db_url = (
        env.get("NEXORA_DATABASE_URL")
        or env.get("DATABASE_URL")
        or env.get("POSTGRES_URL")
        or env.get("POSTGRES_PRISMA_URL")
    )
    has_db = bool(db_url and db_url.strip())
    has_session_override = "session_factory" in overrides or "auth_service" in overrides
    if not has_db and not has_session_override:
        raise ConfigurationError(
            "Production deployment requires a PostgreSQL database. "
            "Please configure NEXORA_DATABASE_URL (or DATABASE_URL) with a valid connection string."
        )

    if has_db and not has_session_override and run_migrations is not False:
        _run_startup_migrations_once(env)

    # 3. Embedding provider & service construction
    embedding_service = overrides.get("embedding_service")
    emb_config: Any = None
    if embedding_service is None:
        try:
            emb_config = EmbeddingConfig.from_env(env)
            # Resolve dimension from provider defaults if not explicitly set
            if emb_config.dimension is None:
                dim = _resolve_dimension_for_config(emb_config)
                emb_config = dataclasses.replace(emb_config, dimension=dim)
            emb_provider = create_embedding_provider(emb_config)
            embedding_service = EmbeddingService(emb_provider)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid embedding configuration: {exc}") from exc
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(f"Failed to initialize embedding provider: {exc}") from exc
    else:
        dim = getattr(embedding_service, "dimension", None)
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            dim = DEFAULT_HASH_DIMENSION
        emb_config = EmbeddingConfig(dimension=dim)

    # 4. Production application settings
    try:
        settings = AppSettings.from_env(
            env,
            embedding=emb_config,
            qdrant=qdrant_config,
        )
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Production settings error: {exc}") from exc

    # Enforce that live generation mode does not silently use fallback demo providers
    if generator_mode == "live":
        if backend == "memory" and "vector_store" not in overrides:
            raise ConfigurationError(
                "Live mode (NEXORA_RAG_GENERATOR_MODE=live) requires persistent vector storage. "
                "Please set NEXORA_VECTOR_STORE=qdrant and provide QDRANT_URL."
            )
        if emb_config.provider in ("hash", "hash-sha256") and "embedding_service" not in overrides:
            raise ConfigurationError(
                "Live mode (NEXORA_RAG_GENERATOR_MODE=live) requires a live embedding provider. "
                "Please set NEXORA_EMBEDDING_PROVIDER=openai and provide OPENAI_API_KEY."
            )

    # 5. Extract dependency overrides
    vector_store = overrides.get("vector_store")
    llm_provider = overrides.get("llm_provider")
    qdrant_client = overrides.get("qdrant_client")
    session_factory = overrides.get("session_factory")
    auth_service = overrides.get("auth_service")

    other_deps = {
        k: v
        for k, v in overrides.items()
        if k
        not in (
            "embedding_service",
            "vector_store",
            "llm_provider",
            "qdrant_client",
            "session_factory",
            "auth_service",
        )
    }

    try:
        return build_http_app(
            settings,
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_provider=llm_provider,
            qdrant_client=qdrant_client,
            session_factory=session_factory,
            auth_service=auth_service,
            environ=env,
            **other_deps,
        )
    except (ConfigurationError, InvalidConfigurationError):
        raise
    except Exception as exc:
        raise ConfigurationError(f"Production startup assembly failed: {exc}") from exc


class _LazyProductionASGIApp:
    """Lazy ASGI application that initializes create_production_app on first request or lifespan."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None
        self._init_lock = threading.Lock()
        self._migration_lock = threading.Lock()
        self._migrations_checked = False

    @staticmethod
    def _is_liveness_scope(scope: Any) -> bool:
        if scope.get("type") != "http":
            return False
        path = (scope.get("path") or "").rstrip("/") or "/"
        return path == "/health"

    def _ensure_migrations_once(self) -> None:
        if self._migrations_checked:
            return
        with self._migration_lock:
            if not self._migrations_checked:
                _run_startup_migrations_once(os.environ)
                self._migrations_checked = True

    def _get_app(self, *, run_migrations: bool = True) -> FastAPI:
        if self._app is None:
            with self._init_lock:
                if self._app is None:
                    self._app = create_production_app(run_migrations=run_migrations)
                    if run_migrations:
                        self._migrations_checked = True
        if run_migrations and not self._migrations_checked:
            self._ensure_migrations_once()
        return self._app

    @property
    def state(self) -> Any:
        return self._get_app().state

    @property
    def openapi_schema(self) -> Any:
        return self._get_app().openapi_schema

    def openapi(self) -> dict[str, Any]:
        return self._get_app().openapi()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "lifespan":
            try:
                target_app = self._get_app(run_migrations=True)
            except Exception as exc:
                message = await receive()
                if message.get("type") == "lifespan.startup":
                    await send({
                        "type": "lifespan.startup.failed",
                        "message": str(exc),
                    })
                raise
            await target_app(scope, receive, send)
            return

        is_liveness = self._is_liveness_scope(scope)
        target_app = self._get_app(run_migrations=not is_liveness)
        await target_app(scope, receive, send)


app = _LazyProductionASGIApp()
