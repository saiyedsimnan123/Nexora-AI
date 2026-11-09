"""Readiness and diagnostic helpers for Nexora API."""

from __future__ import annotations

import os
from typing import Any
from fastapi import Request, Response


def check_readiness_status(request: Request, response: Response) -> dict[str, Any]:
    """Readiness probe: validates availability of required backend dependencies.

    Checks:
    - database: verifies connectivity via session_factory if configured.
    - auth: checks authentication service readiness when enforce_auth is True.
    - research: checks research service availability.
    - vector_store: checks vector store connectivity via count() if configured.
    - llm: checks LLM API key presence and model configuration.
    - embedding: checks embedding provider readiness (flags hash fallback in live mode).
    - environment: safe variable presence diagnostics (booleans only).

    Returns HTTP 200 with status='ready' when all required components are operational.
    Returns HTTP 503 with status='unavailable' when any required component is failing.
    """
    session_factory = getattr(request.app.state, "session_factory", None)
    enforce_auth = getattr(request.app.state, "enforce_auth", False)
    research_service = getattr(request.app.state, "research_service", None)
    auth_service = getattr(request.app.state, "auth_service", None)

    checks: dict[str, Any] = {}
    is_ready = True

    # 1. Database check
    if session_factory is not None:
        try:
            from nexora.database.readiness import check_database_readiness

            check_database_readiness(session_factory)
            checks["database"] = {"status": "ready"}
        except Exception as exc:
            checks["database"] = {
                "status": "unavailable",
                "error": type(exc).__name__,
            }
            is_ready = False
    else:
        if enforce_auth:
            checks["database"] = {
                "status": "unavailable",
                "error": "database_not_configured_for_enforced_auth",
            }
            is_ready = False
        else:
            checks["database"] = {"status": "not_configured"}

    # 2. Authentication check
    if enforce_auth:
        if auth_service is None and session_factory is not None and is_ready:
            from nexora.auth.service import AuthenticationService

            try:
                auth_service = AuthenticationService(session_factory)
                request.app.state.auth_service = auth_service
            except Exception:
                pass

        if auth_service is not None and checks.get("database", {}).get("status") == "ready":
            checks["auth"] = {"status": "ready"}
        else:
            checks["auth"] = {"status": "unavailable"}
            is_ready = False
    else:
        checks["auth"] = {"status": "ready"}

    # 3. Research Service check
    if research_service is not None and callable(getattr(research_service, "ask", None)):
        checks["research"] = {"status": "ready"}
    else:
        checks["research"] = {"status": "unavailable"}
        is_ready = False

    # 4. Vector store / Ingestion check
    ingestion_service = getattr(request.app.state, "ingestion_service", None)
    rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
    vector_store = getattr(request.app.state, "vector_store", None) or getattr(ingestion_service, "_vector_store", None) or getattr(rag_pipeline, "vector_store", None)
    settings = getattr(request.app.state, "settings", None)
    app_env = getattr(request.app.state, "environ", None) or os.environ
    is_live_generation = (
        getattr(settings, "require_llm_key", False)
        or (app_env.get("NEXORA_RAG_GENERATOR_MODE") or app_env.get("RAG_GENERATOR_MODE") or "").strip().lower() == "live"
    )

    if vector_store is not None:
        readiness_fn = getattr(vector_store, "check_readiness", None) or getattr(vector_store, "count", None)
        vs_type = type(vector_store).__name__
        backend_name = "qdrant" if "Qdrant" in vs_type else ("memory" if "Memory" in vs_type else "custom")
        if callable(readiness_fn):
            try:
                readiness_fn()
                checks["vector_store"] = {"status": "ready", "backend": backend_name}
            except Exception as exc:
                checks["vector_store"] = {
                    "status": "unavailable",
                    "backend": backend_name,
                    "error": type(exc).__name__,
                }
                is_ready = False
        else:
            checks["vector_store"] = {"status": "ready", "backend": backend_name}

        if is_live_generation and backend_name == "memory":
            checks["vector_store"] = {
                "status": "degraded_or_fallback",
                "backend": "memory",
                "error": "in_memory_fallback_not_allowed_in_live_mode",
            }
            is_ready = False

    # 5. LLM Provider check
    llm_config = getattr(settings, "llm", None) if settings else None
    if llm_config is not None:
        has_key = bool(llm_config.api_key and llm_config.api_key.strip())
        checks["llm"] = {
            "status": "ready" if has_key else "not_configured",
            "provider": llm_config.provider,
            "model": llm_config.model,
        }
        if not has_key and is_live_generation:
            is_ready = False

    # 6. Embedding Service check
    embedding_service = getattr(request.app.state, "embedding_service", None) or getattr(ingestion_service, "_embedding_service", None)
    if embedding_service is not None:
        emb_provider = getattr(embedding_service, "_provider", None) or embedding_service
        emb_name = getattr(emb_provider, "name", type(emb_provider).__name__)
        is_hash = "Hash" in emb_name
        checks["embedding"] = {
            "status": "fallback" if is_hash else "ready",
            "provider": emb_name,
            "dimension": getattr(embedding_service, "dimension", None),
        }
        if is_hash and is_live_generation:
            checks["embedding"]["error"] = "hash_embedding_not_allowed_in_live_mode"
            is_ready = False

    # 7. Safe environment variable diagnostics (presence boolean only, NO values)
    llm_config = getattr(settings, "llm", None) if settings else None
    active_llm_provider = (
        (llm_config.provider if llm_config else None)
        or (app_env.get("NEXORA_LLM_PROVIDER") or os.environ.get("NEXORA_LLM_PROVIDER") or "")
        or "openai"
    ).strip().lower()

    diagnostic_env_vars = (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "NEXORA_LLM_MODEL",
        "NEXORA_RAG_GENERATOR_MODE",
        "NEXORA_EMBEDDING_PROVIDER",
        "NEXORA_EMBEDDING_MODEL",
        "NEXORA_VECTOR_STORE",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "NEXORA_QDRANT_COLLECTION",
    )
    env_status = {
        var: bool((app_env.get(var) or os.environ.get(var) or "").strip())
        for var in diagnostic_env_vars
    }

    if active_llm_provider == "gemini":
        has_active_llm_key = env_status.get("GEMINI_API_KEY", False) or bool(
            (app_env.get("NEXORA_GEMINI_API_KEY") or os.environ.get("NEXORA_GEMINI_API_KEY") or "").strip()
        )
    else:
        has_active_llm_key = env_status.get("OPENAI_API_KEY", False)

    required_non_key_vars = (
        "NEXORA_LLM_MODEL",
        "NEXORA_RAG_GENERATOR_MODE",
        "NEXORA_EMBEDDING_PROVIDER",
        "NEXORA_EMBEDDING_MODEL",
        "NEXORA_VECTOR_STORE",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "NEXORA_QDRANT_COLLECTION",
    )
    is_env_configured = has_active_llm_key and all(
        env_status.get(v, False) for v in required_non_key_vars
    )

    checks["environment"] = {
        "status": "configured" if is_env_configured else "incomplete",
        "variables": env_status,
        "runtime": {
            "vercel": bool(app_env.get("VERCEL") or os.environ.get("VERCEL")),
            "vercel_env": app_env.get("VERCEL_ENV") or os.environ.get("VERCEL_ENV"),
            "vercel_branch": app_env.get("VERCEL_GIT_COMMIT_REF") or os.environ.get("VERCEL_GIT_COMMIT_REF"),
        },
    }

    # 8. Rate limit storage check (when rate limiting is enabled)
    rate_limit_config = getattr(request.app.state, "rate_limit_config", None)
    rate_limit_storage = getattr(request.app.state, "rate_limit_storage", None)
    if rate_limit_config is not None and getattr(rate_limit_config, "enabled", False):
        if rate_limit_storage is not None and hasattr(rate_limit_storage, "check_readiness"):
            ready_ok, ready_msg = rate_limit_storage.check_readiness()
            if ready_ok:
                checks["rate_limit_storage"] = {"status": "ready"}
            else:
                checks["rate_limit_storage"] = {
                    "status": "unavailable",
                    "error": ready_msg,
                }
                is_ready = False
        else:
            checks["rate_limit_storage"] = {
                "status": "unavailable",
                "error": "storage_not_configured_for_enabled_rate_limiting",
            }
            is_ready = False

    if not is_ready:
        response.status_code = 503
        return {"status": "unavailable", "checks": checks}

    return {"status": "ready", "checks": checks}


def get_environment_diagnostics(request: Request) -> dict[str, Any]:
    """Safe runtime environment diagnostics without exposing secret values."""
    diagnostic_env_vars = (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "NEXORA_LLM_MODEL",
        "NEXORA_RAG_GENERATOR_MODE",
        "NEXORA_EMBEDDING_PROVIDER",
        "NEXORA_EMBEDDING_MODEL",
        "NEXORA_VECTOR_STORE",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "NEXORA_QDRANT_COLLECTION",
    )
    app_env = getattr(request.app.state, "environ", None) or os.environ
    env_status = {
        var: bool((app_env.get(var) or os.environ.get(var) or "").strip())
        for var in diagnostic_env_vars
    }
    settings = getattr(request.app.state, "settings", None)
    vector_store = getattr(request.app.state, "vector_store", None)
    embedding_service = getattr(request.app.state, "embedding_service", None)
    emb_provider = getattr(embedding_service, "_provider", None) if embedding_service else None
    llm_config = getattr(settings, "llm", None) if settings else None

    vs_type = type(vector_store).__name__ if vector_store else None
    backend_name = (
        "qdrant" if vs_type and "Qdrant" in vs_type
        else ("memory" if vs_type and "Memory" in vs_type else vs_type)
    )

    return {
        "status": "ok",
        "environment": env_status,
        "runtime": {
            "vercel": bool(app_env.get("VERCEL") or os.environ.get("VERCEL")),
            "vercel_env": app_env.get("VERCEL_ENV") or os.environ.get("VERCEL_ENV"),
            "vercel_branch": app_env.get("VERCEL_GIT_COMMIT_REF") or os.environ.get("VERCEL_GIT_COMMIT_REF"),
            "vercel_commit": app_env.get("VERCEL_GIT_COMMIT_SHA") or os.environ.get("VERCEL_GIT_COMMIT_SHA"),
        },
        "active_configuration": {
            "vector_store_backend": backend_name,
            "llm_provider": llm_config.provider if llm_config else None,
            "llm_model": llm_config.model if llm_config else None,
            "embedding_provider": getattr(emb_provider, "name", type(emb_provider).__name__) if emb_provider else None,
            "embedding_dimension": getattr(embedding_service, "dimension", None) if embedding_service else None,
            "database_configured": bool(getattr(request.app.state, "session_factory", None)),
        },
    }
