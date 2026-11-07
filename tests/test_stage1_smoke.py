"""Stage 1 Final Smoke Check Test Suite.

Verifies:
1. Backend boots with production frontend build available.
2. /health and /ready endpoints return status ok/ready.
3. Frontend root (/) and assets exist, and SPA client routing matches _SPAMount.
4. API routing (/api/..., /health, /ready, /openapi.json) takes precedence over SPA mount.
5. Vercel subscriber registration and configuration in pyproject.toml & vercel.json.
6. Local background worker starts by default, and Vercel environment disables it.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from starlette.routing import Match
from vercel.queue import get_subscriptions

from nexora.api.app import create_app, should_start_background_worker
from nexora.api.routes import health, readiness
import nexora.graph.queue_consumer  # Registers @subscribe with vercel.queue

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def app_instance():
    """Build application instance with frontend static mount enabled."""
    dist_dir = REPO_ROOT / "frontend" / "dist"
    assert (dist_dir / "index.html").exists(), "frontend/dist/index.html must exist for production smoke check"

    mock_research_service = MagicMock()
    app = create_app(
        research_service=mock_research_service,
        enforce_auth=False,
    )
    return app


def test_smoke_health_and_readiness_handlers():
    """Check 2: /health and /ready handler logic returns ready."""
    # 1. Health check
    h_res = health()
    assert h_res == {"status": "ok"}

    # 2. Readiness probe
    req = SimpleNamespace()
    req.app = SimpleNamespace()
    req.app.state = SimpleNamespace(
        session_factory=None,
        enforce_auth=False,
        research_service=MagicMock(),
        vector_store=None,
        rate_limit_config=None,
        rate_limit_storage=None,
    )
    resp = SimpleNamespace(status_code=200)
    r_res = readiness(req, resp)

    assert r_res["status"] == "ready"
    assert r_res["checks"]["research"]["status"] == "ready"
    assert r_res["checks"]["auth"]["status"] == "ready"
    assert r_res["checks"]["database"]["status"] == "not_configured"


def test_smoke_frontend_and_spa_routing(app_instance):
    """Check 1 & 3: Production frontend build exists and SPA mount matches client routes."""
    dist_dir = REPO_ROOT / "frontend" / "dist"
    index_html = dist_dir / "index.html"
    assert index_html.exists()
    content = index_html.read_text(encoding="utf-8")
    assert "<div id=\"root\"></div>" in content or "<html" in content

    assets_dir = dist_dir / "assets"
    assert assets_dir.exists()
    js_files = list(assets_dir.glob("*.js"))
    assert len(js_files) > 0, "Expected js assets in frontend/dist/assets"

    # Find the _SPAMount route in app.routes
    spa_mount = next((r for r in app_instance.routes if getattr(r, "name", "") == "frontend"), None)
    assert spa_mount is not None, "Expected frontend SPA mount in app.routes"

    # Verify SPA mount matches / and client-side routes
    match_root, _ = spa_mount.matches({"type": "http", "method": "GET", "path": "/"})
    assert match_root == Match.FULL

    match_spa, _ = spa_mount.matches({"type": "http", "method": "GET", "path": "/papers/explore"})
    assert match_spa == Match.FULL

    match_asset, _ = spa_mount.matches({"type": "http", "method": "GET", "path": f"/assets/{js_files[0].name}"})
    assert match_asset == Match.FULL


def test_smoke_api_routing_precedence(app_instance):
    """Check 4: API routing (/api/..., /health, /ready, /openapi.json) is not intercepted by SPA mount."""
    spa_mount = next((r for r in app_instance.routes if getattr(r, "name", "") == "frontend"), None)
    assert spa_mount is not None

    # SPA mount MUST yield (return Match.NONE) for all backend API routes
    assert spa_mount.matches({"type": "http", "method": "GET", "path": "/health"})[0] == Match.NONE
    assert spa_mount.matches({"type": "http", "method": "GET", "path": "/ready"})[0] == Match.NONE
    assert spa_mount.matches({"type": "http", "method": "GET", "path": "/docs"})[0] == Match.NONE
    assert spa_mount.matches({"type": "http", "method": "GET", "path": "/openapi.json"})[0] == Match.NONE
    assert spa_mount.matches({"type": "http", "method": "GET", "path": "/api/v1/research"})[0] == Match.NONE
    assert spa_mount.matches({"type": "http", "method": "GET", "path": "/api/v1/discovery/search"})[0] == Match.NONE

    # Lifespan non-HTTP scope MUST NOT be intercepted
    assert spa_mount.matches({"type": "lifespan"})[0] == Match.NONE


def test_smoke_vercel_subscriber_and_config():
    """Check 5: Vercel subscriber registration and pyproject.toml / vercel.json config."""
    # vercel-queue subscriber verification
    subs = get_subscriptions()
    matching = [s for s in subs if s.topic == "graph-extraction-jobs"]
    assert len(matching) >= 1
    assert any(s.func is not None and s.func.__name__ == "handle_graph_job" for s in matching)

    # pyproject.toml verification
    pyproject_file = REPO_ROOT / "pyproject.toml"
    assert pyproject_file.exists()
    pyproject_data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
    assert pyproject_data["tool"]["vercel"]["entrypoint"] == "nexora.server:app"
    subscribers = pyproject_data.get("tool", {}).get("vercel", {}).get("subscribers", [])
    assert any(
        s.get("topic") == "graph-extraction-jobs" and s.get("entrypoint") == "nexora.graph.queue_consumer"
        for s in subscribers
    )

    # vercel.json verification
    vercel_file = REPO_ROOT / "vercel.json"
    assert vercel_file.exists()
    vercel_data = json.loads(vercel_file.read_text(encoding="utf-8"))
    assert vercel_data["services"]["frontend"]["root"] == "frontend"
    assert vercel_data["services"]["backend"]["entrypoint"] == "src/nexora/server.py"
    assert "rewrites" in vercel_data
    assert "crons" not in vercel_data


def test_smoke_worker_lifecycle_guards():
    """Check 6: Local background worker starts by default, disabled on Vercel."""
    # Local dev defaults
    assert should_start_background_worker({}) is True
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "true"}) is True
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "1"}) is True

    # Serverless guards
    assert should_start_background_worker({"VERCEL": "1"}) is False
    assert should_start_background_worker({"VERCEL": "1", "ENABLE_BACKGROUND_WORKER": "true"}) is False
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "false"}) is False
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "0"}) is False
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "no"}) is False
