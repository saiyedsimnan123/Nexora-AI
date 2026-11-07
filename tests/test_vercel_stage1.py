"""Unit and integration tests for Vercel Deployment Foundation (Stage 1 Scope Corrected).

Covers:
- Serverless worker deactivation guards (VERCEL=1, ENABLE_BACKGROUND_WORKER=false).
- Local worker startup preservation when flags are unset.
- Vercel Queue consumer subscription and invocation of worker.run_once().
- Vercel Services schema and routing structure in vercel.json.
- pyproject.toml configuration for Vercel entrypoint and queue subscribers.
"""

from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
import unittest.mock as mock

import pytest
from vercel.queue import get_subscriptions

from nexora.api.app import should_start_background_worker
from nexora.graph.queue_consumer import (
    handle_graph_job,
    process_queue_trigger,
    set_global_worker,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Serverless Worker Deactivation Tests
# ---------------------------------------------------------------------------

def test_should_start_background_worker_guards():
    """Verify worker start guards for Vercel and serverless environments."""
    # When VERCEL=1, background worker must NOT start
    assert should_start_background_worker({"VERCEL": "1"}) is False
    assert should_start_background_worker({"VERCEL": "1", "ENABLE_BACKGROUND_WORKER": "true"}) is False

    # When ENABLE_BACKGROUND_WORKER is false, 0, or no, must NOT start
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "false"}) is False
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "0"}) is False
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "no"}) is False
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "FALSE"}) is False

    # Default local dev environment (empty or explicitly true) starts worker
    assert should_start_background_worker({}) is True
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "true"}) is True
    assert should_start_background_worker({"ENABLE_BACKGROUND_WORKER": "1"}) is True


# ---------------------------------------------------------------------------
# 2. Vercel Queue Consumer Invocation of worker.run_once()
# ---------------------------------------------------------------------------

def test_queue_consumer_invokes_run_once():
    """Verify queue consumer delegates to worker.run_once() and returns processed count."""
    mock_worker = mock.MagicMock()
    mock_worker.run_once.return_value = 1

    set_global_worker(mock_worker)
    try:
        # Synchronous helper
        res = process_queue_trigger({"event": "job_enqueued"})
        assert res == {"processed_jobs": 1}
        mock_worker.run_once.assert_called_once()

        # Subscriber entrypoint
        mock_worker.run_once.reset_mock()
        sub_res = handle_graph_job({"event": "job_enqueued"})
        assert sub_res == {"processed_jobs": 1}
        mock_worker.run_once.assert_called_once()
    finally:
        set_global_worker(None)


def test_queue_subscription_registered_with_sdk():
    """Verify handle_graph_job is properly registered with vercel-queue SDK."""
    subs = get_subscriptions()
    matching = [s for s in subs if s.topic == "graph-extraction-jobs"]
    assert len(matching) >= 1
    assert any(s.func is not None and s.func.__name__ == "handle_graph_job" for s in matching)


# ---------------------------------------------------------------------------
# 3. Vercel Configuration Verification
# ---------------------------------------------------------------------------

def test_vercel_json_services_schema_structure():
    """Verify vercel.json defines valid services and rewrites without invalid root fields."""
    vercel_json_path = REPO_ROOT / "vercel.json"
    assert vercel_json_path.exists(), "vercel.json must exist at repository root"

    data = json.loads(vercel_json_path.read_text(encoding="utf-8"))

    # Services schema check
    assert "services" in data
    assert "frontend" in data["services"]
    assert data["services"]["frontend"]["root"] == "frontend"
    assert data["services"]["frontend"]["framework"] == "vite"

    assert "backend" in data["services"]
    assert data["services"]["backend"]["root"] == "."
    assert data["services"]["backend"]["framework"] == "fastapi"
    assert data["services"]["backend"]["entrypoint"] == "src/nexora/server.py"

    # Rewrites check
    assert "rewrites" in data
    assert any(r.get("source") == "/api/:path*" and r.get("destination", {}).get("service") == "backend" for r in data["rewrites"])
    assert any(r.get("source") == "/(.*)" and r.get("destination", {}).get("service") == "frontend" for r in data["rewrites"])

    # Disallowed fields that cause conflicts with Vercel Services
    assert "crons" not in data
    assert "builds" not in data
    assert "functions" not in data


def test_pyproject_toml_configuration():
    """Verify pyproject.toml defines entrypoint and queue subscribers."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist at repository root"

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    # Tool vercel check
    assert "tool" in data and "vercel" in data["tool"]
    assert data["tool"]["vercel"]["entrypoint"] == "nexora.server:app"

    # Subscribers check
    subscribers = data.get("tool", {}).get("vercel", {}).get("subscribers", [])
    assert len(subscribers) >= 1
    assert any(
        s.get("topic") == "graph-extraction-jobs" and s.get("entrypoint") == "nexora.graph.queue_consumer"
        for s in subscribers
    )
