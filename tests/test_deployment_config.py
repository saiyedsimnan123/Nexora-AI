"""Validation tests for Docker containerization, Compose, and dependency configurations (Milestone 35)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


def test_requirements_txt_contains_only_runtime_dependencies() -> None:
    req_file = REPO_ROOT / "requirements.txt"
    assert req_file.is_file(), "requirements.txt must exist in repo root"

    content = req_file.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    # Must NOT contain testing frameworks
    packages = [re.split(r"[><=~]", line)[0].strip().lower() for line in lines]
    assert "pytest" not in packages, "pytest must not be in production runtime requirements.txt"

    # Core required runtime packages must be present
    expected_runtime_packages = {
        "fastapi",
        "uvicorn[standard]",
        "sqlalchemy",
        "alembic",
        "psycopg[binary]",
        "qdrant-client",
        "pymupdf",
        "httpx",
        "openai",
        "python-dotenv",
        "python-multipart",
        "redis",
    }
    for expected in expected_runtime_packages:
        assert any(expected in pkg for pkg in packages), f"Missing {expected} in requirements.txt"


def test_requirements_dev_references_runtime_and_includes_pytest() -> None:
    dev_file = REPO_ROOT / "requirements-dev.txt"
    assert dev_file.is_file(), "requirements-dev.txt must exist"

    content = dev_file.read_text(encoding="utf-8")
    assert "-r requirements.txt" in content, "requirements-dev.txt must reference runtime requirements"
    assert "pytest" in content, "requirements-dev.txt must include pytest"


def test_requirements_lock_pins_exact_versions() -> None:
    lock_file = REPO_ROOT / "requirements.lock"
    assert lock_file.is_file(), "requirements.lock must exist"

    content = lock_file.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert len(lines) >= 15, f"Lockfile should contain all runtime packages, found {len(lines)}"

    # All package lines must be pinned with exact == versions
    for line in lines:
        assert "==" in line, f"Package in lockfile is not pinned with exact version: {line}"
        parts = line.split("==")
        assert len(parts) == 2, f"Malformed pinned entry: {line}"
        assert parts[0].strip() and parts[1].strip()


def test_dockerfile_configuration() -> None:
    dockerfile = REPO_ROOT / "Dockerfile"
    assert dockerfile.is_file(), "Dockerfile must exist in repo root"

    content = dockerfile.read_text(encoding="utf-8")

    # Base image: Python 3.12 slim
    assert re.search(r"FROM\s+python:3\.12-slim", content), "Must use python:3.12-slim base image"

    # Non-root user creation & usage
    assert "useradd" in content, "Must create a dedicated non-root user"
    assert re.search(r"USER\s+\w+", content), "Must switch to non-root USER before execution"

    # Uses verified production entrypoint, not offline demo
    assert "nexora.server:app" in content, "Must target production entrypoint nexora.server:app"
    assert "create_offline_app" not in content, "Must not target offline demo in Dockerfile"

    # Signal propagation entrypoint & healthcheck
    assert "ENTRYPOINT" in content
    assert "docker-entrypoint.sh" in content
    assert "HEALTHCHECK" in content
    assert "EXPOSE 8000" in content


def test_dockerignore_configuration() -> None:
    dockerignore = REPO_ROOT / ".dockerignore"
    assert dockerignore.is_file(), ".dockerignore must exist in repo root"

    content = dockerignore.read_text(encoding="utf-8")
    patterns = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    # Must ignore version control, virtual environments, secrets, caches, and test files
    assert ".git" in patterns
    assert ".venv" in patterns
    assert ".env" in patterns
    assert "tests/" in patterns


def test_docker_entrypoint_script_semantics() -> None:
    entrypoint = REPO_ROOT / "docker-entrypoint.sh"
    assert entrypoint.is_file(), "docker-entrypoint.sh must exist"

    content = entrypoint.read_text(encoding="utf-8")

    # Starts with posix shebang and set -e
    assert content.startswith("#!/bin/sh"), "Must start with #!/bin/sh"
    assert "set -e" in content, "Must enforce set -e for strict error handling"

    # Defaults RUN_MIGRATIONS to false
    assert 'RUN_MIGRATIONS="${RUN_MIGRATIONS:-false}"' in content or "RUN_MIGRATIONS:-false" in content

    # Executes alembic upgrade head when opt-in enabled
    assert "alembic upgrade head" in content

    # Uses exec "$@" to pass signals directly to child process
    assert re.search(r'exec\s+"\$@"', content), 'Must use exec "$@" for proper signal delivery'


def test_docker_compose_validity_and_service_wiring() -> None:
    compose_file = REPO_ROOT / "docker-compose.yml"
    assert compose_file.is_file(), "docker-compose.yml must exist"

    content = compose_file.read_text(encoding="utf-8")
    config = yaml.safe_load(content)

    assert "services" in config, "Compose file must define services"
    services = config["services"]

    # 1. PostgreSQL service
    assert "postgres" in services, "Must define postgres service"
    pg = services["postgres"]
    assert "postgres:16" in pg["image"]
    assert "healthcheck" in pg
    assert "postgres_data" in str(pg.get("volumes", []))

    # 2. Qdrant service
    assert "qdrant" in services, "Must define qdrant service"
    qdrant = services["qdrant"]
    assert "qdrant" in qdrant["image"]
    assert "healthcheck" in qdrant
    assert "qdrant_data" in str(qdrant.get("volumes", []))

    # 3. Redis service
    assert "redis" in services, "Must define redis service"
    redis_svc = services["redis"]
    assert "redis:7" in redis_svc["image"]
    assert "healthcheck" in redis_svc
    assert "127.0.0.1:6379:6379" in redis_svc.get("ports", [])

    # 4. One-shot migration service
    assert "migrate" in services, "Must define migrate service"
    migrate = services["migrate"]
    assert migrate["command"] == ["alembic", "upgrade", "head"]
    assert migrate.get("restart") == "no"
    assert "postgres" in migrate.get("depends_on", {})

    # 5. App service
    assert "app" in services, "Must define app service"
    app = services["app"]
    assert "postgres" in app.get("depends_on", {})
    assert "qdrant" in app.get("depends_on", {})
    assert "redis" in app.get("depends_on", {})
    assert "migrate" in app.get("depends_on", {})
    # Must wait for migration service to complete successfully
    assert app["depends_on"]["migrate"]["condition"] == "service_completed_successfully"

    # Named persistent volumes
    assert "volumes" in config
    assert "postgres_data" in config["volumes"]
    assert "qdrant_data" in config["volumes"]


def test_ci_workflow_configuration() -> None:
    ci_file = REPO_ROOT / ".github" / "workflows" / "tests.yml"
    assert ci_file.is_file(), ".github/workflows/tests.yml must exist"

    content = ci_file.read_text(encoding="utf-8")
    ci_config = yaml.safe_load(content)

    test_job = ci_config["jobs"]["test"]
    services = test_job.get("services", {})

    # Must contain both postgres and redis service containers
    assert "postgres" in services, "CI must define postgres service container"
    assert "redis" in services, "CI must define redis service container"

    # Environment variables in steps must supply live URLs
    steps = test_job.get("steps", [])
    step_runs = [s.get("run", "") for s in steps]

    # Must have docker build step
    assert any("docker build" in run_cmd for run_cmd in step_runs), "CI must verify Docker image build"

    # Must have migration step
    assert any("alembic upgrade head" in run_cmd for run_cmd in step_runs), "CI must run Alembic migrations"


def test_gitattributes_enforces_lf_line_endings() -> None:
    gitattributes_file = REPO_ROOT / ".gitattributes"
    assert gitattributes_file.is_file(), ".gitattributes must exist"

    content = gitattributes_file.read_text(encoding="utf-8")
    assert "*.sh" in content
    assert "eol=lf" in content
    assert "Dockerfile" in content
