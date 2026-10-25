"""API integration and security tests for Stage 4 Research Planner Agent and Agent Orchestrator.

Endpoints tested:
- POST /api/v1/agents/plan
- POST /api/v1/agents/run
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from nexora.agents.models import (
    PLAN_STANDARD_DIMENSIONS,
    AgentAction,
    AgentContext,
    AgentRunResult,
    AgentStatus,
    AgentStep,
    AgentStepStatus,
    AgentTrace,
    PlannerServiceError,
    ResearchPlan,
    ResearchPlanDimension,
)
from nexora.agents.orchestrator import AgentOrchestrator
from nexora.agents.planner import ResearchPlannerAgent
from nexora.agents.tools import ToolRegistry, create_default_tool_registry
from nexora.api.app import create_app
from nexora.auth.service import AuthenticationService
from nexora.config.rate_limit import RateLimitConfig
from nexora.database.models import Base
from nexora.database.models.collection import Collection
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.research.synthesis import (
    LiteratureSynthesisReport,
    SynthesisDivergence,
    SynthesisFinding,
    SynthesisLimitation,
)
from nexora.research.trend_service import ResearchTrendService
from nexora.retrieval.in_memory import InMemoryVectorStore
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_session_factory():
    """In-memory SQLite session factory with schema initialized."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def populated_agent_data(sqlite_session_factory):
    """Seed SQLite database with two isolated tenants and research collections."""
    session = sqlite_session_factory()
    try:
        user_a = User(email="tenant-a@example.com")
        user_b = User(email="tenant-b@example.com")
        session.add_all([user_a, user_b])
        session.flush()

        ws_a = Workspace(name="Workspace A", user_id=user_a.id)
        ws_b = Workspace(name="Workspace B", user_id=user_b.id)
        session.add_all([ws_a, ws_b])
        session.flush()

        col_a = Collection(name="Collection A", workspace_id=ws_a.id)
        col_b = Collection(name="Collection B", workspace_id=ws_b.id)
        session.add_all([col_a, col_b])
        session.flush()

        paper_repo = PaperRepository(session)
        doc_repo = DocumentRepository(session)
        chunk_repo = DocumentChunkRepository(session)

        p1 = paper_repo.create(
            collection_id=col_a.id,
            title="Transformer Attention Models",
            abstract="Self-attention mechanisms for sequence modeling.",
            publication_date=datetime.date(2021, 6, 15),
        )
        d1 = doc_repo.create(
            paper_id=p1.id,
            filename="paper1.pdf",
            file_size=1024,
            checksum="1" * 64,
            storage_key="s3://paper1.pdf",
            indexing_status="completed",
        )
        chunk_repo.create(
            document_id=d1.id,
            chunk_index=0,
            text="The Transformer architecture relies on self-attention and Adam optimizer for WMT benchmark translation.",
        )

        key_repo = ApiKeyRepository(session)
        _, raw_token_a = key_repo.generate_key(user_id=user_a.id, name="Key A")
        _, raw_token_b = key_repo.generate_key(user_id=user_b.id, name="Key B")

        session.commit()

        return {
            "user_a_id": user_a.id,
            "user_b_id": user_b.id,
            "col_a_id": col_a.id,
            "col_b_id": col_b.id,
            "token_a": raw_token_a,
            "token_b": raw_token_b,
        }
    finally:
        session.close()


class FakeMockLLMService:
    """Mock LLM Service that returns deterministic agent responses."""

    _is_mock = True

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        if self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = json.dumps({"action": "final_answer", "answer": "Deterministic final answer."})
        self._call_count += 1
        return LLMResponse(
            text=text,
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )


def _build_test_app(
    sqlite_session_factory,
    *,
    enforce_auth: bool = True,
    llm_service: Any = None,
    domain_llm: Any = None,
    orchestrator: Any = None,
    planner: Any = None,
    rate_limit_config: Any = None,
) -> TestClient:
    """Compose test application with wired agents and authentication."""
    vector_store = InMemoryVectorStore(dimension=128)
    auth_service = AuthenticationService(session_factory=sqlite_session_factory)
    active_agent_llm = llm_service or FakeMockLLMService()
    active_domain_llm = domain_llm or FakeMockLLMService()

    trend_service = ResearchTrendService(
        active_domain_llm,
        session_factory=sqlite_session_factory,
        vector_store=vector_store,
    )

    if planner is None:
        planner = ResearchPlannerAgent(
            active_agent_llm,
            trend_service=trend_service,
            session_factory=sqlite_session_factory,
        )

    if orchestrator is None:
        tool_reg = create_default_tool_registry(
            trend_service=trend_service,
            session_factory=sqlite_session_factory,
        )
        orchestrator = AgentOrchestrator(
            llm_service=active_agent_llm,
            tool_registry=tool_reg,
        )

    class DummyResearchService:
        def ask(self, req):
            pass

    app = create_app(
        DummyResearchService(),
        trend_service=trend_service,
        planner_agent=planner,
        orchestrator=orchestrator,
        session_factory=sqlite_session_factory,
        auth_service=auth_service,
        enforce_auth=enforce_auth,
        rate_limit_config=rate_limit_config,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/v1/agents/plan Tests
# ---------------------------------------------------------------------------

def test_plan_endpoint_unauthenticated_fails_when_enforced(sqlite_session_factory, populated_agent_data):
    """Calling POST /api/v1/agents/plan without credentials fails with 401 Unauthorized."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=True)

    resp = client.post(
        "/api/v1/agents/plan",
        json={"goal": "Develop an efficient transformer for long contexts"},
    )
    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["code"] == "unauthorized"


def test_plan_endpoint_invalid_token_fails(sqlite_session_factory, populated_agent_data):
    """Calling POST /api/v1/agents/plan with an invalid token fails with 401 Unauthorized."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=True)

    resp = client.post(
        "/api/v1/agents/plan",
        headers={"Authorization": "Bearer invalid-or-expired-token"},
        json={"goal": "Develop an efficient transformer for long contexts"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_plan_endpoint_authenticated_success_without_collection(sqlite_session_factory, populated_agent_data):
    """Authenticated caller generates a full 7-dimension plan without collection grounding."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=True)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/plan",
        headers=headers,
        json={"goal": "Synthesize a robust evaluation benchmark for multi-agent reasoning"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "plan_id" in data
    assert data["title"]
    assert data["research_problem"]
    assert len(data["research_questions"]) >= 1
    assert data["collection_id"] is None
    assert data["is_mock"] is True

    # Validate all 7 standard dimensions
    dims = data["dimensions"]
    for dim_name in PLAN_STANDARD_DIMENSIONS:
        assert dim_name in dims, f"Missing standard dimension: {dim_name}"
        dim = dims[dim_name]
        assert "title" in dim
        assert "description" in dim
        assert "details" in dim
        assert isinstance(dim["is_grounded"], bool)

    # Hypotheses must not be represented as verified evidence
    methodology = dims["methodology"]
    assert methodology["is_grounded"] is False
    assert methodology["details"].get("recommendation_type") == "proposed_hypothesis"


def test_plan_endpoint_authenticated_success_with_authorized_collection(sqlite_session_factory, populated_agent_data):
    """Authenticated caller generates an evidence-grounded plan using their owned collection."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=True)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}
    col_a_id = str(populated_agent_data["col_a_id"])

    resp = client.post(
        "/api/v1/agents/plan",
        headers=headers,
        json={
            "goal": "Empirical comparison of transformer attention mechanisms",
            "collection_id": col_a_id,
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["collection_id"] == col_a_id
    dims = data["dimensions"]
    assert "literature_strategy" in dims
    assert "documented_gaps" in dims


def test_plan_endpoint_cross_tenant_collection_access_forbidden(sqlite_session_factory, populated_agent_data):
    """User B attempting to plan using User A's collection is rejected with 403 Forbidden."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=True)
    headers_b = {"Authorization": f"Bearer {populated_agent_data['token_b']}"}
    col_a_id = str(populated_agent_data["col_a_id"])

    resp = client.post(
        "/api/v1/agents/plan",
        headers=headers_b,
        json={
            "goal": "Attempting to inspect tenant A's literature",
            "collection_id": col_a_id,
        },
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"


def test_plan_endpoint_missing_collection_returns_not_found(sqlite_session_factory, populated_agent_data):
    """Referencing a non-existent collection UUID returns 404 Not Found."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=True)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}
    random_uuid = str(uuid.uuid4())

    resp = client.post(
        "/api/v1/agents/plan",
        headers=headers,
        json={"goal": "Valid research problem", "collection_id": random_uuid},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_plan_endpoint_malformed_collection_uuid_returns_400(sqlite_session_factory, populated_agent_data):
    """Supplying a malformed collection_id string returns 400 Bad Request."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=True)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/plan",
        headers=headers,
        json={"goal": "Valid research problem", "collection_id": "not-a-valid-uuid"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_plan_endpoint_validation_errors(sqlite_session_factory, populated_agent_data):
    """Empty or oversized goals and extra fields are rejected at the HTTP schema level with 422."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=True)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    # 1. Empty goal
    resp_empty = client.post(
        "/api/v1/agents/plan",
        headers=headers,
        json={"goal": ""},
    )
    assert resp_empty.status_code == 422

    # 2. Oversized goal (>2000 chars)
    resp_oversized = client.post(
        "/api/v1/agents/plan",
        headers=headers,
        json={"goal": "x" * 2001},
    )
    assert resp_oversized.status_code == 422

    # 3. Extra forbidden field
    resp_extra = client.post(
        "/api/v1/agents/plan",
        headers=headers,
        json={"goal": "Valid goal", "malicious_injected_field": "exploit"},
    )
    assert resp_extra.status_code == 422


def test_plan_endpoint_provider_failure_returns_500(sqlite_session_factory, populated_agent_data):
    """When the planner agent raises PlannerServiceError, API returns 500 planner_failed."""
    mock_planner = MagicMock()
    mock_planner.plan.side_effect = PlannerServiceError("Upstream model generation error")

    client = _build_test_app(sqlite_session_factory, enforce_auth=True, planner=mock_planner)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/plan",
        headers=headers,
        json={"goal": "Valid research goal"},
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "planner_failed"


# ---------------------------------------------------------------------------
# POST /api/v1/agents/run Tests
# ---------------------------------------------------------------------------

def test_run_endpoint_unauthenticated_fails_when_enforced(sqlite_session_factory, populated_agent_data):
    """Calling POST /api/v1/agents/run without credentials fails with 401 Unauthorized."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=True)

    resp = client.post(
        "/api/v1/agents/run",
        json={"goal": "Analyze recent literature on reinforcement learning"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_run_endpoint_authenticated_success_immediate_final_answer(sqlite_session_factory, populated_agent_data):
    """Authenticated agent run succeeds immediately when model returns a final answer."""
    llm = FakeMockLLMService([
        '{"action": "final_answer", "answer": "The transformer models have scaled to trillions of parameters."}'
    ])
    client = _build_test_app(sqlite_session_factory, enforce_auth=True, llm_service=llm)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/run",
        headers=headers,
        json={"goal": "Summarize transformer scale limits"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "completed"
    assert data["final_answer"] == "The transformer models have scaled to trillions of parameters."
    assert data["total_steps"] == 0
    assert data["trace"]["status"] == "completed"


def test_run_endpoint_multi_step_tool_execution(sqlite_session_factory, populated_agent_data):
    """Agent executes an allowlisted tool, observes output, and concludes with final answer."""
    llm = FakeMockLLMService([
        '{"action": "call_tool", "tool_name": "analyze_trends", "tool_input": {"collection_id": "' + str(populated_agent_data["col_a_id"]) + '"}}',
        '{"action": "final_answer", "answer": "The analysis shows growing adoption of self-attention mechanisms."}',
    ])
    client = _build_test_app(sqlite_session_factory, enforce_auth=True, llm_service=llm)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/run",
        headers=headers,
        json={"goal": "Analyze trends in collection A"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "completed"
    assert data["total_steps"] == 1
    assert data["trace"]["steps"][0]["tool_name"] == "analyze_trends"
    assert data["trace"]["steps"][0]["status"] == "success"
    assert "growing adoption" in data["final_answer"]


def test_run_endpoint_unknown_tool_halts_safely(sqlite_session_factory, populated_agent_data):
    """Model requesting a tool outside the allowlist halts execution safely without exposing internals."""
    llm = FakeMockLLMService([
        '{"action": "call_tool", "tool_name": "execute_arbitrary_shell", "tool_input": {"cmd": "whoami"}}',
    ])
    client = _build_test_app(sqlite_session_factory, enforce_auth=True, llm_service=llm)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/run",
        headers=headers,
        json={"goal": "Attempting arbitrary tool invocation"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "failed"
    assert "unknown tool" in data["trace"]["error"].lower()
    assert data["trace"]["steps"][0]["status"] == "failed"


def test_run_endpoint_step_limit_enforcement(sqlite_session_factory, populated_agent_data):
    """Agent halting at configured max_steps without final answer transitions to max_steps_exceeded."""
    col_id = str(populated_agent_data["col_a_id"])
    # Return different tool inputs each step to avoid loop detection
    llm = FakeMockLLMService([
        '{"action": "call_tool", "tool_name": "analyze_trends", "tool_input": {"collection_id": "' + col_id + '", "time_unit": "year"}}',
        '{"action": "call_tool", "tool_name": "analyze_trends", "tool_input": {"collection_id": "' + col_id + '", "time_unit": "month"}}',
        '{"action": "call_tool", "tool_name": "analyze_trends", "tool_input": {"collection_id": "' + col_id + '", "time_unit": "quarter"}}',
    ])
    client = _build_test_app(sqlite_session_factory, enforce_auth=True, llm_service=llm)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/run",
        headers=headers,
        json={"goal": "Indefinite trend loop", "max_steps": 2},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "max_steps_exceeded"
    assert data["total_steps"] == 2
    assert "Maximum execution steps (2) reached" in data["trace"]["error"]


def test_run_endpoint_loop_protection(sqlite_session_factory, populated_agent_data):
    """Identical tool call repeated twice triggers loop protection and aborts."""
    col_id = str(populated_agent_data["col_a_id"])
    identical_call = '{"action": "call_tool", "tool_name": "analyze_trends", "tool_input": {"collection_id": "' + col_id + '"}}'
    llm = FakeMockLLMService([identical_call, identical_call])

    client = _build_test_app(sqlite_session_factory, enforce_auth=True, llm_service=llm)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/run",
        headers=headers,
        json={"goal": "Loop test"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "failed"
    assert "Loop protection triggered" in data["trace"]["error"]


def test_run_endpoint_cross_tenant_tool_access_fails_closed(sqlite_session_factory, populated_agent_data):
    """When a tool call attempts cross-tenant access to another tenant's collection, run halts immediately."""
    col_b_id = str(populated_agent_data["col_b_id"])
    # User A tries to invoke synthesize_literature on Tenant B's collection
    llm = FakeMockLLMService([
        '{"action": "call_tool", "tool_name": "synthesize_literature", "tool_input": {"collection_id": "' + col_b_id + '"}}',
    ])
    client = _build_test_app(sqlite_session_factory, enforce_auth=True, llm_service=llm)
    headers_a = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/run",
        headers=headers_a,
        json={"goal": "Attempting cross-tenant tool access"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "failed"
    assert "Authorization check failed" in data["trace"]["error"]


def test_run_endpoint_no_secret_leakage_in_trace(sqlite_session_factory, populated_agent_data):
    """Tool inputs with sensitive credentials are automatically redacted in the returned trace."""
    llm = FakeMockLLMService([
        '{"action": "call_tool", "tool_name": "analyze_trends", "tool_input": {"collection_id": "' + str(populated_agent_data["col_a_id"]) + '", "api_key": "super-secret-key-12345"}}',
        '{"action": "final_answer", "answer": "Done."}',
    ])
    client = _build_test_app(sqlite_session_factory, enforce_auth=True, llm_service=llm)
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/run",
        headers=headers,
        json={"goal": "Redaction test"},
    )
    assert resp.status_code == 200
    data = resp.json()

    step_input = data["trace"]["steps"][0]["tool_input"]
    assert step_input["api_key"] == "[REDACTED]"
    assert "super-secret-key-12345" not in json.dumps(data)


# ---------------------------------------------------------------------------
# Rate Limiting Tests on Agent Endpoints
# ---------------------------------------------------------------------------

def test_rate_limiting_on_agents_endpoints(sqlite_session_factory, populated_agent_data):
    """Rate limiting middleware categorizes agent endpoints under heavy_analysis and enforces limits."""
    rate_cfg = RateLimitConfig(
        enabled=True,
        backend="memory",
        heavy_analysis_limit=2,
        window_seconds=60,
    )
    client = _build_test_app(
        sqlite_session_factory,
        enforce_auth=True,
        rate_limit_config=rate_cfg,
    )
    headers = {"Authorization": f"Bearer {populated_agent_data['token_a']}"}

    # Request 1: allowed
    r1 = client.post("/api/v1/agents/plan", headers=headers, json={"goal": "Research goal 1"})
    assert r1.status_code == 200

    # Request 2: allowed
    r2 = client.post("/api/v1/agents/plan", headers=headers, json={"goal": "Research goal 2"})
    assert r2.status_code == 200

    # Request 3: rate limited
    r3 = client.post("/api/v1/agents/plan", headers=headers, json={"goal": "Research goal 3"})
    assert r3.status_code == 429
    assert r3.json()["error"]["code"] == "rate_limited"


# ---------------------------------------------------------------------------
# Offline Demo Mode Unauthenticated Access
# ---------------------------------------------------------------------------

def test_offline_demo_mode_unauthenticated_access(sqlite_session_factory):
    """In unauthenticated offline demo mode (enforce_auth=False), plan and run work without tokens."""
    client = _build_test_app(sqlite_session_factory, enforce_auth=False)

    # 1. Unauthenticated plan without collection succeeds
    resp_plan = client.post(
        "/api/v1/agents/plan",
        json={"goal": "Design an offline benchmark for NLP models"},
    )
    assert resp_plan.status_code == 200
    assert resp_plan.json()["is_mock"] is True

    # 2. Unauthenticated run succeeds
    resp_run = client.post(
        "/api/v1/agents/run",
        json={"goal": "Summarize offline agent capabilities"},
    )
    assert resp_run.status_code == 200
    assert resp_run.json()["status"] == "completed"
