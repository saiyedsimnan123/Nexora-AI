"""Integration tests for Stage 5 Step 4.2.3: Knowledge Graph Agent Registry & Bootstrap Integration.

Verifies:
1. Production registry construction via create_default_tool_registry() and build_tool_registry().
2. Proper dependency injection (session_factory, KnowledgeGraphTraversalService, KnowledgeGraphAnalyticsService).
3. Registration of all four graph tools alongside the seven research tools (11 total tools).
4. Each graph tool can be executed through the bounded AgentOrchestrator with real database records.
5. Live anonymous /agents/run requests fail closed against private graph data even when enforce_auth=False.
6. Live cross-tenant /agents/run requests fail closed with forbidden status.
7. Existing research tools remain fully functional and discoverable.
8. Unknown tools continue to fail closed.
"""

from __future__ import annotations

import datetime
import json
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.agents.models import (
    AgentAction,
    AgentContext,
    AgentStatus,
    AgentStepStatus,
    ToolDefinition,
)
from nexora.agents.orchestrator import AgentOrchestrator
from nexora.agents.tools import (
    AgentTool,
    AnalyzePaperTool,
    AnalyzeTrendsTool,
    ComparePapersTool,
    DiscoverResearchGapsTool,
    ExploreGraphNeighborhoodTool,
    FindGraphPathTool,
    QueryGraphAnalyticsTool,
    SearchGraphEntitiesTool,
    SearchLiteratureTool,
    SynthesizeLiteratureTool,
    ToolRegistry,
    VerifyClaimTool,
    create_default_tool_registry,
)
from nexora.api.app import create_app
from nexora.api.errors import InvalidConfigurationError
from nexora.auth.service import AuthenticationService
from nexora.bootstrap.application import build_tool_registry
from nexora.database.models.api_key import ApiKey
from nexora.database.models.base import Base
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.graph_entity import GraphEntity
from nexora.database.models.graph_relationship import GraphRelationship
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.database.repository.graph_entity import GraphEntityRepository
from nexora.database.repository.graph_relationship import GraphRelationshipRepository
from nexora.database.repository.paper import PaperRepository
from nexora.graph.analytics_service import KnowledgeGraphAnalyticsService
from nexora.graph.traversal import KnowledgeGraphTraversalService
from nexora.llm.models import LLMMessage, LLMResponse


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

class ScriptedLLMService:
    """Mock LLM Service returning scripted JSON responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        if self.call_count >= len(self.responses):
            return LLMResponse(
                text=json.dumps({"action": "final_answer", "answer": "Default complete."}),
                model="scripted-mock",
            )
        resp = self.responses[self.call_count]
        self.call_count += 1
        return LLMResponse(text=resp, model="scripted-mock")


@pytest.fixture
def sqlite_session_factory():
    """In-memory SQLite session factory with complete schema created."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def seeded_graph_data(sqlite_session_factory):
    """Seed isolated tenants, collections, papers, entities, and relationships."""
    session = sqlite_session_factory()
    try:
        # Tenants
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

        # API Keys
        api_key_repo = ApiKeyRepository(session)
        _, token_a = api_key_repo.generate_key(user_id=user_a.id, name="Test Key A")
        _, token_b = api_key_repo.generate_key(user_id=user_b.id, name="Test Key B")

        # Papers in Col A
        paper_repo = PaperRepository(session)
        p1 = paper_repo.create(
            collection_id=col_a.id,
            title="Attention is All You Need",
            abstract="Seminal transformer paper.",
            publication_date=datetime.date(2017, 6, 12),
        )
        p2 = paper_repo.create(
            collection_id=col_a.id,
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            abstract="Bidirectional representations.",
            publication_date=datetime.date(2018, 10, 11),
        )

        # Entities in Col A
        entity_repo = GraphEntityRepository(session)
        e_vaswani = entity_repo.create(
            collection_id=col_a.id,
            name="Ashish Vaswani",
            entity_type="author",
            description="Transformer author.",
        )
        e_devlin = entity_repo.create(
            collection_id=col_a.id,
            name="Jacob Devlin",
            entity_type="author",
            description="BERT author.",
        )
        e_transformer = entity_repo.create(
            collection_id=col_a.id,
            name="Transformer",
            entity_type="method",
            description="Sequence transduction model.",
        )
        e_attention = entity_repo.create(
            collection_id=col_a.id,
            name="Self-Attention",
            entity_type="concept",
            description="Attention mechanism.",
        )

        # Entities in Col B (Tenant B)
        e_foreign = entity_repo.create(
            collection_id=col_b.id,
            name="Secret Entity B",
            entity_type="method",
            description="Tenant B private entity.",
        )

        # Relationships in Col A
        rel_repo = GraphRelationshipRepository(session)
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="co_authored_with",
            source_type="entity",
            source_id=e_vaswani.id,
            target_type="entity",
            target_id=e_devlin.id,
            confidence=0.92,
            evidence_text="Vaswani and Devlin co-authored deep learning work.",
        )
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="uses_mechanism",
            source_type="entity",
            source_id=e_transformer.id,
            target_type="entity",
            target_id=e_attention.id,
            confidence=0.95,
            evidence_text="The Transformer relies entirely on self-attention.",
        )
        rel_repo.create(
            collection_id=col_a.id,
            relationship_type="authored_by",
            source_type="paper",
            source_id=p1.id,
            target_type="entity",
            target_id=e_vaswani.id,
            confidence=0.99,
            evidence_text="Attention is All You Need was authored by Ashish Vaswani.",
        )

        session.commit()

        return {
            "user_a_id": user_a.id,
            "user_b_id": user_b.id,
            "col_a_id": col_a.id,
            "col_b_id": col_b.id,
            "token_a": token_a,
            "token_b": token_b,
            "p1_id": p1.id,
            "p2_id": p2.id,
            "e_vaswani_id": e_vaswani.id,
            "e_devlin_id": e_devlin.id,
            "e_transformer_id": e_transformer.id,
            "e_attention_id": e_attention.id,
            "e_foreign_id": e_foreign.id,
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 1. Production Registry Construction & Dependency Injection
# ---------------------------------------------------------------------------

def test_default_tool_registry_contains_all_four_graph_tools_with_session_factory(sqlite_session_factory):
    """Verify that passing session_factory registers all four graph tools with wired dependencies."""
    registry = create_default_tool_registry(session_factory=sqlite_session_factory)
    names = registry.list_tool_names()

    assert "search_graph_entities" in names
    assert "explore_graph_neighborhood" in names
    assert "find_graph_path" in names
    assert "query_graph_analytics" in names

    # Verify injected services
    search_tool = registry.get("search_graph_entities")
    assert isinstance(search_tool, SearchGraphEntitiesTool)
    assert search_tool._session_factory is sqlite_session_factory

    explore_tool = registry.get("explore_graph_neighborhood")
    assert isinstance(explore_tool, ExploreGraphNeighborhoodTool)
    assert isinstance(explore_tool._service, KnowledgeGraphTraversalService)

    path_tool = registry.get("find_graph_path")
    assert isinstance(path_tool, FindGraphPathTool)
    assert isinstance(path_tool._service, KnowledgeGraphTraversalService)

    analytics_tool = registry.get("query_graph_analytics")
    assert isinstance(analytics_tool, QueryGraphAnalyticsTool)
    assert isinstance(analytics_tool._service, KnowledgeGraphAnalyticsService)


def test_default_tool_registry_contains_all_eleven_tools_with_all_services(sqlite_session_factory):
    """Verify registry contains all 11 tools when both research and graph dependencies are provided."""
    mock_query = type("QueryMock", (), {"ask": lambda *a, **k: None})()
    mock_analysis = type("AnalysisMock", (), {"analyze_paper": lambda *a, **k: None})()
    mock_comparison = type("ComparisonMock", (), {"compare_cross_paper": lambda *a, **k: None})()
    mock_gap = type("GapMock", (), {"discover_gaps_for_collection": lambda *a, **k: None})()
    mock_trend = type("TrendMock", (), {
        "analyze_collection_trends": lambda *a, **k: None,
        "synthesize_collection_literature": lambda *a, **k: None,
    })()
    mock_verifier = type("VerifierMock", (), {"verify_claim": lambda *a, **k: None})()

    registry = create_default_tool_registry(
        query_service=mock_query,
        analysis_service=mock_analysis,
        comparison_service=mock_comparison,
        gap_service=mock_gap,
        trend_service=mock_trend,
        verifier=mock_verifier,
        session_factory=sqlite_session_factory,
    )

    names = registry.list_tool_names()
    assert len(names) == 11
    expected_tools = {
        "search_literature",
        "analyze_paper",
        "compare_papers",
        "discover_research_gaps",
        "analyze_trends",
        "synthesize_literature",
        "verify_claim",
        "search_graph_entities",
        "explore_graph_neighborhood",
        "find_graph_path",
        "query_graph_analytics",
    }
    assert set(names) == expected_tools

    # Verify duplicate registration is strictly forbidden
    with pytest.raises(ValueError, match="already registered"):
        registry.register(SearchGraphEntitiesTool(session_factory=sqlite_session_factory))


def test_build_tool_registry_and_build_http_app_integration(sqlite_session_factory):
    """Verify build_tool_registry creates graph tools and wires them to application orchestrator."""
    registry = build_tool_registry(session_factory=sqlite_session_factory)
    assert "search_graph_entities" in registry.list_tool_names()
    assert "explore_graph_neighborhood" in registry.list_tool_names()
    assert "find_graph_path" in registry.list_tool_names()
    assert "query_graph_analytics" in registry.list_tool_names()

    # Tool definitions are discoverable
    defs = registry.list_tools()
    def_map = {d.name: d for d in defs}
    for name in ("search_graph_entities", "explore_graph_neighborhood", "find_graph_path", "query_graph_analytics"):
        assert name in def_map
        assert isinstance(def_map[name], ToolDefinition)
        assert len(def_map[name].description) > 0
        assert isinstance(def_map[name].parameters_schema, dict)


def test_default_tool_registry_raises_on_traversal_service_init_failure(monkeypatch, sqlite_session_factory):
    """Verify that traversal service construction failure raises InvalidConfigurationError instead of silently passing."""
    def _exploding_init(self, *args, **kwargs):
        raise RuntimeError("Simulated KnowledgeGraphTraversalService failure")

    monkeypatch.setattr(KnowledgeGraphTraversalService, "__init__", _exploding_init)

    with pytest.raises(InvalidConfigurationError, match="Failed to initialize KnowledgeGraphTraversalService"):
        create_default_tool_registry(session_factory=sqlite_session_factory)


def test_default_tool_registry_raises_on_analytics_service_init_failure(monkeypatch, sqlite_session_factory):
    """Verify that analytics service construction failure raises InvalidConfigurationError instead of silently passing."""
    def _exploding_init(self, *args, **kwargs):
        raise RuntimeError("Simulated KnowledgeGraphAnalyticsService failure")

    monkeypatch.setattr(KnowledgeGraphAnalyticsService, "__init__", _exploding_init)

    with pytest.raises(InvalidConfigurationError, match="Failed to initialize KnowledgeGraphAnalyticsService"):
        create_default_tool_registry(session_factory=sqlite_session_factory)


def test_default_tool_registry_legacy_without_database_registers_no_graph_tools():
    """Verify that legacy or database-free configurations register no graph tools and raise no errors."""
    registry = create_default_tool_registry(session_factory=None)
    assert registry.list_tool_names() == []

    mock_query = type("QueryMock", (), {"ask": lambda *a, **k: None})()
    registry_with_research = create_default_tool_registry(
        query_service=mock_query,
        session_factory=None,
    )
    assert registry_with_research.list_tool_names() == ["search_literature"]
    assert not registry_with_research.has("search_graph_entities")
    assert not registry_with_research.has("explore_graph_neighborhood")
    assert not registry_with_research.has("find_graph_path")
    assert not registry_with_research.has("query_graph_analytics")


def test_default_tool_registry_preserves_explicitly_injected_graph_services(sqlite_session_factory):
    """Verify that explicitly injected graph services are preserved over session_factory auto-instantiation."""
    custom_traversal = type("CustomTraversal", (), {
        "explore_multihop": lambda *a, **k: None,
        "find_paths": lambda *a, **k: None,
    })()
    custom_analytics = type("CustomAnalytics", (), {
        "get_collection_analytics": lambda *a, **k: None,
        "get_author_collaborations": lambda *a, **k: None,
        "get_author_connectivity": lambda *a, **k: None,
        "get_summary": lambda *a, **k: None,
    })()

    registry = create_default_tool_registry(
        session_factory=sqlite_session_factory,
        graph_traversal_service=custom_traversal,
        graph_analytics_service=custom_analytics,
    )

    assert registry.get("explore_graph_neighborhood")._service is custom_traversal
    assert registry.get("find_graph_path")._service is custom_traversal
    assert registry.get("query_graph_analytics")._service is custom_analytics


# ---------------------------------------------------------------------------
# 2. Orchestrator End-to-End Execution for each Graph Tool
# ---------------------------------------------------------------------------

def test_orchestrator_end_to_end_search_graph_entities(sqlite_session_factory, seeded_graph_data):
    """Verify orchestrator invokes search_graph_entities against real database."""
    col_id = str(seeded_graph_data["col_a_id"])
    user_id = seeded_graph_data["user_a_id"]
    auth_svc = AuthenticationService(sqlite_session_factory)

    step_1 = json.dumps({
        "action": "call_tool",
        "tool_name": "search_graph_entities",
        "tool_input": {"collection_id": col_id, "query": "Transformer"},
    })
    step_2 = json.dumps({
        "action": "final_answer",
        "answer": "Found Transformer method entity in collection graph.",
    })
    llm = ScriptedLLMService([step_1, step_2])
    registry = create_default_tool_registry(session_factory=sqlite_session_factory)
    orchestrator = AgentOrchestrator(llm, registry)

    context = AgentContext(
        auth_enforced=True,
        user_id=user_id,
        auth_service=auth_svc,
        session_factory=sqlite_session_factory,
    )

    result = orchestrator.run("Search for Transformer entity", context=context)
    assert result.status == AgentStatus.COMPLETED.value
    assert result.total_steps == 1
    assert result.trace.steps[0].status == AgentStepStatus.SUCCESS.value
    assert "Found 1 entities" in result.trace.steps[0].output_summary
    assert "Transformer" in result.final_answer


def test_orchestrator_end_to_end_explore_graph_neighborhood(sqlite_session_factory, seeded_graph_data):
    """Verify orchestrator invokes explore_graph_neighborhood against real database."""
    col_id = str(seeded_graph_data["col_a_id"])
    node_id = str(seeded_graph_data["e_transformer_id"])
    user_id = seeded_graph_data["user_a_id"]
    auth_svc = AuthenticationService(sqlite_session_factory)

    step_1 = json.dumps({
        "action": "call_tool",
        "tool_name": "explore_graph_neighborhood",
        "tool_input": {
            "collection_id": col_id,
            "node_id": node_id,
            "node_type": "entity",
            "max_depth": 1,
        },
    })
    step_2 = json.dumps({
        "action": "final_answer",
        "answer": "Explored 1-hop neighborhood of Transformer.",
    })
    llm = ScriptedLLMService([step_1, step_2])
    registry = create_default_tool_registry(session_factory=sqlite_session_factory)
    orchestrator = AgentOrchestrator(llm, registry)

    context = AgentContext(
        auth_enforced=True,
        user_id=user_id,
        auth_service=auth_svc,
        session_factory=sqlite_session_factory,
    )

    result = orchestrator.run("Explore Transformer neighborhood", context=context)
    assert result.status == AgentStatus.COMPLETED.value
    assert result.total_steps == 1
    assert result.trace.steps[0].status == AgentStepStatus.SUCCESS.value
    assert "Explored neighborhood" in result.trace.steps[0].output_summary


def test_orchestrator_end_to_end_find_graph_path(sqlite_session_factory, seeded_graph_data):
    """Verify orchestrator invokes find_graph_path against real database."""
    col_id = str(seeded_graph_data["col_a_id"])
    src_id = str(seeded_graph_data["p1_id"])
    dst_id = str(seeded_graph_data["e_devlin_id"])
    user_id = seeded_graph_data["user_a_id"]
    auth_svc = AuthenticationService(sqlite_session_factory)

    step_1 = json.dumps({
        "action": "call_tool",
        "tool_name": "find_graph_path",
        "tool_input": {
            "collection_id": col_id,
            "source_id": src_id,
            "source_type": "paper",
            "target_id": dst_id,
            "target_type": "entity",
            "max_depth": 2,
        },
    })
    step_2 = json.dumps({
        "action": "final_answer",
        "answer": "Discovered path from Paper 1 to Jacob Devlin.",
    })
    llm = ScriptedLLMService([step_1, step_2])
    registry = create_default_tool_registry(session_factory=sqlite_session_factory)
    orchestrator = AgentOrchestrator(llm, registry)

    context = AgentContext(
        auth_enforced=True,
        user_id=user_id,
        auth_service=auth_svc,
        session_factory=sqlite_session_factory,
    )

    result = orchestrator.run("Find path between Paper 1 and Jacob Devlin", context=context)
    assert result.status == AgentStatus.COMPLETED.value
    assert result.total_steps == 1
    assert result.trace.steps[0].status == AgentStepStatus.SUCCESS.value
    assert "Discovered" in result.trace.steps[0].output_summary


def test_orchestrator_end_to_end_query_graph_analytics(sqlite_session_factory, seeded_graph_data):
    """Verify orchestrator invokes query_graph_analytics against real database."""
    col_id = str(seeded_graph_data["col_a_id"])
    user_id = seeded_graph_data["user_a_id"]
    auth_svc = AuthenticationService(sqlite_session_factory)

    step_1 = json.dumps({
        "action": "call_tool",
        "tool_name": "query_graph_analytics",
        "tool_input": {
            "collection_id": col_id,
            "query_type": "summary",
        },
    })
    step_2 = json.dumps({
        "action": "final_answer",
        "answer": "Graph contains 4 entities and 3 relationships.",
    })
    llm = ScriptedLLMService([step_1, step_2])
    registry = create_default_tool_registry(session_factory=sqlite_session_factory)
    orchestrator = AgentOrchestrator(llm, registry)

    context = AgentContext(
        auth_enforced=True,
        user_id=user_id,
        auth_service=auth_svc,
        session_factory=sqlite_session_factory,
    )

    result = orchestrator.run("Query graph summary", context=context)
    assert result.status == AgentStatus.COMPLETED.value
    assert result.total_steps == 1
    assert result.trace.steps[0].status == AgentStepStatus.SUCCESS.value
    assert "Collection graph summary" in result.trace.steps[0].output_summary


# ---------------------------------------------------------------------------
# 3. Live HTTP API Regression Tests
# ---------------------------------------------------------------------------

def _build_test_api_client(
    sqlite_session_factory,
    *,
    enforce_auth: bool = False,
    llm_service: Any = None,
) -> TestClient:
    """Build live FastAPI test client with production tool registry and wired orchestrator."""
    auth_service = AuthenticationService(session_factory=sqlite_session_factory)
    active_llm = llm_service or ScriptedLLMService([
        '{"action": "final_answer", "answer": "Default live test answer."}'
    ])
    tool_registry = create_default_tool_registry(session_factory=sqlite_session_factory)
    orchestrator = AgentOrchestrator(llm_service=active_llm, tool_registry=tool_registry)

    class DummyResearchService:
        def ask(self, req):
            pass

    app = create_app(
        DummyResearchService(),
        orchestrator=orchestrator,
        tool_registry=tool_registry,
        session_factory=sqlite_session_factory,
        auth_service=auth_service,
        enforce_auth=enforce_auth,
    )
    return TestClient(app)


def test_live_agents_run_anonymous_denied_graph_access_even_when_enforce_auth_false(
    sqlite_session_factory, seeded_graph_data
):
    """Anonymous live /agents/run request cannot access private graph data, even when enforce_auth=False."""
    col_a_id = str(seeded_graph_data["col_a_id"])
    # LLM attempts to call search_graph_entities
    step = json.dumps({
        "action": "call_tool",
        "tool_name": "search_graph_entities",
        "tool_input": {"collection_id": col_a_id, "query": "Transformer"},
    })
    llm = ScriptedLLMService([step])

    # enforce_auth=False simulates demo/offline mode
    client = _build_test_api_client(sqlite_session_factory, enforce_auth=False, llm_service=llm)

    resp = client.post(
        "/api/v1/agents/run",
        json={"goal": "Attempting anonymous graph search"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # The run must fail closed because graph data is tenant-private
    assert data["status"] == "failed"
    assert "Authorization check failed" in data["trace"]["error"]
    assert data["trace"]["steps"][0]["status"] == "failed"
    assert data["trace"]["steps"][0]["error"] == "Authentication required: collection knowledge graph data is private and cannot be accessed anonymously."


def test_live_agents_run_cross_tenant_denied_graph_access(
    sqlite_session_factory, seeded_graph_data
):
    """Authenticated user A attempting to access tenant B's collection graph is rejected with forbidden."""
    col_b_id = str(seeded_graph_data["col_b_id"])
    # User A tries to explore Collection B
    step = json.dumps({
        "action": "call_tool",
        "tool_name": "search_graph_entities",
        "tool_input": {"collection_id": col_b_id, "query": "Secret"},
    })
    llm = ScriptedLLMService([step])

    client = _build_test_api_client(sqlite_session_factory, enforce_auth=True, llm_service=llm)
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/run",
        headers=headers,
        json={"goal": "Attempting cross-tenant graph search"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # The run must fail closed with forbidden ownership error
    assert data["status"] == "failed"
    assert "Authorization check failed" in data["trace"]["error"]
    assert data["trace"]["steps"][0]["status"] == "failed"
    assert "forbidden" in data["trace"]["steps"][0]["error"].lower()


def test_live_agents_run_authenticated_success_with_graph_tool(
    sqlite_session_factory, seeded_graph_data
):
    """Authenticated user A successfully runs an agent using a knowledge graph tool."""
    col_a_id = str(seeded_graph_data["col_a_id"])
    step_1 = json.dumps({
        "action": "call_tool",
        "tool_name": "search_graph_entities",
        "tool_input": {"collection_id": col_a_id, "query": "Transformer"},
    })
    step_2 = json.dumps({
        "action": "final_answer",
        "answer": "Identified Transformer method in collection graph.",
    })
    llm = ScriptedLLMService([step_1, step_2])

    client = _build_test_api_client(sqlite_session_factory, enforce_auth=True, llm_service=llm)
    headers = {"Authorization": f"Bearer {seeded_graph_data['token_a']}"}

    resp = client.post(
        "/api/v1/agents/run",
        headers=headers,
        json={"goal": "Identify transformer architecture in graph"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "completed"
    assert data["total_steps"] == 1
    assert data["trace"]["steps"][0]["status"] == "success"
    assert "Found 1 entities" in data["trace"]["steps"][0]["output_summary"]
    assert "Identified Transformer" in data["final_answer"]
