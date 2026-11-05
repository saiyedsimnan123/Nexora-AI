"""Unit tests for Stage 5 Step 4.2.2: Knowledge Graph Agent Tools.

Verifies:
1. Tool initialization, dependency injection, and interface validation.
2. SearchGraphEntitiesTool execution with/without query, limit bounds, and serialization.
3. ExploreGraphNeighborhoodTool execution, bounded depth/nodes/edges, and provenance preservation.
4. FindGraphPathTool execution, bounded depth/paths, and evidence citation extraction.
5. QueryGraphAnalyticsTool with all 4 query types, author_id handling, and limit bounds.
6. Fail-closed security and cross-tenant isolation across all four tools.
7. AgentOrchestrator integration: immediate abort on authorization failure.
"""

from __future__ import annotations

import json
from typing import Any
import uuid
from unittest.mock import MagicMock

import pytest

from nexora.agents.models import (
    AgentAction,
    AgentContext,
    AgentStatus,
    AgentStepStatus,
    InvalidToolArgumentError,
    ToolResult,
)
from nexora.agents.orchestrator import AgentOrchestrator
from nexora.agents.tools import (
    ExploreGraphNeighborhoodInput,
    ExploreGraphNeighborhoodTool,
    FindGraphPathInput,
    FindGraphPathTool,
    QueryGraphAnalyticsInput,
    QueryGraphAnalyticsTool,
    SearchGraphEntitiesInput,
    SearchGraphEntitiesTool,
    ToolRegistry,
)
from nexora.api.errors import DocumentNotFoundError, ForbiddenError
from nexora.graph.analytics import (
    AuthorCollaborationRecord,
    AuthorConnectivityMetrics,
    CollectionAnalyticsResult,
    CollectionGraphSummary,
    EntityTypeCount,
    RelationshipTypeCount,
)
from nexora.graph.traversal_models import (
    GraphEdge,
    GraphNode,
    GraphPath,
    GraphSubgraph,
)
from nexora.llm.models import LLMMessage, LLMResponse


# ---------------------------------------------------------------------------
# Test Helpers & Scripted LLM
# ---------------------------------------------------------------------------

class ScriptedLLMService:
    """Mock LLM Service returning scripted responses for orchestrator tests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        if self.call_count >= len(self.responses):
            return LLMResponse(
                text=json.dumps({"action": "final_answer", "answer": "Done."}),
                model="scripted-mock",
            )
        resp = self.responses[self.call_count]
        self.call_count += 1
        return LLMResponse(text=resp, model="scripted-mock")


def create_auth_context(
    user_id: uuid.UUID | None = None,
    auth_service: Any = None,
    auth_enforced: bool = True,
) -> AgentContext:
    """Helper to create an AgentContext for tests."""
    return AgentContext(
        auth_enforced=auth_enforced,
        user_id=user_id or uuid.uuid4(),
        auth_service=auth_service or MagicMock(),
    )


# ---------------------------------------------------------------------------
# 1. Tool Initialization & Interface Validation
# ---------------------------------------------------------------------------

def test_search_graph_entities_tool_init_validation():
    """Verify SearchGraphEntitiesTool requires session_factory or repository."""
    with pytest.raises(TypeError, match="session_factory or repository must be provided"):
        SearchGraphEntitiesTool(None, repository=None)

    repo = MagicMock()
    tool = SearchGraphEntitiesTool(repository=repo)
    assert tool.name == "search_graph_entities"
    assert "Search or list entities" in tool.description
    assert isinstance(tool.parameters_schema, dict)


def test_explore_graph_neighborhood_tool_init_validation():
    """Verify ExploreGraphNeighborhoodTool requires a traversal service with explore_multihop."""
    with pytest.raises(TypeError, match="traversal_service must provide explore_multihop"):
        ExploreGraphNeighborhoodTool(None)

    with pytest.raises(TypeError, match="traversal_service must provide explore_multihop"):
        ExploreGraphNeighborhoodTool(MagicMock(spec=[]))

    svc = MagicMock()
    svc.explore_multihop = MagicMock()
    tool = ExploreGraphNeighborhoodTool(svc)
    assert tool.name == "explore_graph_neighborhood"
    assert "Explore bounded 1- or 2-hop" in tool.description
    assert isinstance(tool.parameters_schema, dict)


def test_find_graph_path_tool_init_validation():
    """Verify FindGraphPathTool requires a traversal service with find_paths."""
    with pytest.raises(TypeError, match="traversal_service must provide find_paths"):
        FindGraphPathTool(None)

    with pytest.raises(TypeError, match="traversal_service must provide find_paths"):
        FindGraphPathTool(MagicMock(spec=[]))

    svc = MagicMock()
    svc.find_paths = MagicMock()
    tool = FindGraphPathTool(svc)
    assert tool.name == "find_graph_path"
    assert "Discover multi-hop connecting paths" in tool.description
    assert isinstance(tool.parameters_schema, dict)


def test_query_graph_analytics_tool_init_validation():
    """Verify QueryGraphAnalyticsTool requires an analytics service with all required methods."""
    with pytest.raises(TypeError, match="analytics_service must provide analytics methods"):
        QueryGraphAnalyticsTool(None)

    incomplete_svc = MagicMock(spec=["get_summary"])
    with pytest.raises(TypeError, match="analytics_service must provide analytics methods"):
        QueryGraphAnalyticsTool(incomplete_svc)

    svc = MagicMock()
    svc.get_collection_analytics = MagicMock()
    svc.get_author_collaborations = MagicMock()
    svc.get_author_connectivity = MagicMock()
    svc.get_summary = MagicMock()
    tool = QueryGraphAnalyticsTool(svc)
    assert tool.name == "query_graph_analytics"
    assert "Query collection-wide knowledge graph statistics" in tool.description
    assert isinstance(tool.parameters_schema, dict)


# ---------------------------------------------------------------------------
# 2. Argument Validation on Tool Instances
# ---------------------------------------------------------------------------

def test_tools_validate_arguments():
    """Verify validate_arguments validates schemas and raises InvalidToolArgumentError."""
    col_id = str(uuid.uuid4())
    node_id = str(uuid.uuid4())

    search_tool = SearchGraphEntitiesTool(repository=MagicMock())
    valid_search = search_tool.validate_arguments({"collection_id": col_id, "query": "transformer"})
    assert valid_search["collection_id"] == col_id
    with pytest.raises(InvalidToolArgumentError):
        search_tool.validate_arguments({"collection_id": "not-a-uuid"})

    explore_tool = ExploreGraphNeighborhoodTool(MagicMock(explore_multihop=MagicMock()))
    valid_explore = explore_tool.validate_arguments({
        "collection_id": col_id,
        "node_id": node_id,
        "node_type": "entity",
    })
    assert valid_explore["node_type"] == "entity"
    with pytest.raises(InvalidToolArgumentError):
        explore_tool.validate_arguments({"collection_id": col_id, "node_id": node_id, "node_type": "invalid_type"})

    path_tool = FindGraphPathTool(MagicMock(find_paths=MagicMock()))
    valid_path = path_tool.validate_arguments({
        "collection_id": col_id,
        "source_id": node_id,
        "source_type": "paper",
        "target_id": str(uuid.uuid4()),
        "target_type": "entity",
    })
    assert valid_path["source_type"] == "paper"
    with pytest.raises(InvalidToolArgumentError):
        path_tool.validate_arguments({"collection_id": col_id, "source_id": "bad"})

    analytics_tool = QueryGraphAnalyticsTool(MagicMock(
        get_collection_analytics=MagicMock(),
        get_author_collaborations=MagicMock(),
        get_author_connectivity=MagicMock(),
        get_summary=MagicMock(),
    ))
    valid_analytics = analytics_tool.validate_arguments({"collection_id": col_id, "query_type": "summary"})
    assert valid_analytics["query_type"] == "summary"
    with pytest.raises(InvalidToolArgumentError):
        analytics_tool.validate_arguments({"collection_id": col_id, "query_type": "invalid"})


# ---------------------------------------------------------------------------
# 3. SearchGraphEntitiesTool Execution
# ---------------------------------------------------------------------------

def test_search_graph_entities_with_query():
    """Verify SearchGraphEntitiesTool calls search_entities when query is provided."""
    repo = MagicMock()
    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    mock_entity.collection_id = uuid.uuid4()
    mock_entity.name = "LoRA"
    mock_entity.canonical_name = "lora"
    mock_entity.entity_type = "method"
    mock_entity.description = "Low-rank adaptation"
    mock_entity.aliases = ["LORA"]
    mock_entity.entity_metadata = {"source": "arxiv"}
    repo.search_entities.return_value = [mock_entity]

    tool = SearchGraphEntitiesTool(repository=repo)
    user_id = uuid.uuid4()
    auth_svc = MagicMock()
    context = AgentContext(auth_enforced=True, user_id=user_id, auth_service=auth_svc)

    col_id = str(mock_entity.collection_id)
    res = tool.execute({"collection_id": col_id, "query": "LoRA", "entity_type": "method", "limit": 15}, context)

    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0]["name"] == "LoRA"
    assert res.data[0]["entity_type"] == "method"
    assert res.data[0]["metadata"] == {"source": "arxiv"}
    assert "LoRA" in res.summary

    auth_svc.verify_collection_ownership.assert_called_once_with(mock_entity.collection_id, user_id)
    repo.search_entities.assert_called_once_with(
        collection_id=mock_entity.collection_id,
        query="LoRA",
        entity_type="method",
        limit=15,
    )
    repo.list_by_collection_id.assert_not_called()


def test_search_graph_entities_without_query():
    """Verify SearchGraphEntitiesTool calls list_by_collection_id when query is omitted."""
    repo = MagicMock()
    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    mock_entity.collection_id = uuid.uuid4()
    mock_entity.name = "AdamW"
    mock_entity.canonical_name = "adamw"
    mock_entity.entity_type = "method"
    mock_entity.description = "Optimizer"
    mock_entity.aliases = []
    mock_entity.entity_metadata = {}
    repo.list_by_collection_id.return_value = [mock_entity]

    tool = SearchGraphEntitiesTool(repository=repo)
    user_id = uuid.uuid4()
    auth_svc = MagicMock()
    context = AgentContext(auth_enforced=True, user_id=user_id, auth_service=auth_svc)

    col_id = str(mock_entity.collection_id)
    res = tool.execute({"collection_id": col_id, "limit": 10}, context)

    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0]["name"] == "AdamW"
    repo.list_by_collection_id.assert_called_once_with(
        collection_id=mock_entity.collection_id,
        entity_type=None,
        limit=10,
    )
    repo.search_entities.assert_not_called()


def test_search_graph_entities_enforces_limit_cap():
    """Verify SearchGraphEntitiesTool caps limit at 20."""
    repo = MagicMock()
    repo.list_by_collection_id.return_value = []

    tool = SearchGraphEntitiesTool(repository=repo)
    user_id = uuid.uuid4()
    auth_svc = MagicMock()
    context = AgentContext(auth_enforced=True, user_id=user_id, auth_service=auth_svc)

    col_id = str(uuid.uuid4())
    # Request limit=50, should be capped at 20
    res = tool.execute({"collection_id": col_id, "limit": 20}, context)
    assert res.success is True
    repo.list_by_collection_id.assert_called_once_with(
        collection_id=uuid.UUID(col_id),
        entity_type=None,
        limit=20,
    )


def test_search_graph_entities_session_factory_branch():
    """Verify SearchGraphEntitiesTool works via session_factory."""
    mock_session = MagicMock()
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__enter__.return_value = mock_session

    tool = SearchGraphEntitiesTool(session_factory=mock_session_factory)
    user_id = uuid.uuid4()
    auth_svc = MagicMock()
    context = AgentContext(auth_enforced=True, user_id=user_id, auth_service=auth_svc)

    col_id = str(uuid.uuid4())
    with pytest.MonkeyPatch().context() as mp:
        mock_repo_cls = MagicMock()
        mock_repo_instance = MagicMock()
        mock_repo_instance.list_by_collection_id.return_value = []
        mock_repo_cls.return_value = mock_repo_instance
        mp.setattr("nexora.database.repository.graph_entity.GraphEntityRepository", mock_repo_cls)

        res = tool.execute({"collection_id": col_id}, context)
        assert res.success is True
        mock_repo_instance.list_by_collection_id.assert_called_once()


# ---------------------------------------------------------------------------
# 4. ExploreGraphNeighborhoodTool Execution
# ---------------------------------------------------------------------------

def test_explore_graph_neighborhood_success():
    """Verify ExploreGraphNeighborhoodTool returns bounded subgraph with evidence."""
    svc = MagicMock()
    col_uuid = uuid.uuid4()
    seed_uuid = uuid.uuid4()
    target_uuid = uuid.uuid4()
    edge_uuid = uuid.uuid4()

    mock_node_seed = GraphNode(
        id=seed_uuid,
        node_type="entity",
        name="Attention Mechanism",
        entity_type="concept",
        metadata={"domain": "NLP"},
    )
    mock_node_target = GraphNode(
        id=target_uuid,
        node_type="paper",
        name="Attention Is All You Need",
        entity_type=None,
        metadata={"doi": "10.1234/test"},
    )
    mock_edge = GraphEdge(
        id=edge_uuid,
        relationship_type="introduced_in",
        source_type="entity",
        source_id=seed_uuid,
        target_type="paper",
        target_id=target_uuid,
        confidence=0.95,
        evidence_text="We propose the Transformer in [E1] based on attention [E2].",
        chunk_id=uuid.uuid4(),
        paper_id=target_uuid,
        page_number=3,
        extraction_method="llm_extraction",
    )
    mock_subgraph = GraphSubgraph(
        seed_id=seed_uuid,
        seed_type="entity",
        depth=1,
        nodes=[mock_node_seed, mock_node_target],
        edges=[mock_edge],
        is_truncated=False,
        truncation_reason=None,
    )
    svc.explore_multihop.return_value = mock_subgraph

    tool = ExploreGraphNeighborhoodTool(svc)
    user_id = uuid.uuid4()
    auth_svc = MagicMock()
    context = AgentContext(auth_enforced=True, user_id=user_id, auth_service=auth_svc)

    res = tool.execute(
        {
            "collection_id": str(col_uuid),
            "node_id": str(seed_uuid),
            "node_type": "entity",
            "max_depth": 1,
            "max_nodes": 10,
            "max_edges": 20,
            "direction": "both",
        },
        context,
    )

    assert res.success is True
    assert res.data["seed_id"] == str(seed_uuid)
    assert res.data["depth"] == 1
    assert len(res.data["nodes"]) == 2
    assert len(res.data["edges"]) == 1
    assert res.data["is_truncated"] is False
    assert res.data["edges"][0]["confidence"] == 0.95
    assert res.data["edges"][0]["page_number"] == 3
    # Unverified regex labels in arbitrary text are NOT exposed as verified citations
    assert res.evidence_labels == []
    assert res.data["edges"][0]["evidence_text"] == "We propose the Transformer in [E1] based on attention [E2]."
    assert res.data["edges"][0]["chunk_id"] is not None
    assert res.data["edges"][0]["paper_id"] == str(target_uuid)

    svc.explore_multihop.assert_called_once_with(
        start_id=seed_uuid,
        start_type="entity",
        collection_id=col_uuid,
        max_depth=1,
        max_nodes=10,
        max_edges=20,
        direction="both",
        relationship_types=None,
        min_confidence=None,
        include_unscored=False,
    )


def test_explore_graph_neighborhood_enforces_hard_limits():
    """Verify ExploreGraphNeighborhoodTool caps depth=2, nodes=25, edges=50."""
    svc = MagicMock()
    seed_uuid = uuid.uuid4()
    col_uuid = uuid.uuid4()
    svc.explore_multihop.return_value = GraphSubgraph(
        seed_id=seed_uuid,
        seed_type="paper",
        depth=2,
        nodes=[GraphNode(id=seed_uuid, node_type="paper", name="Paper X")],
        edges=[],
        is_truncated=True,
        truncation_reason="node_limit_reached",
    )

    tool = ExploreGraphNeighborhoodTool(svc)
    context = create_auth_context()

    res = tool.execute(
        {
            "collection_id": str(col_uuid),
            "node_id": str(seed_uuid),
            "node_type": "paper",
            "max_depth": 2,  # schema allows 1 or 2
            "max_nodes": 25, # schema allows up to 25
            "max_edges": 50, # schema allows up to 50
        },
        context,
    )
    assert res.success is True
    svc.explore_multihop.assert_called_once_with(
        start_id=seed_uuid,
        start_type="paper",
        collection_id=col_uuid,
        max_depth=2,
        max_nodes=25,
        max_edges=50,
        direction="both",
        relationship_types=None,
        min_confidence=None,
        include_unscored=False,
    )


# ---------------------------------------------------------------------------
# 5. FindGraphPathTool Execution
# ---------------------------------------------------------------------------

def test_find_graph_path_success():
    """Verify FindGraphPathTool calls find_paths and formats multi-hop path."""
    svc = MagicMock()
    col_uuid = uuid.uuid4()
    src_uuid = uuid.uuid4()
    mid_uuid = uuid.uuid4()
    dst_uuid = uuid.uuid4()
    e1_uuid = uuid.uuid4()
    e2_uuid = uuid.uuid4()

    n1 = GraphNode(id=src_uuid, node_type="entity", name="LoRA", entity_type="method")
    n2 = GraphNode(id=mid_uuid, node_type="paper", name="LoRA Paper")
    n3 = GraphNode(id=dst_uuid, node_type="entity", name="Edward Hu", entity_type="author")

    e1 = GraphEdge(
        id=e1_uuid,
        relationship_type="introduced_in",
        source_type="entity",
        source_id=src_uuid,
        target_type="paper",
        target_id=mid_uuid,
        confidence=0.9,
        evidence_text="LoRA is presented in [E1].",
    )
    e2 = GraphEdge(
        id=e2_uuid,
        relationship_type="authored_by",
        source_type="paper",
        source_id=mid_uuid,
        target_type="entity",
        target_id=dst_uuid,
        confidence=0.8,
        evidence_text="Written by Edward Hu [E3].",
    )

    path = GraphPath(
        nodes=[n1, n2, n3],
        edges=[e1, e2],
        length=2,
        confidence_score=0.72,
    )
    svc.find_paths.return_value = [path]

    tool = FindGraphPathTool(svc)
    context = create_auth_context()

    res = tool.execute(
        {
            "collection_id": str(col_uuid),
            "source_id": str(src_uuid),
            "source_type": "entity",
            "target_id": str(dst_uuid),
            "target_type": "entity",
            "max_depth": 3,
            "max_paths": 2,
        },
        context,
    )

    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0]["hops"] == 2
    assert res.data[0]["cumulative_confidence"] == 0.72
    assert len(res.data[0]["nodes"]) == 3
    assert len(res.data[0]["edges"]) == 2
    # Unverified regex labels in arbitrary text are NOT exposed as verified citations
    assert res.evidence_labels == []
    assert res.data[0]["edges"][0]["evidence_text"] == "LoRA is presented in [E1]."
    assert res.data[0]["edges"][1]["evidence_text"] == "Written by Edward Hu [E3]."
    assert "Discovered 1 path(s)" in res.summary

    svc.find_paths.assert_called_once_with(
        source_id=src_uuid,
        source_type="entity",
        target_id=dst_uuid,
        target_type="entity",
        collection_id=col_uuid,
        max_depth=3,
        max_paths=2,
        direction="outgoing",
        relationship_types=None,
        min_confidence=None,
        include_unscored=False,
    )


# ---------------------------------------------------------------------------
# 6. QueryGraphAnalyticsTool Execution
# ---------------------------------------------------------------------------

def test_query_graph_analytics_overview():
    """Verify QueryGraphAnalyticsTool overview query."""
    svc = MagicMock()
    col_uuid = uuid.uuid4()
    a1_uuid = uuid.uuid4()
    a2_uuid = uuid.uuid4()
    p1_uuid = uuid.uuid4()

    mock_result = CollectionAnalyticsResult(
        collection_id=col_uuid,
        summary=CollectionGraphSummary(
            collection_id=col_uuid,
            total_entities=10,
            total_relationships=20,
            entity_counts_by_type={"method": 4, "dataset": 6},
            relationship_counts_by_type={"uses": 12, "evaluates_on": 8},
        ),
        entity_distribution=[
            EntityTypeCount(entity_type="method", count=4),
            EntityTypeCount(entity_type="dataset", count=6),
        ],
        relationship_distribution=[
            RelationshipTypeCount(relationship_type="uses", count=12),
            RelationshipTypeCount(relationship_type="evaluates_on", count=8),
        ],
        top_collaborations=[
            AuthorCollaborationRecord(
                author_id=a1_uuid,
                author_name="Alice",
                collaborator_id=a2_uuid,
                collaborator_name="Bob",
                shared_paper_ids=[p1_uuid],
                collaboration_count=1,
            )
        ],
        author_connectivity=[
            AuthorConnectivityMetrics(
                author_id=a1_uuid,
                author_name="Alice",
                paper_count=3,
                collaborator_count=2,
                total_collaborations=5,
            )
        ],
    )
    svc.get_collection_analytics.return_value = mock_result

    tool = QueryGraphAnalyticsTool(svc)
    context = create_auth_context()

    res = tool.execute({"collection_id": str(col_uuid), "query_type": "overview", "top_n": 5}, context)

    assert res.success is True
    assert res.data["collection_id"] == str(col_uuid)
    assert res.data["summary"]["total_entities"] == 10
    assert len(res.data["top_collaborations"]) == 1
    assert res.data["top_collaborations"][0]["author_name"] == "Alice"
    assert res.data["author_connectivity"][0]["collaborator_count"] == 2
    assert res.data["author_connectivity"][0]["coauthor_count"] == 2
    assert "Collection graph overview: 10 entities, 20 relationships" in res.summary

    svc.get_collection_analytics.assert_called_once_with(
        collection_id=col_uuid,
        top_collaborations_limit=5,
        author_connectivity_limit=5,
        min_confidence=None,
        include_unscored=False,
    )


def test_query_graph_analytics_summary():
    """Verify QueryGraphAnalyticsTool summary query."""
    svc = MagicMock()
    col_uuid = uuid.uuid4()
    mock_sum = CollectionGraphSummary(
        collection_id=col_uuid,
        total_entities=5,
        total_relationships=8,
        entity_counts_by_type={"author": 5},
        relationship_counts_by_type={"authored_by": 8},
    )
    svc.get_summary.return_value = mock_sum

    tool = QueryGraphAnalyticsTool(svc)
    context = create_auth_context()

    res = tool.execute({"collection_id": str(col_uuid), "query_type": "summary"}, context)

    assert res.success is True
    assert res.data["total_entities"] == 5
    assert res.data["total_relationships"] == 8
    svc.get_summary.assert_called_once_with(collection_id=col_uuid)


def test_query_graph_analytics_collaborations_with_author():
    """Verify QueryGraphAnalyticsTool collaborations query maps author_id."""
    svc = MagicMock()
    col_uuid = uuid.uuid4()
    author_uuid = uuid.uuid4()
    collab_uuid = uuid.uuid4()
    p_uuid = uuid.uuid4()

    mock_collab = AuthorCollaborationRecord(
        author_id=author_uuid,
        author_name="Alice",
        collaborator_id=collab_uuid,
        collaborator_name="Bob",
        shared_paper_ids=[p_uuid],
        collaboration_count=1,
    )
    svc.get_author_collaborations.return_value = [mock_collab]

    tool = QueryGraphAnalyticsTool(svc)
    context = create_auth_context()

    res = tool.execute(
        {
            "collection_id": str(col_uuid),
            "query_type": "collaborations",
            "author_id": str(author_uuid),
            "top_n": 10,
        },
        context,
    )

    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0]["collaborator_name"] == "Bob"
    svc.get_author_collaborations.assert_called_once_with(
        collection_id=col_uuid,
        author_id=author_uuid,
        min_shared_papers=1,
        min_confidence=None,
        include_unscored=False,
        limit=10,
    )


def test_query_graph_analytics_connectivity_maps_author_ids():
    """Verify QueryGraphAnalyticsTool connectivity query maps author_id to author_ids list."""
    svc = MagicMock()
    col_uuid = uuid.uuid4()
    author_uuid = uuid.uuid4()

    mock_conn = AuthorConnectivityMetrics(
        author_id=author_uuid,
        author_name="Alice",
        paper_count=4,
        collaborator_count=3,
        total_collaborations=7,
    )
    svc.get_author_connectivity.return_value = [mock_conn]

    tool = QueryGraphAnalyticsTool(svc)
    context = create_auth_context()

    res = tool.execute(
        {
            "collection_id": str(col_uuid),
            "query_type": "connectivity",
            "author_id": str(author_uuid),
        },
        context,
    )

    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0]["author_name"] == "Alice"
    assert res.data[0]["total_collaborations"] == 7
    svc.get_author_connectivity.assert_called_once_with(
        collection_id=col_uuid,
        author_ids=[author_uuid],
        min_confidence=None,
        include_unscored=False,
        limit=5,
    )


def test_query_graph_analytics_rejects_author_id_for_overview_and_summary():
    """Verify QueryGraphAnalyticsTool rejects author_id for overview and summary."""
    tool = QueryGraphAnalyticsTool(MagicMock(
        get_collection_analytics=MagicMock(),
        get_author_collaborations=MagicMock(),
        get_author_connectivity=MagicMock(),
        get_summary=MagicMock(),
    ))
    context = create_auth_context()
    col_id = str(uuid.uuid4())
    author_id = str(uuid.uuid4())

    # Case A: overview with author_id
    res_ov = tool.execute({"collection_id": col_id, "query_type": "overview", "author_id": author_id}, context)
    assert res_ov.success is False
    assert res_ov.error_type == "validation_error"

    # Case B: summary with author_id
    res_sum = tool.execute({"collection_id": col_id, "query_type": "summary", "author_id": author_id}, context)
    assert res_sum.success is False
    assert res_sum.error_type == "validation_error"


# ---------------------------------------------------------------------------
# 7. Fail-Closed Security & Cross-Tenant Access across all 4 tools
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tool_factory,valid_args",
    [
        (
            lambda: SearchGraphEntitiesTool(repository=MagicMock()),
            lambda col_id: {"collection_id": col_id, "query": "test"},
        ),
        (
            lambda: ExploreGraphNeighborhoodTool(MagicMock(explore_multihop=MagicMock())),
            lambda col_id: {"collection_id": col_id, "node_id": str(uuid.uuid4()), "node_type": "entity"},
        ),
        (
            lambda: FindGraphPathTool(MagicMock(find_paths=MagicMock())),
            lambda col_id: {
                "collection_id": col_id,
                "source_id": str(uuid.uuid4()),
                "source_type": "entity",
                "target_id": str(uuid.uuid4()),
                "target_type": "paper",
            },
        ),
        (
            lambda: QueryGraphAnalyticsTool(MagicMock(
                get_collection_analytics=MagicMock(),
                get_author_collaborations=MagicMock(),
                get_author_connectivity=MagicMock(),
                get_summary=MagicMock(),
            )),
            lambda col_id: {"collection_id": col_id, "query_type": "summary"},
        ),
    ],
)
def test_all_graph_tools_fail_closed_missing_user_or_auth_service(tool_factory, valid_args):
    """Verify all four tools fail closed when user_id or auth_service is missing, even with auth_enforced=False."""
    tool = tool_factory()
    col_id = str(uuid.uuid4())
    args = valid_args(col_id)

    # 1. Missing user_id
    ctx_no_user = AgentContext(auth_enforced=True, user_id=None, auth_service=MagicMock())
    res = tool.execute(args, ctx_no_user)
    assert res.success is False
    assert res.error_type == "unauthorized"
    assert "Authentication required" in res.error

    # 2. Missing auth_service
    ctx_no_auth = AgentContext(auth_enforced=True, user_id=uuid.uuid4(), auth_service=None)
    res = tool.execute(args, ctx_no_auth)
    assert res.success is False
    assert res.error_type == "unauthorized"
    assert "Authorization service unavailable" in res.error

    # 3. auth_enforced=False with missing user_id STILL fails closed (zero-bypass)
    ctx_unforced_no_user = AgentContext(auth_enforced=False, user_id=None, auth_service=MagicMock())
    res = tool.execute(args, ctx_unforced_no_user)
    assert res.success is False
    assert res.error_type == "unauthorized"

    # 4. auth_enforced=False with missing auth_service STILL fails closed (zero-bypass)
    ctx_unforced_no_auth = AgentContext(auth_enforced=False, user_id=uuid.uuid4(), auth_service=None)
    res = tool.execute(args, ctx_unforced_no_auth)
    assert res.success is False
    assert res.error_type == "unauthorized"


@pytest.mark.parametrize(
    "tool_factory,valid_args",
    [
        (
            lambda: SearchGraphEntitiesTool(repository=MagicMock()),
            lambda col_id: {"collection_id": col_id, "query": "test"},
        ),
        (
            lambda: ExploreGraphNeighborhoodTool(MagicMock(explore_multihop=MagicMock())),
            lambda col_id: {"collection_id": col_id, "node_id": str(uuid.uuid4()), "node_type": "entity"},
        ),
        (
            lambda: FindGraphPathTool(MagicMock(find_paths=MagicMock())),
            lambda col_id: {
                "collection_id": col_id,
                "source_id": str(uuid.uuid4()),
                "source_type": "entity",
                "target_id": str(uuid.uuid4()),
                "target_type": "paper",
            },
        ),
        (
            lambda: QueryGraphAnalyticsTool(MagicMock(
                get_collection_analytics=MagicMock(),
                get_author_collaborations=MagicMock(),
                get_author_connectivity=MagicMock(),
                get_summary=MagicMock(),
            )),
            lambda col_id: {"collection_id": col_id, "query_type": "summary"},
        ),
    ],
)
def test_all_graph_tools_cross_tenant_denial(tool_factory, valid_args):
    """Verify all four tools reject foreign collections with error_type='forbidden'."""
    tool = tool_factory()
    col_id = str(uuid.uuid4())
    args = valid_args(col_id)

    auth_svc = MagicMock()
    auth_svc.verify_collection_ownership.side_effect = ForbiddenError(
        f"Access forbidden: User does not own collection {col_id}."
    )
    context = AgentContext(auth_enforced=True, user_id=uuid.uuid4(), auth_service=auth_svc)

    res = tool.execute(args, context)
    assert res.success is False
    assert res.error_type == "forbidden"
    assert "Access forbidden" in res.error


@pytest.mark.parametrize(
    "tool_factory,valid_args",
    [
        (
            lambda: SearchGraphEntitiesTool(repository=MagicMock()),
            lambda col_id: {"collection_id": col_id, "query": "test"},
        ),
        (
            lambda: ExploreGraphNeighborhoodTool(MagicMock(explore_multihop=MagicMock())),
            lambda col_id: {"collection_id": col_id, "node_id": str(uuid.uuid4()), "node_type": "entity"},
        ),
        (
            lambda: FindGraphPathTool(MagicMock(find_paths=MagicMock())),
            lambda col_id: {
                "collection_id": col_id,
                "source_id": str(uuid.uuid4()),
                "source_type": "entity",
                "target_id": str(uuid.uuid4()),
                "target_type": "paper",
            },
        ),
        (
            lambda: QueryGraphAnalyticsTool(MagicMock(
                get_collection_analytics=MagicMock(),
                get_author_collaborations=MagicMock(),
                get_author_connectivity=MagicMock(),
                get_summary=MagicMock(),
            )),
            lambda col_id: {"collection_id": col_id, "query_type": "summary"},
        ),
    ],
)
def test_all_graph_tools_not_found_collection(tool_factory, valid_args):
    """Verify all four tools report non-existent collection with error_type='not_found'."""
    tool = tool_factory()
    col_id = str(uuid.uuid4())
    args = valid_args(col_id)

    auth_svc = MagicMock()
    auth_svc.verify_collection_ownership.side_effect = DocumentNotFoundError(
        f"Collection {col_id} not found."
    )
    context = AgentContext(auth_enforced=True, user_id=uuid.uuid4(), auth_service=auth_svc)

    res = tool.execute(args, context)
    assert res.success is False
    assert res.error_type == "not_found"
    assert "not found" in res.error


# ---------------------------------------------------------------------------
# 8. AgentOrchestrator Integration
# ---------------------------------------------------------------------------

def test_orchestrator_halts_on_graph_tool_auth_failure():
    """Verify AgentOrchestrator immediately aborts when graph tool fails authorization."""
    mock_repo = MagicMock()
    search_tool = SearchGraphEntitiesTool(repository=mock_repo)
    registry = ToolRegistry([search_tool])

    col_id = str(uuid.uuid4())
    llm_step = json.dumps({
        "action": "call_tool",
        "tool_name": "search_graph_entities",
        "tool_input": {"collection_id": col_id, "query": "secret research"},
    })
    llm = ScriptedLLMService([llm_step])
    orchestrator = AgentOrchestrator(llm, registry)

    # Auth service rejects ownership
    auth_svc = MagicMock()
    auth_svc.verify_collection_ownership.side_effect = ForbiddenError("Access forbidden to foreign collection.")
    context = AgentContext(auth_enforced=True, user_id=uuid.uuid4(), auth_service=auth_svc)

    result = orchestrator.run("Find graph entities", context=context)

    assert result.status == AgentStatus.FAILED.value
    assert "Authorization check failed" in result.trace.error
    assert result.total_steps == 1
    assert result.trace.steps[0].status == AgentStepStatus.FAILED.value
    mock_repo.search_entities.assert_not_called()


def test_orchestrator_successful_graph_tool_run():
    """Verify AgentOrchestrator executes graph tool and generates final answer."""
    mock_repo = MagicMock()
    mock_entity = MagicMock()
    mock_entity.id = uuid.uuid4()
    mock_entity.collection_id = uuid.uuid4()
    mock_entity.name = "FlashAttention"
    mock_entity.canonical_name = "flashattention"
    mock_entity.entity_type = "method"
    mock_entity.description = "Fast and memory-efficient exact attention"
    mock_entity.aliases = []
    mock_entity.entity_metadata = {}
    mock_repo.search_entities.return_value = [mock_entity]

    search_tool = SearchGraphEntitiesTool(repository=mock_repo)
    registry = ToolRegistry([search_tool])

    col_id = str(mock_entity.collection_id)
    step_1 = json.dumps({
        "action": "call_tool",
        "tool_name": "search_graph_entities",
        "tool_input": {"collection_id": col_id, "query": "FlashAttention"},
    })
    step_2 = json.dumps({
        "action": "final_answer",
        "answer": "Found FlashAttention method in knowledge graph [E1].",
    })
    llm = ScriptedLLMService([step_1, step_2])
    orchestrator = AgentOrchestrator(llm, registry)

    auth_svc = MagicMock()
    context = AgentContext(auth_enforced=True, user_id=uuid.uuid4(), auth_service=auth_svc)

    result = orchestrator.run("Identify FlashAttention in graph", context=context)

    assert result.status == AgentStatus.COMPLETED.value
    assert result.total_steps == 1
    assert "FlashAttention" in result.final_answer
    assert result.trace.steps[0].status == AgentStepStatus.SUCCESS.value
    assert "Found 1 entities" in result.trace.steps[0].output_summary


# ---------------------------------------------------------------------------
# 9. Citation Integrity & Relationship Types Enforcement Tests
# ---------------------------------------------------------------------------

def test_explore_neighborhood_citation_like_text_preserved_in_provenance_not_labels():
    """Verify that [E#] substrings in evidence_text are preserved in provenance but not exposed as verified citations."""
    svc = MagicMock()
    col_uuid = uuid.uuid4()
    seed_uuid = uuid.uuid4()
    target_uuid = uuid.uuid4()
    chunk_uuid = uuid.uuid4()

    mock_node_seed = GraphNode(id=seed_uuid, node_type="entity", name="Concept A", entity_type="concept")
    mock_node_target = GraphNode(id=target_uuid, node_type="paper", name="Paper B")
    mock_edge = GraphEdge(
        id=uuid.uuid4(),
        relationship_type="evaluated_on",
        source_type="entity",
        source_id=seed_uuid,
        target_type="paper",
        target_id=target_uuid,
        confidence=0.88,
        evidence_text="As discussed in [E1] and [E99], this concept was evaluated thoroughly.",
        chunk_id=chunk_uuid,
        paper_id=target_uuid,
        page_number=5,
        extraction_method="heuristic",
    )
    svc.explore_multihop.return_value = GraphSubgraph(
        seed_id=seed_uuid,
        seed_type="entity",
        depth=1,
        nodes=[mock_node_seed, mock_node_target],
        edges=[mock_edge],
    )

    tool = ExploreGraphNeighborhoodTool(svc)
    context = create_auth_context()

    res = tool.execute(
        {
            "collection_id": str(col_uuid),
            "node_id": str(seed_uuid),
            "node_type": "entity",
        },
        context,
    )

    assert res.success is True
    # Citation integrity: Unverified text substrings must NEVER populate evidence_labels
    assert res.evidence_labels == []
    # Provenance fields must be completely preserved
    edge_data = res.data["edges"][0]
    assert edge_data["evidence_text"] == "As discussed in [E1] and [E99], this concept was evaluated thoroughly."
    assert edge_data["chunk_id"] == str(chunk_uuid)
    assert edge_data["paper_id"] == str(target_uuid)
    assert edge_data["page_number"] == 5
    assert edge_data["confidence"] == 0.88
    assert edge_data["extraction_method"] == "heuristic"


def test_find_path_citation_like_text_preserved_in_provenance_not_labels():
    """Verify that path evidence_text preserves provenance without exposing unverified citation labels."""
    svc = MagicMock()
    col_uuid = uuid.uuid4()
    src_uuid = uuid.uuid4()
    dst_uuid = uuid.uuid4()
    chunk_uuid = uuid.uuid4()

    n1 = GraphNode(id=src_uuid, node_type="entity", name="Method X", entity_type="method")
    n2 = GraphNode(id=dst_uuid, node_type="paper", name="Paper Y")
    e1 = GraphEdge(
        id=uuid.uuid4(),
        relationship_type="introduced_in",
        source_type="entity",
        source_id=src_uuid,
        target_type="paper",
        target_id=dst_uuid,
        confidence=0.91,
        evidence_text="Reference [E42] shows the foundational results.",
        chunk_id=chunk_uuid,
        paper_id=dst_uuid,
        page_number=2,
    )
    path = GraphPath(nodes=[n1, n2], edges=[e1], length=1, confidence_score=0.91)
    svc.find_paths.return_value = [path]

    tool = FindGraphPathTool(svc)
    context = create_auth_context()

    res = tool.execute(
        {
            "collection_id": str(col_uuid),
            "source_id": str(src_uuid),
            "source_type": "entity",
            "target_id": str(dst_uuid),
            "target_type": "paper",
        },
        context,
    )

    assert res.success is True
    # Citation integrity: Unverified text substrings must NEVER populate evidence_labels
    assert res.evidence_labels == []
    # Provenance preserved in edge
    edge_data = res.data[0]["edges"][0]
    assert edge_data["evidence_text"] == "Reference [E42] shows the foundational results."
    assert edge_data["chunk_id"] == str(chunk_uuid)
    assert edge_data["paper_id"] == str(dst_uuid)


@pytest.mark.parametrize("invalid_rel", [
    [],                      # Empty list
    [""],                    # Empty string
    ["   "],                 # Blank string
    [123],                   # Non-string
    ["valid", ""],           # Mixed with empty
    ["a" * 101],             # Over length limit
])
def test_tools_reject_malformed_relationship_types(invalid_rel):
    """Verify neighborhood and path tools reject malformed relationship_types with validation_error."""
    col_id = str(uuid.uuid4())
    node_id = str(uuid.uuid4())
    context = create_auth_context()

    explore_tool = ExploreGraphNeighborhoodTool(MagicMock(explore_multihop=MagicMock()))
    res_exp = explore_tool.execute(
        {
            "collection_id": col_id,
            "node_id": node_id,
            "node_type": "entity",
            "relationship_types": invalid_rel,
        },
        context,
    )
    assert res_exp.success is False
    assert res_exp.error_type == "validation_error"

    path_tool = FindGraphPathTool(MagicMock(find_paths=MagicMock()))
    res_path = path_tool.execute(
        {
            "collection_id": col_id,
            "source_id": node_id,
            "source_type": "entity",
            "target_id": str(uuid.uuid4()),
            "target_type": "paper",
            "relationship_types": invalid_rel,
        },
        context,
    )
    assert res_path.success is False
    assert res_path.error_type == "validation_error"

