"""Tests for Stage 5 Step 4.2.1 Knowledge Graph Agent Schemas & Fail-Closed Authorization."""

from __future__ import annotations

from uuid import UUID, uuid4
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nexora.agents.models import AgentContext, AgentSecurityError, InvalidToolArgumentError
from nexora.agents.tools import (
    ExploreGraphNeighborhoodInput,
    FindGraphPathInput,
    QueryGraphAnalyticsInput,
    SearchGraphEntitiesInput,
    _verify_graph_collection_access,
)
from nexora.api.errors import DocumentNotFoundError, ForbiddenError


# ===========================================================================
# 1. SearchGraphEntitiesInput Tests
# ===========================================================================

def test_search_graph_entities_input_valid():
    col_id = str(uuid4())
    data = SearchGraphEntitiesInput(
        collection_id=col_id,
        query="Transformer",
        entity_type="method",
        limit=15,
    )
    assert data.collection_id == col_id
    assert data.query == "Transformer"
    assert data.entity_type == "method"
    assert data.limit == 15


def test_search_graph_entities_input_defaults():
    col_id = str(uuid4())
    data = SearchGraphEntitiesInput(collection_id=col_id)
    assert data.collection_id == col_id
    assert data.query is None
    assert data.entity_type is None
    assert data.limit == 10


def test_search_graph_entities_input_invalid_uuid():
    with pytest.raises(ValidationError) as exc:
        SearchGraphEntitiesInput(collection_id="invalid-uuid")
    assert "collection_id" in str(exc.value)


def test_search_graph_entities_input_extra_fields_forbidden():
    col_id = str(uuid4())
    with pytest.raises(ValidationError) as exc:
        SearchGraphEntitiesInput(collection_id=col_id, unexpected_field="malicious")
    assert "extra_forbidden" in str(exc.value)


def test_search_graph_entities_input_limit_bounds():
    col_id = str(uuid4())
    # Upper bound max 20
    with pytest.raises(ValidationError) as exc:
        SearchGraphEntitiesInput(collection_id=col_id, limit=21)
    assert "limit" in str(exc.value)

    # Lower bound min 1
    with pytest.raises(ValidationError) as exc:
        SearchGraphEntitiesInput(collection_id=col_id, limit=0)
    assert "limit" in str(exc.value)


def test_search_graph_entities_input_entity_type_validation():
    col_id = str(uuid4())
    # Valid canonical types
    for et in ("author", "concept", "dataset", "method", "metric", "other"):
        obj = SearchGraphEntitiesInput(collection_id=col_id, entity_type=et)
        assert obj.entity_type == et

    # Invalid type rejected
    with pytest.raises(ValidationError) as exc:
        SearchGraphEntitiesInput(collection_id=col_id, entity_type="invalid_type")
    assert "Invalid entity_type" in str(exc.value)


# ===========================================================================
# 2. ExploreGraphNeighborhoodInput Tests
# ===========================================================================

def test_explore_graph_neighborhood_input_valid():
    col_id = str(uuid4())
    node_id = str(uuid4())
    data = ExploreGraphNeighborhoodInput(
        collection_id=col_id,
        node_id=node_id,
        node_type="entity",
        max_depth=2,
        max_nodes=25,
        max_edges=50,
        direction="outgoing",
        relationship_types=["based_on", "uses_mechanism"],
        min_confidence=0.85,
    )
    assert data.collection_id == col_id
    assert data.node_id == node_id
    assert data.node_type == "entity"
    assert data.max_depth == 2
    assert data.max_nodes == 25
    assert data.max_edges == 50
    assert data.direction == "outgoing"
    assert data.relationship_types == ["based_on", "uses_mechanism"]
    assert data.min_confidence == 0.85


def test_explore_graph_neighborhood_input_defaults():
    col_id = str(uuid4())
    node_id = str(uuid4())
    data = ExploreGraphNeighborhoodInput(
        collection_id=col_id,
        node_id=node_id,
        node_type="paper",
    )
    assert data.max_depth == 1
    assert data.max_nodes == 15
    assert data.max_edges == 30
    assert data.direction == "both"
    assert data.relationship_types is None
    assert data.min_confidence is None


def test_explore_graph_neighborhood_input_limits_enforced():
    col_id = str(uuid4())
    node_id = str(uuid4())

    # Depth > 2 rejected
    with pytest.raises(ValidationError) as exc:
        ExploreGraphNeighborhoodInput(collection_id=col_id, node_id=node_id, node_type="entity", max_depth=3)
    assert "max_depth" in str(exc.value)

    # Nodes > 25 rejected
    with pytest.raises(ValidationError) as exc:
        ExploreGraphNeighborhoodInput(collection_id=col_id, node_id=node_id, node_type="entity", max_nodes=26)
    assert "max_nodes" in str(exc.value)

    # Edges > 50 rejected
    with pytest.raises(ValidationError) as exc:
        ExploreGraphNeighborhoodInput(collection_id=col_id, node_id=node_id, node_type="entity", max_edges=51)
    assert "max_edges" in str(exc.value)


def test_explore_graph_neighborhood_input_node_type_and_direction():
    col_id = str(uuid4())
    node_id = str(uuid4())

    # Invalid node_type
    with pytest.raises(ValidationError) as exc:
        ExploreGraphNeighborhoodInput(collection_id=col_id, node_id=node_id, node_type="invalid_type")
    assert "node_type" in str(exc.value)

    # Invalid direction
    with pytest.raises(ValidationError) as exc:
        ExploreGraphNeighborhoodInput(collection_id=col_id, node_id=node_id, node_type="entity", direction="sideways")
    assert "direction" in str(exc.value)


def test_explore_graph_neighborhood_input_extra_fields_forbidden():
    col_id = str(uuid4())
    node_id = str(uuid4())
    with pytest.raises(ValidationError) as exc:
        ExploreGraphNeighborhoodInput(collection_id=col_id, node_id=node_id, node_type="entity", extra="forbidden")
    assert "extra_forbidden" in str(exc.value)


# ===========================================================================
# 3. FindGraphPathInput Tests
# ===========================================================================

def test_find_graph_path_input_valid():
    col_id = str(uuid4())
    src_id = str(uuid4())
    tgt_id = str(uuid4())
    data = FindGraphPathInput(
        collection_id=col_id,
        source_id=src_id,
        source_type="entity",
        target_id=tgt_id,
        target_type="paper",
        max_depth=3,
        max_paths=3,
        direction="both",
        min_confidence=0.7,
    )
    assert data.collection_id == col_id
    assert data.source_id == src_id
    assert data.source_type == "entity"
    assert data.target_id == tgt_id
    assert data.target_type == "paper"
    assert data.max_depth == 3
    assert data.max_paths == 3
    assert data.direction == "both"
    assert data.min_confidence == 0.7


def test_find_graph_path_input_defaults():
    col_id = str(uuid4())
    src_id = str(uuid4())
    tgt_id = str(uuid4())
    data = FindGraphPathInput(
        collection_id=col_id,
        source_id=src_id,
        source_type="entity",
        target_id=tgt_id,
        target_type="entity",
    )
    assert data.max_depth == 2
    assert data.max_paths == 2
    assert data.direction == "outgoing"
    assert data.min_confidence is None


def test_find_graph_path_input_limits_enforced():
    col_id = str(uuid4())
    src_id = str(uuid4())
    tgt_id = str(uuid4())

    # Depth > 3 rejected
    with pytest.raises(ValidationError) as exc:
        FindGraphPathInput(
            collection_id=col_id,
            source_id=src_id,
            source_type="entity",
            target_id=tgt_id,
            target_type="entity",
            max_depth=4,
        )
    assert "max_depth" in str(exc.value)

    # Paths > 3 rejected
    with pytest.raises(ValidationError) as exc:
        FindGraphPathInput(
            collection_id=col_id,
            source_id=src_id,
            source_type="entity",
            target_id=tgt_id,
            target_type="entity",
            max_paths=4,
        )
    assert "max_paths" in str(exc.value)


def test_find_graph_path_input_extra_fields_forbidden():
    col_id = str(uuid4())
    src_id = str(uuid4())
    tgt_id = str(uuid4())
    with pytest.raises(ValidationError) as exc:
        FindGraphPathInput(
            collection_id=col_id,
            source_id=src_id,
            source_type="entity",
            target_id=tgt_id,
            target_type="entity",
            bogus="extra",
        )
    assert "extra_forbidden" in str(exc.value)


# ===========================================================================
# 4. QueryGraphAnalyticsInput Tests
# ===========================================================================

def test_query_graph_analytics_input_valid():
    col_id = str(uuid4())
    author_id = str(uuid4())
    data = QueryGraphAnalyticsInput(
        collection_id=col_id,
        query_type="collaborations",
        top_n=15,
        min_confidence=0.9,
        include_unscored=True,
        author_id=author_id,
    )
    assert data.collection_id == col_id
    assert data.query_type == "collaborations"
    assert data.top_n == 15
    assert data.min_confidence == 0.9
    assert data.include_unscored is True
    assert data.author_id == author_id


def test_query_graph_analytics_input_defaults_include_unscored_false():
    col_id = str(uuid4())
    data = QueryGraphAnalyticsInput(collection_id=col_id)
    assert data.query_type == "overview"
    assert data.top_n == 5
    assert data.min_confidence is None
    # Mandatory policy: include_unscored defaults to False for agent tools
    assert data.include_unscored is False
    assert data.author_id is None


def test_query_graph_analytics_input_top_n_bounds():
    col_id = str(uuid4())
    # Max top_n 15
    with pytest.raises(ValidationError) as exc:
        QueryGraphAnalyticsInput(collection_id=col_id, top_n=16)
    assert "top_n" in str(exc.value)

    # Min top_n 1
    with pytest.raises(ValidationError) as exc:
        QueryGraphAnalyticsInput(collection_id=col_id, top_n=0)
    assert "top_n" in str(exc.value)


def test_query_graph_analytics_input_min_confidence_bounds():
    col_id = str(uuid4())
    with pytest.raises(ValidationError) as exc:
        QueryGraphAnalyticsInput(collection_id=col_id, min_confidence=1.1)
    assert "min_confidence" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        QueryGraphAnalyticsInput(collection_id=col_id, min_confidence=-0.1)
    assert "min_confidence" in str(exc.value)


def test_query_graph_analytics_input_query_type_validation():
    col_id = str(uuid4())
    for qt in ("overview", "summary", "collaborations", "connectivity"):
        obj = QueryGraphAnalyticsInput(collection_id=col_id, query_type=qt)
        assert obj.query_type == qt

    with pytest.raises(ValidationError) as exc:
        QueryGraphAnalyticsInput(collection_id=col_id, query_type="invalid_analytics")
    assert "Invalid query_type" in str(exc.value)


def test_query_graph_analytics_input_extra_fields_forbidden():
    col_id = str(uuid4())
    with pytest.raises(ValidationError) as exc:
        QueryGraphAnalyticsInput(collection_id=col_id, disallowed="field")
    assert "extra_forbidden" in str(exc.value)


# ===========================================================================
# 5. _verify_graph_collection_access Authorization Tests
# ===========================================================================

def test_verify_graph_collection_access_success():
    """Verify that valid context with user_id and auth_service passes and delegates to verify_collection_ownership."""
    user_id = uuid4()
    col_id = uuid4()
    mock_auth_svc = MagicMock()
    mock_auth_svc.verify_collection_ownership.return_value = None

    context = AgentContext(
        user_id=user_id,
        auth_service=mock_auth_svc,
        auth_enforced=True,
    )

    resolved_col = _verify_graph_collection_access(context, str(col_id))
    assert resolved_col == col_id
    mock_auth_svc.verify_collection_ownership.assert_called_once_with(col_id, user_id)


def test_verify_graph_collection_access_missing_user_fails_closed():
    """Verify missing user_id raises AgentSecurityError (never permits anonymous access)."""
    col_id = uuid4()
    mock_auth_svc = MagicMock()

    context = AgentContext(
        user_id=None,
        auth_service=mock_auth_svc,
        auth_enforced=True,
    )

    with pytest.raises(AgentSecurityError) as exc:
        _verify_graph_collection_access(context, col_id)
    assert "Authentication required" in str(exc.value)
    mock_auth_svc.verify_collection_ownership.assert_not_called()


def test_verify_graph_collection_access_missing_auth_service_fails_closed():
    """Verify missing auth_service raises AgentSecurityError."""
    user_id = uuid4()
    col_id = uuid4()

    context = AgentContext(
        user_id=user_id,
        auth_service=None,
        auth_enforced=True,
    )

    with pytest.raises(AgentSecurityError) as exc:
        _verify_graph_collection_access(context, col_id)
    assert "Authorization service unavailable" in str(exc.value)


def test_verify_graph_collection_access_auth_enforced_false_does_not_bypass():
    """Verify that auth_enforced=False NEVER bypasses authorization when user_id is missing."""
    col_id = uuid4()
    mock_auth_svc = MagicMock()

    context = AgentContext(
        user_id=None,
        auth_service=mock_auth_svc,
        auth_enforced=False,  # Should NOT bypass!
    )

    with pytest.raises(AgentSecurityError) as exc:
        _verify_graph_collection_access(context, col_id)
    assert "Authentication required" in str(exc.value)
    mock_auth_svc.verify_collection_ownership.assert_not_called()


def test_verify_graph_collection_access_auth_enforced_false_with_user_calls_auth_service():
    """Verify that even when auth_enforced=False, collection ownership is strictly verified."""
    user_id = uuid4()
    col_id = uuid4()
    mock_auth_svc = MagicMock()

    context = AgentContext(
        user_id=user_id,
        auth_service=mock_auth_svc,
        auth_enforced=False,
    )

    resolved_col = _verify_graph_collection_access(context, col_id)
    assert resolved_col == col_id
    mock_auth_svc.verify_collection_ownership.assert_called_once_with(col_id, user_id)


def test_verify_graph_collection_access_forbidden_propagates():
    """Verify that ForbiddenError raised by auth_service propagates directly."""
    user_id = uuid4()
    col_id = uuid4()
    mock_auth_svc = MagicMock()
    mock_auth_svc.verify_collection_ownership.side_effect = ForbiddenError("Access forbidden to foreign collection.")

    context = AgentContext(
        user_id=user_id,
        auth_service=mock_auth_svc,
        auth_enforced=True,
    )

    with pytest.raises(ForbiddenError, match="Access forbidden to foreign collection"):
        _verify_graph_collection_access(context, col_id)


def test_verify_graph_collection_access_not_found_propagates():
    """Verify that DocumentNotFoundError raised by auth_service propagates directly."""
    user_id = uuid4()
    col_id = uuid4()
    mock_auth_svc = MagicMock()
    mock_auth_svc.verify_collection_ownership.side_effect = DocumentNotFoundError("Collection not found.")

    context = AgentContext(
        user_id=user_id,
        auth_service=mock_auth_svc,
        auth_enforced=True,
    )

    with pytest.raises(DocumentNotFoundError, match="Collection not found"):
        _verify_graph_collection_access(context, col_id)


def test_verify_graph_collection_access_invalid_uuid():
    """Verify invalid UUID string raises InvalidToolArgumentError."""
    user_id = uuid4()
    mock_auth_svc = MagicMock()

    context = AgentContext(
        user_id=user_id,
        auth_service=mock_auth_svc,
        auth_enforced=True,
    )

    with pytest.raises(InvalidToolArgumentError) as exc:
        _verify_graph_collection_access(context, "not-a-valid-uuid")
    assert "collection_id must be a valid UUID" in str(exc.value)


def test_verify_graph_collection_access_invalid_context_type():
    """Verify non-AgentContext object raises AgentSecurityError."""
    with pytest.raises(AgentSecurityError) as exc:
        _verify_graph_collection_access("not-a-context", uuid4())  # type: ignore[arg-type]
    assert "Operation requires a valid AgentContext" in str(exc.value)


# ---------------------------------------------------------------------------
# Relationship Types Validator Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("invalid_rel_types", [
    [],                      # Empty list cannot silently convert to None
    [""],                    # Empty string
    ["   "],                 # Whitespace only
    [123],                   # Non-string integer
    [None],                  # None entry
    ["valid", ""],           # Mixed valid and empty
    ["valid", 456],          # Mixed valid and non-string
    ["a" * 101],             # Exceeds max length of 100
])
def test_explore_neighborhood_relationship_types_validation(invalid_rel_types):
    """Verify ExploreGraphNeighborhoodInput rejects malformed relationship_types."""
    col_id = str(uuid4())
    node_id = str(uuid4())
    with pytest.raises(ValidationError):
        ExploreGraphNeighborhoodInput(
            collection_id=col_id,
            node_id=node_id,
            node_type="entity",
            relationship_types=invalid_rel_types,
        )


@pytest.mark.parametrize("invalid_rel_types", [
    [],                      # Empty list cannot silently convert to None
    [""],                    # Empty string
    ["   "],                 # Whitespace only
    [123],                   # Non-string integer
    [None],                  # None entry
    ["valid", ""],           # Mixed valid and empty
    ["valid", 456],          # Mixed valid and non-string
    ["a" * 101],             # Exceeds max length of 100
])
def test_find_path_relationship_types_validation(invalid_rel_types):
    """Verify FindGraphPathInput rejects malformed relationship_types."""
    col_id = str(uuid4())
    src_id = str(uuid4())
    dst_id = str(uuid4())
    with pytest.raises(ValidationError):
        FindGraphPathInput(
            collection_id=col_id,
            source_id=src_id,
            source_type="entity",
            target_id=dst_id,
            target_type="paper",
            relationship_types=invalid_rel_types,
        )

