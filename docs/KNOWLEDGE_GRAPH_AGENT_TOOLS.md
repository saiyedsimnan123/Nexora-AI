# Knowledge Graph Agent Tools Guide

The **Nexora AI Knowledge Graph Agent Tools** connect autonomous agent workflows to collection-scoped, tenant-isolated knowledge graphs. They enable multi-hop topological reasoning, entity discovery, relationship pathfinding, and community analytics over scientific literature with strict fail-closed security.

---

## 1. Core Principles

- **Zero-Trust Collection Scoping**: Knowledge graph data is strictly private and collection-scoped. The agent context must prove authenticated identity and ownership of the target collection.
- **Fail-Closed Security Invariance**: Anonymous requests, cross-tenant requests, and requests with missing authorization dependencies are immediately rejected (`unauthorized` or `forbidden`). `auth_enforced=False` is never an authorization bypass.
- **Immediate Execution Abort**: When an agent step fails an authorization or ownership check, the `AgentOrchestrator` halts immediately, marking the trace as failed without leaking private graph structure or metadata.
- **Strict Evidence Grounding**: Relationships discovered in the graph retain evidence provenance (`chunk_id`, `paper_id`, `page_number`, `evidence_text`, `extraction_method`). Ungrounded citations are never fabricated.
- **Deterministic Traversal & Multiplicative Confidence**: Pathfinding evaluates path confidence as the mathematical product of edge confidences ($\prod_{e \in E} \text{conf}(e)$). Bounded Breadth-First Search (BFS) enforces hard depth and node/edge caps to prevent denial-of-service or cyclic loops.

---

## 2. Architecture & Production Bootstrap Flow

```text
HTTP Client (REST API)
          │
          ▼ POST /api/v1/agents/run
┌────────────────────────────────────────────────────────┐
│ FastAPI Application State (app.state)                  │
├────────────────────────────────────────────────────────┤
│ 1. Dependency Resolution & Context Injection           │
│    - current_user -> context.user_id                   │
│    - auth_service -> context.auth_service              │
│    - session_factory -> context.session_factory        │
├────────────────────────────────────────────────────────┤
│ 2. AgentOrchestrator                                   │
│    - Tool Registry Allowlist                           │
│    - Loop Protection & Step-Budget Enforcement         │
│    - Reactive Security Fail-Closed Abort               │
├────────────────────────────────────────────────────────┤
│ 3. ToolRegistry (Stage 5 Step 4.2)                     │
│    ├── search_graph_entities                           │
│    ├── explore_graph_neighborhood                      │
│    ├── find_graph_path                                 │
│    └── query_graph_analytics                           │
└────────────────────────────────────────────────────────┘
          │
          ▼ _verify_graph_collection_access(context, collection_id)
┌────────────────────────────────────────────────────────┐
│ Knowledge Graph Services & Database Engine             │
├────────────────────────────────────────────────────────┤
│ • KnowledgeGraphTraversalService (BFS & Pathfinding)   │
│ • KnowledgeGraphAnalyticsService (Metrics & Networks)  │
│ • GraphEntityRepository / GraphRelationshipRepository  │
└────────────────────────────────────────────────────────┘
```

### Application Assembly Chain

1. `create_production_app()` in `src/nexora/server.py` delegates to `build_http_app()`.
2. `build_http_app()` in `src/nexora/bootstrap/application.py` constructs a single database engine and `session_factory` from application settings.
3. `build_tool_registry(session_factory=...)` delegates to `create_default_tool_registry()` in `src/nexora/agents/tools.py`.
4. `create_default_tool_registry()`:
   - Registers `SearchGraphEntitiesTool(session_factory=session_factory)`.
   - Constructs `KnowledgeGraphTraversalService(session_factory=session_factory)` and registers `ExploreGraphNeighborhoodTool` and `FindGraphPathTool`.
   - Constructs `KnowledgeGraphAnalyticsService(session_factory=session_factory)` and registers `QueryGraphAnalyticsTool`.
   - If `session_factory` is supplied and service initialization fails, it raises an explicit `InvalidConfigurationError`.
   - If no database is configured (`session_factory is None`), graph tools are omitted, preserving legacy database-free configurations.
5. The assembled registry is injected into `AgentOrchestrator`, which is stored on `app.state.orchestrator` and resolved by FastAPI dependency injection in `run_agent_endpoint()`.

---

## 3. Tool Reference & Schemas

### 3.1 `search_graph_entities`

Searches or lists canonical entities (`author`, `concept`, `dataset`, `method`, `metric`, `other`) within an authorized collection.

#### Input Schema (`SearchGraphEntitiesInput`)
| Parameter | Type | Required | Default | Bounds / Validation | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `collection_id` | `str` | Yes | — | Valid UUID | UUID of authorized collection. |
| `query` | `str` | No | `None` | Max 200 chars | Substring query to match against `name` and `canonical_name`. |
| `entity_type` | `str` | No | `None` | Enum | Classification filter: `author`, `concept`, `dataset`, `method`, `metric`, `other`. |
| `limit` | `int` | No | `10` | $1 \le \text{limit} \le 20$ | Maximum entities to return (hard capped at 20). |

*Note: Extra parameters are strictly rejected (`extra="forbid"`).*

#### Output Data (`list[dict]`)
Each entity contains:
- `id` (`str`): Entity UUID.
- `collection_id` (`str`): Collection UUID.
- `name` (`str`): Display name.
- `canonical_name` (`str`): Normalized lowercase identifier.
- `entity_type` (`str`): Entity classification category.
- `description` (`str` | `None`): Summary description.
- `aliases` (`list[str]`): List of known aliases.
- `metadata` (`dict`): Entity attributes and properties.

---

### 3.2 `explore_graph_neighborhood`

Performs bounded 1- or 2-hop local Breadth-First Search (BFS) around a seed entity or paper.

#### Input Schema (`ExploreGraphNeighborhoodInput`)
| Parameter | Type | Required | Default | Bounds / Validation | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `collection_id` | `str` | Yes | — | Valid UUID | UUID of authorized collection. |
| `node_id` | `str` | Yes | — | Valid UUID | UUID of seed paper or entity. |
| `node_type` | `str` | Yes | — | `'paper'`, `'entity'` | Seed node type. |
| `max_depth` | `int` | No | `1` | $1 \le \text{depth} \le 2$ | Maximum exploration depth (hard ceiling: 2). |
| `max_nodes` | `int` | No | `15` | $1 \le \text{nodes} \le 25$ | Total node discovery budget (hard cap: 25). |
| `max_edges` | `int` | No | `30` | $1 \le \text{edges} \le 50$ | Total edge discovery budget (hard cap: 50). |
| `direction` | `str` | No | `'both'` | `'both'`, `'outgoing'`, `'incoming'` | Directional traversal filter. |
| `relationship_types` | `list[str]` | No | `None` | Max 20 strings | Allowed relationship types (empty list `[]` rejected). |
| `min_confidence` | `float` | No | `None` | $0.0 \le \text{conf} \le 1.0$ | Minimum edge confidence filter. |

#### Output Data (`dict`)
- `seed_id` (`str`), `seed_type` (`str`): Starting point of exploration.
- `depth` (`int`): Actual depth reached by BFS traversal.
- `nodes` (`list[dict]`): Discovered nodes with `id`, `node_type`, `name`, `entity_type`, `description`, `metadata`.
- `edges` (`list[dict]`): Grounded edges with `source_type`, `source_id`, `target_type`, `target_id`, `relationship_type`, `confidence`, `evidence_text`, `chunk_id`, `paper_id`, `page_number`, `extraction_method`.
- `is_truncated` (`bool`): Whether budgets terminated expansion early.
- `truncation_reason` (`str` | `None`): `"max_nodes_reached"`, `"max_edges_reached"`, or `None`.

---

### 3.3 `find_graph_path`

Discovers connecting multi-hop relationship paths between two nodes (up to 3 hops).

#### Input Schema (`FindGraphPathInput`)
| Parameter | Type | Required | Default | Bounds / Validation | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `collection_id` | `str` | Yes | — | Valid UUID | UUID of authorized collection. |
| `source_id` | `str` | Yes | — | Valid UUID | Source node UUID. |
| `source_type` | `str` | Yes | — | `'paper'`, `'entity'` | Source node type. |
| `target_id` | `str` | Yes | — | Valid UUID | Target node UUID. |
| `target_type` | `str` | Yes | — | `'paper'`, `'entity'` | Target node type. |
| `max_depth` | `int` | No | `2` | $1 \le \text{depth} \le 3$ | Maximum hops allowed (hard cap: 3). |
| `max_paths` | `int` | No | `2` | $1 \le \text{paths} \le 3$ | Maximum candidate paths to discover (hard cap: 3). |
| `direction` | `str` | No | `'outgoing'` | `'outgoing'`, `'incoming'`, `'both'` | Edge traversal direction. |
| `relationship_types` | `list[str]` | No | `None` | Max 20 strings | Filter relationship types. |
| `min_confidence` | `float` | No | `None` | $0.0 \le \text{conf} \le 1.0$ | Minimum edge confidence. |

#### Output Data (`list[dict]`)
Each discovered path contains:
- `hops` (`int`), `length` (`int`): Number of edges in path ($0$ if source equals target).
- `confidence_score` (`float` | `None`): Multiplicative confidence score $\prod_{e} \text{conf}(e)$ (rounded to 6 decimals). `None` if any edge is unscored.
- `cumulative_confidence` (`float` | `None`): Alias of confidence score.
- `nodes` (`list[dict]`): Ordered node chain from source to target.
- `edges` (`list[dict]`): Ordered edge chain with full provenance grounding (`chunk_id`, `paper_id`, `page_number`, `evidence_text`).

---

### 3.4 `query_graph_analytics`

Queries collection-wide knowledge graph statistics, author collaboration networks, or author connectivity metrics.

#### Input Schema (`QueryGraphAnalyticsInput`)
| Parameter | Type | Required | Default | Bounds / Validation | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `collection_id` | `str` | Yes | — | Valid UUID | UUID of authorized collection. |
| `query_type` | `str` | Yes | — | Enum | `'overview'`, `'summary'`, `'collaborations'`, `'connectivity'`. |
| `top_n` | `int` | No | `5` | $1 \le \text{top\_n} \le 15$ | Result limit for rankings (hard cap: 15). |
| `author_id` | `str` | No | `None` | Valid UUID | Author UUID (permitted only for `collaborations` and `connectivity`). |
| `min_confidence` | `float` | No | `None` | $0.0 \le \text{conf} \le 1.0$ | Minimum confidence filter. |
| `include_unscored` | `bool` | No | `False` | `True`, `False` | Whether to include relationships with null confidence. |

#### Output Data by `query_type`
- **`summary`**: Overall counts: `total_entities`, `total_relationships`, `entity_counts_by_type`, `relationship_counts_by_type`.
- **`overview`**: Combined dashboard payload containing `summary`, `entity_distribution`, `relationship_distribution`, `top_collaborations`, and `author_connectivity`.
- **`collaborations`**: Pairwise author co-authorship records: `author_id`, `author_name`, `collaborator_id`, `collaborator_name`, `collaboration_count`, `shared_paper_ids`.
- **`connectivity`**: Author degree metrics: `author_id`, `author_name`, `paper_count`, `collaborator_count`, `total_collaborations`, `coauthor_count`, `total_degree`.

---

## 4. Multi-Tenant Authorization & Fail-Closed Security

All 4 tools call the central security gate `_verify_graph_collection_access()` before executing any query:

```python
def _verify_graph_collection_access(context: AgentContext, collection_id: str | UUID) -> UUID:
    if not isinstance(context, AgentContext):
        raise AgentSecurityError("Operation requires a valid AgentContext.")

    if context.user_id is None:
        raise AgentSecurityError(
            "Authentication required: collection knowledge graph data is private and cannot be accessed anonymously."
        )

    if context.auth_service is None:
        raise AgentSecurityError(
            "Authorization service unavailable: cannot verify collection ownership."
        )

    col_uuid = UUID(str(collection_id).strip())
    context.auth_service.verify_collection_ownership(col_uuid, context.user_id)
    return col_uuid
```

### Security Guarantees

1. **Anonymous Rejection**: If `context.user_id is None`, the call fails closed immediately with `error_type="unauthorized"`.
2. **`auth_enforced=False` Is Never a Bypass**: Server-level flags like `--no-enforce-auth` affect endpoint routing for legacy demos, but `_verify_graph_collection_access` **never** checks `context.auth_enforced`. Graph data cannot be accessed anonymously under any flag.
3. **Cross-Tenant Isolation**: `verify_collection_ownership` validates that `collection.workspace.user_id == user_id`. Querying another tenant's collection raises `ForbiddenError`, converted to `error_type="forbidden"`.
4. **Immediate Orchestrator Abort**: In `AgentOrchestrator.run()`:
   ```python
   if not tool_result.success and tool_result.error_type in ("forbidden", "unauthorized"):
       trace.mark_failed(f"Authorization check failed during {tool_name}: {tool_result.error}")
       break
   ```
   No further steps or LLM calls are executed. Zero graph topology is leaked.

---

## 5. Error Handling & Two-Tier Status Architecture

Nexora AI maintains a strict two-tier separation between **HTTP endpoint-level exceptions** and **tool-level execution results**:

### 5.1 Endpoint-Level HTTP Status Codes
HTTP exceptions occur at the FastAPI router boundary in `run_agent_endpoint()` before or surrounding the autonomous agent loop:

- **HTTP 401 Unauthorized (`AuthenticationError`)**: Raised if the application was started with `enforce_auth=True` and no valid credentials are provided (`current_user is None`).
- **HTTP 403 Forbidden (`ForbiddenError`)**: Raised if endpoint-level access checks fail or if `AgentSecurityError` is caught at the top-level route boundary.
- **HTTP 400 Bad Request (`InvalidRequestError`)**: Raised if the outer JSON request payload is malformed or violates endpoint schema constraints.
- **HTTP 500 Internal Server Error (`InvalidConfigurationError` / `AgentServiceError`)**: Raised if dependency injection fails during application startup (e.g. required graph service construction fails with a session factory) or if an unrecoverable runtime exception occurs outside the tool execution loop.

### 5.2 Tool-Level Errors inside `ToolResult` and `AgentOrchestrator`
During autonomous execution, tools do **not** raise HTTP exceptions directly. Instead, `tool.execute()` safely traps internal exceptions and returns structured `ToolResult` records:

- `ToolResult.failure(error_type="validation_error")`: Returned when tool input arguments fail schema validation (e.g. invalid UUID format, unallowed extra fields, invalid query types).
- `ToolResult.failure(error_type="unauthorized")`: Returned by `_verify_graph_collection_access` when `context.user_id` is missing or `context.auth_service` is unavailable.
- `ToolResult.failure(error_type="forbidden")`: Returned by `_verify_graph_collection_access` when collection ownership verification fails.
- `ToolResult.failure(error_type="not_found")`: Returned when the requested collection or seed node cannot be found in the database.
- `ToolResult.failure(error_type="execution_error")`: Returned when an unexpected backend or database exception occurs during service execution.

### 5.3 Orchestrator Fail-Closed Abort Behavior
**Important Architectural Distinction**:
Tool-level `unauthorized` and `forbidden` errors do **not** map to HTTP 401 or 403 status codes. Because the HTTP request to `/api/v1/agents/run` was itself valid and authorized to start an agent, the endpoint completes execution and returns an HTTP 200 response with `AgentRunResponseBody`:

- When a tool fails with `error_type in ("forbidden", "unauthorized")`, `AgentOrchestrator` halts immediately:
  ```python
  if not tool_result.success and tool_result.error_type in ("forbidden", "unauthorized"):
      trace.mark_failed(f"Authorization check failed during {tool_name}: {tool_result.error}")
      break
  ```
- The response body sets `status: "failed"`, logs the error inside `trace.steps[step_index].error`, and ceases further steps.
- **Result**: Zero private collection data is leaked, and the orchestrator terminates without crashing the client connection.

---

## 6. Usage Examples

### 6.1 Direct Python Execution (within Agent Workspace)

```python
from nexora.agents.models import AgentContext
from nexora.agents.tools import create_default_tool_registry

# 1. Initialize production tool registry with active database session factory
registry = create_default_tool_registry(session_factory=app_session_factory)

# 2. Build authenticated tenant context
context = AgentContext(
    user_id=authenticated_user_id,
    auth_enforced=True,
    auth_service=app_auth_service,
    session_factory=app_session_factory,
)

# 3. Execute 2-hop neighborhood exploration
result = registry.execute(
    "explore_graph_neighborhood",
    {
        "collection_id": str(collection_uuid),
        "node_id": str(paper_uuid),
        "node_type": "paper",
        "max_depth": 2,
        "direction": "outgoing",
    },
    context,
)

if result.success:
    print(f"Exploration Succeeded: {result.summary}")
    for node in result.data["nodes"]:
        print(f"  Discovered: {node['name']} ({node['node_type']})")
else:
    print(f"Execution Failed ({result.error_type}): {result.error}")
```

### 6.2 HTTP Agent API Request (`POST /api/v1/agents/run`)

```bash
curl -X POST "http://localhost:8000/api/v1/agents/run" \
  -H "Authorization: Bearer nx_live_a1b2c3d4..." \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Find multi-hop connecting paths between Transformer model and Self-Attention in collection 550e8400-e29b-41d4-a716-446655440000",
    "max_steps": 5,
    "timeout_seconds": 30.0
  }'
```

#### Successful JSON Response
```json
{
  "status": "completed",
  "final_answer": "Discovered path from Transformer to Self-Attention via 'uses_method' relationship with confidence 0.90.",
  "total_steps": 1,
  "trace": {
    "status": "completed",
    "steps": [
      {
        "step_index": 0,
        "tool_name": "find_graph_path",
        "status": "success",
        "output_summary": "Discovered 1 path(s) connecting entity '20000000-...' to entity '20000000-...'.",
        "duration_ms": 11.2,
        "error": null,
        "evidence_labels": []
      }
    ],
    "total_duration_ms": 154.2
  }
}
```

---

## 7. Empirical Evaluation & Benchmark Results

### 7.1 Benchmark Execution Context & Limitations
The latency and accuracy figures below were measured using the dedicated performance test suite (`tests/test_knowledge_graph_agent_benchmarks.py`).

**Important Evaluation Context & Boundaries**:
- **Small In-Memory SQLite Fixture**: Benchmarks were executed across 20 warm iterations ($N=20$) on a lightweight, controlled in-memory SQLite graph corpus ($\approx 20$ records). These numbers reflect pure algorithmic traversal, serialization, and SQL parsing overhead in a single process without network delays, concurrent connection contention, or disk persistence I/O.
- **Mock LLM Execution**: The `AgentOrchestrator` benchmark executes using a deterministic scripted mock LLM (`ScriptedBenchmarkLLM`) and **completely bypasses external LLM API inference**. Production agent latency will be overwhelmingly dominated by external frontier model network roundtrips and token generation times.
- **Fixture-Specific Precision & Recall**: The reported 100% Precision@1 and Recall metrics are **fixture-specific measurements** verifying that exact and substring SQL filters match expected ground-truth records in this specific dataset. They must **not** be construed as general search-quality guarantees across large, noisy, or open-ended unstructured corpora.

### 7.2 Measured Evaluation Metrics

| Evaluation Category | Tool / Workflow Evaluated | Measured Metric | Measured Result | Evaluation Status |
| :--- | :--- | :--- | :---: | :---: |
| **Search Precision** | `search_graph_entities` | Exact Match Precision@1 (Fixture) | **100.0%** (1.0) | Verified on benchmark fixture |
| **Search Recall** | `search_graph_entities` | Substring Match Recall (Fixture) | **100.0%** (1.0) | Verified on benchmark fixture |
| **Path Correctness** | `find_graph_path` | 1-Hop & 2-Hop Sequence Accuracy | **100.0%** (1.0) | Verified on benchmark fixture |
| **Confidence Integrity** | `find_graph_path` | Multiplicative Confidence Match | **100.0%** ($0.95 \times 0.90 = 0.855$) | Verified mathematically |
| **Budget Enforcement** | `explore_graph_neighborhood` | Node & Edge Budget Compliance | **100.0%** (`is_truncated=True`) | Verified on benchmark fixture |
| **Security Isolation** | All 4 Graph Tools | Anonymous Denial Rate | **100.0%** (`unauthorized`) | Verified fail-closed |
| **Tenant Isolation** | All 4 Graph Tools | Cross-Tenant Denial Rate | **100.0%** (`forbidden`) | Verified fail-closed |
| **Input Validation** | All 4 Graph Tools | Adversarial Input Rejection Rate | **100.0%** (`validation_error`) | Verified schema boundary |
| **Latency Benchmark** | `search_graph_entities` | Latency (p50 / p95 / max) | **1.76 ms** / **4.95 ms** / 5.52 ms | Passed ($N=20$, warm SQLite) |
| **Latency Benchmark** | `explore_graph_neighborhood` | Latency (p50 / p95 / max) | **8.11 ms** / **9.73 ms** / 10.59 ms | Passed ($N=20$, warm SQLite) |
| **Latency Benchmark** | `find_graph_path` | Latency (p50 / p95 / max) | **11.01 ms** / **25.63 ms** / 52.15 ms | Passed ($N=20$, warm SQLite) |
| **Latency Benchmark** | `query_graph_analytics` | Latency (p50 / p95 / max) | **11.85 ms** / **19.88 ms** / 23.79 ms | Passed ($N=20$, warm SQLite) |
| **Orchestrator Run** | Full Agent Step Execution | End-to-End Latency (p50 / p95 / max) | **4.67 ms** / **8.00 ms** / 8.83 ms | Passed ($N=20$, mock LLM) |

---

## 8. Known Limitations & Gaps

1. **Traversal Budget Hard Ceilings**:
   - `explore_graph_neighborhood` enforces a maximum depth of **2** and a maximum of **25** nodes.
   - `find_graph_path` enforces a maximum depth of **3** hops and at most **3** candidate paths.
   Requests with higher limits are clamped to these hard ceilings to prevent unbounded database query expansion.
2. **Unscored Relationship Policy**:
   - Agent traversal tools enforce `include_unscored=False` by default. Relationships without a defined confidence score are ignored during traversal unless explicitly scored.
3. **No Vector-Graph Semantic Search Join**:
   - The current tools perform relational graph traversals and exact substring entity matching. Dynamic hybrid vector semantic search joined with graph traversal (e.g. searching concepts via vector embeddings before walking relationships) is an identified gap for future milestones.
4. **Entity Aliases Not Indexed in Substring Search**:
   - `GraphEntityRepository.search_entities()` searches `name` and `canonical_name`. While `aliases` are stored and returned in entity payloads, they are not queried in the SQL `search_entities` WHERE clause.
