"""Empirical latency and throughput performance benchmarks for Knowledge Graph Agent Tools.

Stage 5 Step 4.3 — Performance Benchmarking.

Runs separated from deterministic CI tests:
- Measures 20 warm iterations per tool across:
  1. SearchGraphEntitiesTool
  2. ExploreGraphNeighborhoodTool
  3. FindGraphPathTool
  4. QueryGraphAnalyticsTool
  5. AgentOrchestrator end-to-end execution
- Computes empirical statistics: p50 (median), p95 (95th percentile), p99, and maximum latency (ms).
- Reports results with zero synthetic fabrication.
"""

from __future__ import annotations

import datetime
import json
import statistics
import time
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.agents.models import AgentContext
from nexora.agents.orchestrator import AgentOrchestrator
from nexora.agents.tools import (
    ExploreGraphNeighborhoodTool,
    FindGraphPathTool,
    QueryGraphAnalyticsTool,
    SearchGraphEntitiesTool,
    create_default_tool_registry,
)
from nexora.auth.service import AuthenticationService
from nexora.database.models.base import Base
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.graph_entity import GraphEntity
from nexora.database.models.graph_relationship import GraphRelationship
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.graph.analytics_service import KnowledgeGraphAnalyticsService
from nexora.graph.traversal import KnowledgeGraphTraversalService
from nexora.llm.models import LLMMessage, LLMResponse


BENCHMARK_ITERATIONS = 20

USER_ID = UUID("11111111-1111-1111-1111-111111111111")
WS_ID = UUID("aaaaaaaa-1111-1111-1111-111111111111")
COL_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

PAPER_ID = UUID("10000000-0000-0000-0000-000000000001")
DOC_ID = UUID("10000000-0000-0000-0000-000000000011")
CHUNK_ID = UUID("10000000-0000-0000-0000-000000000021")

ENTITY_TRANSFORMER_ID = UUID("20000000-0000-0000-0000-000000000001")
ENTITY_ATTENTION_ID = UUID("20000000-0000-0000-0000-000000000002")
ENTITY_VASWANI_ID = UUID("20000000-0000-0000-0000-000000000003")
ENTITY_SHAZEER_ID = UUID("20000000-0000-0000-0000-000000000004")
ENTITY_WMT_ID = UUID("20000000-0000-0000-0000-000000000005")


class ScriptedBenchmarkLLM:
    """Mock LLM returning deterministic script responses without external network overhead."""

    def __init__(self, step_responses: list[str]) -> None:
        self.responses = list(step_responses)
        self.call_count = 0

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return LLMResponse(text=self.responses[idx], model="scripted-benchmark")


@pytest.fixture(scope="module")
def benchmark_env():
    """Module-scoped SQLite database seeded with warm benchmark graph corpus."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    session = session_factory()
    try:
        user = User(id=USER_ID, email="benchmark@example.com")
        session.add(user)
        session.flush()

        ws = Workspace(id=WS_ID, name="Bench Workspace", user_id=USER_ID)
        session.add(ws)
        session.flush()

        col = Collection(id=COL_ID, name="Bench Collection", workspace_id=WS_ID)
        session.add(col)
        session.flush()

        paper = Paper(
            id=PAPER_ID,
            collection_id=COL_ID,
            title="Attention Is All You Need",
            publication_date=datetime.date(2017, 6, 12),
        )
        session.add(paper)
        session.flush()

        doc = Document(
            id=DOC_ID,
            paper_id=PAPER_ID,
            filename="bench.pdf",
            mime_type="application/pdf",
            file_size=1024,
            checksum="b" * 64,
            storage_key="papers/bench.pdf",
            indexing_status="completed",
        )
        session.add(doc)
        session.flush()

        chunk = DocumentChunkRecord(
            id=CHUNK_ID,
            document_id=DOC_ID,
            chunk_index=0,
            text="Benchmark chunk text for attention mechanism.",
        )
        session.add(chunk)
        session.flush()

        e1 = GraphEntity(id=ENTITY_TRANSFORMER_ID, collection_id=COL_ID, name="Transformer", canonical_name="transformer", entity_type="method")
        e2 = GraphEntity(id=ENTITY_ATTENTION_ID, collection_id=COL_ID, name="Self-Attention", canonical_name="self-attention", entity_type="method")
        e3 = GraphEntity(id=ENTITY_VASWANI_ID, collection_id=COL_ID, name="Ashish Vaswani", canonical_name="ashish vaswani", entity_type="author")
        e4 = GraphEntity(id=ENTITY_SHAZEER_ID, collection_id=COL_ID, name="Noam Shazeer", canonical_name="noam shazeer", entity_type="author")
        e5 = GraphEntity(id=ENTITY_WMT_ID, collection_id=COL_ID, name="WMT 2014 En-De", canonical_name="wmt 2014 en-de", entity_type="dataset")
        session.add_all([e1, e2, e3, e4, e5])

        r1 = GraphRelationship(collection_id=COL_ID, source_type="paper", source_id=PAPER_ID, source_paper_id=PAPER_ID, target_type="entity", target_id=ENTITY_VASWANI_ID, target_entity_id=ENTITY_VASWANI_ID, relationship_type="authored_by", confidence=0.99)
        r2 = GraphRelationship(collection_id=COL_ID, source_type="paper", source_id=PAPER_ID, source_paper_id=PAPER_ID, target_type="entity", target_id=ENTITY_SHAZEER_ID, target_entity_id=ENTITY_SHAZEER_ID, relationship_type="authored_by", confidence=0.99)
        r3 = GraphRelationship(collection_id=COL_ID, source_type="paper", source_id=PAPER_ID, source_paper_id=PAPER_ID, target_type="entity", target_id=ENTITY_TRANSFORMER_ID, target_entity_id=ENTITY_TRANSFORMER_ID, relationship_type="proposes", confidence=0.95)
        r4 = GraphRelationship(collection_id=COL_ID, source_type="entity", source_id=ENTITY_TRANSFORMER_ID, source_entity_id=ENTITY_TRANSFORMER_ID, target_type="entity", target_id=ENTITY_ATTENTION_ID, target_entity_id=ENTITY_ATTENTION_ID, relationship_type="uses_method", confidence=0.90)
        r5 = GraphRelationship(collection_id=COL_ID, source_type="paper", source_id=PAPER_ID, source_paper_id=PAPER_ID, target_type="entity", target_id=ENTITY_WMT_ID, target_entity_id=ENTITY_WMT_ID, relationship_type="evaluates_on", confidence=0.85)
        session.add_all([r1, r2, r3, r4, r5])
        session.commit()

        auth_service = AuthenticationService(session_factory=session_factory)
        traversal_service = KnowledgeGraphTraversalService(session_factory=session_factory)
        analytics_service = KnowledgeGraphAnalyticsService(session_factory=session_factory)

        context = AgentContext(
            user_id=USER_ID,
            auth_enforced=True,
            auth_service=auth_service,
            session_factory=session_factory,
        )

        return {
            "session_factory": session_factory,
            "auth_service": auth_service,
            "traversal_service": traversal_service,
            "analytics_service": analytics_service,
            "context": context,
        }
    finally:
        session.close()


def _calculate_percentiles(durations_ms: list[float]) -> dict[str, float]:
    """Calculate p50, p95, p99, and max from a list of duration measurements."""
    sorted_durs = sorted(durations_ms)
    n = len(sorted_durs)
    p50_idx = int(0.50 * (n - 1))
    p95_idx = int(0.95 * (n - 1))
    p99_idx = int(0.99 * (n - 1))
    return {
        "count": n,
        "mean_ms": round(statistics.mean(sorted_durs), 3),
        "p50_ms": round(sorted_durs[p50_idx], 3),
        "p95_ms": round(sorted_durs[p95_idx], 3),
        "p99_ms": round(sorted_durs[p99_idx], 3),
        "max_ms": round(sorted_durs[-1], 3),
        "min_ms": round(sorted_durs[0], 3),
    }


def test_benchmark_search_graph_entities_latency(benchmark_env):
    """Empirical latency benchmark for search_graph_entities over 20 warm iterations."""
    tool = SearchGraphEntitiesTool(session_factory=benchmark_env["session_factory"])
    args = {"collection_id": str(COL_ID), "query": "Transformer", "entity_type": "method"}
    ctx = benchmark_env["context"]

    durations: list[float] = []
    # Warmup
    tool.execute(args, ctx)

    for _ in range(BENCHMARK_ITERATIONS):
        t0 = time.perf_counter()
        res = tool.execute(args, ctx)
        t1 = time.perf_counter()
        assert res.success is True
        durations.append((t1 - t0) * 1000)

    stats = _calculate_percentiles(durations)
    print(f"\n[BENCHMARK] search_graph_entities (N={BENCHMARK_ITERATIONS}): p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms, max={stats['max_ms']}ms")
    assert stats["p95_ms"] < 50.0
    assert stats["count"] == BENCHMARK_ITERATIONS


def test_benchmark_explore_graph_neighborhood_latency(benchmark_env):
    """Empirical latency benchmark for explore_graph_neighborhood over 20 warm iterations."""
    tool = ExploreGraphNeighborhoodTool(benchmark_env["traversal_service"])
    args = {
        "collection_id": str(COL_ID),
        "node_id": str(PAPER_ID),
        "node_type": "paper",
        "direction": "outgoing",
        "max_depth": 2,
    }
    ctx = benchmark_env["context"]

    durations: list[float] = []
    # Warmup
    tool.execute(args, ctx)

    for _ in range(BENCHMARK_ITERATIONS):
        t0 = time.perf_counter()
        res = tool.execute(args, ctx)
        t1 = time.perf_counter()
        assert res.success is True
        durations.append((t1 - t0) * 1000)

    stats = _calculate_percentiles(durations)
    print(f"\n[BENCHMARK] explore_graph_neighborhood (N={BENCHMARK_ITERATIONS}): p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms, max={stats['max_ms']}ms")
    assert stats["p95_ms"] < 50.0
    assert stats["count"] == BENCHMARK_ITERATIONS


def test_benchmark_find_graph_path_latency(benchmark_env):
    """Empirical latency benchmark for find_graph_path over 20 warm iterations."""
    tool = FindGraphPathTool(benchmark_env["traversal_service"])
    args = {
        "collection_id": str(COL_ID),
        "source_id": str(PAPER_ID),
        "source_type": "paper",
        "target_id": str(ENTITY_ATTENTION_ID),
        "target_type": "entity",
        "direction": "outgoing",
        "max_depth": 2,
    }
    ctx = benchmark_env["context"]

    durations: list[float] = []
    # Warmup
    tool.execute(args, ctx)

    for _ in range(BENCHMARK_ITERATIONS):
        t0 = time.perf_counter()
        res = tool.execute(args, ctx)
        t1 = time.perf_counter()
        assert res.success is True
        durations.append((t1 - t0) * 1000)

    stats = _calculate_percentiles(durations)
    print(f"\n[BENCHMARK] find_graph_path (N={BENCHMARK_ITERATIONS}): p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms, max={stats['max_ms']}ms")
    assert stats["p95_ms"] < 50.0
    assert stats["count"] == BENCHMARK_ITERATIONS


def test_benchmark_query_graph_analytics_latency(benchmark_env):
    """Empirical latency benchmark for query_graph_analytics over 20 warm iterations."""
    tool = QueryGraphAnalyticsTool(benchmark_env["analytics_service"])
    args = {"collection_id": str(COL_ID), "query_type": "overview", "top_n": 5}
    ctx = benchmark_env["context"]

    durations: list[float] = []
    # Warmup
    tool.execute(args, ctx)

    for _ in range(BENCHMARK_ITERATIONS):
        t0 = time.perf_counter()
        res = tool.execute(args, ctx)
        t1 = time.perf_counter()
        assert res.success is True
        durations.append((t1 - t0) * 1000)

    stats = _calculate_percentiles(durations)
    print(f"\n[BENCHMARK] query_graph_analytics (N={BENCHMARK_ITERATIONS}): p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms, max={stats['max_ms']}ms")
    assert stats["p95_ms"] < 50.0
    assert stats["count"] == BENCHMARK_ITERATIONS


def test_benchmark_orchestrator_end_to_end_agent_step_latency(benchmark_env):
    """Empirical latency benchmark for full AgentOrchestrator tool loop execution."""
    registry = create_default_tool_registry(session_factory=benchmark_env["session_factory"])
    step_call = json.dumps({
        "action": "call_tool",
        "tool_name": "search_graph_entities",
        "tool_input": {"collection_id": str(COL_ID), "query": "Transformer"},
    })
    step_final = json.dumps({
        "action": "final_answer",
        "answer": "Identified Transformer entity in graph.",
    })
    llm = ScriptedBenchmarkLLM([step_call, step_final])
    orchestrator = AgentOrchestrator(llm_service=llm, tool_registry=registry)
    ctx = benchmark_env["context"]

    durations: list[float] = []
    # Warmup
    orchestrator.run("Find Transformer in graph", context=ctx)

    for _ in range(BENCHMARK_ITERATIONS):
        t0 = time.perf_counter()
        result = orchestrator.run("Find Transformer in graph", context=ctx)
        t1 = time.perf_counter()
        assert result.status == "completed"
        durations.append((t1 - t0) * 1000)

    stats = _calculate_percentiles(durations)
    print(f"\n[BENCHMARK] AgentOrchestrator graph run (N={BENCHMARK_ITERATIONS}): p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms, max={stats['max_ms']}ms")
    assert stats["p95_ms"] < 100.0
    assert stats["count"] == BENCHMARK_ITERATIONS
