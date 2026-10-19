"""Multi-Tenant Vector Isolation and RAG Security Regression Tests (Milestone 27).

Verifies that:
1. Indexed vector chunks authoritatively map to their owning collection_id.
2. Direct retrieval (RetrievalService) enforces collection boundaries.
3. RAG pipeline (pipeline.answer and POST /api/v1/research/query) guarantees tenant isolation:
   User A can never retrieve, cite, or view User B's private research evidence.
4. Cross-collection access attempts (User A targeting Collection B or Paper B) return HTTP 403 Forbidden.
5. Authenticated users with zero collections fail closed (return insufficient evidence without searching globally).
6. Legacy vectors with missing collection_id metadata fail closed and are not exposed.
7. Paper analysis, comparison, and verification routes enforce ownership and reject cross-tenant access.
8. Qdrant vector-store pushes payload filters directly to query_points.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from nexora.api.app import create_app
from nexora.auth.models import AuthenticatedUser
from nexora.auth.service import AuthenticationService
from nexora.database.models.base import Base
from nexora.database.models.collection import Collection
from nexora.database.models.document import Document, IndexingStatus
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.paper import Paper
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.paper import PaperRepository
from nexora.database.repository.user import UserRepository
from nexora.database.repository.workspace import WorkspaceRepository
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.service import DocumentPersistenceService
from nexora.embedding.hash_provider import HashEmbeddingProvider
from nexora.embedding.service import EmbeddingService
from nexora.rag.pipeline import INSUFFICIENT_EVIDENCE_ANSWER, create_rag_pipeline
from nexora.retrieval.in_memory import InMemoryVectorStore
from nexora.retrieval.models import VectorRecord
from nexora.retrieval.qdrant import QdrantVectorStore
from nexora.retrieval.qdrant_config import QdrantConfig
from nexora.api.models import ResearchVerifyRequest
from nexora.api.service import ResearchApplicationService
from nexora.research.models import ResearchAnswer
from nexora.research.query import ResearchQueryService
from nexora.research.verification import ClaimToEvidenceVerifier
from nexora.retrieval.evidence import EvidenceContextBuilder
from nexora.retrieval.quality import RetrievalQualityPolicy, RetrievalQualityService
from nexora.retrieval.service import RetrievalService


class FakeResearchService:
    def ask(self, request: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "text": "Fake research answer",
                "query": getattr(request, "query", "test"),
                "model": "fake-model",
                "evidence_count": 1,
            }
        )


class FakeAnalysisService:
    def analyze_paper(self, paper_id: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_id": str(paper_id),
                "title": "Fake Paper Title",
                "problem_and_objectives": {"summary": "Objectives"},
                "methodology_and_algorithms": {"summary": "Methodology"},
                "models_and_datasets": {"summary": "Dataset"},
                "experiments_and_metrics": {"summary": "Experiments"},
                "results_and_findings": {"summary": "Results"},
                "limitations": {"summary": "Limitations"},
                "future_work": {"summary": "Future work"},
                "evidence_count": 1,
                "model": "fake-model",
                "usage": None,
            }
        )


class FakeComparisonService:
    def compare_papers(self, paper_ids: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_ids": [str(p) for p in paper_ids],
                "dimensions": {"methods": {"claim": "c"}},
                "overall_summary": "Comparison summary",
                "model": "fake-model",
                "usage": None,
            }
        )

    def compare_cross_paper(self, paper_ids: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            to_dict=lambda: {
                "paper_ids": [str(p) for p in paper_ids],
                "paper_titles": {str(p): f"Title {p}" for p in paper_ids},
                "matrix": {"methods": {}},
                "conclusion": "Demonstrated improvement",
                "conclusion_reasoning": "Strong evidence",
                "comparability_status": "Comparable",
                "overall_summary": "Overall summary",
                "model": "fake-model",
                "usage": None,
            }
        )


# ---------------------------------------------------------------------------
# Database & Multi-Tenant Setup Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_tenant_env():
    """Sets up an in-memory SQLite DB with two separate users, workspaces, and collections."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    dim = 8
    provider = HashEmbeddingProvider(dimension=dim)
    embedding_service = EmbeddingService(provider)
    vector_store = InMemoryVectorStore(dimension=dim)
    retrieval_service = RetrievalService(embedding_service, vector_store)
    rag_pipeline = create_rag_pipeline(retrieval_service=retrieval_service, default_mode="vector")

    with session_factory() as session:
        user_repo = UserRepository(session)
        ws_repo = WorkspaceRepository(session)
        col_repo = CollectionRepository(session)
        key_repo = ApiKeyRepository(session)

        # 1. User A
        user_a = user_repo.create(email="user_a@example.com")
        ws_a = ws_repo.create(user_id=user_a.id, name="Workspace Alpha")
        col_a = col_repo.create(workspace_id=ws_a.id, name="Collection Alpha")
        key_a, raw_key_a = key_repo.generate_key(user_id=user_a.id, name="Key Alpha")

        # 2. User B
        user_b = user_repo.create(email="user_b@example.com")
        ws_b = ws_repo.create(user_id=user_b.id, name="Workspace Beta")
        col_b = col_repo.create(workspace_id=ws_b.id, name="Collection Beta")
        key_b, raw_key_b = key_repo.generate_key(user_id=user_b.id, name="Key Beta")

        # 3. User C (User with NO collections)
        user_c = user_repo.create(email="user_c@example.com")
        ws_c = ws_repo.create(user_id=user_c.id, name="Workspace Charlie")
        key_c, raw_key_c = key_repo.generate_key(user_id=user_c.id, name="Key Charlie")

        session.commit()

        # Capture IDs
        u_a_id, c_a_id = user_a.id, col_a.id
        u_b_id, c_b_id = user_b.id, col_b.id
        u_c_id = user_c.id

    return {
        "engine": engine,
        "session_factory": session_factory,
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "retrieval_service": retrieval_service,
        "rag_pipeline": rag_pipeline,
        "user_a_id": u_a_id,
        "col_a_id": c_a_id,
        "key_a": raw_key_a,
        "user_b_id": u_b_id,
        "col_b_id": c_b_id,
        "key_b": raw_key_b,
        "user_c_id": u_c_id,
        "key_c": raw_key_c,
    }


def _seed_paper_and_chunks(
    env: dict[str, Any],
    collection_id: uuid.UUID,
    title: str,
    chunk_texts: list[str],
) -> tuple[Paper, Document]:
    """Helper to seed paper, document, relational chunks, and vector store records."""
    sf = env["session_factory"]
    vs: InMemoryVectorStore = env["vector_store"]
    embed = env["embedding_service"]

    with sf() as session:
        paper_repo = PaperRepository(session)
        paper = paper_repo.create(
            collection_id=collection_id,
            title=title,
            abstract=f"Abstract of {title}",
        )
        session.flush()

        persistence = DocumentPersistenceService(session)
        proc_doc = ProcessedDocument(
            document_id=str(uuid.uuid4()),
            source=f"{title.lower().replace(' ', '_')}.pdf",
            chunks=[
                DocumentChunk(
                    text=t,
                    embedding=embed.embed_text(t),
                    index=i,
                )
                for i, t in enumerate(chunk_texts)
            ],
            metadata={"title": title, "collection_id": str(collection_id)},
        )
        doc = persistence.persist_processed_document(
            paper.id,
            proc_doc,
            filename=f"{title}.pdf",
            storage_key=f"storage/{title}.pdf",
        )
        raw_chunks = list(persistence.list_chunks_for_document(doc.id))
        session.commit()

        # Index into vector store with authoritative collection_id
        records = []
        for c in raw_chunks:
            vec = embed.embed_text(c.text)
            records.append(
                VectorRecord(
                    id=str(c.id),
                    vector=vec,
                    payload={
                        "document_id": str(doc.id),
                        "chunk_id": str(c.id),
                        "paper_id": str(paper.id),
                        "collection_id": str(collection_id),
                        "title": title,
                        "text": c.text,
                    },
                )
            )
        vs.upsert(records)
        return paper, doc


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_direct_retrieval_isolation_between_tenants(multi_tenant_env):
    """RetrievalService.retrieve respects collection boundaries and does not leak cross-tenant chunks."""
    env = multi_tenant_env
    retrieval: RetrievalService = env["retrieval_service"]

    # Seed User A's paper: Cryogenic Memory
    paper_a, _ = _seed_paper_and_chunks(
        env,
        env["col_a_id"],
        "Quantum Cryogenic Memory Architectures",
        ["Cryogenic flux qubit memory cells operating at 15 millikelvin with superconducting loops."],
    )

    # Seed User B's paper: Synaptic Plasticity
    paper_b, _ = _seed_paper_and_chunks(
        env,
        env["col_b_id"],
        "Neural Synaptic Plasticity Networks",
        ["Synaptic weight updates in spiking neural networks using spike-timing-dependent plasticity."],
    )

    # 1. Search scoped to Collection A
    res_a = retrieval.retrieve("memory neural quantum", collection_id=env["col_a_id"])
    assert len(res_a.results) > 0
    assert all(c.metadata["collection_id"] == str(env["col_a_id"]) for c in res_a.results)
    assert all("Cryogenic" in c.text for c in res_a.results)
    assert not any("Synaptic" in c.text for c in res_a.results)

    # 2. Search scoped to Collection B
    res_b = retrieval.retrieve("memory neural quantum", collection_id=env["col_b_id"])
    assert len(res_b.results) > 0
    assert all(c.metadata["collection_id"] == str(env["col_b_id"]) for c in res_b.results)
    assert all("Synaptic" in c.text for c in res_b.results)
    assert not any("Cryogenic" in c.text for c in res_b.results)

    # 3. Explicit empty collection scope fails closed
    res_empty = retrieval.retrieve("memory neural quantum", collection_ids=[])
    assert len(res_empty.results) == 0


def test_rag_pipeline_answer_scoping_and_fail_closed(multi_tenant_env):
    """RAGPipeline.answer isolates evidence and citations to authorized collections."""
    env = multi_tenant_env
    rag = env["rag_pipeline"]

    _seed_paper_and_chunks(
        env,
        env["col_a_id"],
        "Paper Alpha",
        ["Superconducting qubits in aluminum cavities demonstrate 100 microsecond coherence."],
    )
    _seed_paper_and_chunks(
        env,
        env["col_b_id"],
        "Paper Beta",
        ["Dopamine modulation in prefrontal cortex regulates working memory gating."],
    )

    # Query with Collection A scope: only Paper Alpha chunks cited
    resp_a = rag.answer(
        "What coherence or memory is described?",
        collection_ids=[str(env["col_a_id"])],
    )
    assert resp_a.evidence_count > 0
    assert all(s.metadata.get("collection_id") == str(env["col_a_id"]) for s in resp_a.sources)
    assert all("qubits" in s.text.lower() for s in resp_a.sources)

    # Query with Collection B scope: only Paper Beta chunks cited
    resp_b = rag.answer(
        "What coherence or memory is described?",
        collection_ids=[str(env["col_b_id"])],
    )
    assert resp_b.evidence_count > 0
    assert all(s.metadata.get("collection_id") == str(env["col_b_id"]) for s in resp_b.sources)
    assert all("dopamine" in s.text.lower() for s in resp_b.sources)

    # Query with empty collection scope: fails closed with insufficient evidence
    resp_empty = rag.answer(
        "What coherence or memory is described?",
        collection_ids=[],
    )
    assert resp_empty.evidence_count == 0
    assert resp_empty.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert resp_empty.sources == []


def test_api_rag_query_tenant_isolation_and_unauthorized_rejection(multi_tenant_env):
    """HTTP POST /api/v1/research/query strictly enforces tenant isolation and 403 on cross-access."""
    env = multi_tenant_env
    app = create_app(
        FakeResearchService(),
        rag_pipeline=env["rag_pipeline"],
        session_factory=env["session_factory"],
        enforce_auth=True,
    )
    client = TestClient(app)

    _seed_paper_and_chunks(
        env,
        env["col_a_id"],
        "Alpha Secret Research",
        ["Confidential quantum error correction code for Falcon-27 processor."],
    )
    paper_b, _ = _seed_paper_and_chunks(
        env,
        env["col_b_id"],
        "Beta Secret Research",
        ["Proprietary neuromorphic spike encoding algorithm on Loihi-2 chip."],
    )

    secret_a = "Confidential quantum error correction code for Falcon-27 processor."
    secret_b = "Proprietary neuromorphic spike encoding algorithm on Loihi-2 chip."

    # 1. User A queries unscoped: retrieves only User A's secret
    resp_a = client.post(
        "/api/v1/research/query",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={"query": secret_a, "retrieval_mode": "vector"},
    )
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert len(data_a["sources"]) > 0
    for src in data_a["sources"]:
        assert "Falcon" in src["text"]
        assert "Loihi" not in src["text"]

    # 2. User B queries unscoped: retrieves only User B's secret
    resp_b = client.post(
        "/api/v1/research/query",
        headers={"Authorization": f"Bearer {env['key_b']}"},
        json={"query": secret_b, "retrieval_mode": "vector"},
    )
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert len(data_b["sources"]) > 0
    for src in data_b["sources"]:
        assert "Loihi" in src["text"]
        assert "Falcon" not in src["text"]

    # 2b. User A queries unscoped using User B's secret text: receives NO sources from B
    resp_a_cross = client.post(
        "/api/v1/research/query",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={"query": secret_b, "retrieval_mode": "vector"},
    )
    assert resp_a_cross.status_code == 200
    assert not any("Loihi" in s["text"] for s in resp_a_cross.json()["sources"])

    # 3. User A attempts to explicitly query User B's collection: 403 Forbidden
    resp_forbidden = client.post(
        "/api/v1/research/query",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={
            "query": "neuromorphic spike",
            "collection_id": str(env["col_b_id"]),
            "retrieval_mode": "vector",
        },
    )
    assert resp_forbidden.status_code == 403
    assert "forbidden" in str(resp_forbidden.json()).lower()

    # 4. User A attempts to target User B's paper_id: 403 Forbidden
    resp_paper_forbidden = client.post(
        "/api/v1/research/query",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={
            "query": "neuromorphic spike",
            "paper_id": str(paper_b.id),
            "retrieval_mode": "vector",
        },
    )
    assert resp_paper_forbidden.status_code == 403
    assert "forbidden" in str(resp_paper_forbidden.json()).lower()


def test_api_rag_query_user_with_no_collections_fails_closed(multi_tenant_env):
    """User C has valid credentials but 0 collections; query must fail closed without leaking data."""
    env = multi_tenant_env
    app = create_app(
        FakeResearchService(),
        rag_pipeline=env["rag_pipeline"],
        session_factory=env["session_factory"],
        enforce_auth=True,
    )
    client = TestClient(app)

    _seed_paper_and_chunks(
        env,
        env["col_a_id"],
        "Alpha Paper",
        ["Quantum flux teleportation protocol."],
    )

    resp = client.post(
        "/api/v1/research/query",
        headers={"Authorization": f"Bearer {env['key_c']}"},
        json={"query": "quantum flux teleportation", "retrieval_mode": "vector"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert data["sources"] == []
    assert data["evidence_count"] == 0


def test_legacy_vector_without_collection_metadata_is_not_leaked(multi_tenant_env):
    """Vectors stored without collection_id in payload must fail filter match and never be exposed."""
    env = multi_tenant_env
    vs: InMemoryVectorStore = env["vector_store"]
    embed = env["embedding_service"]
    retrieval: RetrievalService = env["retrieval_service"]

    # Seed legitimate Paper A
    _seed_paper_and_chunks(
        env,
        env["col_a_id"],
        "Alpha Legitimate",
        ["Authorized Alpha text."],
    )

    # Inject legacy un-scoped vector into the store
    legacy_text = "Legacy un-scoped sensitive research chunk from old schema version."
    legacy_rec = VectorRecord(
        id=str(uuid.uuid4()),
        vector=embed.embed_text(legacy_text),
        payload={
            "document_id": "legacy-doc-1",
            "chunk_id": "legacy-chunk-1",
            "paper_id": "legacy-paper-1",
            # collection_id is intentionally omitted
            "text": legacy_text,
        },
    )
    vs.upsert([legacy_rec])

    # User A queries Collection A
    res_a = retrieval.retrieve(legacy_text, collection_id=env["col_a_id"])
    assert not any(c.text == legacy_text for c in res_a.results)

    # User B queries Collection B
    res_b = retrieval.retrieve(legacy_text, collection_id=env["col_b_id"])
    assert not any(c.text == legacy_text for c in res_b.results)


def test_cross_tenant_rejection_on_analysis_and_compare_routes(multi_tenant_env):
    """Analysis and comparison endpoints reject cross-tenant paper IDs with 403."""
    env = multi_tenant_env
    app = create_app(
        FakeResearchService(),
        rag_pipeline=env["rag_pipeline"],
        analysis_service=FakeAnalysisService(),
        comparison_service=FakeComparisonService(),
        session_factory=env["session_factory"],
        enforce_auth=True,
    )
    client = TestClient(app)

    paper_a, _ = _seed_paper_and_chunks(env, env["col_a_id"], "Paper Alpha", ["Alpha text."])
    paper_b, _ = _seed_paper_and_chunks(env, env["col_b_id"], "Paper Beta", ["Beta text."])

    # User A attempts to analyze Paper B: 403
    resp1 = client.post(
        "/api/v1/papers/analyze",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={"paper_id": str(paper_b.id)},
    )
    assert resp1.status_code == 403

    # User A attempts to analyze Paper B via path param: 403
    resp2 = client.post(
        f"/api/v1/papers/{paper_b.id}/analyze",
        headers={"Authorization": f"Bearer {env['key_a']}"},
    )
    assert resp2.status_code == 403

    # User A attempts to compare Paper A with Paper B: 403
    resp3 = client.post(
        "/api/v1/papers/compare",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={"paper_ids": [str(paper_a.id), str(paper_b.id)]},
    )
    assert resp3.status_code == 403

    # User A attempts cross-paper compare with Paper B: 403
    resp4 = client.post(
        "/api/v1/research/compare",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={"paper_ids": [str(paper_a.id), str(paper_b.id)]},
    )
    assert resp4.status_code == 403

    # User A attempts verification specifying Paper B: 403
    resp5 = client.post(
        "/api/v1/research/verify",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={"query": "test query", "expected_paper_id": str(paper_b.id)},
    )
    assert resp5.status_code == 403


def test_qdrant_vector_store_pushes_filter_to_query_points():
    """QdrantVectorStore.search builds and forwards models.Filter (MatchValue & MatchAny) to query_points."""
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_client.query_points.return_value = MagicMock(points=[])

    config = QdrantConfig(url="http://localhost:6333", collection_name="test_col")
    store = QdrantVectorStore(config, dimension=4, client=fake_client)

    # 1. Single collection ID: MatchValue
    store.search(
        [0.1, 0.2, 0.3, 0.4],
        limit=5,
        filter={"collection_id": "test-uuid-1234"},
    )
    fake_client.query_points.assert_called_once()
    call_kwargs = fake_client.query_points.call_args.kwargs
    query_filter = call_kwargs.get("query_filter")
    assert query_filter is not None
    assert hasattr(query_filter, "must")
    condition = query_filter.must[0]
    assert condition.key == "collection_id"
    assert condition.match.value == "test-uuid-1234"

    # 2. Multiple collection IDs: MatchAny
    fake_client.reset_mock()
    store.search(
        [0.1, 0.2, 0.3, 0.4],
        limit=5,
        filter={"collection_id": ["col-a", "col-b"]},
    )
    fake_client.query_points.assert_called_once()
    call_kwargs2 = fake_client.query_points.call_args.kwargs
    query_filter2 = call_kwargs2.get("query_filter")
    assert query_filter2 is not None
    assert hasattr(query_filter2, "must")
    cond_any = query_filter2.must[0]
    assert cond_any.key == "collection_id"
    assert hasattr(cond_any.match, "any")
    assert cond_any.match.any == ["col-a", "col-b"]

    # 3. Unfiltered search: query_filter is None
    fake_client.reset_mock()
    store.search([0.1, 0.2, 0.3, 0.4], limit=5)
    fake_client.query_points.assert_called_once()
    call_kwargs3 = fake_client.query_points.call_args.kwargs
    assert call_kwargs3.get("query_filter") is None


def test_research_verify_tenant_isolation_unscoped_query(multi_tenant_env):
    """Tenant B verifies a query without expected_paper_id: evidence must come only from Tenant B's collections."""
    env = multi_tenant_env
    retrieval_service = env["retrieval_service"]
    quality = RetrievalQualityService(retrieval_service, RetrievalQualityPolicy())
    evidence_builder = EvidenceContextBuilder(quality)

    class MockAnswerService:
        def answer(self, query: str, context: Any, **kwargs: Any) -> ResearchAnswer:
            cites = " ".join(f"[{item.label.strip('[]')}]" for item in context.items)
            text = f"Findings for {query}: {cites}" if cites else "No evidence found in authorized research collections."
            return ResearchAnswer(
                text=text,
                query=query,
                model="mock-model",
                usage={"prompt_tokens": 10, "completion_tokens": 10},
                evidence_count=len(context.items),
            )

    query_service = ResearchQueryService(evidence_builder, MockAnswerService())
    verifier = ClaimToEvidenceVerifier()
    research_service = ResearchApplicationService(query_service, verifier=verifier)

    app = create_app(
        research_service,
        rag_pipeline=env["rag_pipeline"],
        session_factory=env["session_factory"],
        enforce_auth=True,
    )
    client = TestClient(app)

    # Seed Paper A (User A, Collection Alpha) and Paper B (User B, Collection Beta)
    paper_a, _ = _seed_paper_and_chunks(
        env,
        env["col_a_id"],
        "Paper Alpha Secret",
        ["Quantum stealth breakthrough alpha unique proprietary findings."],
    )
    paper_b, _ = _seed_paper_and_chunks(
        env,
        env["col_b_id"],
        "Paper Beta Public",
        ["Neural network architecture beta unique proprietary findings."],
    )

    # Tenant B verifies a query matching "findings" without providing expected_paper_id
    resp_b = client.post(
        "/api/v1/research/verify",
        headers={"Authorization": f"Bearer {env['key_b']}"},
        json={"query": "findings"},
    )
    assert resp_b.status_code == 200
    data_b = resp_b.json()

    # Must have retrieved chunks from Collection Beta, NEVER Collection Alpha
    assert data_b["evidence_count"] > 0
    verification_b = data_b["verification"]
    for claim in verification_b.get("claims", []):
        for source in claim.get("sources", []):
            assert source.get("paper_id") == str(paper_b.id)
            assert source.get("paper_id") != str(paper_a.id)

    # Tenant A verifies the same query: gets Paper A, NEVER Paper B
    resp_a = client.post(
        "/api/v1/research/verify",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={"query": "findings"},
    )
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert data_a["evidence_count"] > 0
    for claim in data_a["verification"].get("claims", []):
        for source in claim.get("sources", []):
            assert source.get("paper_id") == str(paper_a.id)
            assert source.get("paper_id") != str(paper_b.id)


def test_research_verify_cross_tenant_expected_paper_rejected_with_403(multi_tenant_env):
    """Tenant B supplies Tenant A's expected_paper_id: rejected with 403 before retrieval."""
    env = multi_tenant_env
    retrieval_service = env["retrieval_service"]
    quality = RetrievalQualityService(retrieval_service, RetrievalQualityPolicy())
    evidence_builder = EvidenceContextBuilder(quality)

    class MockAnswerService:
        def answer(self, query: str, context: Any, **kwargs: Any) -> ResearchAnswer:
            return ResearchAnswer(text="answer", query=query, model="mock-model", usage=None, evidence_count=0)

    query_service = ResearchQueryService(evidence_builder, MockAnswerService())
    research_service = ResearchApplicationService(query_service)

    app = create_app(
        research_service,
        rag_pipeline=env["rag_pipeline"],
        session_factory=env["session_factory"],
        enforce_auth=True,
    )
    client = TestClient(app)

    paper_a, _ = _seed_paper_and_chunks(env, env["col_a_id"], "Paper Alpha", ["Alpha text content."])

    # User B passes Paper A as expected_paper_id -> 403 Forbidden
    resp = client.post(
        "/api/v1/research/verify",
        headers={"Authorization": f"Bearer {env['key_b']}"},
        json={"query": "Alpha content", "expected_paper_id": str(paper_a.id)},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"


def test_research_verify_user_with_no_collections_fails_closed(multi_tenant_env):
    """A tenant with zero collections receives insufficient evidence and cannot trigger unscoped retrieval."""
    env = multi_tenant_env
    retrieval_service = env["retrieval_service"]
    quality = RetrievalQualityService(retrieval_service, RetrievalQualityPolicy())
    evidence_builder = EvidenceContextBuilder(quality)

    class MockAnswerService:
        def answer(self, query: str, context: Any, **kwargs: Any) -> ResearchAnswer:
            return ResearchAnswer(
                text="Should not be called for empty collections",
                query=query,
                model="mock-model",
                usage=None,
                evidence_count=len(context.items),
            )

    query_service = ResearchQueryService(evidence_builder, MockAnswerService())
    research_service = ResearchApplicationService(query_service)

    app = create_app(
        research_service,
        rag_pipeline=env["rag_pipeline"],
        session_factory=env["session_factory"],
        enforce_auth=True,
    )
    client = TestClient(app)

    # Seed papers in Collection Alpha
    _seed_paper_and_chunks(env, env["col_a_id"], "Paper Alpha", ["Confidential alpha text."])

    # User C (owns 0 collections) calls /research/verify without text
    resp = client.post(
        "/api/v1/research/verify",
        headers={"Authorization": f"Bearer {env['key_c']}"},
        json={"query": "Confidential"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evidence_count"] == 0
    assert "No evidence found" in data["text"]
    assert data["verification"]["is_grounded"] is True

    # User C calls /research/verify with explicit text
    resp_text = client.post(
        "/api/v1/research/verify",
        headers={"Authorization": f"Bearer {env['key_c']}"},
        json={"query": "Confidential", "text": "Confidential alpha text is proven [E1]."},
    )
    assert resp_text.status_code == 200
    data_text = resp_text.json()
    assert data_text["evidence_count"] == 0
    assert data_text["verification"]["is_grounded"] is False
    assert data_text["verification"]["claims"][0]["status"] in ("insufficient", "unresolved")


def test_research_verify_authorized_expected_paper_succeeds(multi_tenant_env):
    """Authorized verification with a valid expected paper continues to work."""
    env = multi_tenant_env
    retrieval_service = env["retrieval_service"]
    quality = RetrievalQualityService(retrieval_service, RetrievalQualityPolicy())
    evidence_builder = EvidenceContextBuilder(quality)

    class MockAnswerService:
        def answer(self, query: str, context: Any, **kwargs: Any) -> ResearchAnswer:
            label = context.items[0].label if context.items else "[E1]"
            return ResearchAnswer(
                text=f"Valid answer with citation {label}.",
                query=query,
                model="mock-model",
                usage=None,
                evidence_count=len(context.items),
            )

    query_service = ResearchQueryService(evidence_builder, MockAnswerService())
    research_service = ResearchApplicationService(query_service)

    app = create_app(
        research_service,
        rag_pipeline=env["rag_pipeline"],
        session_factory=env["session_factory"],
        enforce_auth=True,
    )
    client = TestClient(app)

    paper_a, _ = _seed_paper_and_chunks(env, env["col_a_id"], "Paper Alpha", ["Alpha authoritative text."])

    # User A verifies with their own paper
    resp = client.post(
        "/api/v1/research/verify",
        headers={"Authorization": f"Bearer {env['key_a']}"},
        json={"query": "authoritative", "expected_paper_id": str(paper_a.id)},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evidence_count"] > 0
    for claim in data["verification"].get("claims", []):
        for source in claim.get("sources", []):
            assert source.get("paper_id") == str(paper_a.id)


def test_research_verify_service_collection_ids_and_paper_id_propagation(multi_tenant_env):
    """Direct invocation of ResearchApplicationService.verify respects collection_ids and paper_id."""
    env = multi_tenant_env
    retrieval_service = env["retrieval_service"]
    quality = RetrievalQualityService(retrieval_service, RetrievalQualityPolicy())
    evidence_builder = EvidenceContextBuilder(quality)

    class MockAnswerService:
        def answer(self, query: str, context: Any, **kwargs: Any) -> ResearchAnswer:
            return ResearchAnswer(
                text=f"Answer for {query}",
                query=query,
                model="test",
                usage=None,
                evidence_count=len(context.items),
            )

    query_service = ResearchQueryService(evidence_builder, MockAnswerService())
    service = ResearchApplicationService(query_service)

    paper_a, _ = _seed_paper_and_chunks(env, env["col_a_id"], "Alpha Paper", ["Specific alpha content chunk."])
    paper_b, _ = _seed_paper_and_chunks(env, env["col_b_id"], "Beta Paper", ["Specific beta content chunk."])

    # 1. Scoped to Collection A
    req_a = ResearchVerifyRequest(query="content", collection_ids=[str(env["col_a_id"])])
    res_a = service.verify(req_a)
    assert res_a.evidence_count > 0

    # 2. Scoped to empty collections -> fail closed
    req_empty = ResearchVerifyRequest(query="content", collection_ids=[])
    res_empty = service.verify(req_empty)
    assert res_empty.evidence_count == 0
    assert "No evidence found" in res_empty.text

    # 3. Scoped to specific paper_id
    req_paper = ResearchVerifyRequest(query="content", paper_id=str(paper_b.id))
    res_paper = service.verify(req_paper)
    assert res_paper.evidence_count > 0
