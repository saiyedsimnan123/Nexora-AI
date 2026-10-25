"""Comprehensive tests for Stage 3 Research Intelligence:
- Temporal research trend analysis across paper collections
- Evidence-grounded literature review synthesis
- API endpoints and tenant authorization
"""

import datetime
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from nexora.api.errors import (
    DocumentNotFoundError,
    InvalidRequestError,
    TrendServiceError,
)
from nexora.database.models.api_key import ApiKey
from nexora.database.models.user import User
from nexora.database.models.workspace import Workspace
from nexora.database.repository.api_key import ApiKeyRepository
from nexora.auth.service import AuthenticationService
from nexora.bootstrap.application import build_http_app, build_trend_service
from nexora.config.settings import AppSettings
from nexora.database.connection import create_database_engine
from nexora.database.models import Base
from nexora.database.models.collection import Collection
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.database.session import create_session_factory
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.research.analysis import AnalysisField, StructuredPaperAnalysis
from nexora.research.synthesis import (
    LiteratureSynthesisReport,
    SynthesisDivergence,
    SynthesisFinding,
    SynthesisLimitation,
)
from nexora.research.trend_service import ResearchTrendService
from nexora.research.trends import (
    ResearchTrendsReport,
    TemporalBucket,
    TrendCategory,
    TrendCitation,
    TrendItem,
    TrendTrajectory,
)
from nexora.retrieval.in_memory import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


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
def populated_trend_data(sqlite_session_factory):
    """Seed SQLite database with 3 papers spanning 2021, 2022, and 2023."""
    session = sqlite_session_factory()
    try:
        user_a = User(email="author-a@example.com")
        user_b = User(email="author-b@example.com")
        session.add_all([user_a, user_b])
        session.flush()

        workspace_a = Workspace(name="Workspace A", user_id=user_a.id)
        workspace_b = Workspace(name="Workspace B", user_id=user_b.id)
        session.add_all([workspace_a, workspace_b])
        session.flush()

        col_a = Collection(name="ML Trends Collection", workspace_id=workspace_a.id)
        col_b = Collection(name="Tenant B Collection", workspace_id=workspace_b.id)
        session.add_all([col_a, col_b])
        session.flush()

        paper_repo = PaperRepository(session)
        doc_repo = DocumentRepository(session)
        chunk_repo = DocumentChunkRepository(session)

        # Paper 1: 2021
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
        chunk_repo.create(
            document_id=d1.id,
            chunk_index=1,
            text="Evaluated on BLEU metric showing 28.4 BLEU score. Limitation: quadratic memory complexity.",
        )

        # Paper 2: 2022
        p2 = paper_repo.create(
            collection_id=col_a.id,
            title="Optimized BERT Pretraining",
            abstract="Improving Transformer robustness with AdamW.",
            publication_date=datetime.date(2022, 5, 10),
        )
        d2 = doc_repo.create(
            paper_id=p2.id,
            filename="paper2.pdf",
            file_size=1024,
            checksum="2" * 64,
            storage_key="s3://paper2.pdf",
            indexing_status="completed",
        )
        chunk_repo.create(
            document_id=d2.id,
            chunk_index=0,
            text="We evaluate Transformer models using AdamW optimization on GLUE benchmark datasets.",
        )
        chunk_repo.create(
            document_id=d2.id,
            chunk_index=1,
            text="Experiments show 88.5 accuracy metric. Limitation: high compute cost during pretraining.",
        )

        # Paper 3: 2023
        p3 = paper_repo.create(
            collection_id=col_a.id,
            title="Mamba Linear State Space Models",
            abstract="Selective state space architectures for linear-time sequence modeling.",
            publication_date=datetime.date(2023, 12, 1),
        )
        d3 = doc_repo.create(
            paper_id=p3.id,
            filename="paper3.pdf",
            file_size=1024,
            checksum="3" * 64,
            storage_key="s3://paper3.pdf",
            indexing_status="completed",
        )
        chunk_repo.create(
            document_id=d3.id,
            chunk_index=0,
            text="Mamba architecture introduces selective state spaces achieving linear scaling without attention.",
        )
        chunk_repo.create(
            document_id=d3.id,
            chunk_index=1,
            text="Evaluated on Pile benchmark achieving 5x throughput. Future work: multi-modal extensions.",
        )

        # Paper in Tenant B
        p_b = paper_repo.create(
            collection_id=col_b.id,
            title="Tenant B Private Paper",
            abstract="Confidential research.",
            publication_date=datetime.date(2023, 1, 1),
        )
        d_b = doc_repo.create(
            paper_id=p_b.id,
            filename="tenant_b.pdf",
            file_size=1024,
            checksum="b" * 64,
            storage_key="s3://tenant_b.pdf",
            indexing_status="completed",
        )
        chunk_repo.create(
            document_id=d_b.id,
            chunk_index=0,
            text="Confidential findings for Tenant B only.",
        )

        session.commit()

        return {
            "user_a_id": user_a.id,
            "user_b_id": user_b.id,
            "col_a_id": col_a.id,
            "col_b_id": col_b.id,
            "paper1_id": p1.id,
            "paper2_id": p2.id,
            "paper3_id": p3.id,
            "paper_b_id": p_b.id,
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Unit Tests: Models and Trajectory Logic
# ---------------------------------------------------------------------------

def test_trend_models():
    """Verify initialization and serialization of trend data models."""
    citation = TrendCitation(
        paper_id=str(uuid.uuid4()),
        chunk_id="chunk-1",
        evidence_label="[E1]",
        text_excerpt="Sample excerpt",
        period="2023",
    )
    assert citation.to_dict()["evidence_label"] == "[E1]"

    item = TrendItem(
        name="Transformer",
        category="architecture",
        trajectory="established",
        trajectory_reasoning="Used across all periods.",
        occurrences_by_period={"2021": 1, "2022": 2},
        citations=[citation.to_dict()],
        evidence_labels=["[E1]"],
    )
    d = item.to_dict()
    assert d["name"] == "Transformer"
    assert d["category"] == "architecture"
    assert d["trajectory"] == "established"
    assert d["total_occurrences"] == 3
    assert d["first_seen_period"] == "2021"
    assert d["last_seen_period"] == "2022"

    bucket = TemporalBucket(
        period="2023",
        paper_count=1,
        paper_ids=["pid-1"],
        paper_titles={"pid-1": "Title 1"},
        highlights=["Highlight 1"],
    )
    b_dict = bucket.to_dict()
    assert b_dict["period"] == "2023"
    assert b_dict["paper_count"] == 1

    report = ResearchTrendsReport(
        collection_id=str(uuid.uuid4()),
        paper_ids=["pid-1"],
        time_unit="year",
        temporal_buckets=[bucket],
        trends=[item],
        missing_date_paper_ids=[],
        summary="Trend report summary",
    )
    r_dict = report.to_dict()
    assert len(r_dict["temporal_buckets"]) == 1
    assert len(r_dict["trends"]) == 1
    assert r_dict["time_unit"] == "year"


def test_synthesis_models():
    """Verify initialization and serialization of synthesis data models."""
    finding = SynthesisFinding(
        claim="Transformers scale effectively with compute.",
        status="supported",
        evidence_labels=["[E1]"],
        sources=[{"chunk_id": "c1"}],
        supporting_paper_ids=["p1"],
    )
    assert finding.to_dict()["claim"].startswith("Transformers")

    divergence = SynthesisDivergence(
        topic="Compute Efficiency",
        description="Linear state spaces vs quadratic attention.",
        perspectives={"p1": "Attention", "p2": "State Space"},
        evidence_labels=["[E1]", "[E2]"],
    )
    assert divergence.to_dict()["topic"] == "Compute Efficiency"

    limitation = SynthesisLimitation(
        description="Quadratic memory complexity on long contexts.",
        affected_paper_ids=["p1"],
        evidence_labels=["[E1]"],
    )
    assert limitation.to_dict()["description"].startswith("Quadratic")

    report = LiteratureSynthesisReport(
        collection_id=str(uuid.uuid4()),
        paper_ids=["p1", "p2"],
        executive_summary="Executive synthesis.",
        common_findings=[finding],
        disagreements=[divergence],
        primary_methods=["Self-Attention", "State Space"],
        recurring_limitations=[limitation],
        open_research_gaps=["Long context scaling"],
        evidence_count=2,
        is_grounded=True,
    )
    r_dict = report.to_dict()
    assert r_dict["is_grounded"] is True
    assert len(r_dict["common_findings"]) == 1
    assert len(r_dict["disagreements"]) == 1


def test_trend_trajectory_classification_logic():
    """Test deterministic trajectory classification across various temporal distributions."""
    mock_llm = MagicMock()
    service = ResearchTrendService(mock_llm)

    # 1. Fewer than 2 valid periods in corpus -> INSUFFICIENT_DATA
    traj, reasoning = service._classify_trajectory({"2023": 5}, ["2023"])
    assert traj == TrendTrajectory.INSUFFICIENT_DATA.value
    assert "Fewer than 2 distinct temporal periods" in reasoning

    # 2. Present only in latest period (2023) across 2021-2023 -> EMERGING
    traj, reasoning = service._classify_trajectory({"2023": 2}, ["2021", "2022", "2023"])
    assert traj == TrendTrajectory.EMERGING.value
    assert "exclusively in the most recent period" in reasoning

    # 3. Present only in earliest period (2021) across 2021-2023 -> DECLINING
    traj, reasoning = service._classify_trajectory({"2021": 3}, ["2021", "2022", "2023"])
    assert traj == TrendTrajectory.DECLINING.value
    assert "absent from recent periods" in reasoning

    # 4. Present across multiple spanning periods up to latest -> ESTABLISHED
    traj, reasoning = service._classify_trajectory(
        {"2021": 1, "2022": 2, "2023": 2}, ["2021", "2022", "2023"]
    )
    assert traj == TrendTrajectory.ESTABLISHED.value
    assert "Consistently utilized across multiple temporal periods" in reasoning

    # 5. Present in 2021 and 2022, but absent in 2023 -> DECLINING
    traj, reasoning = service._classify_trajectory(
        {"2021": 2, "2022": 1}, ["2021", "2022", "2023"]
    )
    assert traj == TrendTrajectory.DECLINING.value
    assert "absent from the most recent period" in reasoning

    # 6. Single occurrence across corpus (< 2) -> INSUFFICIENT_DATA (inadequate sample size)
    traj, reasoning = service._classify_trajectory({"2023": 1}, ["2021", "2022", "2023"])
    assert traj == TrendTrajectory.INSUFFICIENT_DATA.value
    assert "Fewer than 2 total occurrences" in reasoning

    traj, reasoning = service._classify_trajectory({"2021": 1}, ["2021", "2022", "2023"])
    assert traj == TrendTrajectory.INSUFFICIENT_DATA.value
    assert "Fewer than 2 total occurrences" in reasoning


# ---------------------------------------------------------------------------
# Integration Tests: ResearchTrendService with Database
# ---------------------------------------------------------------------------

def test_research_trend_service_analysis_with_dates(sqlite_session_factory, populated_trend_data):
    """Test full temporal trend analysis across dated papers in SQLite database."""
    col_id = populated_trend_data["col_a_id"]

    mock_llm = MagicMock()
    # Mock LLM indicates mock mode for deterministic analysis
    type(mock_llm).__name__ = "MockLLMService"

    service = ResearchTrendService(
        mock_llm,
        session_factory=sqlite_session_factory,
    )

    report = service.analyze_collection_trends(col_id)

    assert report.collection_id == str(col_id)
    assert len(report.paper_ids) == 3
    assert len(report.temporal_buckets) == 3

    # Check temporal buckets: 2021, 2022, 2023
    bucket_periods = [b.period for b in report.temporal_buckets]
    assert bucket_periods == ["2021", "2022", "2023"]

    # Verify no missing dates in this dataset
    assert len(report.missing_date_paper_ids) == 0

    # Verify trends were identified and classified
    assert len(report.trends) > 0
    trajectories = {t.trajectory for t in report.trends}
    # Emerging or established or declining should be present
    assert any(t in trajectories for t in ["emerging", "established", "declining", "insufficient_data"])

    # Check citations provenance
    for trend in report.trends:
        if trend.citations:
            for cit in trend.citations:
                assert "paper_id" in cit
                assert "evidence_label" in cit
                assert cit["evidence_label"].startswith("[E")


def test_research_trend_service_missing_dates_honesty(sqlite_session_factory):
    """Test that papers lacking publication dates are honestly tracked and not fabricated."""
    session = sqlite_session_factory()
    try:
        user = User(email="undated@example.com")
        session.add(user)
        session.flush()
        ws = Workspace(name="WS", user_id=user.id)
        session.add(ws)
        session.flush()
        col = Collection(name="Undated Col", workspace_id=ws.id)
        session.add(col)
        session.flush()

        # Create paper without publication_date
        paper_repo = PaperRepository(session)
        doc_repo = DocumentRepository(session)
        chunk_repo = DocumentChunkRepository(session)

        p = paper_repo.create(
            collection_id=col.id,
            title="Undated Paper",
            abstract="No date here.",
            publication_date=None,  # No date
        )
        d = doc_repo.create(
            paper_id=p.id,
            filename="undated.pdf",
            file_size=100,
            checksum="a" * 64,
            storage_key="s3://u.pdf",
            indexing_status="completed",
        )
        chunk_repo.create(
            document_id=d.id,
            chunk_index=0,
            text="Methodology based on Diffusion models for image synthesis.",
        )
        session.commit()
        col_id = col.id
        paper_id = p.id
    finally:
        session.close()

    mock_llm = MagicMock()
    type(mock_llm).__name__ = "MockLLMService"

    service = ResearchTrendService(mock_llm, session_factory=sqlite_session_factory)
    report = service.analyze_collection_trends(col_id)

    # Missing date must be honestly tracked
    assert str(paper_id) in report.missing_date_paper_ids
    # Trajectories cannot be established without temporal span
    for t in report.trends:
        assert t.trajectory == TrendTrajectory.INSUFFICIENT_DATA.value


def test_literature_synthesis_service(sqlite_session_factory, populated_trend_data):
    """Test literature review synthesis generating grounded findings, divergences, and gaps."""
    col_id = populated_trend_data["col_a_id"]

    mock_llm = MagicMock()
    type(mock_llm).__name__ = "MockLLMService"

    service = ResearchTrendService(
        mock_llm,
        session_factory=sqlite_session_factory,
    )

    synthesis = service.synthesize_collection_literature(col_id, topic="Sequence Modeling")

    assert synthesis.collection_id == str(col_id)
    assert synthesis.topic == "Sequence Modeling"
    assert synthesis.is_grounded is True
    assert len(synthesis.paper_ids) == 3
    assert len(synthesis.executive_summary) > 0

    # Common findings
    assert len(synthesis.common_findings) > 0
    for finding in synthesis.common_findings:
        assert finding.status == "supported"
        assert len(finding.evidence_labels) > 0

    # Disagreements/divergences identified across different architectures
    assert len(synthesis.disagreements) > 0

    # Primary methods and recurring limitations
    assert len(synthesis.primary_methods) > 0
    assert len(synthesis.recurring_limitations) > 0


def test_empty_and_invalid_collection_errors(sqlite_session_factory):
    """Test error handling for non-existent, empty, or malformed collection requests."""
    mock_llm = MagicMock()
    service = ResearchTrendService(mock_llm, session_factory=sqlite_session_factory)

    # Malformed collection_id
    with pytest.raises(InvalidRequestError):
        service.analyze_collection_trends("not-a-uuid")

    # Non-existent collection
    random_col = uuid.uuid4()
    with pytest.raises(DocumentNotFoundError):
        service.analyze_collection_trends(random_col)

    # Empty collection
    session = sqlite_session_factory()
    try:
        user = User(email="empty@example.com")
        session.add(user)
        session.flush()
        ws = Workspace(name="Empty WS", user_id=user.id)
        session.add(ws)
        session.flush()
        empty_col = Collection(name="Empty Collection", workspace_id=ws.id)
        session.add(empty_col)
        session.commit()
        empty_col_id = empty_col.id
    finally:
        session.close()

    with pytest.raises(DocumentNotFoundError):
        service.analyze_collection_trends(empty_col_id)

    with pytest.raises(DocumentNotFoundError):
        service.synthesize_collection_literature(empty_col_id)


# ---------------------------------------------------------------------------
# API Integration & Tenant Authorization Tests
# ---------------------------------------------------------------------------

from nexora.api.app import create_app


def _build_test_api_client(sqlite_session_factory, enforce_auth=True):
    """Build a test HTTP app with auth enforcement and wired trend service."""
    vector_store = InMemoryVectorStore(dimension=128)
    auth_service = AuthenticationService(session_factory=sqlite_session_factory)

    mock_llm = MagicMock()
    type(mock_llm).__name__ = "MockLLMService"

    trend_service = ResearchTrendService(
        mock_llm,
        session_factory=sqlite_session_factory,
        vector_store=vector_store,
    )

    class DummyResearchService:
        def ask(self, req):
            pass

    app = create_app(
        DummyResearchService(),
        trend_service=trend_service,
        session_factory=sqlite_session_factory,
        auth_service=auth_service,
        enforce_auth=enforce_auth,
    )
    return TestClient(app)


def test_api_trends_and_synthesis_authorized_success(sqlite_session_factory, populated_trend_data):
    """Test successful 200 OK responses on /papers/trends and /papers/synthesis when authorized."""
    client = _build_test_api_client(sqlite_session_factory, enforce_auth=True)

    session = sqlite_session_factory()
    try:
        key_repo = ApiKeyRepository(session)
        key_a, raw_token_a = key_repo.generate_key(
            user_id=populated_trend_data["user_a_id"],
            name="Key A",
        )
    finally:
        session.close()

    col_a_id = str(populated_trend_data["col_a_id"])
    headers = {"Authorization": f"Bearer {raw_token_a}"}

    # 1. POST /api/v1/papers/trends
    resp_trends = client.post(
        "/api/v1/papers/trends",
        headers=headers,
        json={"collection_id": col_a_id, "time_unit": "year"},
    )
    assert resp_trends.status_code == 200
    data_trends = resp_trends.json()
    assert data_trends["collection_id"] == col_a_id
    assert len(data_trends["temporal_buckets"]) == 3
    assert len(data_trends["trends"]) > 0

    # 2. POST /api/v1/collections/{id}/trends convenience route
    resp_col_trends = client.post(
        f"/api/v1/collections/{col_a_id}/trends",
        headers=headers,
    )
    assert resp_col_trends.status_code == 200
    assert resp_col_trends.json()["collection_id"] == col_a_id

    # 3. POST /api/v1/papers/synthesis
    resp_synth = client.post(
        "/api/v1/papers/synthesis",
        headers=headers,
        json={"collection_id": col_a_id, "topic": "Sequence Modeling"},
    )
    assert resp_synth.status_code == 200
    data_synth = resp_synth.json()
    assert data_synth["collection_id"] == col_a_id
    assert data_synth["is_grounded"] is True
    assert len(data_synth["common_findings"]) > 0

    # 4. POST /api/v1/collections/{id}/synthesis convenience route
    resp_col_synth = client.post(
        f"/api/v1/collections/{col_a_id}/synthesis",
        headers=headers,
    )
    assert resp_col_synth.status_code == 200
    assert resp_col_synth.json()["collection_id"] == col_a_id


def test_api_trends_and_synthesis_tenant_isolation_401_and_403(
    sqlite_session_factory, populated_trend_data
):
    """Verify strict tenant isolation: 401 unauthenticated, 403 cross-tenant, and zero leakage."""
    client = _build_test_api_client(sqlite_session_factory, enforce_auth=True)

    session = sqlite_session_factory()
    try:
        key_repo = ApiKeyRepository(session)
        key_b, raw_token_b = key_repo.generate_key(
            user_id=populated_trend_data["user_b_id"],
            name="Key B",
        )
    finally:
        session.close()

    col_a_id = str(populated_trend_data["col_a_id"])
    headers_b = {"Authorization": f"Bearer {raw_token_b}"}

    # 1. Calling /papers/trends without credentials -> 401 Unauthorized
    resp_unauth_trends = client.post(
        "/api/v1/papers/trends",
        json={"collection_id": col_a_id},
    )
    assert resp_unauth_trends.status_code == 401
    assert resp_unauth_trends.json()["error"]["code"] == "unauthorized"

    # 2. Calling /papers/synthesis without credentials -> 401 Unauthorized
    resp_unauth_synth = client.post(
        "/api/v1/papers/synthesis",
        json={"collection_id": col_a_id},
    )
    assert resp_unauth_synth.status_code == 401
    assert resp_unauth_synth.json()["error"]["code"] == "unauthorized"

    # 3. User B calling /papers/trends on User A's collection -> 403 Forbidden
    resp_cross_trends = client.post(
        "/api/v1/papers/trends",
        headers=headers_b,
        json={"collection_id": col_a_id},
    )
    assert resp_cross_trends.status_code == 403
    assert resp_cross_trends.json()["error"]["code"] == "forbidden"
    # Verify zero UUID leakage in message
    assert col_a_id not in resp_cross_trends.json()["error"]["message"]

    # 4. User B calling /papers/synthesis on User A's collection -> 403 Forbidden
    resp_cross_synth = client.post(
        "/api/v1/papers/synthesis",
        headers=headers_b,
        json={"collection_id": col_a_id},
    )
    assert resp_cross_synth.status_code == 403
    assert resp_cross_synth.json()["error"]["code"] == "forbidden"
    assert col_a_id not in resp_cross_synth.json()["error"]["message"]

    # 5. User B calling path-based convenience endpoints on User A's collection -> 403 Forbidden
    resp_cross_col_trends = client.post(
        f"/api/v1/collections/{col_a_id}/trends",
        headers=headers_b,
    )
    assert resp_cross_col_trends.status_code == 403

    resp_cross_col_synth = client.post(
        f"/api/v1/collections/{col_a_id}/synthesis",
        headers=headers_b,
    )
    assert resp_cross_col_synth.status_code == 403


def test_api_trends_and_synthesis_request_validation(
    sqlite_session_factory, populated_trend_data
):
    """Test API request validation: invalid UUIDs, negative temperature, excessive papers."""
    client = _build_test_api_client(sqlite_session_factory, enforce_auth=True)

    session = sqlite_session_factory()
    try:
        key_repo = ApiKeyRepository(session)
        key_a, raw_token_a = key_repo.generate_key(
            user_id=populated_trend_data["user_a_id"],
            name="Key A",
        )
    finally:
        session.close()

    headers = {"Authorization": f"Bearer {raw_token_a}"}

    # Invalid UUID
    resp_bad_uuid = client.post(
        "/api/v1/papers/trends",
        headers=headers,
        json={"collection_id": "not-a-valid-uuid"},
    )
    assert resp_bad_uuid.status_code == 400

    # Negative temperature
    resp_bad_temp = client.post(
        "/api/v1/papers/trends",
        headers=headers,
        json={
            "collection_id": str(populated_trend_data["col_a_id"]),
            "temperature": -0.5,
        },
    )
    assert resp_bad_temp.status_code == 400

    # More than 10 papers
    many_pids = [str(uuid.uuid4()) for _ in range(11)]
    resp_too_many = client.post(
        "/api/v1/papers/trends",
        headers=headers,
        json={
            "collection_id": str(populated_trend_data["col_a_id"]),
            "paper_ids": many_pids,
        },
    )
    assert resp_too_many.status_code == 400



def test_synthesis_citation_integrity_and_unsupported_claim_filtering(
    sqlite_session_factory, populated_trend_data
):
    """Verify that live synthesis rejects fabricated citations and marks unsupported claims insufficient_evidence."""
    col_id = populated_trend_data["col_a_id"]

    mock_llm_response = {
        "executive_summary": "Testing citation integrity.",
        "common_findings": [
            {
                "claim": "Grounded finding supported by real evidence.",
                "status": "supported",
                "evidence_labels": ["[E1]"],
                "supporting_paper_ids": [str(populated_trend_data["paper1_id"])],
            },
            {
                "claim": "Ungrounded finding citing a non-existent chunk.",
                "status": "supported",
                "evidence_labels": ["[E999]"],  # Fabricated citation
                "supporting_paper_ids": ["unknown-pid"],
            },
        ],
        "disagreements": [
            {
                "topic": "Grounded Disagreement",
                "description": "Valid divergence between models.",
                "perspectives": {"p1": "A", "p2": "B"},
                "evidence_labels": ["[E1]", "[E2]"],
            },
            {
                "topic": "Fabricated Disagreement",
                "description": "Hallucinated debate without evidence.",
                "perspectives": {"p1": "A", "p2": "B"},
                "evidence_labels": ["[E999]"],  # Fabricated citation
            },
        ],
        "primary_methods": ["Transformer"],
        "recurring_limitations": [
            {
                "description": "Grounded limitation.",
                "affected_paper_ids": [str(populated_trend_data["paper1_id"])],
                "evidence_labels": ["[E1]"],
            },
            {
                "description": "Fabricated limitation.",
                "affected_paper_ids": ["unknown-pid"],
                "evidence_labels": ["[E999]"],  # Fabricated citation
            },
        ],
        "open_research_gaps": ["Real gap"],
        "limitations": ["Scope limitation"],
    }

    import json
    class ControllableLLM:
        def generate(self, messages, **kwargs):
            return LLMResponse(
                text=json.dumps(mock_llm_response),
                model="test-llm-v1",
                usage={"prompt_tokens": 100, "completion_tokens": 50},
            )

    service = ResearchTrendService(
        ControllableLLM(),
        session_factory=sqlite_session_factory,
    )

    synthesis = service.synthesize_collection_literature(col_id)

    # Finding 1: real citation -> supported
    f1 = synthesis.common_findings[0]
    assert f1.claim == "Grounded finding supported by real evidence."
    assert f1.status == "supported"
    assert len(f1.sources) == 1
    assert f1.sources[0]["evidence_label"] == "[E1]"

    # Finding 2: fabricated citation -> marked insufficient_evidence, sources empty
    f2 = synthesis.common_findings[1]
    assert f2.claim == "Ungrounded finding citing a non-existent chunk."
    assert f2.status == "insufficient_evidence"
    assert len(f2.sources) == 0

    # Disagreements: fabricated one is omitted
    assert len(synthesis.disagreements) == 1
    assert synthesis.disagreements[0].topic == "Grounded Disagreement"

    # Limitations: fabricated one is omitted
    assert len(synthesis.recurring_limitations) == 1
    assert synthesis.recurring_limitations[0].description == "Grounded limitation."


def test_api_fail_closed_without_session_factory():
    """Verify private trend and synthesis endpoints fail closed when database session factory is missing."""
    dummy_uuid = str(uuid.uuid4())

    class DummyResearchService:
        def ask(self, req):
            pass

    app = create_app(
        DummyResearchService(),
        session_factory=None,
        enforce_auth=False,
    )
    client = TestClient(app, raise_server_exceptions=False)

    resp_tr = client.post("/api/v1/papers/trends", json={"collection_id": dummy_uuid})
    assert resp_tr.status_code == 500
    assert resp_tr.json()["error"]["code"] == "server_misconfigured"

    resp_sy = client.post("/api/v1/papers/synthesis", json={"collection_id": dummy_uuid})
    assert resp_sy.status_code == 500
    assert resp_sy.json()["error"]["code"] == "server_misconfigured"
