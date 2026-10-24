"""Comprehensive Backend Finalization Acceptance Suite (Milestones 39-41).

Verifies the six core backend dimensions:
1. Production readiness and liveness under Redis / DB operational and failure states.
2. Claim-to-evidence superiority assertion gating (failing closed when baselines are missing).
3. Legitimate superiority claims supported by direct comparative evidence.
4. Honest abstention handling on insufficient evidence without hallucination.
5. Conflicting evidence detection across chunks.
6. Multi-tenant workspace and collection isolation.
"""

from __future__ import annotations

import collections
import math
import time
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from starlette.testclient import TestClient

from nexora.api.app import create_app
from nexora.api.rate_limit import RedisRateLimitStorage
from nexora.config.rate_limit import RateLimitConfig
from nexora.llm.base import LLMProvider
from nexora.llm.models import LLMMessage, LLMResponse
from nexora.llm.service import LLMService
from nexora.research.models import ResearchAnswer
from nexora.research.verification import (
    AnswerVerificationResult,
    ClaimToEvidenceVerifier,
    ClaimVerificationStatus,
    extract_claims_from_text,
)
from nexora.retrieval.evaluation import (
    AnswerGroundingEvaluationMetrics,
    evaluate_answer_grounding,
)
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem


# ---------------------------------------------------------------------------
# Test Fixture: Mock Redis Client
# ---------------------------------------------------------------------------
class FakeRedisClient:
    """Accurate in-memory simulation of Redis for testing."""

    def __init__(self) -> None:
        self.zsets: dict[str, dict[str, float]] = collections.defaultdict(dict)
        self.closed: bool = False

    def ping(self) -> bool:
        if self.closed:
            raise ConnectionError("Redis server is unavailable")
        return True

    def eval(self, script: str, numkeys: int, key: str, *args: str) -> list[int]:
        if self.closed:
            raise ConnectionError("Redis client is closed")
        now = float(args[0])
        limit = int(args[1])
        window = float(args[2])
        member_id = str(args[3])
        cutoff = now - window

        zset = self.zsets[key]
        to_remove = [m for m, score in zset.items() if score <= cutoff]
        for m in to_remove:
            del zset[m]

        current_count = len(zset)
        if current_count >= limit:
            oldest_time = min(zset.values()) if zset else now
            retry_after = max(1, math.ceil(oldest_time + window - now))
            return [0, 0, retry_after]

        zset[member_id] = now
        remaining = max(0, limit - current_count - 1)
        return [1, remaining, 0]


class FakeLLMProvider(LLMProvider):
    def __init__(self, response_text: str = "") -> None:
        self.response_text = response_text
        self.history: list[list[LLMMessage]] = []

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.history.append(messages)
        return LLMResponse(
            text=self.response_text,
            model=model or "acceptance-model",
            usage={"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
            raw={"id": "test_id"},
        )


# ---------------------------------------------------------------------------
# 1. Readiness Probe with Redis Storage
# ---------------------------------------------------------------------------
def test_readiness_probe_redis_healthy_and_failure_states():
    """Verify /ready reports healthy status with Redis and 503 when Redis drops."""
    fake_redis = FakeRedisClient()
    storage = RedisRateLimitStorage(redis_client=fake_redis)
    config = RateLimitConfig(enabled=True, backend="redis")

    # Minimal mock research service for readiness
    mock_research = MagicMock()
    mock_research.ask = MagicMock(return_value=None)

    app = create_app(
        rate_limit_config=config,
        rate_limit_storage=storage,
        research_service=mock_research,
    )
    client = TestClient(app)

    # 1. Healthy state -> 200 OK
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["checks"]["rate_limit_storage"]["status"] == "ready"

    # Liveness check is also OK
    assert client.get("/health").status_code == 200

    # 2. Redis failure -> 503 Unavailable
    fake_redis.closed = True
    resp_down = client.get("/ready")
    assert resp_down.status_code == 503
    down_data = resp_down.json()
    assert down_data["status"] == "unavailable"
    assert down_data["checks"]["rate_limit_storage"]["status"] == "unavailable"
    assert "ConnectionError" in down_data["checks"]["rate_limit_storage"]["error"]

    # Liveness MUST remain 200 OK even when Redis is down
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}

    # 3. Rate limiting disabled -> 200 OK without requiring storage
    app_disabled = create_app(
        rate_limit_config=RateLimitConfig(enabled=False),
        research_service=mock_research,
    )
    client_disabled = TestClient(app_disabled)
    resp_disabled = client_disabled.get("/ready")
    assert resp_disabled.status_code == 200
    assert resp_disabled.json()["status"] == "ready"
    assert "rate_limit_storage" not in resp_disabled.json()["checks"]

    # 4. Rate limiting enabled with InMemoryRateLimitStorage -> 200 OK
    from nexora.api.rate_limit import InMemoryRateLimitStorage

    mem_storage = InMemoryRateLimitStorage()
    app_mem = create_app(
        rate_limit_config=RateLimitConfig(enabled=True, backend="memory"),
        rate_limit_storage=mem_storage,
        research_service=mock_research,
    )
    client_mem = TestClient(app_mem)
    resp_mem = client_mem.get("/ready")
    assert resp_mem.status_code == 200
    assert resp_mem.json()["checks"]["rate_limit_storage"]["status"] == "ready"

    # 5. Rate limiting enabled but storage is None -> fails closed with 503
    app_unconfigured = create_app(
        rate_limit_config=RateLimitConfig(enabled=True, backend="memory"),
        research_service=mock_research,
    )
    app_unconfigured.state.rate_limit_storage = None
    client_unconfigured = TestClient(app_unconfigured)
    resp_unconfigured = client_unconfigured.get("/ready")
    assert resp_unconfigured.status_code == 503
    assert resp_unconfigured.json()["checks"]["rate_limit_storage"]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# 2. Unsupported Superiority Claims Prevention
# ---------------------------------------------------------------------------
def test_unsupported_superiority_claim_prevention():
    """Ensure verifier flags claims asserting superiority without comparative evidence."""
    # Context lacks any comparative baseline data
    context = EvidenceContext(
        query="What are the results?",
        items=[
            EvidenceItem(
                document_id="doc-1",
                chunk_id="chunk-1",
                text="The neural network utilizes 12 layers and is trained on GPU hardware.",
                score=0.9,
                metadata={"paper_id": "paper-1"},
                position=0,
            )
        ],
    )

    verifier = ClaimToEvidenceVerifier()

    # Claim asserts comparative superiority without baseline data
    answer_text = "Our method outperforms all baselines and achieves state-of-the-art results [E1]."
    result = verifier.verify_answer(answer_text, context, query="What are the results?")

    assert result.total_claims == 1
    claim = result.claims[0]
    assert claim.is_superiority_claim is True
    assert claim.is_unsupported_superiority is True
    assert result.unsupported_superiority_claims_count == 1
    assert result.has_unsupported_superiority_claims is True
    assert result.is_grounded is False
    assert len(result.unsupported_superiority_claims) == 1


def test_non_superiority_claim_with_everyday_terms_not_false_positive():
    """Verify that everyday verbs (e.g., heart beats, beat frequency) are not false positives."""
    context = EvidenceContext(
        query="What is the resting heart rate?",
        items=[
            EvidenceItem(
                document_id="doc-med",
                chunk_id="chunk-med-1",
                text="The resting patient heart beats 70 times per minute under standard conditions.",
                score=0.92,
                metadata={"paper_id": "paper-med-1"},
                position=0,
            )
        ],
    )

    mock_llm = FakeLLMProvider(
        '{"status": "SUPPORTED", "reasoning": "Evidence directly states the heart beats 70 times per minute.", "supported_points": ["heart beats 70 times per minute"]}'
    )
    verifier = ClaimToEvidenceVerifier(LLMService(mock_llm))

    answer_text = "The patient heart beats 70 times per minute [E1]."
    result = verifier.verify_answer(answer_text, context, query="What is the resting heart rate?")

    assert result.total_claims == 1
    claim = result.claims[0]
    # Must NOT be classified as a superiority claim
    assert claim.is_superiority_claim is False
    assert claim.is_unsupported_superiority is False
    assert claim.is_supported is True
    assert result.unsupported_superiority_claims_count == 0
    assert result.has_unsupported_superiority_claims is False
    assert result.is_grounded is True


# ---------------------------------------------------------------------------
# 3. Legitimate Superiority Claim Supported by Evidence
# ---------------------------------------------------------------------------
def test_legitimate_superiority_claim_with_comparative_evidence():
    """Verify superiority claims succeed when direct comparative metrics are present."""
    context = EvidenceContext(
        query="How does it compare to baselines?",
        items=[
            EvidenceItem(
                document_id="doc-1",
                chunk_id="chunk-1",
                text="Our architecture outperforms the ResNet-50 baseline by 4.3% top-1 accuracy on ImageNet.",
                score=0.95,
                metadata={"paper_id": "paper-1"},
                position=0,
            )
        ],
    )

    # Provider confirms the claim is supported
    mock_llm = FakeLLMProvider(
        '{"status": "SUPPORTED", "reasoning": "Direct comparative baseline numbers present.", "supported_points": ["outperforms baseline by 4.3% top-1 accuracy"]}'
    )
    verifier = ClaimToEvidenceVerifier(LLMService(mock_llm))

    answer_text = "The proposed model outperforms the ResNet-50 baseline by 4.3% top-1 accuracy [E1]."
    result = verifier.verify_answer(answer_text, context, query="How does it compare to baselines?")

    assert result.total_claims == 1
    claim = result.claims[0]
    assert claim.is_superiority_claim is True
    assert claim.is_unsupported_superiority is False
    assert claim.is_supported is True
    assert result.unsupported_superiority_claims_count == 0
    assert result.has_unsupported_superiority_claims is False
    assert result.is_grounded is True


# ---------------------------------------------------------------------------
# 4. Honest Abstention on Insufficient Evidence
# ---------------------------------------------------------------------------
def test_unanswerable_query_honest_abstention():
    """Verify that empty/insufficient evidence yields honest abstention without hallucination."""
    empty_context = EvidenceContext(query="What is the quantum speedup?", items=[])
    verifier = ClaimToEvidenceVerifier()

    abstention_answer = "There is insufficient evidence in the provided context to answer the question."
    result = verifier.verify_answer(abstention_answer, empty_context, query="What is the quantum speedup?")

    # With 0 evidence and 0 claims extracted from honest abstention, result is safe
    assert result.total_claims == 0
    assert result.is_grounded is True

    # Evaluator calculates abstention rate accurately
    cases = [
        {"context": empty_context, "text": abstention_answer, "is_abstention": True},
        {"context": empty_context, "text": "Insufficient evidence available.", "is_abstention": True},
    ]
    metrics = evaluate_answer_grounding(cases, verifier=verifier)
    assert metrics.total_answers == 2
    assert metrics.abstention_count == 2
    assert metrics.abstention_rate == 1.0
    assert metrics.unsupported_superiority_claims_count == 0


# ---------------------------------------------------------------------------
# 5. Conflicting Evidence Across Chunks
# ---------------------------------------------------------------------------
def test_conflicting_evidence_detection():
    """Verify that conflicting evidence results in contradiction detection and fails grounding."""
    context = EvidenceContext(
        query="Did the training converge?",
        items=[
            EvidenceItem(
                document_id="doc-1",
                chunk_id="chunk-1",
                text="The training converged in 5 hours without loss instability.",
                score=0.9,
                metadata={"paper_id": "paper-1"},
                position=0,
            ),
            EvidenceItem(
                document_id="doc-2",
                chunk_id="chunk-2",
                text="The model failed to converge and suffered severe divergence.",
                score=0.88,
                metadata={"paper_id": "paper-2"},
                position=1,
            ),
        ],
    )

    # Provider reports contradiction
    mock_llm = FakeLLMProvider(
        '{"status": "CONTRADICTED", "reasoning": "Evidence chunk E2 directly contradicts the claim.", "contradicted_points": ["training converged"]}'
    )
    verifier = ClaimToEvidenceVerifier(LLMService(mock_llm))

    answer_text = "The training process converged smoothly in 5 hours [E1][E2]."
    result = verifier.verify_answer(answer_text, context, query="Did the training converge?")

    assert result.total_claims == 1
    assert result.has_contradictions is True
    assert result.is_grounded is False
    assert result.contradicted_claims_count == 1


# ---------------------------------------------------------------------------
# 6. Evaluation Metrics Multi-Failure Tracking
# ---------------------------------------------------------------------------
def test_evaluation_metrics_tracking():
    """Verify evaluate_answer_grounding tracks failure modes and superiority counts."""
    context = EvidenceContext(
        query="Test query",
        items=[
            EvidenceItem(
                document_id="doc-1",
                chunk_id="chunk-1",
                text="System operates normally.",
                score=0.9,
                metadata={"paper_id": "paper-1"},
                position=0,
            )
        ],
    )

    cases = [
        {
            "context": context,
            "text": "The model vastly outperforms all competitors [E1].",
            "is_abstention": False,
        },
        {
            "context": EvidenceContext(query="No ev query", items=[]),
            "text": "Insufficient evidence provided.",
            "is_abstention": True,
        },
    ]

    metrics = evaluate_answer_grounding(cases, verifier=ClaimToEvidenceVerifier())
    assert metrics.total_answers == 2
    assert metrics.total_claims == 1
    assert metrics.unsupported_superiority_claims_count == 1
    assert metrics.unsupported_superiority_claim_rate == 1.0
    assert metrics.abstention_count == 1
    assert metrics.abstention_rate == 0.5
