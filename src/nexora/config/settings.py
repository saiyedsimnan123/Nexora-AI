"""Typed, immutable application settings composed from existing config objects."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from nexora.llm.config import LLMConfig
from nexora.retrieval.qdrant_config import QdrantConfig
from nexora.retrieval.quality import RetrievalQualityPolicy

SUPPORTED_VECTOR_STORES = ("memory", "qdrant")
SUPPORTED_LLM_PROVIDERS = ("openai",)


class ConfigurationError(ValueError):
    """Invalid application configuration. Messages never contain secret values."""


def _env_number(env: Mapping[str, str], name: str, convert: Any) -> Any:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return convert(raw.strip())
    except ValueError:
        raise ConfigurationError(f"{name} is not a valid number") from None


@dataclass(frozen=True, repr=False)
class AppSettings:
    """Everything the composition root needs.

    ``embedding`` is the existing embedding config object (anything with a
    positive integer ``dimension``); ``llm`` and ``qdrant`` are the existing
    LLMConfig / QdrantConfig. Nothing here connects to a service.
    """

    llm: LLMConfig
    embedding: Any
    vector_store: str = "memory"
    qdrant: QdrantConfig | None = None
    retrieval: RetrievalQualityPolicy = field(default_factory=RetrievalQualityPolicy)
    max_evidence_characters: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.llm, LLMConfig):
            raise ConfigurationError("llm must be an LLMConfig")
        if self.llm.provider.strip().lower() not in SUPPORTED_LLM_PROVIDERS:
            raise ConfigurationError(
                f"llm.provider must be one of {SUPPORTED_LLM_PROVIDERS}"
            )
        if self.llm.provider.strip().lower() == "openai" and not self.llm.api_key:
            raise ConfigurationError("llm.api_key is required for the openai provider")
        dimension = getattr(self.embedding, "dimension", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ConfigurationError("embedding.dimension must be a positive integer")
        if self.vector_store not in SUPPORTED_VECTOR_STORES:
            raise ConfigurationError(
                f"vector_store must be one of {SUPPORTED_VECTOR_STORES}"
            )
        if self.vector_store == "qdrant" and not isinstance(self.qdrant, QdrantConfig):
            raise ConfigurationError("qdrant config is required when vector_store is 'qdrant'")
        if not isinstance(self.retrieval, RetrievalQualityPolicy):
            raise ConfigurationError("retrieval must be a RetrievalQualityPolicy")
        limit = self.max_evidence_characters
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise ConfigurationError("max_evidence_characters must be a non-negative integer or None")

    @property
    def embedding_dimension(self) -> int:
        return self.embedding.dimension

    def __repr__(self) -> str:  # deliberately omits nested configs (they hold secrets)
        return (
            f"AppSettings(vector_store={self.vector_store!r}, "
            f"llm_provider={self.llm.provider!r}, llm_model={self.llm.model!r}, "
            f"embedding_dimension={self.embedding_dimension!r})"
        )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        embedding: Any,
        qdrant: QdrantConfig | None = None,
    ) -> AppSettings:
        """Load settings from the environment (read only here, never at import).

        LLM settings use ``LLMConfig.from_env``. The already-loaded embedding
        and Qdrant config objects are supplied by the caller so their own
        loaders stay the single source of truth for those variables.

        Variables: NEXORA_LLM_*, NEXORA_VECTOR_STORE (memory|qdrant),
        NEXORA_RETRIEVAL_MIN_SCORE, NEXORA_RETRIEVAL_MAX_RESULTS,
        NEXORA_RETRIEVAL_MAX_RESULTS_PER_DOCUMENT,
        NEXORA_RETRIEVAL_CANDIDATE_LIMIT, NEXORA_MAX_EVIDENCE_CHARACTERS.
        """
        env = os.environ if environ is None else environ
        try:
            llm = LLMConfig.from_env(env)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(str(exc)) from None

        policy_kwargs: dict[str, Any] = {}
        for var, key, convert in (
            ("NEXORA_RETRIEVAL_MIN_SCORE", "min_score", float),
            ("NEXORA_RETRIEVAL_MAX_RESULTS", "max_results", int),
            ("NEXORA_RETRIEVAL_MAX_RESULTS_PER_DOCUMENT", "max_results_per_document", int),
            ("NEXORA_RETRIEVAL_CANDIDATE_LIMIT", "candidate_limit", int),
        ):
            value = _env_number(env, var, convert)
            if value is not None:
                policy_kwargs[key] = value
        try:
            policy = RetrievalQualityPolicy(**policy_kwargs)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid retrieval settings: {exc}") from None

        backend = (env.get("NEXORA_VECTOR_STORE") or "memory").strip().lower() or "memory"
        return cls(
            llm=llm,
            embedding=embedding,
            vector_store=backend,
            qdrant=qdrant,
            retrieval=policy,
            max_evidence_characters=_env_number(env, "NEXORA_MAX_EVIDENCE_CHARACTERS", int),
        )
