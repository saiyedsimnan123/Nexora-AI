import dataclasses
from types import SimpleNamespace

import pytest

from nexora.config.settings import AppSettings, ConfigurationError
from nexora.llm.config import LLMConfig
from nexora.retrieval.qdrant_config import QdrantConfig
from nexora.retrieval.quality import RetrievalQualityPolicy

KEY = "placeholder-key-123"
EMBEDDING = SimpleNamespace(dimension=3)
BASE_ENV = {"NEXORA_LLM_MODEL": "m", "NEXORA_LLM_API_KEY": KEY}


def llm(**kw):
    base = dict(model="m", api_key=KEY)
    base.update(kw)
    return LLMConfig(**base)


def qdrant():
    return QdrantConfig(url="http://localhost:6333", api_key=KEY, collection_name="chunks")


def test_defaults_from_env():
    s = AppSettings.from_env(BASE_ENV, embedding=EMBEDDING)
    assert s.vector_store == "memory" and s.qdrant is None
    assert s.retrieval == RetrievalQualityPolicy()
    assert s.max_evidence_characters is None and s.embedding_dimension == 3


def test_env_loading_of_all_variables():
    env = dict(BASE_ENV, NEXORA_VECTOR_STORE=" Qdrant ", NEXORA_RETRIEVAL_MIN_SCORE="0.4",
               NEXORA_RETRIEVAL_MAX_RESULTS="5", NEXORA_RETRIEVAL_MAX_RESULTS_PER_DOCUMENT="2",
               NEXORA_RETRIEVAL_CANDIDATE_LIMIT="20", NEXORA_MAX_EVIDENCE_CHARACTERS="4000",
               NEXORA_LLM_TEMPERATURE="0.1")
    s = AppSettings.from_env(env, embedding=EMBEDDING, qdrant=qdrant())
    assert s.vector_store == "qdrant"
    assert (s.retrieval.min_score, s.retrieval.max_results) == (0.4, 5)
    assert (s.retrieval.max_results_per_document, s.retrieval.candidate_limit) == (2, 20)
    assert s.max_evidence_characters == 4000 and s.llm.temperature == 0.1


def test_explicit_construction_overrides():
    s = AppSettings(llm=llm(), embedding=EMBEDDING, vector_store="qdrant", qdrant=qdrant(),
                    retrieval=RetrievalQualityPolicy(max_results=2), max_evidence_characters=0)
    assert s.retrieval.max_results == 2 and s.max_evidence_characters == 0


def test_settings_are_frozen():
    s = AppSettings(llm=llm(), embedding=EMBEDDING)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.vector_store = "qdrant"


@pytest.mark.parametrize("var,value", [
    ("NEXORA_RETRIEVAL_MIN_SCORE", "abc"), ("NEXORA_RETRIEVAL_MAX_RESULTS", "1.5"),
    ("NEXORA_RETRIEVAL_MAX_RESULTS", "0"), ("NEXORA_RETRIEVAL_MAX_RESULTS", "-3"),
    ("NEXORA_RETRIEVAL_CANDIDATE_LIMIT", "x"), ("NEXORA_MAX_EVIDENCE_CHARACTERS", "lots"),
    ("NEXORA_MAX_EVIDENCE_CHARACTERS", "-1"), ("NEXORA_VECTOR_STORE", "postgres"),
    ("NEXORA_LLM_TIMEOUT", "fast"), ("NEXORA_LLM_TEMPERATURE", "9"),
])
def test_invalid_env_values_fail_clearly(var, value):
    with pytest.raises(ConfigurationError) as info:
        AppSettings.from_env(dict(BASE_ENV, **{var: value}), embedding=EMBEDDING)
    assert KEY not in str(info.value)


def test_missing_model_and_missing_openai_key():
    with pytest.raises(ConfigurationError):
        AppSettings.from_env({}, embedding=EMBEDDING)
    with pytest.raises(ConfigurationError) as info:
        AppSettings.from_env({"NEXORA_LLM_MODEL": "m"}, embedding=EMBEDDING)
    assert "api_key" in str(info.value)


def test_unsupported_llm_provider():
    with pytest.raises(ConfigurationError):
        AppSettings(llm=llm(provider="mystery"), embedding=EMBEDDING)


@pytest.mark.parametrize("embedding", [SimpleNamespace(dimension=0), SimpleNamespace(dimension=-2),
                                       SimpleNamespace(dimension=True), SimpleNamespace(dimension="3"),
                                       SimpleNamespace(), None])
def test_invalid_embedding_dimension(embedding):
    with pytest.raises(ConfigurationError):
        AppSettings(llm=llm(), embedding=embedding)


def test_qdrant_backend_requires_qdrant_config():
    with pytest.raises(ConfigurationError):
        AppSettings(llm=llm(), embedding=EMBEDDING, vector_store="qdrant")
    with pytest.raises(ConfigurationError):
        AppSettings(llm=llm(), embedding=EMBEDDING, vector_store="qdrant", qdrant={"url": "x"})


@pytest.mark.parametrize("kw", [{"llm": "m"}, {"retrieval": {"max_results": 2}},
                                {"max_evidence_characters": True}, {"max_evidence_characters": -5},
                                {"max_evidence_characters": 1.5}])
def test_other_invalid_fields(kw):
    base = dict(llm=llm(), embedding=EMBEDDING)
    base.update(kw)
    with pytest.raises(ConfigurationError):
        AppSettings(**base)


def test_secrets_never_appear_in_repr_or_str():
    s = AppSettings(llm=llm(), embedding=EMBEDDING, vector_store="qdrant", qdrant=qdrant())
    assert KEY not in repr(s) and KEY not in str(s)
    assert "AppSettings" in repr(s) and "'m'" in repr(s)


def test_from_env_reads_only_the_supplied_mapping(monkeypatch=None):
    # A different, empty mapping must not pick up values from the real environment.
    with pytest.raises(ConfigurationError):
        AppSettings.from_env({}, embedding=EMBEDDING)
