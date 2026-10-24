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


def test_missing_openai_key_behavior():
    # In default offline/mock mode, AppSettings does not fail solely because an API key is missing
    s = AppSettings.from_env({}, embedding=EMBEDDING)
    assert s.require_llm_key is False
    assert s.llm.api_key is None
    assert s.llm.model == "gpt-4o-mini"

    # When live mode or require_llm_key is explicitly requested, missing API key fails clearly
    with pytest.raises(ConfigurationError) as info:
        AppSettings.from_env({"NEXORA_RAG_GENERATOR_MODE": "live"}, embedding=EMBEDDING)
    assert "api_key" in str(info.value)

    with pytest.raises(ConfigurationError) as info2:
        AppSettings.from_env({"NEXORA_REQUIRE_LLM_KEY": "true"}, embedding=EMBEDDING)
    assert "api_key" in str(info2.value)


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


def test_from_env_reads_only_the_supplied_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    # A supplied mapping requiring live generation must not pick up api_key from real os.environ.
    monkeypatch.setenv("NEXORA_LLM_API_KEY", "real-env-secret-key")
    with pytest.raises(ConfigurationError) as exc_info:
        AppSettings.from_env({"NEXORA_RAG_GENERATOR_MODE": "live"}, embedding=EMBEDDING)
    assert "api_key" in str(exc_info.value)


def test_defaults_from_env_enforces_auth_by_default():
    s = AppSettings.from_env(BASE_ENV, embedding=EMBEDDING)
    assert s.enforce_auth is True


@pytest.mark.parametrize("raw,expected", [
    ("false", False), ("0", False), ("no", False), ("off", False), (" False ", False),
    ("true", True), ("1", True), ("yes", True), ("on", True), (" TRUE ", True),
])
def test_env_loading_of_enforce_auth(raw, expected):
    env = dict(BASE_ENV, NEXORA_ENFORCE_AUTH=raw)
    s = AppSettings.from_env(env, embedding=EMBEDDING)
    assert s.enforce_auth is expected


@pytest.mark.parametrize("bad_val", ["not_a_bool", "maybe", "2", "-1", "enabled", "disabled"])
def test_env_loading_of_invalid_enforce_auth_fails(bad_val):
    env = dict(BASE_ENV, NEXORA_ENFORCE_AUTH=bad_val)
    with pytest.raises(ConfigurationError) as exc:
        AppSettings.from_env(env, embedding=EMBEDDING)
    assert "NEXORA_ENFORCE_AUTH must be one of" in str(exc.value)


@pytest.mark.parametrize("bad_type", ["true", 1, 0, None, [True]])
def test_explicit_non_bool_enforce_auth_fails(bad_type):
    with pytest.raises(ConfigurationError) as exc:
        AppSettings(llm=llm(), embedding=EMBEDDING, enforce_auth=bad_type)
    assert "enforce_auth must be a boolean" in str(exc.value)


def test_cors_and_rate_limit_defaults_on_app_settings():
    from nexora.config.cors import CORSConfig
    from nexora.config.rate_limit import RateLimitConfig

    s = AppSettings(llm=llm(), embedding=EMBEDDING)
    assert isinstance(s.cors, CORSConfig)
    assert s.cors.enabled is False
    assert s.cors.allow_origins == ()

    assert isinstance(s.rate_limit, RateLimitConfig)
    assert s.rate_limit.enabled is False
    assert s.rate_limit.heavy_analysis_limit == 10


def test_cors_and_rate_limit_invalid_types_raise_configuration_error():
    with pytest.raises(ConfigurationError) as exc:
        AppSettings(llm=llm(), embedding=EMBEDDING, cors="not_a_cors_config")
    assert "cors must be a CORSConfig" in str(exc.value)

    with pytest.raises(ConfigurationError) as exc:
        AppSettings(llm=llm(), embedding=EMBEDDING, rate_limit="not_a_rate_limit_config")
    assert "rate_limit must be a RateLimitConfig" in str(exc.value)


def test_env_loading_of_cors_and_rate_limit():
    env = dict(
        BASE_ENV,
        NEXORA_CORS_ALLOWED_ORIGINS="http://localhost:3000",
        NEXORA_CORS_ALLOW_CREDENTIALS="false",
        NEXORA_RATE_LIMIT_ENABLED="true",
        NEXORA_RATE_LIMIT_HEAVY_ANALYSIS="7",
    )
    s = AppSettings.from_env(env, embedding=EMBEDDING)
    assert s.cors.enabled is True
    assert s.cors.allow_origins == ("http://localhost:3000",)
    assert s.rate_limit.enabled is True
    assert s.rate_limit.heavy_analysis_limit == 7


def test_database_config_on_app_settings():
    from nexora.database.config import DatabaseConfig

    db = DatabaseConfig(url="postgresql://localhost/db")
    s = AppSettings(llm=llm(), embedding=EMBEDDING, database=db)
    assert s.database is db


def test_database_config_invalid_type_raises():
    with pytest.raises(ConfigurationError) as exc:
        AppSettings(llm=llm(), embedding=EMBEDDING, database="invalid")
    assert "database must be a DatabaseConfig" in str(exc.value)


def test_env_loading_of_database_config():
    env = dict(
        BASE_ENV,
        NEXORA_DATABASE_URL="postgresql://localhost/db",
        NEXORA_DATABASE_POOL_SIZE="12",
    )
    s = AppSettings.from_env(env, embedding=EMBEDDING)
    assert s.database is not None
    assert s.database.url == "postgresql://localhost/db"
    assert s.database.pool_size == 12


def test_env_loading_of_database_config_fallback():
    env = dict(
        BASE_ENV,
        DATABASE_URL="postgresql://localhost/db",
        DATABASE_POOL_SIZE="15",
    )
    s = AppSettings.from_env(env, embedding=EMBEDDING)
    assert s.database is not None
    assert s.database.url == "postgresql://localhost/db"
    assert s.database.pool_size == 15

