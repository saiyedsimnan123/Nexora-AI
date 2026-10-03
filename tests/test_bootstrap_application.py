import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import nexora.bootstrap.application as bootstrap_module
import nexora.config.settings as settings_module
from nexora.api.models import ResearchRequest
from nexora.api.service import ResearchApplicationService
from nexora.bootstrap.application import build_http_app, build_research_application
from nexora.config.settings import AppSettings, ConfigurationError
from nexora.llm.models import LLMResponse
from nexora.llm.openai_provider import OpenAIProvider
from nexora.llm.config import LLMConfig
from nexora.retrieval.models import RetrievedChunk, SearchResult
from nexora.retrieval.qdrant import QdrantVectorStore
from nexora.retrieval.qdrant_config import QdrantConfig
from nexora.retrieval.quality import RetrievalQualityPolicy

KEY = "placeholder-key-123"


class FakeEmbedding:
    def __init__(self, dimension=None):
        if dimension is not None:
            self.dimension = dimension
        self.queries = []

    def embed_text(self, text):
        self.queries.append(text)
        return [1.0, 0.0, 0.0]


class FakeStore:
    def __init__(self, chunks=None, dimension=None):
        if dimension is not None:
            self.dimension = dimension
        self.chunks = chunks if chunks is not None else [
            RetrievedChunk(document_id="d1", chunk_id="c1", text="evidence text", score=0.9, metadata={})]
        self.searches = []

    def upsert(self, records): pass
    def delete(self, ids): pass
    def count(self): return len(self.chunks)

    def search(self, vector, *, limit=10):
        self.searches.append((vector, limit))
        return SearchResult(query="", results=list(self.chunks))


class FakeProvider:
    def __init__(self):
        self.calls = []

    def generate(self, messages, *, model=None, temperature=None, max_output_tokens=None):
        self.calls.append((messages, model, temperature, max_output_tokens))
        return LLMResponse(text="composed answer", model="fake-model", usage={"total_tokens": 3},
                           raw={"response_id": "r"})


class UntouchedClient:
    """Fails the test if any Qdrant client method is used during composition."""

    def __getattr__(self, name):
        raise AssertionError(f"unexpected client call: {name}")


def settings(**kw):
    base = dict(llm=LLMConfig(model="m", api_key=KEY), embedding=SimpleNamespace(dimension=3))
    base.update(kw)
    return AppSettings(**base)


def test_complete_composition_with_injected_fakes():
    store, provider, emb = FakeStore(), FakeProvider(), FakeEmbedding()
    app = build_research_application(
        settings(retrieval=RetrievalQualityPolicy(max_results=3, candidate_limit=12)),
        embedding_service=emb, vector_store=store, llm_provider=provider)
    assert isinstance(app, ResearchApplicationService)
    response = app.ask(ResearchRequest(query="What is X?", model="m2", temperature=0.2, max_output_tokens=50))
    assert response.text == "composed answer" and response.evidence_count == 1
    assert emb.queries == ["What is X?"]
    assert store.searches == [([1.0, 0.0, 0.0], 12)]  # candidate_limit from settings
    messages, model, temperature, tokens = provider.calls[0]
    assert (model, temperature, tokens) == ("m2", 0.2, 50)
    assert "evidence text" in messages[1].content


def test_evidence_character_limit_from_settings_is_applied():
    store = FakeStore(chunks=[RetrievedChunk(document_id="d", chunk_id="c", text="x" * 100, score=0.5, metadata={})])
    app = build_research_application(settings(max_evidence_characters=10), embedding_service=FakeEmbedding(),
                                     vector_store=store, llm_provider=FakeProvider())
    assert app.ask(ResearchRequest(query="q")).evidence_count == 0


def test_retrieval_policy_from_settings_filters_results():
    store = FakeStore(chunks=[
        RetrievedChunk(document_id="d", chunk_id="hi", text="keep", score=0.9, metadata={}),
        RetrievedChunk(document_id="d", chunk_id="lo", text="drop", score=0.1, metadata={})])
    app = build_research_application(settings(retrieval=RetrievalQualityPolicy(min_score=0.5)),
                                     embedding_service=FakeEmbedding(), vector_store=store,
                                     llm_provider=FakeProvider())
    assert app.ask(ResearchRequest(query="q")).evidence_count == 1


def test_qdrant_store_selected_with_embedding_dimension_and_no_client_calls():
    cfg = QdrantConfig(url="http://localhost:6333", api_key=KEY, collection_name="chunks")
    s = settings(vector_store="qdrant", qdrant=cfg)
    with patch("nexora.bootstrap.application.QdrantVectorStore", wraps=QdrantVectorStore) as spy:
        build_research_application(s, embedding_service=FakeEmbedding(), llm_provider=FakeProvider(),
                                   qdrant_client=UntouchedClient())
    (args, kwargs) = spy.call_args
    assert args[0] is cfg and kwargs["dimension"] == 3
    assert kwargs["client"] is not None


def test_injected_qdrant_client_is_passed_through_and_no_client_is_built():
    cfg = QdrantConfig(url="http://localhost:6333", api_key=KEY, collection_name="chunks")
    fake_client = UntouchedClient()
    with patch("nexora.bootstrap.application._build_qdrant_client") as builder, \
         patch("nexora.bootstrap.application.QdrantVectorStore", wraps=QdrantVectorStore) as spy:
        build_research_application(settings(vector_store="qdrant", qdrant=cfg),
                                   embedding_service=FakeEmbedding(), llm_provider=FakeProvider(),
                                   qdrant_client=fake_client)
    builder.assert_not_called()
    assert spy.call_args[1]["client"] is fake_client


def test_bootstrap_builds_the_qdrant_client_itself_when_none_is_injected():
    cfg = QdrantConfig(url="http://localhost:6333", api_key=KEY, collection_name="chunks")
    built = UntouchedClient()
    with patch("nexora.bootstrap.application._build_qdrant_client", return_value=built) as builder, \
         patch("nexora.bootstrap.application.QdrantVectorStore", wraps=QdrantVectorStore) as spy:
        build_research_application(settings(vector_store="qdrant", qdrant=cfg),
                                   embedding_service=FakeEmbedding(), llm_provider=FakeProvider())
    builder.assert_called_once_with(cfg)
    assert spy.call_args[1]["client"] is built  # the store never has to create its own


def test_memory_backend_never_builds_a_qdrant_client():
    with patch("nexora.bootstrap.application._build_qdrant_client") as builder, \
         patch("nexora.bootstrap.application.InMemoryVectorStore", FakeStore):
        build_research_application(settings(), embedding_service=FakeEmbedding(), llm_provider=FakeProvider())
    builder.assert_not_called()


def test_memory_store_selected_by_default():
    created = []

    class FakeInMemory(FakeStore):
        def __init__(self):
            super().__init__()
            created.append(self)

    with patch("nexora.bootstrap.application.InMemoryVectorStore", FakeInMemory):
        build_research_application(settings(), embedding_service=FakeEmbedding(), llm_provider=FakeProvider())
    assert len(created) == 1


def test_default_llm_provider_is_created_offline_from_settings():
    with patch("nexora.bootstrap.application.create_llm_provider") as factory:
        factory.return_value = FakeProvider()
        build_research_application(settings(), embedding_service=FakeEmbedding(), vector_store=FakeStore())
    assert factory.call_args[0][0].model == "m"


def test_real_openai_provider_construction_makes_no_network_call():
    from nexora.llm.service import create_llm_provider

    assert isinstance(create_llm_provider(LLMConfig(model="m", api_key=KEY)), OpenAIProvider)


@pytest.mark.parametrize("kw", [
    {"embedding_service": object()}, {"embedding_service": None},
    {"embedding_service": FakeEmbedding(), "vector_store": object()},
])
def test_invalid_injected_dependencies(kw):
    kw.setdefault("llm_provider", FakeProvider())
    with pytest.raises(ConfigurationError):
        build_research_application(settings(), **kw)


def test_invalid_settings_object():
    with pytest.raises(ConfigurationError):
        build_research_application({"llm": "x"}, embedding_service=FakeEmbedding())


def test_embedding_and_store_dimension_mismatch_rejected():
    with pytest.raises(ConfigurationError) as info:
        build_research_application(settings(), embedding_service=FakeEmbedding(dimension=8),
                                   vector_store=FakeStore(), llm_provider=FakeProvider())
    assert KEY not in str(info.value)
    with pytest.raises(ConfigurationError):
        build_research_application(settings(), embedding_service=FakeEmbedding(dimension=3),
                                   vector_store=FakeStore(dimension=1536), llm_provider=FakeProvider())
    build_research_application(settings(), embedding_service=FakeEmbedding(dimension=3),
                               vector_store=FakeStore(dimension=3), llm_provider=FakeProvider())


def test_modules_have_no_infrastructure_sdk_imports_or_globals():
    for module in (bootstrap_module, settings_module):
        source = inspect.getsource(module)
        for forbidden in ("import openai", "from openai", "\nglobal ", "_global_"):
            assert forbidden not in source
        # SDK imports must never run at import time (no unindented import lines).
        for line in source.splitlines():
            assert not line.startswith(("import qdrant_client", "from qdrant_client", "import openai"))
    assert "os.environ" not in inspect.getsource(bootstrap_module)


def test_http_app_composition_is_compatible_with_create_app():
    from fastapi.testclient import TestClient

    app = build_http_app(settings(), embedding_service=FakeEmbedding(), vector_store=FakeStore(),
                         llm_provider=FakeProvider())
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/api/v1/research", json={"query": "q"}).json()
    assert body["text"] == "composed answer" and "raw" not in body
