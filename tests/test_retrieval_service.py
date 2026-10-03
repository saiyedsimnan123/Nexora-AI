import pytest

from nexora.retrieval.models import RetrievedChunk, SearchResult
from nexora.retrieval.service import RetrievalService


class FakeEmbeddingService:
    def __init__(self, vector=None, error=None):
        self.vector = vector if vector is not None else [0.1, 0.2, 0.3]
        self.error = error
        self.queries = []

    def embed_text(self, text):
        self.queries.append(text)
        if self.error:
            raise self.error
        return self.vector


class FakeVectorStore:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def upsert(self, records):
        pass

    def delete(self, ids):
        pass

    def search(self, vector, *, limit=10):
        self.calls.append((vector, limit))
        if self.error:
            raise self.error
        return self.result

    def count(self):
        return 0


def make_result():
    chunk = RetrievedChunk(
        document_id="paper-1",
        chunk_id="paper-1#3",
        text="Transformers use attention.",
        score=0.87,
        metadata={"page": 4, "title": "Attention"},
    )
    return SearchResult(query="", results=[chunk])


def make_service(**kw):
    emb = FakeEmbeddingService(**kw.get("emb", {}))
    store = FakeVectorStore(**kw.get("store", {"result": make_result()}))
    return RetrievalService(emb, store), emb, store


def test_valid_construction():
    service, _, _ = make_service()
    assert isinstance(service, RetrievalService)


def test_invalid_dependencies_rejected():
    with pytest.raises(TypeError):
        RetrievalService(object(), FakeVectorStore())
    with pytest.raises(TypeError):
        RetrievalService(FakeEmbeddingService(), object())


def test_valid_query_accepted():
    service, emb, _ = make_service()
    service.retrieve("what is attention?")
    assert emb.queries == ["what is attention?"]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_queries_rejected(query):
    service, emb, store = make_service()
    with pytest.raises(ValueError):
        service.retrieve(query)
    assert emb.queries == [] and store.calls == []


@pytest.mark.parametrize("query", [None, 123, [], {}, b"bytes"])
def test_non_string_queries_rejected(query):
    service, emb, store = make_service()
    with pytest.raises(TypeError):
        service.retrieve(query)
    assert emb.queries == [] and store.calls == []


@pytest.mark.parametrize("limit", [1, 2, 5, 10, 50])
def test_valid_limits_accepted(limit):
    service, _, store = make_service()
    service.retrieve("q", limit=limit)
    assert store.calls[0][1] == limit


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limits_rejected(limit):
    service, emb, store = make_service()
    with pytest.raises(ValueError):
        service.retrieve("q", limit=limit)
    assert emb.queries == [] and store.calls == []


@pytest.mark.parametrize("limit", [True, False, 1.5, "10", None])
def test_wrong_type_limits_rejected(limit):
    service, emb, store = make_service()
    with pytest.raises(TypeError):
        service.retrieve("q", limit=limit)
    assert emb.queries == [] and store.calls == []


def test_vector_from_embedding_is_passed_to_store():
    vector = [0.5, -0.5, 0.25]
    service, emb, store = make_service(emb={"vector": vector})
    service.retrieve("graph neural networks", limit=3)
    assert emb.queries == ["graph neural networks"]
    assert store.calls == [(vector, 3)]
    assert store.calls[0][0] is vector


def test_default_limit_is_ten():
    service, _, store = make_service()
    service.retrieve("q")
    assert store.calls[0][1] == 10


def test_result_returned_unchanged():
    result = make_result()
    service, _, _ = make_service(store={"result": result})
    out = service.retrieve("q")
    assert out is result
    chunk = out.results[0]
    assert chunk.document_id == "paper-1"
    assert chunk.chunk_id == "paper-1#3"
    assert chunk.text == "Transformers use attention."
    assert chunk.score == 0.87
    assert chunk.metadata == {"page": 4, "title": "Attention"}


def test_embedding_failure_propagates_and_skips_search():
    err = RuntimeError("embedding down")
    service, _, store = make_service(emb={"error": err})
    with pytest.raises(RuntimeError) as info:
        service.retrieve("q")
    assert info.value is err
    assert store.calls == []


def test_vector_store_failure_propagates():
    err = ConnectionError("store down")
    service, _, _ = make_service(store={"error": err})
    with pytest.raises(ConnectionError) as info:
        service.retrieve("q")
    assert info.value is err
