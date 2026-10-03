import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexora.retrieval.models import RetrievedChunk, SearchResult, VectorRecord
from nexora.retrieval.qdrant import QdrantStoreError, QdrantVectorStore
from nexora.retrieval.qdrant_config import QdrantConfig
from nexora.retrieval.store import VectorStore

SECRET = "super-secret-key"
ID1 = "doc-1#chunk-0"
ID2 = "doc-2#chunk-7"
UUID_ID = str(uuid.UUID(int=1))


class FakeClient:
    def __init__(self, exists=True, points=None, total=0, fail=False):
        self.exists = exists
        self.points = points or []
        self.total = total
        self.fail = fail
        self.calls = []

    def _rec(self, name, *args, **kwargs):
        if self.fail:
            raise ConnectionError(f"boom {SECRET}")
        self.calls.append((name, args, kwargs))

    def collection_exists(self, name):
        self._rec("collection_exists", name)
        return self.exists

    def create_collection(self, **kwargs):
        self._rec("create_collection", **kwargs)

    def upsert(self, name, points):
        self._rec("upsert", name, points=points)

    def delete(self, name, points_selector):
        self._rec("delete", name, points_selector=points_selector)

    def query_points(self, name, **kwargs):
        self._rec("query_points", name, **kwargs)
        return SimpleNamespace(points=self.points)

    def count(self, name, exact):
        self._rec("count", name, exact=exact)
        return SimpleNamespace(count=self.total)


def make_config(name="test_chunks"):
    return QdrantConfig(url="http://localhost:6333", api_key=SECRET, collection_name=name)


def make_store(client=None, dim=3, name="test_chunks"):
    client = client or FakeClient()
    return QdrantVectorStore(make_config(name), dimension=dim, client=client), client


def point(pid, score, payload):
    return SimpleNamespace(id=pid, score=score, payload=payload)


def test_import_does_not_create_client():
    import nexora.retrieval.qdrant as mod

    assert hasattr(mod, "QdrantVectorStore")


def test_constructor_validation():
    with pytest.raises(TypeError):
        QdrantVectorStore("not-config", dimension=3, client=FakeClient())
    store, client = make_store()
    assert store._client is client
    assert store.dimension == 3


@pytest.mark.parametrize("dim", [0, -1, True, 1.5, "3", None])
def test_invalid_dimensions(dim):
    with pytest.raises(ValueError):
        QdrantVectorStore(make_config(), dimension=dim, client=FakeClient())


def test_ensure_collection_exists_does_not_create():
    store, client = make_store(FakeClient(exists=True))
    store.ensure_collection()
    store.ensure_collection()
    assert [c[0] for c in client.calls] == ["collection_exists"] * 2


def test_ensure_collection_creates_with_size_cosine_and_name():
    store, client = make_store(FakeClient(exists=False), dim=5, name="my_coll")
    store.ensure_collection()
    assert client.calls[0][1] == ("my_coll",)
    _, _, kwargs = client.calls[1]
    assert kwargs["collection_name"] == "my_coll"
    assert kwargs["vectors_config"].size == 5
    assert str(kwargs["vectors_config"].distance.value).lower() == "cosine"


def test_upsert_converts_records_and_preserves_payload():
    store, client = make_store()
    payload = {"document_id": "d", "chunk_id": "c", "text": "hi", "extra": 1}
    store.upsert([VectorRecord(id=ID1, vector=[1.0, 0.0, 0.0], payload=payload)])
    name, args, kwargs = client.calls[0]
    assert name == "upsert" and args == ("test_chunks",)
    pt = kwargs["points"][0]
    assert pt.id != ID1
    assert str(uuid.UUID(pt.id)) == pt.id  # valid Qdrant point ID
    assert list(pt.vector) == [1.0, 0.0, 0.0]
    assert pt.payload == {**payload, "_nexora_id": ID1}
    assert "_nexora_id" not in payload  # caller's payload not mutated


def test_id_mapping_is_deterministic_and_uuid_passthrough():
    store, client = make_store()
    rec = VectorRecord(id=ID1, vector=[1.0, 0.0, 0.0], payload={})
    store.upsert([rec])
    store.upsert([rec])
    assert client.calls[0][2]["points"][0].id == client.calls[1][2]["points"][0].id
    store.upsert([VectorRecord(id=UUID_ID, vector=[1.0, 0.0, 0.0], payload={})])
    assert client.calls[2][2]["points"][0].id == UUID_ID


def test_requirements_include_qdrant_client():
    text = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text()
    assert "qdrant-client" in text


def test_upsert_validation():
    store, client = make_store()
    with pytest.raises(TypeError):
        store.upsert("nope")
    with pytest.raises(TypeError):
        store.upsert([{"id": ID1}])
    with pytest.raises(ValueError):
        store.upsert([VectorRecord(id=ID1, vector=[1.0, 2.0], payload={})])
    with pytest.raises(ValueError):
        store.upsert([VectorRecord(id=ID1, vector=[1.0, float("nan"), 0.0], payload={})])
    with pytest.raises(ValueError):
        store.upsert([VectorRecord(id="", vector=[1.0, 0.0, 0.0], payload={})])
    assert client.calls == []


def test_search_converts_points_and_keeps_order_and_scores():
    pts = [
        point(ID1, 0.9, {"document_id": "d1", "chunk_id": "c1", "text": "alpha"}),
        point(ID2, 0.4, {"document_id": "d2", "chunk_id": "c2", "text": "beta"}),
    ]
    store, client = make_store(FakeClient(points=pts))
    result = store.search([1.0, 0.0, 0.0], limit=2)
    assert isinstance(result, SearchResult) and result.query == ""
    assert all(isinstance(r, RetrievedChunk) for r in result.results)
    assert [r.score for r in result.results] == [0.9, 0.4]
    assert [r.text for r in result.results] == ["alpha", "beta"]
    assert [r.document_id for r in result.results] == ["d1", "d2"]
    assert [r.chunk_id for r in result.results] == ["c1", "c2"]
    assert result.results[0].metadata == pts[0].payload
    assert client.calls[0][2]["limit"] == 2


def test_search_fallbacks():
    qid = str(uuid.UUID(int=9))
    store, _ = make_store(FakeClient(points=[point(qid, 0.5, None)]))
    chunk = store.search([1.0, 0.0, 0.0]).results[0]
    assert chunk.document_id == qid and chunk.chunk_id == qid
    assert chunk.text == "" and chunk.metadata == {}


def test_round_trip_preserves_original_id():
    store, client = make_store()
    store.upsert([VectorRecord(id=ID1, vector=[1.0, 0.0, 0.0], payload={"text": "t"})])
    pt = client.calls[0][2]["points"][0]
    client.points = [point(pt.id, 0.8, pt.payload)]
    chunk = store.search([1.0, 0.0, 0.0]).results[0]
    assert chunk.document_id == ID1 and chunk.chunk_id == ID1
    assert chunk.text == "t" and chunk.score == 0.8
    assert chunk.metadata["_nexora_id"] == ID1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_limits(limit):
    store, _ = make_store()
    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0], limit=limit)


@pytest.mark.parametrize("vec", [[], [1.0], [1, 2, 3, 4], [1, float("inf"), 0], "abc", [True, 0, 0]])
def test_search_invalid_vectors(vec):
    store, client = make_store()
    with pytest.raises(ValueError):
        store.search(vec)
    assert client.calls == []


def test_delete_and_invalid_ids():
    store, client = make_store()
    store.delete([ID1, ID2])
    name, args, kwargs = client.calls[0]
    assert name == "delete"
    pts = kwargs["points_selector"].points
    assert len(pts) == 2 and all(str(uuid.UUID(p)) == p for p in pts)
    for bad in ([""], [5], ["  "]):
        with pytest.raises(ValueError):
            store.delete(bad)
    with pytest.raises(TypeError):
        store.delete(ID1)
    assert store.dimension == 3


def test_count_uses_client_count():
    store, client = make_store(FakeClient(total=7))
    assert store.count() == 7
    assert client.calls[0][2] == {"exact": True}


def test_infrastructure_error_is_distinct_and_hides_key():
    store, _ = make_store(FakeClient(fail=True))
    with pytest.raises(QdrantStoreError) as info:
        store.count()
    assert SECRET not in str(info.value)
    assert not isinstance(info.value, ValueError)


def test_satisfies_vector_store_contract():
    store, _ = make_store()
    for method in ("upsert", "delete", "search", "count"):
        assert callable(getattr(store, method))
    try:
        assert isinstance(store, VectorStore)
    except TypeError:
        pass  # protocol is not runtime_checkable
