import dataclasses

import pytest

from nexora.retrieval.evidence import (
    EvidenceContext,
    EvidenceContextBuilder,
    EvidenceItem,
)
from nexora.retrieval.models import RetrievedChunk, SearchResult


def item(**kw):
    base = dict(document_id="d", chunk_id="c", text="t", score=0.5, metadata={}, position=0)
    base.update(kw)
    return EvidenceItem(**base)


def chunk(cid, text="text", score=0.5, doc="d", meta=None):
    return RetrievedChunk(
        document_id=doc, chunk_id=cid, text=text, score=score, metadata=meta or {"k": [1]}
    )


class FakeQuality:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return SearchResult(query=query, results=list(self.chunks))


def build(chunks, query="q", **kw):
    return EvidenceContextBuilder(FakeQuality(chunks), **kw).build(query)


def test_item_valid_and_frozen():
    i = item(score=1, position=3)
    assert i.position == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        i.text = "x"


@pytest.mark.parametrize("field,bad,exc", [
    ("document_id", "", ValueError), ("document_id", 1, TypeError),
    ("chunk_id", "  ", ValueError), ("chunk_id", None, TypeError),
    ("text", 5, TypeError),
    ("score", float("nan"), ValueError), ("score", float("inf"), ValueError),
    ("score", "0.5", TypeError), ("score", True, TypeError),
    ("metadata", [], TypeError), ("metadata", {1: "a"}, TypeError),
    ("position", -1, ValueError), ("position", 1.5, TypeError),
    ("position", True, TypeError), ("position", False, TypeError),
])
def test_item_invalid_fields(field, bad, exc):
    with pytest.raises(exc):
        item(**{field: bad})


def test_item_metadata_defensive_copy():
    meta = {"a": {"b": 1}}
    i = item(metadata=meta)
    meta["a"]["b"] = 2
    assert i.metadata == {"a": {"b": 1}}


def test_context_valid_and_list_copy():
    items = [item()]
    ctx = EvidenceContext(query="q", items=items, total_candidates=1)
    items.append(item(position=1))
    assert len(ctx.items) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.query = "x"


def test_context_validation():
    with pytest.raises(TypeError):
        EvidenceContext(query=1, items=[])
    with pytest.raises(TypeError):
        EvidenceContext(query="q", items="abc")
    with pytest.raises(TypeError):
        EvidenceContext(query="q", items=[object()])
    for bad, exc in [(-1, ValueError), (True, TypeError), (1.5, TypeError)]:
        with pytest.raises(exc):
            EvidenceContext(query="q", items=[], total_candidates=bad)


def test_builder_construction():
    EvidenceContextBuilder(FakeQuality())
    with pytest.raises(TypeError):
        EvidenceContextBuilder(object())


@pytest.mark.parametrize("bad", [-1, True, 1.5, "10"])
def test_invalid_character_limits(bad):
    with pytest.raises((TypeError, ValueError)):
        EvidenceContextBuilder(FakeQuality(), max_total_characters=bad)


@pytest.mark.parametrize("bad,exc", [("", ValueError), ("  ", ValueError), (None, TypeError), (3, TypeError)])
def test_invalid_query(bad, exc):
    fake = FakeQuality([chunk("1")])
    with pytest.raises(exc):
        EvidenceContextBuilder(fake).build(bad)
    assert fake.queries == []


def test_conversion_preserves_fields_order_and_positions():
    chunks = [chunk("1", "alpha", 0.9, "A", {"p": 1}), chunk("2", "beta", 0.4, "B", {"p": 2}), chunk("3", "", -1.0)]
    ctx = build(chunks, query="my query")
    assert ctx.query == "my query" and ctx.total_candidates == 3
    assert [i.position for i in ctx.items] == [0, 1, 2]
    for i, c in zip(ctx.items, chunks):
        assert (i.document_id, i.chunk_id, i.text, i.score, i.metadata) == (
            c.document_id, c.chunk_id, c.text, c.score, c.metadata)


def test_empty_retrieval():
    ctx = build([])
    assert ctx.items == [] and ctx.total_candidates == 0 and ctx.query == "q"


def test_metadata_independent_and_chunks_unmodified():
    c = chunk("1", meta={"nested": {"x": 1}})
    ctx = build([c])
    ctx.items[0].metadata["nested"]["x"] = 99
    assert c.metadata == {"nested": {"x": 1}}
    assert c.text == "text" and c.score == 0.5


def test_deterministic():
    chunks = [chunk("1"), chunk("2")]
    assert build(chunks) == build(chunks)


def test_retrieval_error_propagates_unchanged():
    err = RuntimeError("boom")
    with pytest.raises(RuntimeError) as info:
        EvidenceContextBuilder(FakeQuality(error=err)).build("q")
    assert info.value is err


def test_no_limit_keeps_everything():
    chunks = [chunk(str(n), "x" * 1000) for n in range(5)]
    assert len(build(chunks).items) == 5
    assert len(build(chunks, max_total_characters=None).items) == 5


def test_limit_keeps_whole_items_only():
    chunks = [chunk("1", "a" * 400), chunk("2", "b" * 300), chunk("3", "c" * 500)]
    ctx = build(chunks, max_total_characters=1000)
    assert [i.chunk_id for i in ctx.items] == ["1", "2"]
    assert sum(len(i.text) for i in ctx.items) == 700
    assert ctx.items[1].text == "b" * 300  # never truncated
    assert [i.position for i in ctx.items] == [0, 1]
    assert ctx.total_candidates == 3


def test_limit_boundaries():
    chunks = [chunk("1", "a" * 5), chunk("2", "b" * 5)]
    assert len(build(chunks, max_total_characters=10).items) == 2
    assert len(build(chunks, max_total_characters=9).items) == 1
    assert len(build(chunks, max_total_characters=4).items) == 0


def test_zero_limit_returns_no_items_but_counts_candidates():
    ctx = build([chunk("1", "a"), chunk("2", "b")], max_total_characters=0)
    assert ctx.items == [] and ctx.total_candidates == 2


def test_first_item_too_large_stops_without_skipping_ahead():
    chunks = [chunk("1", "a" * 50), chunk("2", "b")]
    assert build(chunks, max_total_characters=10).items == []
