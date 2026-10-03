import dataclasses

import pytest

from nexora.retrieval.models import RetrievedChunk, SearchResult
from nexora.retrieval.quality import RetrievalQualityPolicy, RetrievalQualityService


def chunk(doc, cid, score, text="t"):
    return RetrievedChunk(
        document_id=doc, chunk_id=cid, text=text, score=score, metadata={"k": cid}
    )


class FakeRetrieval:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
        self.calls = []

    def retrieve(self, query, *, limit=10):
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return SearchResult(query="", results=list(self.chunks))


def run(chunks, query="q", **policy):
    fake = FakeRetrieval(chunks)
    service = RetrievalQualityService(fake, RetrievalQualityPolicy(**policy))
    return service.retrieve(query), fake


def ids(result):
    return [c.chunk_id for c in result.results]


def test_valid_policy_and_immutable():
    p = RetrievalQualityPolicy(min_score=0.5, max_results=4, max_results_per_document=2, candidate_limit=8)
    assert (p.min_score, p.max_results, p.max_results_per_document, p.candidate_limit) == (0.5, 4, 2, 8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.max_results = 1
    assert RetrievalQualityPolicy(max_results=4).candidate_limit == 12
    assert RetrievalQualityPolicy(min_score=-3).min_score == -3


@pytest.mark.parametrize("bad", [True, "0.5", float("nan"), float("inf")])
def test_invalid_min_score(bad):
    with pytest.raises((TypeError, ValueError)):
        RetrievalQualityPolicy(min_score=bad)


@pytest.mark.parametrize("field", ["max_results", "max_results_per_document", "candidate_limit"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_invalid_integer_fields(field, bad):
    with pytest.raises((TypeError, ValueError)):
        RetrievalQualityPolicy(**{field: bad})


def test_candidate_limit_must_cover_max_results():
    with pytest.raises(ValueError):
        RetrievalQualityPolicy(max_results=5, candidate_limit=4)
    assert RetrievalQualityPolicy(max_results=5, candidate_limit=5).candidate_limit == 5


def test_constructor_rejects_bad_dependencies():
    with pytest.raises(TypeError):
        RetrievalQualityService(object(), RetrievalQualityPolicy())
    with pytest.raises(TypeError):
        RetrievalQualityService(FakeRetrieval(), {"max_results": 3})


def test_basic_passthrough_and_query_preserved():
    chunks = [chunk("a", "1", 0.9), chunk("b", "2", 0.8)]
    result, _ = run(chunks, query="my question")
    assert result.query == "my question"
    assert result.results == chunks


def test_candidate_limit_passed_to_retrieval():
    _, fake = run([], max_results=3, candidate_limit=20)
    assert fake.calls == [("q", 20)]


def test_min_score_filtering_with_boundary():
    chunks = [chunk("a", "1", 0.9), chunk("a", "2", 0.5), chunk("a", "3", 0.49)]
    assert ids(run(chunks, min_score=0.5)[0]) == ["1", "2"]


def test_no_threshold_when_min_score_none():
    chunks = [chunk("a", "1", -5.0), chunk("a", "2", 0.0)]
    assert ids(run(chunks)[0]) == ["1", "2"]


def test_duplicate_chunk_ids_keep_first():
    first, second = chunk("a", "1", 0.9, "first"), chunk("a", "1", 0.8, "second")
    result, _ = run([first, second, chunk("b", "2", 0.7)])
    assert ids(result) == ["1", "2"]
    assert result.results[0] is first


def test_duplicate_chunk_id_across_documents_is_duplicate():
    result, _ = run([chunk("a", "1", 0.9), chunk("b", "1", 0.8)])
    assert [c.document_id for c in result.results] == ["a"]


def test_per_document_limit_example():
    chunks = [
        chunk("A", "A1", 0.9), chunk("A", "A2", 0.8), chunk("A", "A3", 0.7),
        chunk("B", "B1", 0.6), chunk("C", "C1", 0.5),
    ]
    result, _ = run(chunks, max_results=4, max_results_per_document=2)
    assert ids(result) == ["A1", "A2", "B1", "C1"]


def test_max_results_enforced():
    chunks = [chunk("a", str(i), 1.0 - i / 10) for i in range(6)]
    assert ids(run(chunks, max_results=3)[0]) == ["0", "1", "2"]


def test_skipped_candidates_do_not_consume_result_slots():
    chunks = [chunk("a", "1", 0.1), chunk("a", "2", 0.9), chunk("a", "3", 0.8)]
    assert ids(run(chunks, min_score=0.5, max_results=2)[0]) == ["2", "3"]


def test_empty_and_fully_filtered_results():
    result, _ = run([], query="x")
    assert result.query == "x" and result.results == []
    result, _ = run([chunk("a", "1", 0.1)], min_score=0.9)
    assert result.results == []


def test_inputs_not_mutated_and_order_deterministic():
    chunks = [chunk("b", "2", 0.9), chunk("a", "1", 0.8), chunk("b", "3", 0.7)]
    snapshot = [dataclasses.replace(c, metadata=dict(c.metadata)) for c in chunks]
    first, _ = run(chunks, max_results_per_document=1)
    second, _ = run(chunks, max_results_per_document=1)
    assert ids(first) == ids(second) == ["2", "1"]
    assert chunks == snapshot
    assert first.results[0] is chunks[0]


def test_retrieval_errors_propagate_unchanged():
    err = RuntimeError("boom")
    service = RetrievalQualityService(FakeRetrieval(error=err), RetrievalQualityPolicy())
    with pytest.raises(RuntimeError) as info:
        service.retrieve("q")
    assert info.value is err
