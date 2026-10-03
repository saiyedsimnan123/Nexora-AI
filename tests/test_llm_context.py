import copy

import pytest

from nexora.llm.context import EvidenceContextFormatter
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem


def make_item(n, text=None, score=0.5, doc=None):
    return EvidenceItem(
        document_id=doc or f"paper-{n}",
        chunk_id=f"chunk-{n}",
        text=text if text is not None else f"evidence text {n}",
        score=score,
        metadata={"page": n},
        position=n - 1,
    )


def make_context(*items):
    return EvidenceContext(query="q", items=list(items), total_candidates=len(items))


def test_single_item_exact_format():
    out = EvidenceContextFormatter().format(make_context(make_item(1, score=0.87)))
    assert out == "[E1]\nDocument: paper-1\nChunk: chunk-1\nScore: 0.87\nevidence text 1"


def test_multiple_items_labels_and_order():
    out = EvidenceContextFormatter().format(make_context(make_item(1), make_item(2), make_item(3)))
    assert out.index("[E1]") < out.index("[E2]") < out.index("[E3]")
    assert out.count("[E") == 3
    assert out.split("\n\n")[1].startswith("[E2]")


def test_ids_scores_and_text_preserved_exactly():
    text = "  Line one.\n\n  Line two with [E9] and unicode é — and trailing space \n"
    out = EvidenceContextFormatter().format(make_context(make_item(1, text=text, score=-1.25, doc="doc/α")))
    assert text in out
    assert "Document: doc/α" in out and "Chunk: chunk-1" in out
    assert "Score: -1.25" in out


def test_metadata_not_included():
    assert "page" not in EvidenceContextFormatter().format(make_context(make_item(1)))


def test_empty_context():
    fmt = EvidenceContextFormatter()
    out = fmt.format(make_context())
    assert out == "No evidence was retrieved." and out == fmt.format(make_context())


@pytest.mark.parametrize("bad", [None, "text", [], {"items": []}, make_item(1)])
def test_invalid_context_type(bad):
    with pytest.raises(TypeError):
        EvidenceContextFormatter().format(bad)


def test_does_not_mutate_context():
    ctx = make_context(make_item(1), make_item(2))
    before = copy.deepcopy(ctx)
    EvidenceContextFormatter().format(ctx)
    assert ctx == before


def test_deterministic():
    ctx = make_context(make_item(1), make_item(2))
    assert EvidenceContextFormatter().format(ctx) == EvidenceContextFormatter().format(ctx)
