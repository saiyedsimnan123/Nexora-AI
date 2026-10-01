import pytest

from nexora.document.chunker import chunk_text


def _reconstruct(chunks: list[str], overlap: int) -> str:
    """Rebuild the original text by dropping each chunk's overlapped prefix."""
    return chunks[0] + "".join(chunk[overlap:] for chunk in chunks[1:])


def test_empty_input_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_input_returns_empty_list():
    assert chunk_text("   \n\t  \n") == []


def test_short_text_produces_one_chunk():
    assert chunk_text("Short text.", chunk_size=100, chunk_overlap=10) == [
        "Short text."
    ]


def test_text_exactly_chunk_size_produces_one_chunk():
    text = "a" * 50
    assert chunk_text(text, chunk_size=50, chunk_overlap=10) == [text]


def test_long_text_is_split():
    text = "abcdefghij" * 10
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 1


def test_every_chunk_respects_chunk_size():
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=64, chunk_overlap=16)
    assert all(0 < len(chunk) <= 64 for chunk in chunks)


def test_original_text_order_is_preserved():
    text = "".join(str(i % 10) for i in range(300))
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
    assert _reconstruct(chunks, 10) == text


def test_consecutive_chunks_share_the_overlap():
    text = "".join(chr(97 + i % 26) for i in range(200))
    overlap = 12
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=overlap)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-overlap:] == current[:overlap]


def test_zero_overlap_splits_without_sharing():
    text = "abcdefghij"
    assert chunk_text(text, chunk_size=4, chunk_overlap=0) == [
        "abcd",
        "efgh",
        "ij",
    ]


def test_last_chunk_is_not_a_duplicate_of_previous_tail():
    chunks = chunk_text("abcdefghij", chunk_size=5, chunk_overlap=2)
    assert chunks == ["abcde", "defgh", "ghij"]


@pytest.mark.parametrize("bad_size", [0, -1, -100])
def test_non_positive_chunk_size_raises_value_error(bad_size):
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("some text", chunk_size=bad_size, chunk_overlap=0)


@pytest.mark.parametrize("bad_size", [10.5, "100", None, True])
def test_non_int_chunk_size_raises_type_error(bad_size):
    with pytest.raises(TypeError, match="chunk_size"):
        chunk_text("some text", chunk_size=bad_size, chunk_overlap=0)


def test_negative_chunk_overlap_raises_value_error():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("some text", chunk_size=10, chunk_overlap=-1)


@pytest.mark.parametrize("bad_overlap", [2.5, "2", None, False])
def test_non_int_chunk_overlap_raises_type_error(bad_overlap):
    with pytest.raises(TypeError, match="chunk_overlap"):
        chunk_text("some text", chunk_size=10, chunk_overlap=bad_overlap)


@pytest.mark.parametrize("overlap", [10, 11, 50])
def test_overlap_not_smaller_than_chunk_size_raises_value_error(overlap):
    with pytest.raises(ValueError, match="smaller than"):
        chunk_text("some text", chunk_size=10, chunk_overlap=overlap)


@pytest.mark.parametrize("bad_text", [None, 123, ["text"], b"text"])
def test_non_string_text_raises_type_error(bad_text):
    with pytest.raises(TypeError, match="text must be a str"):
        chunk_text(bad_text)


def test_technical_text_is_preserved_exactly():
    text = (
        "Accuracy: 94.5% (p < 0.05); see [12], Eq. (3): y = wx + b, "
        "α ≥ β. BERT-base achieves F1=0.91 on SQuAD v2.0.\n\n"
        "Next paragraph with 3.14159 and e^(iπ) + 1 = 0."
    )
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=8)
    assert _reconstruct(chunks, 8) == text


def test_chunking_is_deterministic():
    text = "Deterministic chunking test. " * 50
    first = chunk_text(text, chunk_size=80, chunk_overlap=20)
    second = chunk_text(text, chunk_size=80, chunk_overlap=20)
    assert first == second


def test_default_arguments():
    text = "x" * 2500
    chunks = chunk_text(text)
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert _reconstruct(chunks, 200) == text
