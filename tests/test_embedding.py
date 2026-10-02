import math

import pytest

from nexora.embedding import DEFAULT_DIMENSION, EmbeddingProvider, HashEmbeddingProvider


def test_provider_can_be_created():
    provider = HashEmbeddingProvider()
    assert isinstance(provider, HashEmbeddingProvider)


def test_default_dimension_works():
    provider = HashEmbeddingProvider()
    assert provider.dimension == DEFAULT_DIMENSION
    assert len(provider.embed_text("research paper")) == DEFAULT_DIMENSION


@pytest.mark.parametrize("dimension", [1, 3, 8, 9, 64, 384, 1000])
def test_custom_dimension_works(dimension):
    provider = HashEmbeddingProvider(dimension=dimension)
    vector = provider.embed_text("graph neural networks")
    assert provider.dimension == dimension
    assert len(vector) == dimension


def test_output_is_a_list_of_floats():
    vector = HashEmbeddingProvider().embed_text("attention is all you need")
    assert isinstance(vector, list)
    assert all(type(value) is float for value in vector)


def test_output_values_are_finite_and_unit_length():
    vector = HashEmbeddingProvider(dimension=50).embed_text("BERT F1=0.91")
    assert all(math.isfinite(value) for value in vector)
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_same_text_produces_exactly_the_same_vector():
    provider = HashEmbeddingProvider(dimension=32)
    assert provider.embed_text("same text") == provider.embed_text("same text")


def test_different_texts_produce_different_vectors():
    provider = HashEmbeddingProvider(dimension=32)
    texts = ["alpha", "beta", "Alpha", "alpha ", "alpha.", "α ≥ β", "p < 0.05"]
    vectors = [tuple(provider.embed_text(text)) for text in texts]
    assert len(set(vectors)) == len(texts)


def test_deterministic_across_separate_instances():
    first = HashEmbeddingProvider(dimension=40).embed_text("Eq. (3): y = wx + b")
    second = HashEmbeddingProvider(dimension=40).embed_text("Eq. (3): y = wx + b")
    assert first == second


def test_known_vector_is_stable_across_processes():
    # Fixed expected values guard against accidental use of hash(), which
    # changes between Python processes, or against algorithm changes.
    vector = HashEmbeddingProvider(dimension=4).embed_text("nexora")
    expected = [0.28432901419429385, 0.8295271989233097, -0.478667309717539, 0.04380918330124561]
    assert vector == pytest.approx(expected, abs=1e-12)


def test_different_dimensions_give_different_length_vectors():
    text = "same text"
    short = HashEmbeddingProvider(dimension=8).embed_text(text)
    long = HashEmbeddingProvider(dimension=16).embed_text(text)
    assert len(short) == 8 and len(long) == 16


def test_empty_text_raises_value_error():
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        HashEmbeddingProvider().embed_text("")


@pytest.mark.parametrize("text", ["   ", "\n", "\t \n  "])
def test_whitespace_only_text_raises_value_error(text):
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        HashEmbeddingProvider().embed_text(text)


@pytest.mark.parametrize("bad_text", [None, 123, 4.5, ["text"], b"text"])
def test_non_string_input_raises_type_error(bad_text):
    with pytest.raises(TypeError, match="text must be a str"):
        HashEmbeddingProvider().embed_text(bad_text)


@pytest.mark.parametrize("bad_dimension", [0, -1, -128])
def test_non_positive_dimension_raises_value_error(bad_dimension):
    with pytest.raises(ValueError, match="dimension must be positive"):
        HashEmbeddingProvider(dimension=bad_dimension)


@pytest.mark.parametrize("bad_dimension", [1.5, "128", None, True])
def test_non_int_dimension_raises_type_error(bad_dimension):
    with pytest.raises(TypeError, match="dimension must be an int"):
        HashEmbeddingProvider(dimension=bad_dimension)


def test_hash_provider_satisfies_embedding_provider():
    assert isinstance(HashEmbeddingProvider(), EmbeddingProvider)


def test_custom_provider_satisfies_embedding_provider():
    class ConstantProvider:
        def embed_text(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    assert isinstance(ConstantProvider(), EmbeddingProvider)


def test_object_without_embed_text_is_not_a_provider():
    assert not isinstance(object(), EmbeddingProvider)


def test_code_can_depend_on_the_abstraction():
    def embed_all(provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
        return [provider.embed_text(text) for text in texts]

    vectors = embed_all(HashEmbeddingProvider(dimension=16), ["one", "two"])
    assert len(vectors) == 2
    assert all(len(vector) == 16 for vector in vectors)
