"""Tests for the embedding integration layer (Step 10I).

All tests use fake pipelines. No test makes a real network request or uses
real credentials.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import socket
from unittest.mock import Mock

import pytest

from nexora.embedding import integration as integration_module
from nexora.embedding.integration import EmbeddedChunk, EmbeddingIntegration
from nexora.embedding.pipeline import EmbeddingPipeline


class FakePipeline:
    """Deterministic fake pipeline that records batch calls."""

    def __init__(self):
        self.batch_calls: list = []
        self.single_calls: list = []

    @staticmethod
    def vector_for(text):
        return [float(len(text)), float(ord(text[0])) if text else 0.0]

    def embed_text(self, text):
        self.single_calls.append(text)
        return self.vector_for(text)

    def embed_texts(self, texts):
        self.batch_calls.append(texts)
        return [self.vector_for(t) for t in texts]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("Network access attempted during integration test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def integration(pipeline):
    return EmbeddingIntegration(pipeline)


CHUNKS = ["chunk A", "chunk B", "chunk C"]


# 1. Normal batch embedding
def test_returns_one_embedded_chunk_per_input(integration):
    result = integration.embed_chunks(list(CHUNKS))
    assert len(result) == 3
    assert all(isinstance(item, EmbeddedChunk) for item in result)


# 2. Text preservation
def test_original_text_is_preserved_exactly(integration):
    texts = ["  leading", "trailing  ", "multi\nline", "Ünïcode ✓", "UPPER lower"]
    result = integration.embed_chunks(list(texts))
    assert [item.text for item in result] == texts


# 3. Embedding association
def test_each_vector_belongs_to_its_chunk(integration, pipeline):
    result = integration.embed_chunks(list(CHUNKS))
    for item in result:
        assert item.embedding == pipeline.vector_for(item.text)


def test_vectors_are_the_pipeline_objects_unchanged():
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    fake = Mock()
    fake.embed_texts.return_value = vectors
    result = EmbeddingIntegration(fake).embed_chunks(["a", "b"])
    assert result[0].embedding is vectors[0]
    assert result[1].embedding is vectors[1]


# 4. Index assignment
def test_indexes_start_at_zero_and_increase_by_one(integration):
    result = integration.embed_chunks(list(CHUNKS))
    assert [item.index for item in result] == [0, 1, 2]


# 5. Order preservation
def test_returned_order_matches_input_order(integration):
    texts = ["zzz", "a", "mmmm", "bb"]
    result = integration.embed_chunks(list(texts))
    assert [item.text for item in result] == texts


def test_duplicate_chunks_get_separate_entries_and_indexes(integration):
    result = integration.embed_chunks(["same", "other", "same"])
    assert [item.index for item in result] == [0, 1, 2]
    assert result[0].text == result[2].text == "same"
    assert result[0].embedding == result[2].embedding
    assert result[0].embedding != result[1].embedding


# 6. Empty input
def test_empty_input_returns_empty_list(integration):
    assert integration.embed_chunks([]) == []


def test_empty_input_does_not_call_pipeline(integration, pipeline):
    integration.embed_chunks([])
    assert pipeline.batch_calls == []
    assert pipeline.single_calls == []


# 7. Pipeline delegation
def test_pipeline_batch_method_called_exactly_once(integration, pipeline):
    integration.embed_chunks(list(CHUNKS))
    assert len(pipeline.batch_calls) == 1


def test_pipeline_receives_the_exact_input_list(integration, pipeline):
    batch = list(CHUNKS)
    integration.embed_chunks(batch)
    assert pipeline.batch_calls[0] is batch


def test_single_text_method_is_never_used(integration, pipeline):
    integration.embed_chunks(list(CHUNKS))
    assert pipeline.single_calls == []


# 8. Error propagation
def test_pipeline_error_propagates_unchanged():
    error = ValueError("provider rejected input")
    fake = Mock()
    fake.embed_texts.side_effect = error
    with pytest.raises(ValueError) as exc_info:
        EmbeddingIntegration(fake).embed_chunks(["a"])
    assert exc_info.value is error


def test_pipeline_error_of_any_type_propagates():
    error = RuntimeError("boom")
    fake = Mock()
    fake.embed_texts.side_effect = error
    with pytest.raises(RuntimeError) as exc_info:
        EmbeddingIntegration(fake).embed_chunks(["a"])
    assert exc_info.value is error


# Integration-level contract checks
@pytest.mark.parametrize("bad", [None, "chunk A", ("a", "b"), 5, {"a": 1}])
def test_non_list_input_is_rejected_without_calling_pipeline(bad):
    fake = Mock()
    with pytest.raises(TypeError, match="list"):
        EmbeddingIntegration(fake).embed_chunks(bad)
    fake.embed_texts.assert_not_called()


def test_vector_count_mismatch_raises_value_error():
    fake = Mock()
    fake.embed_texts.return_value = [[0.1]]
    with pytest.raises(ValueError, match="vectors"):
        EmbeddingIntegration(fake).embed_chunks(["a", "b"])


# 9. Constructor behavior
def test_construction_does_not_call_pipeline():
    fake = Mock()
    EmbeddingIntegration(fake)
    assert fake.method_calls == []


def test_works_with_real_embedding_pipeline_over_fake_service():
    class FakeService:
        def __init__(self):
            self.calls = []

        def embed_text(self, text):
            return [float(len(text))]

        def embed_texts(self, texts):
            self.calls.append(texts)
            return [[float(len(t))] for t in texts]

    service = FakeService()
    real_pipeline = EmbeddingPipeline(service)
    result = EmbeddingIntegration(real_pipeline).embed_chunks(["a", "bbb"])
    assert result == [
        EmbeddedChunk(text="a", embedding=[1.0], index=0),
        EmbeddedChunk(text="bbb", embedding=[3.0], index=1),
    ]
    assert len(service.calls) == 1


def test_works_with_pipeline_shaped_mock():
    # A mock constrained to EmbeddingPipeline's interface (spec) is enough:
    # the integration layer relies on the pipeline's public contract only,
    # and intentionally performs no runtime isinstance check.
    fake = Mock(spec=EmbeddingPipeline)
    fake.embed_texts.return_value = [[1.0]]
    result = EmbeddingIntegration(fake).embed_chunks(["x"])
    assert result == [EmbeddedChunk(text="x", embedding=[1.0], index=0)]


# 10. Provider isolation
def _imports():
    tree = ast.parse(inspect.getsource(integration_module))
    modules, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            for alias in node.names:
                names.add(alias.name)
    return modules, names


def test_integration_imports_no_provider_factory_cache_or_config():
    modules, names = _imports()
    forbidden_modules = {
        "nexora.embedding.hash_provider",
        "nexora.embedding.openai_provider",
        "nexora.embedding.factory",
        "nexora.embedding.cache",
        "nexora.embedding.config",
    }
    forbidden_names = {
        "HashEmbeddingProvider",
        "OpenAIEmbeddingProvider",
        "create_embedding_provider",
        "EmbeddingConfig",
        "EmbeddingCache",
        "InMemoryEmbeddingCache",
    }
    assert not (modules & forbidden_modules)
    assert not (names & forbidden_names)


def test_integration_does_not_import_os_or_sys():
    modules, _ = _imports()
    assert "os" not in modules
    assert "sys" not in modules


def test_lower_layers_do_not_depend_on_integration():
    from nexora.embedding import pipeline as pipeline_module
    from nexora.embedding import service as service_module

    for module in (pipeline_module, service_module):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "nexora.embedding.integration"
            elif isinstance(node, ast.Import):
                assert all(
                    alias.name != "nexora.embedding.integration"
                    for alias in node.names
                )


# 11. No network (the autouse fixture makes any connection attempt fail)
def test_construction_and_use_make_no_network_calls(pipeline):
    integration = EmbeddingIntegration(pipeline)
    assert len(integration.embed_chunks(list(CHUNKS))) == 3


# 12. Dataclass behavior
def test_embedded_chunk_fields():
    item = EmbeddedChunk(text="hello", embedding=[0.5, 0.25], index=7)
    assert item.text == "hello"
    assert item.embedding == [0.5, 0.25]
    assert item.index == 7


def test_embedded_chunk_is_frozen():
    item = EmbeddedChunk(text="hello", embedding=[0.5], index=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.text = "changed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.index = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.embedding = [9.9]


def test_embedded_chunk_is_a_dataclass_with_value_equality():
    assert dataclasses.is_dataclass(EmbeddedChunk)
    a = EmbeddedChunk(text="x", embedding=[1.0], index=0)
    b = EmbeddedChunk(text="x", embedding=[1.0], index=0)
    assert a == b
