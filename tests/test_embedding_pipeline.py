"""Tests for EmbeddingPipeline (Step 10G).

The pipeline is a thin delegation layer, so these tests use fake services.
No test makes a real network request or a real OpenAI call.
"""

from __future__ import annotations

import ast
import inspect
import socket
from unittest.mock import Mock

import pytest

from nexora.embedding import pipeline as pipeline_module
from nexora.embedding.pipeline import EmbeddingPipeline
from nexora.embedding.service import EmbeddingService


class FakeService:
    """Deterministic fake that records calls and never touches a network."""

    def __init__(self):
        self.text_calls: list = []
        self.texts_calls: list = []

    @staticmethod
    def _vector(text):
        return [float(len(text)), float(ord(text[0])) if text else 0.0]

    def embed_text(self, text):
        self.text_calls.append(text)
        return self._vector(text)

    def embed_texts(self, texts):
        self.texts_calls.append(texts)
        return [self._vector(t) for t in texts]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("Network access attempted during pipeline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def service():
    return FakeService()


@pytest.fixture
def pipeline(service):
    return EmbeddingPipeline(service)


# Single-text delegation
def test_embed_text_delegates_to_service(pipeline, service):
    pipeline.embed_text("hello")
    assert service.text_calls == ["hello"]
    assert service.texts_calls == []


# Batch delegation
def test_embed_texts_delegates_to_service(pipeline, service):
    batch = ["a", "bb", "ccc"]
    pipeline.embed_texts(batch)
    assert service.texts_calls == [batch]
    assert service.text_calls == []


# Returned values are preserved
def test_embed_text_returns_service_result_unchanged():
    sentinel = [0.1, 0.2, 0.3]
    service = Mock()
    service.embed_text.return_value = sentinel
    result = EmbeddingPipeline(service).embed_text("x")
    assert result is sentinel


def test_embed_texts_returns_service_result_unchanged():
    sentinel = [[0.1], [0.2]]
    service = Mock()
    service.embed_texts.return_value = sentinel
    result = EmbeddingPipeline(service).embed_texts(["x", "y"])
    assert result is sentinel


def test_values_match_service_output(pipeline, service):
    assert pipeline.embed_text("abc") == service.embed_text("abc")
    assert pipeline.embed_texts(["abc", "de"]) == service.embed_texts(
        ["abc", "de"]
    )


# Duplicates remain represented
def test_duplicate_inputs_remain_represented(pipeline):
    results = pipeline.embed_texts(["same", "other", "same"])
    assert len(results) == 3
    assert results[0] == results[2]
    assert results[0] != results[1]


def test_pipeline_passes_duplicates_to_service_unchanged(pipeline, service):
    batch = ["same", "same", "same"]
    pipeline.embed_texts(batch)
    assert service.texts_calls == [batch]


# Order is preserved
def test_input_order_is_preserved(pipeline):
    texts = ["c", "aaa", "bb", "dddd"]
    results = pipeline.embed_texts(texts)
    assert [r[0] for r in results] == [float(len(t)) for t in texts]


def test_empty_batch_is_delegated(pipeline, service):
    assert pipeline.embed_texts([]) == []
    assert service.texts_calls == [[]]


# Invalid input behavior is delegated
def test_embed_text_error_propagates_unchanged():
    error = ValueError("text must be a non-empty string")
    service = Mock()
    service.embed_text.side_effect = error
    with pytest.raises(ValueError) as exc_info:
        EmbeddingPipeline(service).embed_text("")
    assert exc_info.value is error


def test_embed_texts_error_propagates_unchanged():
    error = TypeError("texts must be a list of strings")
    service = Mock()
    service.embed_texts.side_effect = error
    with pytest.raises(TypeError) as exc_info:
        EmbeddingPipeline(service).embed_texts("not a list")
    assert exc_info.value is error


def test_pipeline_does_not_validate_input_itself():
    service = Mock()
    service.embed_text.return_value = [0.0]
    service.embed_texts.return_value = []
    pipe = EmbeddingPipeline(service)
    pipe.embed_text(123)
    pipe.embed_texts(None)
    service.embed_text.assert_called_once_with(123)
    service.embed_texts.assert_called_once_with(None)


# Accepts an EmbeddingService
def test_pipeline_accepts_an_embedding_service():
    service = Mock(spec=EmbeddingService)
    service.embed_text.return_value = [1.0]
    pipe = EmbeddingPipeline(service)
    assert pipe.embed_text("x") == [1.0]
    service.embed_text.assert_called_once_with("x")


# pipeline.py imports no concrete provider (or provider selection / cache)
def _imported_names():
    tree = ast.parse(inspect.getsource(pipeline_module))
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


def test_pipeline_imports_no_concrete_provider():
    modules, names = _imported_names()
    forbidden_modules = {
        "nexora.embedding.hash_provider",
        "nexora.embedding.openai_provider",
        "nexora.embedding.factory",
        "nexora.embedding.cache",
    }
    assert not (modules & forbidden_modules)
    assert "HashEmbeddingProvider" not in names
    assert "OpenAIEmbeddingProvider" not in names
    assert "create_embedding_provider" not in names


def test_pipeline_source_does_not_reference_concrete_providers():
    source = inspect.getsource(pipeline_module)
    assert "HashEmbeddingProvider" not in source
    assert "OpenAIEmbeddingProvider" not in source


# Construction performs no network activity and no service calls
def test_construction_makes_no_network_calls_and_no_service_calls():
    # The autouse fixture blocks socket.connect; construction must succeed.
    service = Mock()
    EmbeddingPipeline(service)
    assert service.method_calls == []
