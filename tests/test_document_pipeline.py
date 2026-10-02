"""Tests for DocumentPipeline (Step 10J).

These tests verify orchestration only, using small fake stages. No PDF file,
network, credentials or external service is needed.
"""

from __future__ import annotations

import ast
import inspect
import socket

import pytest

from nexora.document import pipeline as pipeline_module
from nexora.document.chunker import chunk_text
from nexora.document.loader import load_pdf_text
from nexora.document.pipeline import DocumentPipeline
from nexora.document.preprocessor import clean_text
from nexora.embedding.integration import EmbeddedChunk


class Recorder:
    """Shared call log so tests can assert exact stage ordering."""

    def __init__(self):
        self.calls: list[tuple] = []


class FakeIntegration:
    def __init__(self, recorder, error=None):
        self._recorder = recorder
        self._error = error
        self.returned = None

    def embed_chunks(self, chunks):
        self._recorder.calls.append(("embed", chunks))
        if self._error is not None:
            raise self._error
        self.returned = [
            EmbeddedChunk(text=text, embedding=[float(i)], index=i)
            for i, text in enumerate(chunks)
        ]
        return self.returned


def make_pipeline(
    recorder,
    *,
    raw="raw text",
    cleaned="clean text",
    chunks=("chunk 1", "chunk 2"),
    loader_error=None,
    preprocessor_error=None,
    chunker_error=None,
    integration_error=None,
    **pipeline_kwargs,
):
    def loader(file_path):
        recorder.calls.append(("load", file_path))
        if loader_error is not None:
            raise loader_error
        return raw

    def preprocessor(text):
        recorder.calls.append(("clean", text))
        if preprocessor_error is not None:
            raise preprocessor_error
        return cleaned

    def chunker(text, **kwargs):
        recorder.calls.append(("chunk", text, kwargs))
        if chunker_error is not None:
            raise chunker_error
        return list(chunks)

    integration = FakeIntegration(recorder, error=integration_error)
    pipe = DocumentPipeline(
        integration,
        loader=loader,
        preprocessor=preprocessor,
        chunker=chunker,
        **pipeline_kwargs,
    )
    return pipe, integration


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("Network access attempted during pipeline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def recorder():
    return Recorder()


# 1. Successful flow
def test_process_pdf_returns_embedded_chunks(recorder):
    pipe, _ = make_pipeline(recorder)
    result = pipe.process_pdf("paper.pdf")
    assert [item.text for item in result] == ["chunk 1", "chunk 2"]
    assert all(isinstance(item, EmbeddedChunk) for item in result)


# 2. Stage order
def test_stages_run_in_order(recorder):
    pipe, _ = make_pipeline(recorder)
    pipe.process_pdf("paper.pdf")
    assert [call[0] for call in recorder.calls] == [
        "load",
        "clean",
        "chunk",
        "embed",
    ]


def test_each_stage_runs_exactly_once(recorder):
    pipe, _ = make_pipeline(recorder)
    pipe.process_pdf("paper.pdf")
    assert len(recorder.calls) == 4


# 3. Data passed between stages
def test_data_flows_between_stages(recorder):
    pipe, _ = make_pipeline(
        recorder,
        raw="raw text",
        cleaned="clean text",
        chunks=("chunk 1", "chunk 2"),
    )
    pipe.process_pdf("some/path.pdf")
    load, clean, chunk, embed = recorder.calls
    assert load == ("load", "some/path.pdf")
    assert clean == ("clean", "raw text")
    assert chunk[0] == "chunk" and chunk[1] == "clean text"
    assert embed == ("embed", ["chunk 1", "chunk 2"])


# 4. Embedded chunks returned unchanged
def test_returned_list_is_the_integration_result(recorder):
    pipe, integration = make_pipeline(recorder)
    result = pipe.process_pdf("paper.pdf")
    assert result is integration.returned


# 5-6. Order and indexes
def test_chunk_order_is_preserved(recorder):
    chunks = ("zeta", "alpha", "mid", "alpha")
    pipe, _ = make_pipeline(recorder, chunks=chunks)
    result = pipe.process_pdf("paper.pdf")
    assert [item.text for item in result] == list(chunks)


def test_indexes_are_zero_based_positions(recorder):
    pipe, _ = make_pipeline(recorder, chunks=("a", "b", "c"))
    result = pipe.process_pdf("paper.pdf")
    assert [item.index for item in result] == [0, 1, 2]


# 7-8. Empty chunk list
def test_no_chunks_returns_empty_list(recorder):
    pipe, _ = make_pipeline(recorder, chunks=())
    assert pipe.process_pdf("paper.pdf") == []


def test_integration_not_called_when_no_chunks(recorder):
    pipe, _ = make_pipeline(recorder, chunks=())
    pipe.process_pdf("paper.pdf")
    assert [call[0] for call in recorder.calls] == ["load", "clean", "chunk"]


# 9-12. Error propagation (and later stages are not reached)
def test_loader_error_propagates(recorder):
    error = FileNotFoundError("missing.pdf")
    pipe, _ = make_pipeline(recorder, loader_error=error)
    with pytest.raises(FileNotFoundError) as exc_info:
        pipe.process_pdf("missing.pdf")
    assert exc_info.value is error
    assert [call[0] for call in recorder.calls] == ["load"]


def test_preprocessor_error_propagates(recorder):
    error = ValueError("bad text")
    pipe, _ = make_pipeline(recorder, preprocessor_error=error)
    with pytest.raises(ValueError) as exc_info:
        pipe.process_pdf("paper.pdf")
    assert exc_info.value is error
    assert [call[0] for call in recorder.calls] == ["load", "clean"]


def test_chunker_error_propagates(recorder):
    error = ValueError("chunk_overlap must be smaller than chunk_size")
    pipe, _ = make_pipeline(recorder, chunker_error=error)
    with pytest.raises(ValueError) as exc_info:
        pipe.process_pdf("paper.pdf")
    assert exc_info.value is error
    assert [call[0] for call in recorder.calls] == ["load", "clean", "chunk"]


def test_embedding_integration_error_propagates(recorder):
    error = RuntimeError("embedding failed")
    pipe, _ = make_pipeline(recorder, integration_error=error)
    with pytest.raises(RuntimeError) as exc_info:
        pipe.process_pdf("paper.pdf")
    assert exc_info.value is error


# 13. Chunk size and overlap
def test_default_chunk_settings_passed_to_chunker(recorder):
    pipe, _ = make_pipeline(recorder)
    pipe.process_pdf("paper.pdf")
    kwargs = recorder.calls[2][2]
    assert kwargs == {"chunk_size": 1000, "chunk_overlap": 200}


def test_custom_chunk_settings_passed_to_chunker(recorder):
    pipe, _ = make_pipeline(recorder, chunk_size=500, chunk_overlap=50)
    pipe.process_pdf("paper.pdf")
    kwargs = recorder.calls[2][2]
    assert kwargs == {"chunk_size": 500, "chunk_overlap": 50}


# Default wiring uses the existing document components
def test_default_stages_are_the_existing_components():
    params = inspect.signature(DocumentPipeline.__init__).parameters
    assert params["loader"].default is load_pdf_text
    assert params["preprocessor"].default is clean_text
    assert params["chunker"].default is chunk_text


def test_real_preprocessor_and_chunker_work_with_fake_loader(recorder):
    integration = FakeIntegration(recorder)
    pipe = DocumentPipeline(
        integration,
        loader=lambda path: "Some research paper text.\n\nSecond paragraph.",
    )
    result = pipe.process_pdf("paper.pdf")
    embedded_input = recorder.calls[0][1]
    assert isinstance(embedded_input, list)
    assert embedded_input
    assert all(isinstance(chunk, str) for chunk in embedded_input)
    assert [item.index for item in result] == list(range(len(embedded_input)))


# 14. No network (autouse fixture blocks connections)
def test_processing_makes_no_network_calls(recorder):
    pipe, _ = make_pipeline(recorder)
    assert len(pipe.process_pdf("paper.pdf")) == 2


# 15. Constructor performs no processing
def test_constructor_does_no_processing(recorder):
    make_pipeline(recorder)
    assert recorder.calls == []


# 16-19. Architectural isolation
def _imports():
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


def test_no_provider_selection_or_provider_imports():
    modules, names = _imports()
    forbidden_modules = {
        "nexora.embedding.hash_provider",
        "nexora.embedding.openai_provider",
        "nexora.embedding.factory",
        "nexora.embedding.config",
        "nexora.embedding.cache",
        "nexora.embedding.service",
        "nexora.embedding.pipeline",
    }
    forbidden_names = {
        "HashEmbeddingProvider",
        "OpenAIEmbeddingProvider",
        "create_embedding_provider",
        "EmbeddingConfig",
        "EmbeddingCache",
        "InMemoryEmbeddingCache",
        "EmbeddingService",
        "EmbeddingPipeline",
    }
    assert not (modules & forbidden_modules)
    assert not (names & forbidden_names)


def test_no_openai_sdk_usage():
    modules, _ = _imports()
    assert not any(m == "openai" or m.startswith("openai.") for m in modules)


def test_no_environment_access():
    modules, _ = _imports()
    assert "os" not in modules
    assert "sys" not in modules
    source = inspect.getsource(pipeline_module)
    assert "environ" not in source
    assert "getenv" not in source


def test_no_database_or_vector_store_imports():
    modules, _ = _imports()
    forbidden_roots = {
        "qdrant_client",
        "psycopg",
        "psycopg2",
        "sqlalchemy",
        "neo4j",
        "redis",
    }
    roots = {m.split(".")[0] for m in modules}
    assert not (roots & forbidden_roots)
