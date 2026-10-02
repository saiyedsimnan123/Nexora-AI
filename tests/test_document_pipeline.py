"""Tests for the Step 10L DocumentPipeline contract.

process_pdf() returns a ProcessedDocument built from DocumentChunk objects
converted from the EmbeddedChunk objects produced by EmbeddingIntegration.
All collaborators are fakes: no network, API keys, database or environment.
"""

from __future__ import annotations

import ast
import inspect
import os
import socket
import sys
import uuid

import pytest

import nexora.document.pipeline as pipeline_module
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.pipeline import DocumentPipeline
from nexora.embedding.integration import EmbeddedChunk


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeIntegration:
    """EmbeddingIntegration-compatible recorder."""

    def __init__(self, returned=None, events=None, error=None):
        self.calls: list[list[str]] = []
        self.returned = returned
        self.events = events
        self.error = error

    def embed_chunks(self, chunks):
        self.calls.append(chunks)
        if self.events is not None:
            self.events.append("embed")
        if self.error is not None:
            raise self.error
        if self.returned is not None:
            return self.returned
        return [
            EmbeddedChunk(text=text, embedding=[float(i), 0.5], index=i)
            for i, text in enumerate(chunks)
        ]


class Recorder:
    """Callable that records calls and returns a fixed value or function result."""

    def __init__(self, name, result=None, events=None, error=None):
        self.name = name
        self.result = result
        self.events = events
        self.error = error
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.events is not None:
            self.events.append(self.name)
        if self.error is not None:
            raise self.error
        return self.result


def make_pipeline(integration=None, **kwargs):
    integration = integration if integration is not None else FakeIntegration()
    kwargs.setdefault("loader", lambda path: "raw text")
    kwargs.setdefault("preprocessor", lambda text: text)
    kwargs.setdefault("chunker", lambda text, **kw: ["a", "b"])
    return DocumentPipeline(integration, **kwargs), integration


# --------------------------------------------------------------------------
# Construction and return type
# --------------------------------------------------------------------------


def test_construction_accepts_integration_and_injected_collaborators():
    pipe, _ = make_pipeline()
    assert isinstance(pipe, DocumentPipeline)


def test_construction_with_only_integration_uses_defaults():
    pipe = DocumentPipeline(FakeIntegration())
    assert isinstance(pipe, DocumentPipeline)


def test_process_pdf_returns_processed_document():
    pipe, _ = make_pipeline()
    result = pipe.process_pdf("paper.pdf")
    assert isinstance(result, ProcessedDocument)


# --------------------------------------------------------------------------
# document_id, source, metadata
# --------------------------------------------------------------------------


def test_document_id_is_deterministic():
    pipe, _ = make_pipeline()
    first = pipe.process_pdf("paper.pdf")
    second = pipe.process_pdf("paper.pdf")
    assert first.document_id == second.document_id


def test_same_path_gives_same_id_across_pipeline_instances():
    first, _ = make_pipeline()
    second, _ = make_pipeline()
    assert (
        first.process_pdf("paper.pdf").document_id
        == second.process_pdf("paper.pdf").document_id
    )


def test_different_paths_give_different_ids():
    pipe, _ = make_pipeline()
    assert (
        pipe.process_pdf("a.pdf").document_id
        != pipe.process_pdf("b.pdf").document_id
    )


def test_document_id_matches_uuid5_namespace_url():
    pipe, _ = make_pipeline()
    result = pipe.process_pdf("docs/paper.pdf")
    assert result.document_id == str(uuid.uuid5(uuid.NAMESPACE_URL, "docs/paper.pdf"))


def test_source_equals_supplied_path():
    pipe, _ = make_pipeline()
    assert pipe.process_pdf("docs/paper.pdf").source == "docs/paper.pdf"


def test_metadata_is_empty_dict():
    pipe, _ = make_pipeline()
    assert pipe.process_pdf("paper.pdf").metadata == {}


# --------------------------------------------------------------------------
# Chunk conversion and ordering
# --------------------------------------------------------------------------


def test_embedded_chunks_are_converted_to_document_chunks():
    embedded = [
        EmbeddedChunk(text="first", embedding=[0.1, 0.2], index=0),
        EmbeddedChunk(text="second", embedding=[0.3, 0.4], index=1),
    ]
    pipe, _ = make_pipeline(FakeIntegration(returned=embedded))
    result = pipe.process_pdf("paper.pdf")

    assert all(isinstance(c, DocumentChunk) for c in result.chunks)
    assert not any(isinstance(c, EmbeddedChunk) for c in result.chunks)


def test_text_embedding_and_index_are_preserved():
    embedded = [
        EmbeddedChunk(text="first", embedding=[0.1, 0.2], index=0),
        EmbeddedChunk(text="second", embedding=[0.3, 0.4], index=1),
    ]
    pipe, _ = make_pipeline(FakeIntegration(returned=embedded))
    result = pipe.process_pdf("paper.pdf")

    assert result.chunks == [
        DocumentChunk(text="first", embedding=[0.1, 0.2], index=0),
        DocumentChunk(text="second", embedding=[0.3, 0.4], index=1),
    ]


def test_chunk_order_is_preserved():
    texts = ["one", "two", "three", "four"]
    pipe, _ = make_pipeline(chunker=lambda text, **kw: texts)
    result = pipe.process_pdf("paper.pdf")

    assert [c.text for c in result.chunks] == texts
    assert [c.index for c in result.chunks] == [0, 1, 2, 3]


def test_duplicate_chunk_text_is_handled_correctly():
    texts = ["same", "other", "same", "same"]
    pipe, _ = make_pipeline(chunker=lambda text, **kw: texts)
    result = pipe.process_pdf("paper.pdf")

    assert [c.text for c in result.chunks] == texts
    assert [c.index for c in result.chunks] == [0, 1, 2, 3]
    assert [c.embedding for c in result.chunks] == [
        [0.0, 0.5],
        [1.0, 0.5],
        [2.0, 0.5],
        [3.0, 0.5],
    ]


def test_integration_result_order_is_not_resorted():
    embedded = [
        EmbeddedChunk(text="x", embedding=[1.0], index=0),
        EmbeddedChunk(text="y", embedding=[2.0], index=1),
        EmbeddedChunk(text="z", embedding=[3.0], index=2),
    ]
    pipe, _ = make_pipeline(FakeIntegration(returned=embedded))
    result = pipe.process_pdf("paper.pdf")
    assert [c.text for c in result.chunks] == ["x", "y", "z"]


# --------------------------------------------------------------------------
# Empty documents
# --------------------------------------------------------------------------


def test_empty_chunks_return_valid_document_without_embedding_call():
    integration = FakeIntegration()
    pipe, _ = make_pipeline(integration, chunker=lambda text, **kw: [])
    result = pipe.process_pdf("empty.pdf")

    assert isinstance(result, ProcessedDocument)
    assert result.chunks == []
    assert result.source == "empty.pdf"
    assert result.document_id == str(uuid.uuid5(uuid.NAMESPACE_URL, "empty.pdf"))
    assert result.metadata == {}
    assert integration.calls == []


# --------------------------------------------------------------------------
# Dependency injection
# --------------------------------------------------------------------------


def test_loader_receives_file_path_and_output_goes_to_preprocessor():
    loader = Recorder("loader", result="loaded text")
    preprocessor = Recorder("preprocessor", result="cleaned")
    pipe, _ = make_pipeline(loader=loader, preprocessor=preprocessor)
    pipe.process_pdf("docs/paper.pdf")

    assert loader.calls == [(("docs/paper.pdf",), {})]
    assert preprocessor.calls == [(("loaded text",), {})]


def test_preprocessor_output_goes_to_chunker():
    preprocessor = Recorder("preprocessor", result="cleaned text")
    chunker = Recorder("chunker", result=["c"])
    pipe, _ = make_pipeline(preprocessor=preprocessor, chunker=chunker)
    pipe.process_pdf("paper.pdf")

    assert len(chunker.calls) == 1
    args, _kwargs = chunker.calls[0]
    assert args == ("cleaned text",)


def test_chunker_receives_chunk_size_and_overlap():
    chunker = Recorder("chunker", result=["c"])
    pipe, _ = make_pipeline(chunker=chunker, chunk_size=300, chunk_overlap=30)
    pipe.process_pdf("paper.pdf")

    _args, kwargs = chunker.calls[0]
    assert kwargs == {"chunk_size": 300, "chunk_overlap": 30}


def test_default_chunk_configuration_is_1000_and_200():
    chunker = Recorder("chunker", result=["c"])
    pipe, _ = make_pipeline(chunker=chunker)
    pipe.process_pdf("paper.pdf")

    _args, kwargs = chunker.calls[0]
    assert kwargs == {"chunk_size": 1000, "chunk_overlap": 200}


def test_custom_size_and_overlap_are_applied_per_call():
    chunker = Recorder("chunker", result=["c"])
    pipe, _ = make_pipeline(chunker=chunker, chunk_size=64, chunk_overlap=8)
    pipe.process_pdf("a.pdf")
    pipe.process_pdf("b.pdf")

    assert [kw for _a, kw in chunker.calls] == [
        {"chunk_size": 64, "chunk_overlap": 8},
        {"chunk_size": 64, "chunk_overlap": 8},
    ]


# --------------------------------------------------------------------------
# Embedding integration
# --------------------------------------------------------------------------


def test_embed_chunks_receives_chunker_output_exactly_once():
    integration = FakeIntegration()
    pipe, _ = make_pipeline(integration, chunker=lambda text, **kw: ["p", "q"])
    pipe.process_pdf("paper.pdf")

    assert integration.calls == [["p", "q"]]


def test_pipeline_runs_each_stage_once_in_order():
    events: list[str] = []
    loader = Recorder("loader", result="raw", events=events)
    preprocessor = Recorder("preprocessor", result="clean", events=events)
    chunker = Recorder("chunker", result=["c1", "c2"], events=events)
    integration = FakeIntegration(events=events)

    pipe = DocumentPipeline(
        integration, loader=loader, preprocessor=preprocessor, chunker=chunker
    )
    result = pipe.process_pdf("paper.pdf")

    assert events == ["loader", "preprocessor", "chunker", "embed"]
    assert isinstance(result, ProcessedDocument)
    assert [c.text for c in result.chunks] == ["c1", "c2"]


# --------------------------------------------------------------------------
# Real preprocessor and chunker
# --------------------------------------------------------------------------


def test_real_clean_text_and_chunk_text_produce_valid_document():
    from nexora.document.chunker import chunk_text
    from nexora.document.preprocessor import clean_text

    raw = ("Nexora research intelligence. " * 60).strip()
    integration = FakeIntegration()
    pipe = DocumentPipeline(
        integration,
        loader=lambda path: raw,
        preprocessor=clean_text,
        chunker=chunk_text,
        chunk_size=200,
        chunk_overlap=20,
    )
    result = pipe.process_pdf("paper.pdf")

    assert isinstance(result, ProcessedDocument)
    assert len(result.chunks) > 0
    assert all(isinstance(c, DocumentChunk) for c in result.chunks)
    assert [c.index for c in result.chunks] == list(range(len(result.chunks)))
    assert [c.text for c in result.chunks] == integration.calls[0]
    assert all(c.text for c in result.chunks)
    assert all(len(c.embedding) > 0 for c in result.chunks)


# --------------------------------------------------------------------------
# Error propagation
# --------------------------------------------------------------------------


def test_loader_exception_propagates_unchanged():
    error = RuntimeError("loader failed")
    pipe, integration = make_pipeline(loader=Recorder("loader", error=error))
    with pytest.raises(RuntimeError) as exc_info:
        pipe.process_pdf("paper.pdf")
    assert exc_info.value is error
    assert integration.calls == []


def test_preprocessor_exception_propagates_unchanged():
    error = ValueError("preprocessor failed")
    pipe, integration = make_pipeline(preprocessor=Recorder("pre", error=error))
    with pytest.raises(ValueError) as exc_info:
        pipe.process_pdf("paper.pdf")
    assert exc_info.value is error
    assert integration.calls == []


def test_chunker_exception_propagates_unchanged():
    error = ValueError("chunker failed")
    pipe, integration = make_pipeline(chunker=Recorder("chunker", error=error))
    with pytest.raises(ValueError) as exc_info:
        pipe.process_pdf("paper.pdf")
    assert exc_info.value is error
    assert integration.calls == []


def test_embedding_integration_exception_propagates_unchanged():
    error = RuntimeError("embedding failed")
    pipe, _ = make_pipeline(FakeIntegration(error=error))
    with pytest.raises(RuntimeError) as exc_info:
        pipe.process_pdf("paper.pdf")
    assert exc_info.value is error


# --------------------------------------------------------------------------
# Safety: no network, no environment access, no extra dependencies
# --------------------------------------------------------------------------


def test_processing_makes_no_network_calls(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted during processing")

    class BlockedSocket:
        def __init__(self, *args, **kwargs):
            blocked()

    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "gethostbyname", blocked)

    pipe, integration = make_pipeline()
    result = pipe.process_pdf("paper.pdf")

    assert isinstance(result, ProcessedDocument)
    assert len(integration.calls) == 1


def test_pipeline_does_not_read_environment(monkeypatch):
    class GuardedEnv(dict):
        def _fail(self, *args, **kwargs):
            raise AssertionError("environment accessed during processing")

        __getitem__ = get = __contains__ = __iter__ = _fail
        keys = values = items = copy = setdefault = pop = _fail

    pipe, _ = make_pipeline()
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", GuardedEnv())
        result = pipe.process_pdf("paper.pdf")

    assert isinstance(result, ProcessedDocument)


def test_pipeline_has_no_unnecessary_external_dependencies():
    allowed_nexora = {
        "nexora.document.chunker",
        "nexora.document.loader",
        "nexora.document.models",
        "nexora.document.preprocessor",
        "nexora.embedding.integration",
    }
    tree = ast.parse(inspect.getsource(pipeline_module))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not expected"
            imported.add(node.module)

    assert imported, "expected the pipeline module to have imports"
    for module in imported:
        top_level = module.split(".")[0]
        if top_level == "nexora":
            assert module in allowed_nexora, f"unexpected Nexora import: {module}"
        else:
            assert (
                top_level in sys.stdlib_module_names
            ), f"non-stdlib import in pipeline: {module}"
