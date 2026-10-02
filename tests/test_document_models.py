"""Tests for the document result models (Step 10K)."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import socket

import pytest

from nexora.document import models as models_module
from nexora.document.models import DocumentChunk, ProcessedDocument


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("Network access attempted during model test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


_UNSET = object()


def make_chunk(text="chunk", embedding=_UNSET, index=0):
    return DocumentChunk(
        text=text,
        embedding=[0.1, 0.2] if embedding is _UNSET else embedding,
        index=index,
    )


def make_document(
    chunks=_UNSET, metadata=_UNSET, document_id="doc-1", source="a.pdf"
):
    return ProcessedDocument(
        document_id=document_id,
        source=source,
        chunks=[] if chunks is _UNSET else chunks,
        metadata={} if metadata is _UNSET else metadata,
    )


# Construction and field values: DocumentChunk
def test_document_chunk_construction_and_fields():
    chunk = DocumentChunk(text="hello", embedding=[0.5, 0.25], index=3)
    assert chunk.text == "hello"
    assert chunk.embedding == [0.5, 0.25]
    assert chunk.index == 3


def test_document_chunk_is_a_dataclass():
    assert dataclasses.is_dataclass(DocumentChunk)
    assert [f.name for f in dataclasses.fields(DocumentChunk)] == [
        "text",
        "embedding",
        "index",
    ]


def test_document_chunk_allows_empty_text_and_empty_embedding():
    chunk = DocumentChunk(text="", embedding=[], index=0)
    assert chunk.text == ""
    assert chunk.embedding == []


# Construction and field values: ProcessedDocument
def test_processed_document_construction_and_fields():
    chunks = [make_chunk("a", index=0), make_chunk("b", index=1)]
    metadata = {"pages": 12, "title": "A Paper"}
    doc = ProcessedDocument(
        document_id="doc-42",
        source="papers/a.pdf",
        chunks=chunks,
        metadata=metadata,
    )
    assert doc.document_id == "doc-42"
    assert doc.source == "papers/a.pdf"
    assert doc.chunks == chunks
    assert doc.metadata == metadata


def test_processed_document_is_a_dataclass():
    assert dataclasses.is_dataclass(ProcessedDocument)
    assert [f.name for f in dataclasses.fields(ProcessedDocument)] == [
        "document_id",
        "source",
        "chunks",
        "metadata",
    ]


# Frozen behavior
@pytest.mark.parametrize(
    "field, value", [("text", "x"), ("embedding", [9.0]), ("index", 5)]
)
def test_document_chunk_is_frozen(field, value):
    chunk = make_chunk()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(chunk, field, value)


@pytest.mark.parametrize(
    "field, value",
    [
        ("document_id", "other"),
        ("source", "other.pdf"),
        ("chunks", []),
        ("metadata", {}),
    ],
)
def test_processed_document_is_frozen(field, value):
    doc = make_document()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(doc, field, value)


def test_cannot_add_new_attributes():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_chunk().extra = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_document().extra = 1


# Empty chunks
def test_document_with_empty_chunks():
    doc = make_document(chunks=[])
    assert doc.chunks == []
    assert len(doc.chunks) == 0


# Metadata handling
def test_empty_metadata_is_allowed():
    assert make_document(metadata={}).metadata == {}


def test_metadata_holds_arbitrary_value_types():
    metadata = {
        "pages": 3,
        "title": "T",
        "authors": ["A", "B"],
        "nested": {"k": [1, 2]},
        "missing": None,
        "ratio": 0.5,
        "flag": True,
    }
    assert make_document(metadata=metadata).metadata == metadata


def test_metadata_requires_string_keys():
    with pytest.raises(TypeError, match="keys"):
        make_document(metadata={1: "x"})


# Multiple chunks
def test_multiple_chunks_keep_order_and_values():
    chunks = [make_chunk(f"c{i}", [float(i)], i) for i in range(5)]
    doc = make_document(chunks=chunks)
    assert [c.text for c in doc.chunks] == ["c0", "c1", "c2", "c3", "c4"]
    assert [c.index for c in doc.chunks] == [0, 1, 2, 3, 4]
    assert [c.embedding for c in doc.chunks] == [[float(i)] for i in range(5)]


def test_duplicate_chunk_text_is_allowed():
    doc = make_document(chunks=[make_chunk("same", index=0), make_chunk("same", index=1)])
    assert [c.text for c in doc.chunks] == ["same", "same"]


# Independence and non-mutation of supplied containers
def test_caller_list_mutation_does_not_change_document():
    chunks = [make_chunk("a", index=0)]
    doc = make_document(chunks=chunks)
    chunks.append(make_chunk("b", index=1))
    assert len(doc.chunks) == 1


def test_caller_metadata_mutation_does_not_change_document():
    metadata = {"k": "v"}
    doc = make_document(metadata=metadata)
    metadata["k"] = "changed"
    metadata["new"] = 1
    assert doc.metadata == {"k": "v"}


def test_caller_embedding_mutation_does_not_change_chunk():
    vector = [0.1, 0.2]
    chunk = DocumentChunk(text="a", embedding=vector, index=0)
    vector.append(0.3)
    vector[0] = 9.9
    assert chunk.embedding == [0.1, 0.2]


def test_construction_does_not_mutate_supplied_containers():
    vector = [0.1, 0.2]
    chunks = [DocumentChunk(text="a", embedding=vector, index=0)]
    metadata = {"k": "v"}
    make_document(chunks=chunks, metadata=metadata)
    assert vector == [0.1, 0.2]
    assert len(chunks) == 1
    assert metadata == {"k": "v"}


def test_separate_instances_are_independent():
    shared_metadata = {"k": "v"}
    first = make_document(document_id="one", metadata=shared_metadata)
    second = make_document(document_id="two", metadata=shared_metadata)
    first.metadata["only_first"] = True
    first.chunks.append(make_chunk("x", index=0))
    assert second.metadata == {"k": "v"}
    assert second.chunks == []
    assert first.metadata is not second.metadata
    assert first.chunks is not second.chunks


def test_separate_chunk_instances_do_not_share_embeddings():
    vector = [1.0, 2.0]
    first = DocumentChunk(text="a", embedding=vector, index=0)
    second = DocumentChunk(text="b", embedding=vector, index=1)
    first.embedding.append(3.0)
    assert second.embedding == [1.0, 2.0]


def test_equal_values_compare_equal():
    assert make_chunk("a", [0.1], 0) == make_chunk("a", [0.1], 0)
    assert make_document() == make_document()
    assert make_document(document_id="x") != make_document(document_id="y")


# Basic structural validation
@pytest.mark.parametrize("bad", [None, 5, b"bytes", ["a"]])
def test_chunk_text_must_be_str(bad):
    with pytest.raises(TypeError, match="text"):
        DocumentChunk(text=bad, embedding=[0.1], index=0)


@pytest.mark.parametrize("bad", [None, (0.1,), "0.1", {0.1}])
def test_chunk_embedding_must_be_list(bad):
    with pytest.raises(TypeError, match="embedding"):
        DocumentChunk(text="a", embedding=bad, index=0)


@pytest.mark.parametrize("bad", [None, "0", 1.5, True])
def test_chunk_index_must_be_int(bad):
    with pytest.raises(TypeError, match="index"):
        DocumentChunk(text="a", embedding=[0.1], index=bad)


def test_chunk_index_must_not_be_negative():
    with pytest.raises(ValueError, match="non-negative"):
        DocumentChunk(text="a", embedding=[0.1], index=-1)


@pytest.mark.parametrize("bad", [None, 1, b"id"])
def test_document_id_must_be_str(bad):
    with pytest.raises(TypeError, match="document_id"):
        make_document(document_id=bad)


@pytest.mark.parametrize("bad", [None, 1, b"src"])
def test_source_must_be_str(bad):
    with pytest.raises(TypeError, match="source"):
        make_document(source=bad)


@pytest.mark.parametrize("bad", [None, (), "abc", {"a": 1}])
def test_chunks_must_be_a_list(bad):
    with pytest.raises(TypeError, match="chunks"):
        make_document(chunks=bad)


def test_chunks_must_contain_document_chunks():
    with pytest.raises(TypeError, match="DocumentChunk"):
        make_document(chunks=[make_chunk(), "not a chunk"])


@pytest.mark.parametrize("bad", [None, [], "meta", [("k", "v")]])
def test_metadata_must_be_a_dict(bad):
    with pytest.raises(TypeError, match="metadata"):
        make_document(metadata=bad)


# No external / network behavior (the autouse fixture blocks connections)
def test_construction_makes_no_network_calls():
    make_document(chunks=[make_chunk()], metadata={"k": "v"})


def _imported_modules():
    tree = ast.parse(inspect.getsource(models_module))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    return modules


def test_models_import_only_the_standard_library_dataclasses():
    assert _imported_modules() <= {"__future__", "dataclasses"}


def test_models_have_no_environment_access_or_nexora_imports():
    source = inspect.getsource(models_module)
    assert "environ" not in source
    assert "getenv" not in source
    assert not any(m.startswith("nexora") for m in _imported_modules())
