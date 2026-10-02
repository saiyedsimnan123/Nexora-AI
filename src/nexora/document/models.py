"""Document result models.

Small, framework-independent data models describing the result of
processing a document into embedded chunks.

The models are plain frozen dataclasses built on the standard library only.
They contain no processing, storage, API or provider logic. Validation is
limited to basic structural correctness (field types and a non-negative
chunk index). Content rules, such as embedding dimensions or chunk text, are
owned by the document pipeline and the embedding subsystem.

Supplied lists and dictionaries are copied (shallowly) on construction, so
later changes to the caller's objects do not alter a model instance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """A text chunk with its embedding vector.

    Attributes:
        text: The chunk text.
        embedding: The vector for ``text``.
        index: Zero-based position of the chunk within its document.
    """

    text: str
    embedding: list[float]
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a str, got {type(self.text).__name__}")
        if not isinstance(self.embedding, list):
            raise TypeError(
                f"embedding must be a list, got {type(self.embedding).__name__}"
            )
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError(f"index must be an int, got {type(self.index).__name__}")
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}")
        object.__setattr__(self, "embedding", list(self.embedding))


@dataclass(frozen=True)
class ProcessedDocument:
    """The result of processing one document.

    Attributes:
        document_id: Identifier of the document.
        source: Where the document came from (for example a file path).
        chunks: The document's embedded chunks, in order. May be empty.
        metadata: Free-form document metadata with string keys.
    """

    document_id: str
    source: str
    chunks: list[DocumentChunk]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str):
            raise TypeError(
                f"document_id must be a str, got {type(self.document_id).__name__}"
            )
        if not isinstance(self.source, str):
            raise TypeError(f"source must be a str, got {type(self.source).__name__}")
        if not isinstance(self.chunks, list):
            raise TypeError(
                f"chunks must be a list, got {type(self.chunks).__name__}"
            )
        for chunk in self.chunks:
            if not isinstance(chunk, DocumentChunk):
                raise TypeError(
                    "chunks must contain only DocumentChunk objects, "
                    f"got {type(chunk).__name__}"
                )
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"metadata must be a dict, got {type(self.metadata).__name__}"
            )
        for key in self.metadata:
            if not isinstance(key, str):
                raise TypeError(
                    f"metadata keys must be str, got {type(key).__name__}"
                )
        object.__setattr__(self, "chunks", list(self.chunks))
        object.__setattr__(self, "metadata", dict(self.metadata))
