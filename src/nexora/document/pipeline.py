"""Document processing pipeline.

Orchestrates PDF loading, preprocessing, chunking and embedding, and returns
the result as a ProcessedDocument.
"""

from __future__ import annotations

import uuid
from typing import Callable

from nexora.document.chunker import chunk_text
from nexora.document.loader import load_pdf_text
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.preprocessor import clean_text
from nexora.embedding.integration import EmbeddedChunk, EmbeddingIntegration


def _document_id(source: str) -> str:
    """Return a deterministic document ID derived from the source path."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source))


def _to_document_chunk(chunk: EmbeddedChunk) -> DocumentChunk:
    """Convert an EmbeddedChunk into a DocumentChunk."""
    return DocumentChunk(
        text=chunk.text,
        embedding=chunk.embedding,
        index=chunk.index,
    )


class DocumentPipeline:
    """PDF -> load -> preprocess -> chunk -> embed -> ProcessedDocument."""

    def __init__(
        self,
        embedding_integration: EmbeddingIntegration,
        *,
        loader: Callable[[str], str] = load_pdf_text,
        preprocessor: Callable[[str], str] = clean_text,
        chunker: Callable[..., list[str]] = chunk_text,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._embedding_integration = embedding_integration
        self._loader = loader
        self._preprocessor = preprocessor
        self._chunker = chunker
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def process_pdf(self, file_path: str) -> ProcessedDocument:
        text = self._loader(file_path)
        cleaned = self._preprocessor(text)
        chunks = self._chunker(
            cleaned,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        embedded_chunks = (
            self._embedding_integration.embed_chunks(chunks) if chunks else []
        )

        return ProcessedDocument(
            document_id=_document_id(file_path),
            source=file_path,
            chunks=[_to_document_chunk(chunk) for chunk in embedded_chunks],
            metadata={},
        )
