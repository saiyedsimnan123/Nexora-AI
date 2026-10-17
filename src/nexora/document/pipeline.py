"""Document processing pipeline.

Orchestrates PDF loading, page extraction, preprocessing, chunking and embedding,
and returns the result as a ProcessedDocument with traceable provenance metadata.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Sequence

from nexora.document.chunker import ProvenanceChunk, chunk_pages, chunk_text
from nexora.document.loader import (
    DocumentPage,
    ExtractedDocument,
    load_pdf_document,
    load_pdf_text,
)
from nexora.document.models import DocumentChunk, ProcessedDocument
from nexora.document.preprocessor import clean_text
from nexora.embedding.integration import EmbeddedChunk, EmbeddingIntegration


def _document_id(source: str) -> str:
    """Return a deterministic document ID derived from the source path."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source))


def compute_chunk_id(document_id: str, chunk_index: int) -> str:
    """Return a deterministic chunk UUID string derived from document_id and chunk_index."""
    try:
        namespace = uuid.UUID(document_id)
    except (ValueError, AttributeError, TypeError):
        namespace = uuid.NAMESPACE_URL
    return str(uuid.uuid5(namespace, f"chunk:{chunk_index}"))


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
        loader: Callable[[str], Any] = load_pdf_text,
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

    def process_extracted_document(
        self,
        extracted: ExtractedDocument,
    ) -> ProcessedDocument:
        """Process an ExtractedDocument into a ProcessedDocument with page provenance."""
        if not isinstance(extracted, ExtractedDocument):
            raise TypeError(
                f"extracted must be an ExtractedDocument, got {type(extracted).__name__}"
            )

        prov_chunks = chunk_pages(
            extracted.pages,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            preprocessor=self._preprocessor,
        )

        texts = [c.text for c in prov_chunks]
        embedded_chunks = (
            self._embedding_integration.embed_chunks(texts) if texts else []
        )

        doc_id = _document_id(extracted.source)
        meta: dict[str, object] = {
            "chunk_pages": [c.page_number for c in prov_chunks],
            "total_pages": extracted.total_pages,
        }
        if extracted.title:
            meta["title"] = extracted.title
        if extracted.author:
            meta["author"] = extracted.author
        for k, v in extracted.metadata.items():
            if k not in meta:
                meta[k] = v

        return ProcessedDocument(
            document_id=doc_id,
            source=extracted.source,
            chunks=[_to_document_chunk(chunk) for chunk in embedded_chunks],
            metadata=meta,
        )

    def process_pdf(self, file_path: str) -> ProcessedDocument:
        """Process a PDF file into embedded chunks.

        When default loader is used and file points to an existing PDF,
        extracts page structure to preserve exact page provenance.
        When a custom string loader is injected, maintains standard pipeline flow.
        """
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a str, got {type(file_path).__name__}")

        # If default loader is in use and fitz is available, extract pages with provenance
        if self._loader is load_pdf_text:
            try:
                extracted = load_pdf_document(file_path, require_extractable_text=False)
                return self.process_extracted_document(extracted)
            except Exception:
                # Fall back to standard loader flow on any page extraction failure
                pass

        # Standard loader flow (supports injected string loaders in tests)
        raw_result = self._loader(file_path)

        if isinstance(raw_result, ExtractedDocument):
            return self.process_extracted_document(raw_result)

        text = raw_result if isinstance(raw_result, str) else str(raw_result)
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
