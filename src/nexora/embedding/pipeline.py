"""Document pipeline.

Thin orchestration layer that turns a PDF file into embedded chunks::

    load_pdf_text -> clean_text -> chunk_text -> EmbeddingIntegration.embed_chunks

The pipeline only coordinates existing components. It does not parse PDFs,
clean text, split text or compute embeddings itself, and it contains no
provider selection, environment access, or storage logic. Errors raised by
any stage propagate unchanged.

Dependency direction::

    DocumentPipeline -> document components, EmbeddingIntegration
"""

from __future__ import annotations

from collections.abc import Callable

from nexora.document.chunker import chunk_text
from nexora.document.loader import load_pdf_text
from nexora.document.preprocessor import clean_text
from nexora.embedding.integration import EmbeddedChunk, EmbeddingIntegration


class DocumentPipeline:
    """Processes a PDF into ``EmbeddedChunk`` objects.

    The stage functions default to the existing Nexora document components
    and can be replaced (for example with fakes in tests).
    """

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
        """Store the collaborators and chunk settings.

        Construction performs no document processing and no network
        activity. ``chunk_size`` and ``chunk_overlap`` are passed through to
        the chunker, which owns their validation.
        """
        self._embedding_integration = embedding_integration
        self._loader = loader
        self._preprocessor = preprocessor
        self._chunker = chunker
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def process_pdf(self, file_path: str) -> list[EmbeddedChunk]:
        """Load, clean, chunk and embed the PDF at ``file_path``.

        Returns:
            The ``EmbeddedChunk`` list from the embedding integration,
            unchanged and in chunk order. ``[]`` if the document yields no
            chunks, in which case the embedding integration is not called.

        Raises:
            Any exception raised by a stage propagates unchanged.
        """
        raw_text = self._loader(file_path)
        cleaned_text = self._preprocessor(raw_text)
        chunks = self._chunker(
            cleaned_text,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        if not chunks:
            return []
        return self._embedding_integration.embed_chunks(chunks)
