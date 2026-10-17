"""Discovery application service.

Coordinates scholarly source search, paper details, safe PDF downloading,
collection deduplication, and document ingestion pipeline integration.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from nexora.api.errors import (
    DocumentNotFoundError,
    InvalidConfigurationError,
    InvalidRequestError,
)
from nexora.database.models.document import IndexingStatus
from nexora.database.repository.collection import CollectionRepository
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.discovery.arxiv import ArxivClient, normalize_and_validate_arxiv_id
from nexora.discovery.downloader import SafePDFDownloader
from nexora.discovery.errors import (
    DiscoveryConnectionError,
    DiscoveryError,
    DiscoveryRateLimitError,
    DiscoveryResponseError,
    DiscoveryTimeoutError,
    InvalidIdentifierError,
    PaperNotFoundError,
    PDFNotFoundError,
)
from nexora.discovery.models import (
    DiscoveryImportResult,
    ScholarlyPaperMetadata,
    ScholarlySearchQuery,
    ScholarlySearchResult,
)
from nexora.discovery.provider import ScholarlySourceProvider


def _is_unpermitted_external_url(raw_id: str) -> bool:
    """Return True if raw_id is an arbitrary external URL that is not an arXiv URL."""
    lowered = raw_id.strip().lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        valid_prefixes = (
            "https://arxiv.org/abs/",
            "http://arxiv.org/abs/",
            "https://arxiv.org/pdf/",
            "http://arxiv.org/pdf/",
            "https://export.arxiv.org/abs/",
            "http://export.arxiv.org/abs/",
            "https://export.arxiv.org/pdf/",
            "http://export.arxiv.org/pdf/",
        )
        return not any(lowered.startswith(prefix) for prefix in valid_prefixes)
    return False


class DiscoveryService:
    """Application service for scholarly search and explicit paper import."""

    def __init__(
        self,
        *,
        provider: ScholarlySourceProvider | None = None,
        downloader: SafePDFDownloader | None = None,
        session_factory: Any = None,
        ingestion_service: Any = None,
        document_pipeline: Any = None,
    ) -> None:
        self._provider = provider if provider is not None else ArxivClient()
        self._downloader = downloader if downloader is not None else SafePDFDownloader()
        self._session_factory = session_factory
        self._ingestion_service = ingestion_service
        self._document_pipeline = document_pipeline

    def search(self, query: ScholarlySearchQuery) -> ScholarlySearchResult:
        """Search scholarly repository for paper metadata matching query."""
        if not isinstance(query, ScholarlySearchQuery):
            raise TypeError(f"query must be a ScholarlySearchQuery, got {type(query).__name__}")
        return self._provider.search(query)

    def get_paper(self, external_id: str) -> ScholarlyPaperMetadata:
        """Retrieve normalized paper metadata for a single paper identifier.

        Raises:
            InvalidRequestError: If identifier is malformed or an unpermitted URL.
            PaperNotFoundError: If paper does not exist on the upstream provider.
            DiscoveryRateLimitError / DiscoveryTimeoutError / DiscoveryConnectionError / DiscoveryResponseError.
        """
        if not isinstance(external_id, str) or not external_id.strip():
            raise InvalidRequestError("external_id must be a non-empty string.")

        if _is_unpermitted_external_url(external_id):
            raise InvalidRequestError(
                f"Arbitrary external URLs are not permitted as paper identifiers: {external_id!r}"
            )

        try:
            clean_id = normalize_and_validate_arxiv_id(external_id)
        except InvalidIdentifierError as exc:
            raise InvalidRequestError(str(exc)) from exc

        paper = self._provider.get_paper(clean_id)
        if paper is None:
            raise PaperNotFoundError(f"Paper {clean_id!r} not found on arXiv.")
        return paper

    def import_paper(
        self,
        collection_id: UUID | str,
        external_id: str,
        *,
        durable: bool = False,
        session: Session | None = None,
    ) -> DiscoveryImportResult:
        """Explicitly download, deduplicate, and ingest a scholarly paper into a collection.

        Execution steps:
        1. Validate collection ID and paper identifier.
        2. Verify target collection exists in relational storage.
        3. Fetch metadata from provider.
        4. Deduplicate: if an already-indexed document exists in the collection,
           return early with 'already_imported' status without downloading or indexing again.
        5. Download PDF strictly through SafePDFDownloader.
        6. Pass downloaded file through DocumentIngestionService.
        7. Ensure temporary files are always cleaned up.
        """
        # 1. Validate collection_id
        if isinstance(collection_id, str):
            try:
                collection_uuid = UUID(collection_id.strip())
            except (ValueError, AttributeError):
                raise InvalidRequestError("Invalid collection_id: must be a valid UUID.")
        elif isinstance(collection_id, UUID):
            collection_uuid = collection_id
        else:
            raise InvalidRequestError("collection_id must be a UUID or valid UUID string.")

        # 2. Validate external_id
        if not isinstance(external_id, str) or not external_id.strip():
            raise InvalidRequestError("external_id must be a non-empty string.")

        if _is_unpermitted_external_url(external_id):
            raise InvalidRequestError(
                f"Arbitrary external URLs are not permitted as paper identifiers: {external_id!r}"
            )

        try:
            clean_id = normalize_and_validate_arxiv_id(external_id)
        except InvalidIdentifierError as exc:
            raise InvalidRequestError(str(exc)) from exc

        # 3. Resolve database session
        owns_session = False
        db_session = session

        if db_session is None:
            if self._session_factory is not None:
                db_session = self._session_factory()
                owns_session = True
            elif self._ingestion_service is not None:
                factory = getattr(self._ingestion_service, "_session_factory", None)
                if factory is not None:
                    db_session = factory()
                    owns_session = True
                else:
                    pers = getattr(self._ingestion_service, "_persistence_service", None)
                    if pers is not None and getattr(pers, "_session", None) is not None:
                        db_session = pers._session

        if db_session is None:
            raise InvalidConfigurationError(
                "DiscoveryService requires a configured session_factory or database session for paper import."
            )

        try:
            # 4. Verify collection existence
            col_repo = CollectionRepository(db_session)
            col = col_repo.get_by_id(collection_uuid)
            if col is None:
                raise DocumentNotFoundError(f"Collection with id {collection_uuid} not found.")

            # 5. Fetch paper metadata
            metadata = self.get_paper(clean_id)

            # 6. Deduplication check in target collection
            paper_repo = PaperRepository(db_session)
            existing_paper, match_type = paper_repo.find_in_collection(
                collection_uuid, external_id=clean_id, doi=metadata.doi
            )

            if existing_paper is not None:
                doc_repo = DocumentRepository(db_session)
                existing_docs = doc_repo.list_by_paper_id(existing_paper.id)
                completed_doc = next(
                    (d for d in existing_docs if d.indexing_status == IndexingStatus.COMPLETED.value),
                    None,
                )
                if completed_doc is not None:
                    chunk_repo = DocumentChunkRepository(db_session)
                    chunks = chunk_repo.list_by_document_id(completed_doc.id)
                    chunk_count = len(chunks)
                    return DiscoveryImportResult(
                        paper_id=str(existing_paper.id),
                        collection_id=str(collection_uuid),
                        external_id=clean_id,
                        status="already_imported",
                        message="Paper is already imported and indexed in this collection.",
                        title=existing_paper.title,
                        document_id=str(completed_doc.id),
                        chunks_count=chunk_count,
                        indexed_chunks_count=chunk_count,
                        doi=existing_paper.doi,
                        metadata=metadata.to_dict(),
                    )

            if self._ingestion_service is None:
                raise InvalidConfigurationError("ingestion_service is not configured for paper import.")

            # 7. Download PDF strictly through SafePDFDownloader
            downloaded = self._downloader.download_arxiv_pdf(clean_id)
            try:
                # Compute file checksum
                hasher = hashlib.sha256()
                with open(downloaded.path, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                file_checksum = hasher.hexdigest()

                # Register or retrieve paper record deterministically
                paper, _ = paper_repo.get_or_create_paper(
                    collection_id=collection_uuid,
                    title=metadata.title,
                    abstract=metadata.abstract,
                    doi=metadata.doi,
                    external_id=metadata.external_id,
                    publication_date=metadata.publication_date,
                )
                db_session.flush()
                if owns_session:
                    db_session.commit()

                # Ingest document
                safe_filename = f"{clean_id.replace('/', '_')}.pdf"
                pipeline = self._document_pipeline

                if hasattr(self._ingestion_service, "process_and_ingest"):
                    try:
                        ingest_res = self._ingestion_service.process_and_ingest(
                            paper_id=paper.id,
                            file_path=str(downloaded.path),
                            pipeline=pipeline,
                            filename=safe_filename,
                            checksum=file_checksum,
                            durable=durable,
                        )
                    except TypeError:
                        try:
                            ingest_res = self._ingestion_service.process_and_ingest(
                                paper_id=paper.id,
                                file_path=str(downloaded.path),
                                pipeline=pipeline,
                                filename=safe_filename,
                                checksum=file_checksum,
                            )
                        except TypeError:
                            ingest_res = self._ingestion_service.process_and_ingest(
                                paper_id=paper.id,
                                file_path=str(downloaded.path),
                                pipeline=pipeline,
                                filename=safe_filename,
                            )
                elif hasattr(self._ingestion_service, "ingest_file"):
                    ingest_res = self._ingestion_service.ingest_file(
                        paper_id=paper.id,
                        file_path=str(downloaded.path),
                        pipeline=pipeline,
                        filename=safe_filename,
                    )
                else:
                    raise InvalidConfigurationError(
                        "ingestion_service does not provide process_and_ingest"
                    )

                doc_id = getattr(ingest_res, "document_id", None)
                chunks_count = getattr(ingest_res, "chunks_count", 0)
                indexed_count = getattr(ingest_res, "indexed_chunks_count", 0)

                return DiscoveryImportResult(
                    paper_id=str(paper.id),
                    collection_id=str(collection_uuid),
                    external_id=clean_id,
                    status="imported",
                    message="Paper successfully downloaded, ingested, and indexed.",
                    title=paper.title,
                    document_id=str(doc_id) if doc_id else None,
                    chunks_count=chunks_count,
                    indexed_chunks_count=indexed_count,
                    doi=paper.doi,
                    metadata=metadata.to_dict(),
                )
            finally:
                downloaded.cleanup()

        finally:
            if owns_session and db_session is not None:
                db_session.close()
