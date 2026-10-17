"""Paper repository."""
from __future__ import annotations

import datetime
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select

from nexora.database.models.paper import Paper
from nexora.database.repository.base import BaseRepository

# Explicit sentinel to distinguish between "not provided" and "provided as None"
_UNSET = object()


class PaperRepository(BaseRepository[Paper]):
    """Repository for Paper models."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, Paper)

    def create(
        self,
        *,
        collection_id: UUID,
        title: str,
        abstract: str | None = None,
        doi: str | None = None,
        external_id: str | None = None,
        publication_date: datetime.date | None = None,
    ) -> Paper:
        """Create a new paper."""
        return self._create(
            collection_id=collection_id,
            title=title,
            abstract=abstract,
            doi=doi,
            external_id=external_id,
            publication_date=publication_date,
        )

    def list_by_collection_id(
        self,
        collection_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Paper]:
        """List papers for a specific collection with validated pagination."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        stmt = (
            select(Paper)
            .where(Paper.collection_id == collection_id)
            .order_by(Paper.created_at, Paper.id)
            .offset(offset)
            .limit(limit)
        )
        return self._session.scalars(stmt).all()

    def update(
        self,
        id: UUID,
        *,
        title: str | Any = _UNSET,
        abstract: str | None | Any = _UNSET,
        doi: str | None | Any = _UNSET,
        external_id: str | None | Any = _UNSET,
        publication_date: datetime.date | None | Any = _UNSET,
        collection_id: UUID | Any = _UNSET,
    ) -> Paper:
        """Update an existing paper. Only updates provided fields."""
        kwargs: dict[str, Any] = {}
        if title is not _UNSET:
            kwargs["title"] = title
        if abstract is not _UNSET:
            kwargs["abstract"] = abstract
        if doi is not _UNSET:
            kwargs["doi"] = doi
        if external_id is not _UNSET:
            kwargs["external_id"] = external_id
        if publication_date is not _UNSET:
            kwargs["publication_date"] = publication_date
        if collection_id is not _UNSET:
            kwargs["collection_id"] = collection_id
        return self._update(id, **kwargs)

    def get_by_id_for_update(self, id: UUID) -> Paper | None:
        """Get a paper by ID with a database row lock (FOR UPDATE), or None if not found."""
        stmt = select(Paper).where(Paper.id == id).with_for_update()
        return self._session.scalars(stmt).first()

    def get_by_external_id(self, collection_id: UUID, external_id: str) -> Paper | None:
        """Find a paper in a collection by external identifier.

        Matches by exact external_id or normalized base (ignoring arXiv version suffix).
        Returns the earliest created matching record deterministically if multiple exist.
        """
        if not isinstance(collection_id, UUID):
            raise TypeError("collection_id must be a UUID")
        if not isinstance(external_id, str) or not external_id.strip():
            raise ValueError("external_id must be a non-empty string")

        clean_id = external_id.strip()
        if clean_id.lower().startswith("arxiv:"):
            clean_id = clean_id[6:].strip()

        base_id = clean_id.split("v")[0] if "v" in clean_id else clean_id

        stmt = (
            select(Paper)
            .where(
                Paper.collection_id == collection_id,
                or_(
                    Paper.external_id == clean_id,
                    Paper.external_id == f"arxiv:{clean_id}",
                    and_(
                        Paper.external_id.is_not(None),
                        or_(
                            Paper.external_id.startswith(base_id + "v"),
                            Paper.external_id == base_id,
                        ),
                    ),
                ),
            )
            .order_by(Paper.created_at.asc(), Paper.id.asc())
        )
        return self._session.scalars(stmt).first()

    def get_by_doi(self, collection_id: UUID, doi: str) -> Paper | None:
        """Find a paper in a collection by normalized DOI.

        Returns the earliest created matching record deterministically if multiple exist.
        """
        if not isinstance(collection_id, UUID):
            raise TypeError("collection_id must be a UUID")
        if not isinstance(doi, str) or not doi.strip():
            raise ValueError("doi must be a non-empty string")

        from nexora.discovery.deduplication import normalize_doi

        clean_doi = normalize_doi(doi)
        if not clean_doi:
            return None

        stmt = (
            select(Paper)
            .where(
                Paper.collection_id == collection_id,
                Paper.doi.is_not(None),
                func.lower(Paper.doi).in_([
                    clean_doi,
                    f"https://doi.org/{clean_doi}",
                    f"http://doi.org/{clean_doi}",
                    f"doi:{clean_doi}",
                ]),
            )
            .order_by(Paper.created_at.asc(), Paper.id.asc())
        )
        return self._session.scalars(stmt).first()

    def find_in_collection(
        self,
        collection_id: UUID,
        *,
        external_id: str | None = None,
        doi: str | None = None,
    ) -> tuple[Paper | None, str]:
        """Find an existing paper in a collection by external ID (priority 1) or DOI (priority 2).

        Returns (paper, match_type) where match_type is 'external_id', 'doi', or 'none'.
        Never matches by title alone.
        """
        if external_id is not None and str(external_id).strip():
            found = self.get_by_external_id(collection_id, str(external_id).strip())
            if found is not None:
                return found, "external_id"

        if doi is not None and str(doi).strip():
            found = self.get_by_doi(collection_id, str(doi).strip())
            if found is not None:
                return found, "doi"

        return None, "none"

    def get_or_create_paper(
        self,
        *,
        collection_id: UUID,
        title: str,
        abstract: str | None = None,
        doi: str | None = None,
        external_id: str | None = None,
        publication_date: datetime.date | None = None,
    ) -> tuple[Paper, bool]:
        """Get an existing paper if found by external ID or DOI, otherwise create a new one.

        Returns (paper, created) where created is True if a new paper was created,
        or False if an existing duplicate was returned.
        """
        existing, _ = self.find_in_collection(
            collection_id, external_id=external_id, doi=doi
        )
        if existing is not None:
            return existing, False

        new_paper = self.create(
            collection_id=collection_id,
            title=title,
            abstract=abstract,
            doi=doi,
            external_id=external_id,
            publication_date=publication_date,
        )
        return new_paper, True
