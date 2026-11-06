"""Knowledge graph extraction application service (Stage 5 Step 2).

Extracts validated scientific entities and evidence-grounded relationships
from paper metadata, documents, chunks, and structured analysis.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from nexora.database.models.document import Document
from nexora.database.models.document_chunk import DocumentChunkRecord
from nexora.database.models.graph_entity import (
    GraphEntity,
    GraphEntityType,
    normalize_entity_name,
)
from nexora.database.models.graph_relationship import (
    GraphEndpointType,
    GraphRelationship,
)
from nexora.database.models.paper import Paper
from nexora.database.repository.author import AuthorRepository
from nexora.database.repository.base import ConstraintError, RecordNotFoundError
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.graph_entity import GraphEntityRepository
from nexora.database.repository.graph_relationship import GraphRelationshipRepository
from nexora.database.repository.paper import PaperRepository
from nexora.database.repository.paper_author import PaperAuthorRepository
from nexora.database.repository.paper_reference import PaperReferenceRepository
from nexora.graph.models import (
    ExtractedEntityCandidate,
    ExtractedGraphPayload,
    ExtractedRelationshipCandidate,
    GraphExtractionError,
    GraphExtractionResult,
    InvalidExtractionPayloadError,
)
from nexora.graph.prompt import GraphExtractionPromptBuilder
from nexora.research.analysis import StructuredPaperAnalysis
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _clean_json_text(text: str) -> str:
    """Extract raw JSON string from optional markdown fences."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


_CITE_LABEL_RE = re.compile(
    r"^(?:\[\s*cite:\s*(\d+)\s*\]|cite:\s*(\d+)|\[\s*(\d+)\s*\])(?:\s*\(\s*page\s*\d+\s*\))?$",
    re.IGNORECASE,
)


def _resolve_evidence_item(
    label: str | None,
    context: EvidenceContext,
) -> tuple[UUID | None, int | None, str | None]:
    """Resolve chunk_id, page_number, and text excerpt from evidence label and context."""
    if not label or not isinstance(label, str):
        return None, None, None

    cleaned = label.strip()

    # 1. Check if label is a direct chunk UUID
    try:
        raw_uuid = UUID(cleaned)
        for item in context.items:
            if item.chunk_id == str(raw_uuid):
                return raw_uuid, item.page_number, item.text
        # Unverified UUID: Not present in actual extraction context -> reject
        return None, None, None
    except ValueError:
        pass

    # 2. Check standard EvidenceContext.get_by_label (e.g. "[E1]", "E1")
    item = context.get_by_label(cleaned)
    if item is not None:
        try:
            return UUID(item.chunk_id), item.page_number, item.text
        except ValueError:
            return None, item.page_number, item.text

    # 3. Check documented citation label formats (e.g. "[cite: 0]", "cite: 0", "[0]")
    match = _CITE_LABEL_RE.match(cleaned)
    if match:
        idx_str = match.group(1) or match.group(2) or match.group(3)
        if idx_str is not None:
            idx = int(idx_str)
            if 0 <= idx < len(context.items):
                target = context.items[idx]
                try:
                    return UUID(target.chunk_id), target.page_number, target.text
                except ValueError:
                    return None, target.page_number, target.text

    return None, None, None


class KnowledgeGraphExtractionService:
    """Extracts, validates, and persists knowledge graph nodes and edges for papers."""

    def __init__(
        self,
        *,
        llm_service: Any = None,
        session_factory: Any = None,
        prompt_builder: GraphExtractionPromptBuilder | None = None,
        max_evidence_characters: int = 40000,
    ) -> None:
        self._llm = llm_service
        self._session_factory = session_factory
        self._prompt_builder = prompt_builder or GraphExtractionPromptBuilder()
        self._max_chars = max_evidence_characters

    def extract_paper(
        self,
        paper_id: UUID,
        *,
        collection_id: UUID,
        session: Session | None = None,
        authors: Sequence[str] | None = None,
        citations: Sequence[UUID | str] | None = None,
        analysis: StructuredPaperAnalysis | None = None,
        run_llm: bool = True,
    ) -> GraphExtractionResult:
        """Extract scientific entities and relationships for a paper within an authorized collection.

        Execution is wrapped in a database savepoint to guarantee that extraction failures
        cannot delete or corrupt pre-existing papers, documents, or chunks.

        Returns:
            GraphExtractionResult reporting created, reused, and linked counts.
        """
        if not isinstance(paper_id, UUID):
            raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        owns_session = False
        db_session = session

        if db_session is None:
            if self._session_factory is not None:
                db_session = self._session_factory()
                owns_session = True
            else:
                raise GraphExtractionError(
                    "KnowledgeGraphExtractionService requires a session_factory or active database session."
                )

        result = GraphExtractionResult(
            paper_id=paper_id,
            collection_id=collection_id,
        )

        try:
            # Enforce paper existence and collection boundary
            paper_repo = PaperRepository(db_session)
            paper = paper_repo.get_by_id(paper_id)
            if paper is None:
                raise RecordNotFoundError(f"Paper with id {paper_id} not found.")
            if paper.collection_id != collection_id:
                raise ConstraintError(
                    f"Cross-collection extraction prohibited: Paper {paper_id} belongs to "
                    f"collection {paper.collection_id}, not requested collection {collection_id}."
                )

            # Protect graph extraction in a nested savepoint
            with db_session.begin_nested():
                self.extract_metadata_deterministic(
                    paper_id=paper.id,
                    collection_id=collection_id,
                    session=db_session,
                    authors=authors,
                    citations=citations,
                    analysis=analysis,
                    result=result,
                )

                # 4. LLM-assisted extraction from document chunks if requested
                if run_llm and self._llm is not None:
                    self._extract_from_chunks_with_llm(
                        paper=paper,
                        collection_id=collection_id,
                        session=db_session,
                        result=result,
                    )

                db_session.flush()

            if owns_session:
                db_session.commit()

        except Exception as exc:
            logger.warning("Knowledge graph extraction failed for paper %s: %s", paper_id, exc)
            result.status = "failed"
            result.errors.append(str(exc))
            # Reset counters and in-memory collections because savepoint rolled back
            result.entities_created = 0
            result.entities_reused = 0
            result.relationships_created = 0
            result.relationships_reused = 0
            result.authors_linked = 0
            result.citations_linked = 0
            result.entities.clear()
            result.relationships.clear()
            if owns_session:
                db_session.rollback()
        finally:
            if owns_session:
                db_session.close()

        return result

    def extract_metadata_deterministic(
        self,
        paper_id: UUID,
        *,
        collection_id: UUID,
        session: Session,
        authors: Sequence[str] | None = None,
        citations: Sequence[UUID | str] | None = None,
        analysis: StructuredPaperAnalysis | None = None,
        result: GraphExtractionResult | None = None,
    ) -> GraphExtractionResult:
        """Deterministically extract and persist author, citation, and analysis graph data."""
        if not isinstance(paper_id, UUID):
            raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        paper_repo = PaperRepository(session)
        paper = paper_repo.get_by_id(paper_id)
        if paper is None:
            raise RecordNotFoundError(f"Paper with id {paper_id} not found.")
        if paper.collection_id != collection_id:
            raise ConstraintError(
                f"Cross-collection extraction prohibited: Paper {paper_id} belongs to "
                f"collection {paper.collection_id}, not requested collection {collection_id}."
            )

        if result is None:
            result = GraphExtractionResult(
                paper_id=paper_id,
                collection_id=collection_id,
            )

        # Self-healing: if authors not provided, discover existing authors for paper
        if authors is None:
            paper_author_repo = PaperAuthorRepository(session)
            author_repo = AuthorRepository(session)
            pa_records = paper_author_repo.list_by_paper_id(paper.id)
            if pa_records:
                discovered_authors: list[str] = []
                for pa in pa_records:
                    a = author_repo.get_by_id(pa.author_id)
                    if a and a.name and a.name not in discovered_authors:
                        discovered_authors.append(a.name)
                authors = discovered_authors

        with session.begin_nested():
            # 1. Deterministic author extraction & linking
            self._extract_authors(
                paper=paper,
                collection_id=collection_id,
                authors=authors,
                session=session,
                result=result,
            )

            # 2. Deterministic citation extraction & linking
            if citations:
                self._extract_citations(
                    paper=paper,
                    collection_id=collection_id,
                    citations=citations,
                    session=session,
                    result=result,
                )

            # 3. Deterministic extraction from StructuredPaperAnalysis if provided
            if analysis is not None:
                self._extract_from_structured_analysis(
                    paper=paper,
                    collection_id=collection_id,
                    analysis=analysis,
                    session=session,
                    result=result,
                )

            session.flush()

        return result

    def _extract_authors(
        self,
        *,
        paper: Paper,
        collection_id: UUID,
        authors: Sequence[str] | None,
        session: Session,
        result: GraphExtractionResult,
    ) -> None:
        """Deterministically extract and persist author relationships from metadata."""
        author_names: list[str] = []
        if authors:
            for a in authors:
                if isinstance(a, str) and a.strip() and a.strip() not in author_names:
                    author_names.append(a.strip())

        if not author_names:
            return

        author_repo = AuthorRepository(session)
        paper_author_repo = PaperAuthorRepository(session)
        ent_repo = GraphEntityRepository(session)
        rel_repo = GraphRelationshipRepository(session)

        for idx, name in enumerate(author_names):
            # 1. Relational author table
            author = author_repo.get_by_name(name)
            if author is None:
                author = author_repo.create(name=name)

            # 2. Relational paper-author link
            existing_pa = paper_author_repo.get_by_paper_and_author(paper.id, author.id)
            if existing_pa is None:
                paper_author_repo.create(
                    paper_id=paper.id,
                    author_id=author.id,
                    author_order=idx + 1,
                )

            # 3. Graph author entity (collection-scoped)
            author_ent, ent_created = ent_repo.get_or_create(
                collection_id=collection_id,
                name=name,
                entity_type=GraphEntityType.AUTHOR.value,
            )
            if ent_created:
                result.entities_created += 1
            else:
                result.entities_reused += 1
            result.entities.append(author_ent)

            # 4. Graph authored_by relationship
            rel, rel_created = rel_repo.get_or_create(
                collection_id=collection_id,
                relationship_type="authored_by",
                source_type=GraphEndpointType.PAPER.value,
                source_id=paper.id,
                target_type=GraphEndpointType.ENTITY.value,
                target_id=author_ent.id,
                confidence=1.0,
                extraction_method="metadata",
            )
            if rel_created:
                result.relationships_created += 1
            else:
                result.relationships_reused += 1
            result.relationships.append(rel)
            result.authors_linked += 1

    def _extract_citations(
        self,
        *,
        paper: Paper,
        collection_id: UUID,
        citations: Sequence[UUID | str],
        session: Session,
        result: GraphExtractionResult,
    ) -> None:
        """Link citation references between papers within the authorized collection."""
        paper_repo = PaperRepository(session)
        ref_repo = PaperReferenceRepository(session)
        rel_repo = GraphRelationshipRepository(session)
        seen_targets: set[UUID] = set()

        for target_ref in citations:
            target_paper: Paper | None = None
            if isinstance(target_ref, UUID):
                target_paper = paper_repo.get_by_id(target_ref)
            elif isinstance(target_ref, str):
                cleaned = target_ref.strip()
                try:
                    target_uuid = UUID(cleaned)
                    target_paper = paper_repo.get_by_id(target_uuid)
                except ValueError:
                    # Look up by external_id or doi in collection
                    existing, _ = paper_repo.find_in_collection(
                        collection_id, external_id=cleaned, doi=cleaned
                    )
                    target_paper = existing

            # Strict collection boundary: Target must exist and belong to the same collection
            if target_paper is None or target_paper.collection_id != collection_id:
                continue

            # Prevent self-citation link
            if target_paper.id == paper.id:
                continue

            # Deduplicate targets within the same run
            if target_paper.id in seen_targets:
                continue
            seen_targets.add(target_paper.id)

            # 1. Relational citation reference
            existing_ref = ref_repo.get_by_source_and_target(paper.id, target_paper.id)
            if existing_ref is None:
                ref_repo.create(
                    source_paper_id=paper.id,
                    target_paper_id=target_paper.id,
                )

            # 2. Graph cites relationship
            rel, rel_created = rel_repo.get_or_create(
                collection_id=collection_id,
                relationship_type="cites",
                source_type=GraphEndpointType.PAPER.value,
                source_id=paper.id,
                target_type=GraphEndpointType.PAPER.value,
                target_id=target_paper.id,
                confidence=1.0,
                extraction_method="citation_match",
            )
            if rel_created:
                result.relationships_created += 1
            else:
                result.relationships_reused += 1
            result.relationships.append(rel)
            result.citations_linked += 1

    def _extract_from_structured_analysis(
        self,
        *,
        paper: Paper,
        collection_id: UUID,
        analysis: StructuredPaperAnalysis,
        session: Session,
        result: GraphExtractionResult,
    ) -> None:
        """Map grounded dimensions from StructuredPaperAnalysis into graph entities and edges."""
        ent_repo = GraphEntityRepository(session)
        rel_repo = GraphRelationshipRepository(session)

        dimension_mappings = [
            ("methodology_and_algorithms", GraphEntityType.METHOD.value, "proposes"),
            ("models_and_datasets", GraphEntityType.DATASET.value, "uses_dataset"),
            ("experiments_and_metrics", GraphEntityType.METRIC.value, "reports_metric"),
            ("problem_and_objectives", GraphEntityType.CONCEPT.value, "discusses"),
        ]

        for field_name, entity_type, rel_type in dimension_mappings:
            field = analysis.get_field(field_name)
            if field.status != "reported" or not field.summary.strip():
                continue
            if field.summary == "Not reported in the provided evidence.":
                continue

            # Use first valid source chunk for grounded provenance
            chunk_id: UUID | None = None
            page_number: int | None = None
            excerpt: str | None = None

            if field.sources:
                src0 = field.sources[0]
                raw_cid = src0.get("chunk_id")
                if raw_cid:
                    try:
                        chunk_id = UUID(str(raw_cid))
                    except (ValueError, TypeError):
                        chunk_id = None
                raw_page = src0.get("page_number")
                if isinstance(raw_page, int) and raw_page >= 1:
                    page_number = raw_page
                raw_text = src0.get("text")
                if isinstance(raw_text, str) and raw_text.strip():
                    excerpt = raw_text.strip()[:500]

            # Create canonical entity from dimension summary
            # (Truncate or extract representative entity name)
            summary_words = field.summary.strip().split(".")
            candidate_name = summary_words[0].strip()
            if len(candidate_name) > 100:
                candidate_name = candidate_name[:97] + "..."
            if not candidate_name:
                continue

            ent, ent_created = ent_repo.get_or_create(
                collection_id=collection_id,
                name=candidate_name,
                entity_type=entity_type,
                description=field.summary[:5000],
            )
            if ent_created:
                result.entities_created += 1
            else:
                result.entities_reused += 1
            result.entities.append(ent)

            rel, rel_created = rel_repo.get_or_create(
                collection_id=collection_id,
                relationship_type=rel_type,
                source_type=GraphEndpointType.PAPER.value,
                source_id=paper.id,
                target_type=GraphEndpointType.ENTITY.value,
                target_id=ent.id,
                paper_id=paper.id,
                chunk_id=chunk_id,
                page_number=page_number,
                evidence_text=excerpt,
                confidence=0.90 if chunk_id else 0.75,
                extraction_method="structured_analysis",
            )
            if rel_created:
                result.relationships_created += 1
            else:
                result.relationships_reused += 1
            result.relationships.append(rel)

    def prepare_llm_evidence_context(
        self,
        paper_id: UUID,
        *,
        collection_id: UUID,
        session: Session,
    ) -> tuple[Paper, EvidenceContext, list[dict[str, Any]]]:
        """Prepare grounded EvidenceContext and prompt messages from paper chunks without holding locks."""
        if not isinstance(paper_id, UUID):
            raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        paper_repo = PaperRepository(session)
        paper = paper_repo.get_by_id(paper_id)
        if paper is None:
            raise RecordNotFoundError(f"Paper with id {paper_id} not found.")
        if paper.collection_id != collection_id:
            raise ConstraintError(
                f"Cross-collection extraction prohibited: Paper {paper_id} belongs to "
                f"collection {paper.collection_id}, not requested collection {collection_id}."
            )

        doc_repo = DocumentRepository(session)
        chunk_repo = DocumentChunkRepository(session)
        documents = doc_repo.list_by_paper_id(paper.id)
        if not documents:
            return paper, EvidenceContext(query=paper.title or "", items=[]), []

        items: list[EvidenceItem] = []
        used_chars = 0

        for doc in documents:
            chunks = chunk_repo.list_by_document_id(doc.id)
            for chunk in chunks:
                if used_chars + len(chunk.text) > self._max_chars:
                    break
                used_chars += len(chunk.text)
                items.append(
                    EvidenceItem(
                        document_id=str(doc.id),
                        chunk_id=str(chunk.id),
                        text=chunk.text,
                        score=1.0,
                        metadata={
                            "paper_id": str(paper.id),
                            "filename": doc.filename,
                            "chunk_index": chunk.chunk_index,
                        },
                        position=chunk.chunk_index,
                    )
                )

        if not items:
            return paper, EvidenceContext(query=paper.title or "", items=[]), []

        context = EvidenceContext(query=paper.title, items=items)
        messages = self._prompt_builder.build(
            paper_id=str(paper.id),
            context=context,
            title=paper.title,
            abstract=paper.abstract,
        )
        return paper, context, messages

    def parse_llm_extraction_response(
        self,
        response_text: str,
    ) -> ExtractedGraphPayload:
        """Parse and validate LLM output into ExtractedGraphPayload."""
        cleaned_json = _clean_json_text(response_text)
        try:
            raw_dict = json.loads(cleaned_json)
        except Exception as exc:
            raise InvalidExtractionPayloadError(f"LLM returned invalid JSON: {exc}") from exc

        try:
            return ExtractedGraphPayload.model_validate(raw_dict)
        except ValidationError as exc:
            raise InvalidExtractionPayloadError(
                f"Extraction payload failed schema validation: {exc}"
            ) from exc

    def persist_llm_extracted_payload(
        self,
        paper_id: UUID,
        *,
        collection_id: UUID,
        session: Session,
        payload: ExtractedGraphPayload,
        context: EvidenceContext,
        result: GraphExtractionResult | None = None,
    ) -> GraphExtractionResult:
        """Persist entities and relationships from parsed LLM payload inside a transaction."""
        if not isinstance(paper_id, UUID):
            raise TypeError(f"paper_id must be a UUID, got {type(paper_id).__name__}")
        if not isinstance(collection_id, UUID):
            raise TypeError(f"collection_id must be a UUID, got {type(collection_id).__name__}")

        paper = PaperRepository(session).get_by_id(paper_id)
        if paper is None:
            raise RecordNotFoundError(f"Paper with id {paper_id} not found.")
        if paper.collection_id != collection_id:
            raise ConstraintError(
                f"Cross-collection extraction prohibited: Paper {paper_id} belongs to "
                f"collection {paper.collection_id}, not requested collection {collection_id}."
            )

        if result is None:
            result = GraphExtractionResult(
                paper_id=paper_id,
                collection_id=collection_id,
            )

        ent_repo = GraphEntityRepository(session)
        rel_repo = GraphRelationshipRepository(session)

        # 1. Persist extracted entities
        resolved_entities: dict[str, GraphEntity] = {}
        for ent_cand in payload.entities:
            ent, created = ent_repo.get_or_create(
                collection_id=collection_id,
                name=ent_cand.name,
                entity_type=ent_cand.entity_type,
                description=ent_cand.description,
                aliases=ent_cand.aliases,
                entity_metadata=ent_cand.entity_metadata,
            )
            resolved_entities[ent_cand.name.lower()] = ent
            resolved_entities[ent_cand.canonical_name] = ent
            if created:
                result.entities_created += 1
            else:
                result.entities_reused += 1
            result.entities.append(ent)

        # 2. Persist extracted relationships
        for rel_cand in payload.relationships:
            # Resolve source endpoint
            source_id: UUID | None = None
            src_type = rel_cand.source_type.lower()
            if src_type == GraphEndpointType.PAPER.value or src_type == "paper":
                source_id = paper.id
            else:
                src_ent = resolved_entities.get(rel_cand.source_name.lower())
                if src_ent is None:
                    src_ent = ent_repo.get_by_canonical_name(
                        collection_id=collection_id,
                        entity_type=GraphEntityType.CONCEPT.value,
                        canonical_name=normalize_entity_name(rel_cand.source_name),
                    )
                if src_ent is not None:
                    source_id = src_ent.id

            # Resolve target endpoint
            target_id: UUID | None = None
            tgt_type = rel_cand.target_type.lower()
            if tgt_type == GraphEndpointType.PAPER.value or tgt_type == "paper":
                clean_target = rel_cand.target_name.strip()
                is_current_paper = (
                    clean_target.lower() == (paper.title or "").strip().lower()
                    or clean_target.lower() in ("this paper", "current paper", "paper")
                    or clean_target.lower() == str(paper.id).lower()
                )
                if is_current_paper:
                    if src_type in (GraphEndpointType.PAPER.value, "paper"):
                        # Genuine self loop: paper referencing itself
                        continue
                    target_id = paper.id
                else:
                    # Search target paper in collection
                    t_paper = None
                    try:
                        t_uuid = UUID(clean_target)
                        t_paper = PaperRepository(session).get_by_id(t_uuid)
                    except ValueError:
                        t_paper, _ = PaperRepository(session).find_in_collection(
                            collection_id,
                            external_id=clean_target,
                            doi=clean_target,
                        )
                    if t_paper is None:
                        stmt = (
                            select(Paper)
                            .where(
                                Paper.collection_id == collection_id,
                                func.lower(Paper.title) == clean_target.lower(),
                            )
                        )
                        t_paper = session.scalars(stmt).first()
                    if t_paper is not None:
                        target_id = t_paper.id
            else:
                tgt_ent = resolved_entities.get(rel_cand.target_name.lower())
                if tgt_ent is None:
                    tgt_ent = ent_repo.get_by_canonical_name(
                        collection_id=collection_id,
                        entity_type=GraphEntityType.CONCEPT.value,
                        canonical_name=normalize_entity_name(rel_cand.target_name),
                    )
                if tgt_ent is not None:
                    target_id = tgt_ent.id

            if source_id is None or target_id is None:
                continue
            if src_type == tgt_type and source_id == target_id:
                continue

            # Ground evidence provenance from context
            chunk_id, page_num, def_text = _resolve_evidence_item(
                rel_cand.evidence_label, context
            )
            excerpt = rel_cand.evidence_quote or (def_text[:500] if def_text else None)
            cand_page = getattr(rel_cand, "page_number", None)
            if cand_page is not None and cand_page >= 1:
                page_num = cand_page

            rel, created = rel_repo.get_or_create(
                collection_id=collection_id,
                relationship_type=rel_cand.relationship_type,
                source_type=src_type,
                source_id=source_id,
                target_type=tgt_type,
                target_id=target_id,
                paper_id=paper.id,
                chunk_id=chunk_id,
                page_number=page_num,
                evidence_text=excerpt,
                confidence=rel_cand.confidence or (0.90 if chunk_id else 0.70),
                extraction_method=rel_cand.extraction_method or "llm_extraction",
            )
            if created:
                result.relationships_created += 1
            else:
                result.relationships_reused += 1
            result.relationships.append(rel)

        return result

    def _extract_from_chunks_with_llm(
        self,
        *,
        paper: Paper,
        collection_id: UUID,
        session: Session,
        result: GraphExtractionResult,
    ) -> None:
        """Perform grounded LLM extraction over document chunks."""
        _, context, messages = self.prepare_llm_evidence_context(
            paper.id, collection_id=collection_id, session=session
        )
        if not messages:
            return

        response = self._llm.generate(messages, temperature=0.0)
        payload = self.parse_llm_extraction_response(response.text)
        self.persist_llm_extracted_payload(
            paper.id,
            collection_id=collection_id,
            session=session,
            payload=payload,
            context=context,
            result=result,
        )
