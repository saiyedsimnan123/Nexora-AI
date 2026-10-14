"""Evidence-backed research gap discovery service (Milestone 12C-26).

Coordinates:
- Structured analysis of individual papers to extract author-documented limitations and future work.
- Cross-paper LLM synthesis to infer candidate hypotheses across contrasting methodologies and unexamined trade-offs.
- Traceable citation linking to concrete evidence chunks.
- Contradiction / qualification detection when one paper resolves another paper's limitation.
- Strict zero-hallucination guarantees and empty evidence short-circuits.
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any, Sequence
from uuid import UUID

from nexora.api.errors import (
    DocumentNotFoundError,
    GapDiscoveryServiceError,
    InvalidRequestError,
)
from nexora.database.repository.paper import PaperRepository
from nexora.research.analysis import StructuredPaperAnalysis
from nexora.research.analysis_service import PaperAnalysisService, _clean_json_text
from nexora.research.gap import GapCategory, ResearchGap, ResearchGapReport
from nexora.research.gap_prompt import ResearchGapPromptBuilder
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem


def _extract_title(prefix: str, text: str, paper_title: str | None = None) -> str:
    """Produce a concise, descriptive title from a limitation or future-work sentence."""
    clean = text.strip().split(".")[0].strip()
    words = clean.split()
    if len(words) > 10:
        clean = " ".join(words[:10]) + "..."
    if not clean:
        clean = "Author Documented Finding"
    label = f" ({paper_title})" if paper_title else ""
    return f"{prefix}{label}: {clean}"


class ResearchGapService:
    """Orchestrates evidence-grounded research gap discovery across literature."""

    def __init__(
        self,
        llm_service: Any,
        *,
        paper_analysis_service: PaperAnalysisService | None = None,
        session_factory: Any = None,
        vector_store: Any = None,
        prompt_builder: ResearchGapPromptBuilder | None = None,
        max_evidence_characters: int | None = None,
        filter_invalid_citations: bool = True,
    ) -> None:
        if not callable(getattr(llm_service, "generate", None)):
            raise TypeError("llm_service must provide generate(messages, ...)")
        self._llm = llm_service
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._prompt_builder = prompt_builder or ResearchGapPromptBuilder()
        self._max_chars = max_evidence_characters
        self._filter_citations = filter_invalid_citations

        if paper_analysis_service is not None:
            self._paper_analysis_service = paper_analysis_service
        else:
            self._paper_analysis_service = PaperAnalysisService(
                llm_service,
                session_factory=session_factory,
                vector_store=vector_store,
                max_evidence_characters=max_evidence_characters,
                filter_invalid_citations=filter_invalid_citations,
            )

    def _load_paper_evidence_and_title(
        self, paper_id_str: str
    ) -> tuple[str | None, list[EvidenceItem]]:
        """Load ordered evidence items and title for a specific paper."""
        paper_uuid = UUID(paper_id_str)

        # 1. Try relational database
        if self._session_factory is not None:
            session = self._session_factory()
            try:
                from nexora.database.repository.document import DocumentRepository
                from nexora.database.repository.document_chunk import DocumentChunkRepository

                paper_repo = PaperRepository(session)
                paper = paper_repo.get_by_id(paper_uuid)
                if paper is None:
                    raise DocumentNotFoundError(f"Paper {paper_uuid} was not found.")

                title = paper.title
                doc_repo = DocumentRepository(session)
                documents = doc_repo.list_by_paper_id(paper_uuid)

                chunk_repo = DocumentChunkRepository(session)
                items: list[EvidenceItem] = []
                used_chars = 0

                for doc in documents:
                    chunks = chunk_repo.list_by_document_id(doc.id)
                    for chunk in chunks:
                        if self._max_chars is not None and used_chars + len(chunk.text) > self._max_chars:
                            break
                        used_chars += len(chunk.text)
                        items.append(
                            EvidenceItem(
                                document_id=str(doc.id),
                                chunk_id=str(chunk.id),
                                text=chunk.text,
                                score=1.0,
                                metadata={
                                    "paper_id": paper_id_str,
                                    "filename": doc.filename,
                                    "chunk_index": chunk.chunk_index,
                                },
                                position=len(items),
                            )
                        )
                return title, items
            finally:
                if hasattr(session, "close"):
                    session.close()

        # 2. Try vector store
        if self._vector_store is not None:
            items: list[EvidenceItem] = []
            records = getattr(self._vector_store, "_records", {})
            used_chars = 0
            title = None

            if isinstance(records, dict):
                matched = [
                    rec
                    for rec in records.values()
                    if str(rec.payload.get("paper_id")) == paper_id_str
                ]
                matched.sort(key=lambda r: int(r.payload.get("chunk_index", 0)))
                for rec in matched:
                    text = str(rec.payload.get("text", ""))
                    if not title and "title" in rec.payload:
                        title = str(rec.payload["title"])
                    if self._max_chars is not None and used_chars + len(text) > self._max_chars:
                        break
                    used_chars += len(text)
                    items.append(
                        EvidenceItem(
                            document_id=str(rec.payload.get("document_id", "doc-default")),
                            chunk_id=rec.id,
                            text=text,
                            score=1.0,
                            metadata=rec.payload,
                            position=len(items),
                        )
                    )
            return title, items

        # 3. Fallback: delegate to analysis service internal loaders if accessible
        if hasattr(self._paper_analysis_service, "_session_factory") and self._paper_analysis_service._session_factory is not None:
            title, ctx = self._paper_analysis_service._load_evidence_from_database(paper_uuid)
            return title, list(ctx.items)
        if hasattr(self._paper_analysis_service, "_vector_store") and self._paper_analysis_service._vector_store is not None:
            title, ctx = self._paper_analysis_service._load_evidence_from_store(paper_id_str)
            return title, list(ctx.items)

        return None, []

    def _load_collection_paper_ids(self, col_uuid: UUID) -> list[str]:
        """Fetch all paper IDs belonging to a collection."""
        if self._session_factory is not None:
            session = self._session_factory()
            try:
                repo = PaperRepository(session)
                papers = repo.list_by_collection_id(col_uuid, limit=50)
                return [str(p.id) for p in papers]
            finally:
                if hasattr(session, "close"):
                    session.close()
        elif self._vector_store is not None:
            records = getattr(self._vector_store, "_records", {})
            pids: list[str] = []
            if isinstance(records, dict):
                for rec in records.values():
                    if str(rec.payload.get("collection_id")) == str(col_uuid):
                        pid = rec.payload.get("paper_id")
                        if pid and str(pid) not in pids:
                            pids.append(str(pid))
            return pids
        return []

    def discover_gaps_for_papers(
        self,
        paper_ids: Sequence[str],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        collection_id: str | None = None,
    ) -> ResearchGapReport:
        """Discover evidence-backed research gaps across one or more research papers."""
        if not isinstance(paper_ids, (list, tuple)):
            raise InvalidRequestError("paper_ids must be a list of strings")
        if len(paper_ids) == 0:
            raise InvalidRequestError("paper_ids must not be empty")
        if len(paper_ids) > 5:
            raise InvalidRequestError(
                f"Comparison exceeds maximum of 5 papers (got {len(paper_ids)})."
            )

        cleaned_ids: list[str] = []
        for idx, pid in enumerate(paper_ids):
            if not isinstance(pid, str) or not pid.strip():
                raise InvalidRequestError(f"paper_ids[{idx}] must be a non-empty string")
            try:
                UUID(pid.strip())
            except (ValueError, AttributeError):
                raise InvalidRequestError(f"Invalid paper_id {pid!r}: must be a valid UUID.")
            cleaned_ids.append(str(UUID(pid.strip())))

        if len(set(cleaned_ids)) != len(cleaned_ids):
            raise InvalidRequestError("paper_ids must be unique with no duplicates.")

        paper_titles: dict[str, str] = {}
        all_raw_items: list[EvidenceItem] = []
        analyses: list[StructuredPaperAnalysis] = []
        documented_gaps: list[ResearchGap] = []

        # 1. Analyze each paper and extract documented limitations / future work
        for pid in cleaned_ids:
            analysis = self._paper_analysis_service.analyze_paper(
                pid,
                model=model,
                temperature=temperature,
            )
            analyses.append(analysis)

            title, items = self._load_paper_evidence_and_title(pid)
            paper_title = title or analysis.title or f"Paper {pid[:8]}"
            paper_titles[pid] = paper_title
            all_raw_items.extend(items)

            if analysis.limitations.is_reported and analysis.limitations.summary:
                doc_gap = ResearchGap(
                    gap_id=f"gap-doc-{uuid.uuid4().hex[:8]}",
                    category=GapCategory.DOCUMENTED.value,
                    title=_extract_title("Documented Limitation", analysis.limitations.summary, paper_title),
                    description=analysis.limitations.summary,
                    reasoning=f"Explicitly documented as an open limitation or constraint by the authors of {paper_title}.",
                    source_paper_ids=[pid],
                    evidence_labels=list(analysis.limitations.evidence_labels),
                    supporting_sources=copy.deepcopy(analysis.limitations.sources),
                    contradicting_sources=[],
                    confidence=1.0,
                    limitations=["Author-reported limitation; scope restricted to published experiments in this paper."],
                )
                documented_gaps.append(doc_gap)

            if analysis.future_work.is_reported and analysis.future_work.summary:
                doc_gap = ResearchGap(
                    gap_id=f"gap-doc-{uuid.uuid4().hex[:8]}",
                    category=GapCategory.DOCUMENTED.value,
                    title=_extract_title("Documented Future Work", analysis.future_work.summary, paper_title),
                    description=analysis.future_work.summary,
                    reasoning=f"Explicitly identified as open future research by the authors of {paper_title}.",
                    source_paper_ids=[pid],
                    evidence_labels=list(analysis.future_work.evidence_labels),
                    supporting_sources=copy.deepcopy(analysis.future_work.sources),
                    contradicting_sources=[],
                    confidence=1.0,
                    limitations=["Author-reported future work; feasibility or subsequent solutions not assessed within this paper."],
                )
                documented_gaps.append(doc_gap)

        # 2. Zero-hallucination check: if all papers lack evidence, return empty report
        if len(all_raw_items) == 0:
            return ResearchGapReport.empty(
                paper_ids=cleaned_ids,
                collection_id=collection_id,
                reason="No evidence chunks were found for the provided papers.",
            )

        # 3. Create unified evidence context with contiguous unique labels [E1], [E2], ...
        combined_items: list[EvidenceItem] = []
        for item in all_raw_items:
            meta = dict(item.metadata)
            combined_items.append(
                EvidenceItem(
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    text=item.text,
                    score=item.score,
                    metadata=meta,
                    position=len(combined_items),
                )
            )
        combined_context = EvidenceContext(
            query=f"Gap discovery for papers: {', '.join(cleaned_ids)}",
            items=combined_items,
            total_candidates=len(combined_items),
        )

        # 4. Construct prompt and generate synthesis
        messages = self._prompt_builder.build(
            paper_titles=paper_titles,
            documented_gaps=documented_gaps,
            evidence_items=combined_items,
        )

        try:
            response = self._llm.generate(
                messages,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise GapDiscoveryServiceError(
                f"Research gap LLM synthesis failed ({type(exc).__name__})"
            ) from exc

        cleaned_json = _clean_json_text(response.text)
        try:
            parsed = json.loads(cleaned_json)
        except Exception as exc:
            raise GapDiscoveryServiceError(
                f"Malformed model response; failed to parse JSON ({type(exc).__name__})"
            ) from exc

        if not isinstance(parsed, dict):
            raise GapDiscoveryServiceError(
                "Malformed model response; root element must be a JSON object"
            )

        # 5. Parse inferred candidate gaps
        raw_inferred = parsed.get("inferred_gaps", [])
        if not isinstance(raw_inferred, list):
            raw_inferred = []

        inferred_gaps: list[ResearchGap] = []
        for idx, raw_gap in enumerate(raw_inferred):
            if not isinstance(raw_gap, dict):
                continue

            title = raw_gap.get("title")
            if not isinstance(title, str) or not title.strip():
                title = f"Candidate Inferred Gap #{idx + 1}"

            description = raw_gap.get("description")
            if not isinstance(description, str) or not description.strip():
                continue

            reasoning = raw_gap.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = "Inferred across contrasting paper methodologies and evidence."

            raw_src_pids = raw_gap.get("source_paper_ids", [])
            if isinstance(raw_src_pids, list):
                src_pids = [str(p) for p in raw_src_pids if str(p) in cleaned_ids]
            else:
                src_pids = []
            if not src_pids:
                src_pids = list(cleaned_ids)

            raw_labels = raw_gap.get("evidence_labels", [])
            if not isinstance(raw_labels, list):
                raw_labels = []

            supporting_sources: list[dict[str, Any]] = []
            valid_labels: list[str] = []
            for lbl in raw_labels:
                if not isinstance(lbl, str):
                    continue
                item = combined_context.get_by_label(lbl)
                if item is not None:
                    valid_labels.append(lbl)
                    supporting_sources.append({
                        "label": lbl,
                        "document_id": item.document_id,
                        "chunk_id": item.chunk_id,
                        "paper_id": item.paper_id,
                        "filename": item.metadata.get("filename"),
                        "score": item.score,
                        "text": item.text,
                        "position": item.position,
                    })

            if self._filter_citations and len(supporting_sources) == 0:
                continue

            raw_contra_labels = raw_gap.get("contradicting_evidence_labels", [])
            if not isinstance(raw_contra_labels, list):
                raw_contra_labels = []

            contradicting_sources: list[dict[str, Any]] = []
            for lbl in raw_contra_labels:
                if not isinstance(lbl, str):
                    continue
                item = combined_context.get_by_label(lbl)
                if item is not None:
                    contradicting_sources.append({
                        "label": lbl,
                        "document_id": item.document_id,
                        "chunk_id": item.chunk_id,
                        "paper_id": item.paper_id,
                        "filename": item.metadata.get("filename"),
                        "score": item.score,
                        "text": item.text,
                        "position": item.position,
                    })

            raw_conf = raw_gap.get("confidence", 0.70)
            if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
                conf = 0.70
            else:
                conf = max(0.0, min(1.0, float(raw_conf)))

            raw_lims = raw_gap.get("limitations", [])
            if isinstance(raw_lims, list):
                lims = [str(l) for l in raw_lims if isinstance(l, str) and l.strip()]
            else:
                lims = []
            if not lims:
                lims = [
                    f"Candidate gap inferred across {len(cleaned_ids)} papers; requires empirical verification.",
                    "External literature beyond the provided papers was not evaluated.",
                ]

            inferred_gap = ResearchGap(
                gap_id=f"gap-inf-{uuid.uuid4().hex[:8]}",
                category=GapCategory.INFERRED.value,
                title=title,
                description=description,
                reasoning=reasoning,
                source_paper_ids=src_pids,
                evidence_labels=valid_labels,
                supporting_sources=supporting_sources,
                contradicting_sources=contradicting_sources,
                confidence=conf,
                limitations=lims,
            )
            inferred_gaps.append(inferred_gap)

        # 6. Apply cross-paper qualifications to documented gaps if reported
        raw_quals = parsed.get("documented_gap_qualifications", [])
        if isinstance(raw_quals, list) and raw_quals:
            updated_documented_gaps: list[ResearchGap] = []
            for doc_gap in documented_gaps:
                matching_contra_sources: list[dict[str, Any]] = []
                matching_notes: list[str] = []
                for qual in raw_quals:
                    if not isinstance(qual, dict):
                        continue
                    qual_title = str(qual.get("title", "")).lower()
                    if qual_title and (qual_title in doc_gap.title.lower() or doc_gap.title.lower() in qual_title):
                        contra_labels = qual.get("contradicting_evidence_labels", [])
                        if isinstance(contra_labels, list):
                            for lbl in contra_labels:
                                if isinstance(lbl, str):
                                    item = combined_context.get_by_label(lbl)
                                    if item is not None:
                                        matching_contra_sources.append({
                                            "label": lbl,
                                            "document_id": item.document_id,
                                            "chunk_id": item.chunk_id,
                                            "paper_id": item.paper_id,
                                            "filename": item.metadata.get("filename"),
                                            "score": item.score,
                                            "text": item.text,
                                            "position": item.position,
                                        })
                        if qual.get("notes"):
                            matching_notes.append(str(qual["notes"]))

                if matching_contra_sources:
                    new_reasoning = doc_gap.reasoning
                    if matching_notes:
                        new_reasoning += f" (Qualified by another paper: {'; '.join(matching_notes)})"
                    new_doc_gap = ResearchGap(
                        gap_id=doc_gap.gap_id,
                        category=doc_gap.category,
                        title=doc_gap.title,
                        description=doc_gap.description,
                        reasoning=new_reasoning,
                        source_paper_ids=doc_gap.source_paper_ids,
                        evidence_labels=doc_gap.evidence_labels,
                        supporting_sources=doc_gap.supporting_sources,
                        contradicting_sources=doc_gap.contradicting_sources + matching_contra_sources,
                        confidence=doc_gap.confidence,
                        limitations=doc_gap.limitations + ["Partially qualified or addressed by another paper in this comparison."],
                    )
                    updated_documented_gaps.append(new_doc_gap)
                else:
                    updated_documented_gaps.append(doc_gap)
            documented_gaps = updated_documented_gaps

        # 7. Overall summary and limitations
        summary = parsed.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = f"Synthesized {len(documented_gaps)} documented and {len(inferred_gaps)} inferred research gaps across {len(cleaned_ids)} papers."

        raw_overall_lims = parsed.get("limitations")
        if isinstance(raw_overall_lims, list):
            overall_lims = [str(l) for l in raw_overall_lims if isinstance(l, str) and l.strip()]
        else:
            overall_lims = []
        if not overall_lims:
            overall_lims = [
                f"Discovery bounded to the {len(cleaned_ids)} specified papers.",
                "Candidate gaps require domain experimental validation.",
            ]

        return ResearchGapReport(
            paper_ids=cleaned_ids,
            collection_id=collection_id,
            documented_gaps=documented_gaps,
            inferred_gaps=inferred_gaps,
            total_gaps_count=len(documented_gaps) + len(inferred_gaps),
            summary=summary,
            limitations=overall_lims,
            model=response.model,
            usage=response.usage,
            raw=response.raw,
        )

    def discover_gaps_for_collection(
        self,
        collection_id: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ResearchGapReport:
        """Discover research gaps across papers in an existing collection."""
        if not isinstance(collection_id, str) or not collection_id.strip():
            raise InvalidRequestError("collection_id is required and must be a string")
        try:
            col_uuid = UUID(collection_id.strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError(f"Invalid collection_id {collection_id!r}: must be a valid UUID.")

        paper_ids = self._load_collection_paper_ids(col_uuid)
        if not paper_ids:
            return ResearchGapReport.empty(
                [],
                reason=f"Collection {col_uuid} contains no papers.",
                collection_id=str(col_uuid),
            )

        truncated_note = None
        if len(paper_ids) > 5:
            paper_ids = paper_ids[:5]
            truncated_note = "Collection contains more than 5 papers; gap discovery was bounded to the first 5 papers."

        report = self.discover_gaps_for_papers(
            paper_ids,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            collection_id=str(col_uuid),
        )

        if truncated_note and truncated_note not in report.limitations:
            report = ResearchGapReport(
                paper_ids=report.paper_ids,
                collection_id=report.collection_id,
                documented_gaps=report.documented_gaps,
                inferred_gaps=report.inferred_gaps,
                total_gaps_count=report.total_gaps_count,
                summary=report.summary,
                limitations=report.limitations + [truncated_note],
                model=report.model,
                usage=report.usage,
                raw=report.raw,
            )

        return report
