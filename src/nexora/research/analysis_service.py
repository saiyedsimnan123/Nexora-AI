"""Services for structured paper analysis and multi-paper comparison (Milestone 12C-24)."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import UUID

from nexora.api.errors import (
    AnalysisServiceError,
    ComparisonServiceError,
    DocumentNotFoundError,
    InvalidRequestError,
)
from nexora.database.repository.document import DocumentRepository
from nexora.database.repository.document_chunk import DocumentChunkRepository
from nexora.database.repository.paper import PaperRepository
from nexora.research.analysis import (
    ANALYSIS_DIMENSIONS,
    AnalysisField,
    StructuredPaperAnalysis,
)
from nexora.research.comparison import FieldComparison, PaperComparison
from nexora.research.prompt import (
    PaperAnalysisPromptBuilder,
    PaperComparisonPromptBuilder,
)
from nexora.retrieval.evidence import (
    EvidenceContext,
    EvidenceItem,
    filter_unsupplied_citations,
)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _clean_json_text(text: str) -> str:
    """Extract raw JSON text from markdown code blocks if present."""
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_field_dict(
    raw_dict: Any,
    context: EvidenceContext,
    filter_citations: bool = True,
    expected_paper_id: str | None = None,
) -> AnalysisField:
    """Parse and validate an individual AnalysisField dictionary from model output."""
    if not isinstance(raw_dict, dict):
        return AnalysisField.not_reported()

    summary = raw_dict.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = "Not reported in the provided evidence."

    raw_labels = raw_dict.get("evidence_labels", [])
    if not isinstance(raw_labels, list):
        raw_labels = []

    if filter_citations:
        # Strip unsupplied labels and keep only those present in context
        filtered_text = filter_unsupplied_citations(" ".join(str(l) for l in raw_labels), len(context.items))
        from nexora.retrieval.evidence import extract_citations
        valid_labels = extract_citations(filtered_text)
    else:
        valid_labels = [str(l) for l in raw_labels if isinstance(l, str)]

    sources: list[dict[str, Any]] = []
    strictly_valid_labels: list[str] = []
    for lbl in valid_labels:
        item = context.get_by_label(lbl)
        if item is not None:
            # Paper scoping provenance check: ensure chunk belongs to expected paper
            if expected_paper_id is not None and item.paper_id is not None:
                if str(item.paper_id) != str(expected_paper_id):
                    continue
            strictly_valid_labels.append(lbl)
            sources.append({
                "label": lbl,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "paper_id": item.paper_id,
                "filename": item.metadata.get("filename"),
                "score": item.score,
                "text": item.text,
                "position": item.position,
            })

    status = raw_dict.get("status")
    if status not in ("reported", "not_reported", "uncertain"):
        status = "reported" if summary != "Not reported in the provided evidence." else "not_reported"

    # Provenance hardening: A 'reported' status alone must not qualify without valid source chunks.
    if status == "reported":
        if summary == "Not reported in the provided evidence.":
            status = "not_reported"
        elif len(sources) == 0:
            status = "uncertain"

    details = raw_dict.get("details", {})
    if not isinstance(details, dict):
        details = {}

    return AnalysisField(
        summary=summary,
        status=status,
        evidence_labels=strictly_valid_labels,
        sources=sources,
        details=details,
    )



class PaperAnalysisService:
    """Orchestrates structured paper analysis over ingested evidence and LLM abstractions."""

    def __init__(
        self,
        llm_service: Any,
        *,
        session_factory: Any = None,
        vector_store: Any = None,
        prompt_builder: PaperAnalysisPromptBuilder | None = None,
        max_evidence_characters: int | None = None,
        filter_invalid_citations: bool = True,
    ) -> None:
        if not callable(getattr(llm_service, "generate", None)):
            raise TypeError("llm_service must provide generate(messages, ...)")
        self._llm = llm_service
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._prompt_builder = prompt_builder or PaperAnalysisPromptBuilder()
        self._max_chars = max_evidence_characters
        self._filter_citations = filter_invalid_citations

    def _load_evidence_from_database(
        self, paper_uuid: UUID
    ) -> tuple[str | None, EvidenceContext]:
        """Fetch paper metadata and ordered chunks from relational database."""
        session = self._session_factory()
        try:
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
                                "paper_id": str(paper_uuid),
                                "filename": doc.filename,
                                "chunk_index": chunk.chunk_index,
                            },
                            position=len(items),
                        )
                    )
            return title, EvidenceContext(query=f"Analysis of paper {paper_uuid}", items=items, total_candidates=len(items))
        finally:
            if hasattr(session, "close"):
                session.close()

    def _load_evidence_from_store(
        self, paper_id_str: str
    ) -> tuple[str | None, EvidenceContext]:
        """Fetch chunks matching paper_id from vector store or fallback."""
        items: list[EvidenceItem] = []
        records = getattr(self._vector_store, "_records", {})
        used_chars = 0
        title = None

        if isinstance(records, dict):
            matched = [
                rec for rec in records.values()
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

        return title, EvidenceContext(query=f"Analysis of paper {paper_id_str}", items=items, total_candidates=len(items))

    def analyze_paper(
        self,
        paper_id: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredPaperAnalysis:
        """Perform structured analysis on an ingested paper."""
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise InvalidRequestError("paper_id is required and must be a string")

        try:
            paper_uuid = UUID(paper_id.strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError(f"Invalid paper_id {paper_id!r}: must be a valid UUID.")

        paper_id_str = str(paper_uuid)

        # 1. Retrieve evidence
        if self._session_factory is not None:
            title, context = self._load_evidence_from_database(paper_uuid)
        elif self._vector_store is not None:
            title, context = self._load_evidence_from_store(paper_id_str)
        else:
            title = None
            context = EvidenceContext(query=f"Analysis of paper {paper_id_str}", items=[])

        # 2. Empty evidence short-circuit (predictable, no hallucination)
        if not context.items:
            return StructuredPaperAnalysis.empty(
                paper_id=paper_id_str,
                title=title,
                model=model or "none",
            )

        # 3. Assemble prompt and call LLM
        messages = self._prompt_builder.build(paper_id_str, context, title=title)
        try:
            response = self._llm.generate(
                messages,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise AnalysisServiceError(
                f"Paper analysis LLM call failed ({type(exc).__name__})"
            ) from exc

        # 4. Parse JSON output
        cleaned = _clean_json_text(response.text)
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            raise AnalysisServiceError(
                f"Malformed model response; failed to parse JSON ({type(exc).__name__})"
            ) from exc

        if not isinstance(parsed, dict):
            raise AnalysisServiceError("Malformed model response; root element must be a JSON object")

        # 5. Build validated fields for each dimension
        fields: dict[str, AnalysisField] = {}
        for dim in ANALYSIS_DIMENSIONS:
            raw_field = parsed.get(dim)
            fields[dim] = _parse_field_dict(
                raw_field,
                context,
                self._filter_citations,
                expected_paper_id=paper_id_str,
            )



        return StructuredPaperAnalysis(
            paper_id=paper_id_str,
            title=title,
            problem_and_objectives=fields["problem_and_objectives"],
            methodology_and_algorithms=fields["methodology_and_algorithms"],
            models_and_datasets=fields["models_and_datasets"],
            experiments_and_metrics=fields["experiments_and_metrics"],
            results_and_findings=fields["results_and_findings"],
            limitations=fields["limitations"],
            future_work=fields["future_work"],
            evidence_count=len(context.items),
            model=response.model,
            usage=response.usage,
            raw=response.raw,
        )


class PaperComparisonService:
    """Orchestrates multi-paper comparison across aligned research dimensions."""

    def __init__(
        self,
        analysis_service: PaperAnalysisService,
        llm_service: Any,
        *,
        prompt_builder: PaperComparisonPromptBuilder | None = None,
        max_papers: int = 5,
    ) -> None:
        if not isinstance(analysis_service, PaperAnalysisService):
            raise TypeError("analysis_service must be a PaperAnalysisService")
        if not callable(getattr(llm_service, "generate", None)):
            raise TypeError("llm_service must provide generate(messages, ...)")
        if max_papers < 2:
            raise ValueError("max_papers must be at least 2")

        self._analysis_service = analysis_service
        self._llm = llm_service
        self._prompt_builder = prompt_builder or PaperComparisonPromptBuilder()
        self._max_papers = max_papers

    def compare_papers(
        self,
        paper_ids: list[str],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PaperComparison:
        """Compare two or more papers across aligned dimensions."""
        if not isinstance(paper_ids, (list, tuple)):
            raise InvalidRequestError("paper_ids must be a list of strings")
        if len(paper_ids) < 2:
            raise InvalidRequestError("At least 2 paper IDs are required for comparison.")
        if len(paper_ids) > self._max_papers:
            raise InvalidRequestError(
                f"Comparison exceeds maximum of {self._max_papers} papers (got {len(paper_ids)})."
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

        # 1. Analyze each paper
        analyses: list[StructuredPaperAnalysis] = []
        for pid in cleaned_ids:
            # DocumentNotFoundError will bubble up directly if any paper is missing
            analysis = self._analysis_service.analyze_paper(
                pid,
                model=model,
                temperature=temperature,
            )
            analyses.append(analysis)

        # 2. Check if all papers have no evidence
        all_empty = all(a.evidence_count == 0 for a in analyses)
        if all_empty:
            dimensions: dict[str, FieldComparison] = {}
            for dim in ANALYSIS_DIMENSIONS:
                per_paper = {a.paper_id: a.get_field(dim) for a in analyses}
                dimensions[dim] = FieldComparison(
                    dimension=dim,
                    per_paper=per_paper,
                    comparison_summary="None of the compared papers provided reported evidence for this dimension.",
                    key_differences=[],
                    key_similarities=["All papers lack reported evidence for this dimension."],
                )
            return PaperComparison(
                paper_ids=cleaned_ids,
                dimensions=dimensions,
                overall_summary="No evidence was available for any of the compared papers.",
                model=model or "none",
            )

        # 3. Call LLM for comparative synthesis
        messages = self._prompt_builder.build(analyses)
        try:
            response = self._llm.generate(
                messages,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise ComparisonServiceError(
                f"Multi-paper comparison LLM call failed ({type(exc).__name__})"
            ) from exc

        # 4. Parse comparative JSON
        cleaned = _clean_json_text(response.text)
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            raise ComparisonServiceError(
                f"Malformed comparison response; failed to parse JSON ({type(exc).__name__})"
            ) from exc

        if not isinstance(parsed, dict):
            raise ComparisonServiceError("Malformed comparison response; root element must be a JSON object")

        parsed_dimensions = parsed.get("dimensions", {})
        overall_summary = parsed.get("overall_summary", "Comparative analysis across the specified research papers.")
        if not isinstance(overall_summary, str) or not overall_summary.strip():
            overall_summary = "Comparative analysis across the specified research papers."

        dimensions_result: dict[str, FieldComparison] = {}
        for dim in ANALYSIS_DIMENSIONS:
            per_paper = {a.paper_id: a.get_field(dim) for a in analyses}
            raw_dim = parsed_dimensions.get(dim) if isinstance(parsed_dimensions, dict) else None

            if isinstance(raw_dim, dict):
                comp_summary = raw_dim.get("comparison_summary", f"Comparison for {dim}.")
                if not isinstance(comp_summary, str) or not comp_summary.strip():
                    comp_summary = f"Comparison for {dim}."
                diffs = [str(d) for d in raw_dim.get("key_differences", []) if isinstance(d, str) and d.strip()]
                sims = [str(s) for s in raw_dim.get("key_similarities", []) if isinstance(s, str) and s.strip()]
            else:
                comp_summary = f"Comparison synthesis across papers for {dim}."
                diffs = []
                sims = []

            dimensions_result[dim] = FieldComparison(
                dimension=dim,
                per_paper=per_paper,
                comparison_summary=comp_summary,
                key_differences=diffs,
                key_similarities=sims,
            )

        return PaperComparison(
            paper_ids=cleaned_ids,
            dimensions=dimensions_result,
            overall_summary=overall_summary,
            model=response.model,
            usage=response.usage,
            raw=response.raw,
        )
