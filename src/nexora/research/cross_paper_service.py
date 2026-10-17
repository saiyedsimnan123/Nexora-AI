"""Service for cross-paper comparison and evidence synthesis."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import UUID

from nexora.api.errors import (
    ComparisonServiceError,
    DocumentNotFoundError,
    InvalidRequestError,
)
from nexora.database.repository.paper import PaperRepository
from nexora.research.analysis_service import _clean_json_text
from nexora.research.comparison import (
    COMPARISON_MATRIX_DIMENSIONS,
    ConclusionClassification,
    CrossPaperComparisonResult,
    MatrixDimensionComparison,
    MatrixEvidenceSource,
    MatrixPaperFinding,
    VALID_CONCLUSIONS,
)
from nexora.research.cross_paper_prompt import CrossPaperComparisonPromptBuilder
from nexora.retrieval.evidence import EvidenceContext, EvidenceItem, filter_unsupplied_citations


def _normalize_dataset_tokens(text: str) -> set[str]:
    """Extract lower-cased alphanumeric benchmark and dataset tokens."""
    tokens = set(re.findall(r"[a-z0-9\-]+", text.lower()))
    stopwords = {"and", "the", "with", "on", "in", "of", "for", "a", "an", "evaluated", "trained", "using", "dataset", "benchmark"}
    return tokens - stopwords


def _has_unsupported_numerical_claims(summary: str, sources: list[MatrixEvidenceSource]) -> bool:
    """Detect if specific numerical scores in summary are completely missing from cited sources."""
    if not sources or not summary:
        return False
    claimed_numbers = set(re.findall(r"\b\d+(?:\.\d+)?(?:%|x)?\b", summary.lower()))
    if not claimed_numbers:
        return False
    combined_source_text = " ".join(s.text_excerpt.lower() for s in sources)
    unsupported = [
        num for num in claimed_numbers
        if num.rstrip("%x") not in combined_source_text
    ]
    return len(unsupported) > 0 and len(unsupported) == len(claimed_numbers)


def evaluate_cross_paper_conclusion(
    parsed_conclusion: str,
    parsed_reasoning: str,
    parsed_comparability: str,
    matrix: dict[str, MatrixDimensionComparison],
    paper_ids: list[str],
    per_paper_contexts: dict[str, EvidenceContext],
) -> tuple[str, str, str]:
    """Rigorously classify conclusion and comparability status with fail-closed safety guardrails."""
    # 1. Zero evidence across all papers
    all_empty = all(len(per_paper_contexts[pid].items) == 0 for pid in paper_ids)
    if all_empty:
        return (
            ConclusionClassification.INSUFFICIENT_EVIDENCE.value,
            "No evidence was available for any of the compared papers.",
            "insufficient_evidence",
        )

    # 2. Missing evidence for one or more papers
    empty_papers = [pid for pid in paper_ids if len(per_paper_contexts[pid].items) == 0]
    if empty_papers:
        return (
            ConclusionClassification.INSUFFICIENT_EVIDENCE.value,
            f"Evidence was missing for paper(s): {', '.join(empty_papers)}. Comparative superiority cannot be inferred without evidence from all papers.",
            "insufficient_evidence",
        )

    # 3. Check results dimension reporting
    results_comp = matrix.get("results")
    if results_comp:
        reported_results = [pid for pid, fld in results_comp.per_paper.items() if fld.status != "not_reported"]
        if not reported_results:
            return (
                ConclusionClassification.INSUFFICIENT_EVIDENCE.value,
                "None of the compared papers provided reported empirical results or evaluation metrics.",
                "insufficient_evidence",
            )
        if len(reported_results) < len(paper_ids):
            return (
                ConclusionClassification.INSUFFICIENT_EVIDENCE.value,
                "Results are reported for only a subset of papers. Direct comparative superiority cannot be inferred from a single paper.",
                "partially_comparable" if parsed_comparability not in ("incomparable", "insufficient_evidence") else parsed_comparability,
            )

    # 4. Check comparability / benchmark overlap
    datasets_comp = matrix.get("datasets")
    metrics_comp = matrix.get("metrics")
    comparability_comp = matrix.get("comparability")

    comparability_text = ""
    if comparability_comp:
        comparability_text += f"{comparability_comp.comparison_summary} " + " ".join(comparability_comp.key_differences)

    incomparable_indicators = (
        "incomparable",
        "cannot be directly compared",
        "disparate benchmark",
        "non-overlapping",
        "different evaluation",
        "completely different",
        "different protocol",
        "different metric definition",
        "incompatible splits",
        "different prompt format",
        "disparate datasets",
        "non-comparable",
    )

    is_explicitly_incomparable = (
        parsed_comparability in ("incomparable", "disparate", "not_comparable")
        or any(phrase in comparability_text.lower() for phrase in incomparable_indicators)
    )

    # Check for disjoint datasets if reported
    if datasets_comp:
        paper_tokens: list[set[str]] = []
        for pid in paper_ids:
            fld = datasets_comp.per_paper.get(pid)
            text_to_scan = fld.summary if fld else ""
            if fld and isinstance(fld.details, dict):
                text_to_scan += " " + " ".join(str(v) for v in fld.details.values())
            toks = _normalize_dataset_tokens(text_to_scan)
            if toks:
                paper_tokens.append(toks)

        if len(paper_tokens) >= 2:
            common = set.intersection(*paper_tokens)
            if not common and is_explicitly_incomparable:
                is_explicitly_incomparable = True

    effective_comparability = "incomparable" if is_explicitly_incomparable else (parsed_comparability if parsed_comparability in ("comparable", "partially_comparable") else "comparable")

    # Guardrail: Never infer demonstrated improvement across incomparable benchmarks
    if is_explicitly_incomparable:
        if parsed_conclusion == ConclusionClassification.DEMONSTRATED_IMPROVEMENT.value:
            return (
                ConclusionClassification.LIMITED_EVIDENCE.value,
                "Corrected from model output: Demonstrated improvement cannot be established because evaluation setups, benchmarks, and datasets are disparate and non-overlapping.",
                effective_comparability,
            )

    # 5. Check for experimental noise and statistically insignificant marginal improvements
    combined_results_text = ""
    if results_comp:
        combined_results_text += f"{results_comp.comparison_summary} " + " ".join(results_comp.key_differences)
        for fld in results_comp.per_paper.values():
            combined_results_text += " " + fld.summary

    # Strip negated conflict and noise phrases to prevent false-positive keyword triggers
    def _strip_negated_phrases(txt: str) -> str:
        t = re.sub(
            r"\b(?:no|without|zero|neither|nor|lack\s+of)\s+[^,.;:]*?\b(?:conflic\w*|contradic\w*|regress\w*|trade-?offs?)(?:\s+(?:conflic\w*|contradic\w*|regress\w*|trade-?offs?))*",
            "",
            txt,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r"\b(?:beyond|exceeds?|not|no|without)\s+[^,.;:]*?\b(?:experimental\s+noise|margin\s+of\s+error)",
            "",
            t,
            flags=re.IGNORECASE,
        )
        return t

    cleaned_results_text = _strip_negated_phrases(combined_results_text).lower()
    cleaned_reasoning_text = _strip_negated_phrases(parsed_reasoning).lower()

    noise_indicators = (
        "experimental noise",
        "margin of error",
        "statistically insignificant",
        "statistically insignificant difference",
        "negligible difference",
        "marginal difference",
        "marginal improvement",
        "marginal gain",
        "within noise",
        "no statistical significance",
        "single run without variance",
        "insufficient statistical power",
    )
    is_noise_level = any(
        ind in cleaned_results_text or ind in cleaned_reasoning_text for ind in noise_indicators
    )
    if is_noise_level and parsed_conclusion == ConclusionClassification.DEMONSTRATED_IMPROVEMENT.value:
        return (
            ConclusionClassification.LIMITED_EVIDENCE.value,
            "Corrected from model output: Reported performance differences are within experimental noise or lack statistical significance; cannot establish demonstrated improvement without verified, significant gains.",
            effective_comparability,
        )

    # 6. Check for unsupported numerical claims exceeding cited evidence
    has_unsupported_claims = False
    if results_comp:
        for fld in results_comp.per_paper.values():
            if fld.details.get("unsupported_numerical_claims"):
                has_unsupported_claims = True
                break
    if has_unsupported_claims and parsed_conclusion == ConclusionClassification.DEMONSTRATED_IMPROVEMENT.value:
        return (
            ConclusionClassification.LIMITED_EVIDENCE.value,
            "Corrected from model output: Quantitative claims exceed the verified evidence in cited excerpts; cannot substantiate demonstrated improvement.",
            effective_comparability,
        )

    # 7. Check for conflicting evidence, trade-offs, or split wins across datasets
    conflict_indicators = (
        "conflict",
        "contradict",
        "contradictory",
        "opposing",
        "trade-off",
        "tradeoff",
        "regression",
        "regresses",
        "worse",
        "degradation",
        "discrepancy",
        "mixed result",
        "mixed performance",
        "split win",
        "split result",
        "wins on",
        "trails on",
        "underperforms",
        "lower accuracy",
        "latency regression",
        "latency overhead",
        "resource overhead",
        "memory overhead",
        "compute overhead",
        "costly",
        "dataset-dependent",
        "inconsistent across datasets",
    )
    has_conflict = parsed_conclusion == ConclusionClassification.CONFLICTING_EVIDENCE.value or any(
        ind in cleaned_results_text or ind in cleaned_reasoning_text for ind in conflict_indicators
    )

    if has_conflict:
        return (
            ConclusionClassification.CONFLICTING_EVIDENCE.value,
            parsed_reasoning or "Evaluations exhibit conflicting findings, performance trade-offs, or contradictory benchmark results between papers.",
            effective_comparability,
        )

    # 8. Normalize conclusion
    if parsed_conclusion in VALID_CONCLUSIONS:
        return (parsed_conclusion, parsed_reasoning or "Comparative evidence synthesized across reported paper dimensions.", effective_comparability)

    # Fallback safe classification
    return (
        ConclusionClassification.LIMITED_EVIDENCE.value,
        parsed_reasoning or "Limited overlapping evidence available to establish definitive comparative conclusion.",
        effective_comparability,
    )


class CrossPaperComparisonService:
    """Orchestrates cross-paper comparison with strict evidence isolation and provenance."""

    def __init__(
        self,
        llm_service: Any,
        *,
        session_factory: Any = None,
        vector_store: Any = None,
        retrieval_service: Any = None,
        prompt_builder: CrossPaperComparisonPromptBuilder | None = None,
        max_evidence_characters: int | None = None,
        max_papers: int = 5,
        filter_invalid_citations: bool = True,
    ) -> None:
        if not callable(getattr(llm_service, "generate", None)):
            raise TypeError("llm_service must provide generate(messages, ...)")
        self._llm = llm_service
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._retrieval_service = retrieval_service
        self._prompt_builder = prompt_builder or CrossPaperComparisonPromptBuilder()
        self._max_chars = max_evidence_characters
        self._max_papers = max_papers
        self._filter_citations = filter_invalid_citations

    def _load_evidence_from_database(
        self, paper_uuid: UUID
    ) -> tuple[str | None, list[EvidenceItem]]:
        """Fetch paper metadata and ordered chunks from relational database."""
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

                    # Extract page number if present in chunk, vector store, doc metadata, or storage_key
                    page_number = getattr(chunk, "page_number", None)
                    if page_number is None and self._vector_store is not None:
                        records = getattr(self._vector_store, "_records", {})
                        rec = records.get(str(chunk.id))
                        if rec and hasattr(rec, "payload"):
                            page_number = rec.payload.get("page_number") or rec.payload.get("page")
                    if page_number is None and hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                        chunk_pages = doc.metadata.get("chunk_pages")
                        if isinstance(chunk_pages, (list, tuple)) and chunk.chunk_index < len(chunk_pages):
                            page_number = chunk_pages[chunk.chunk_index]
                    if page_number is None and doc.storage_key and "#page=" in doc.storage_key:
                        try:
                            page_number = int(doc.storage_key.split("#page=")[-1].split("&")[0])
                        except (ValueError, TypeError):
                            pass

                    metadata: dict[str, Any] = {
                        "paper_id": str(paper_uuid),
                        "filename": doc.filename,
                        "chunk_index": chunk.chunk_index,
                        "title": title,
                    }
                    if page_number is not None and isinstance(page_number, int) and page_number > 0:
                        metadata["page_number"] = page_number

                    items.append(
                        EvidenceItem(
                            document_id=str(doc.id),
                            chunk_id=str(chunk.id),
                            text=chunk.text,
                            score=1.0,
                            metadata=metadata,
                            position=len(items),
                        )
                    )
            return title, items
        finally:
            if hasattr(session, "close"):
                session.close()

    def _load_evidence_from_vector_or_retrieval(
        self, paper_id_str: str, query: str | None = None
    ) -> tuple[str | None, list[EvidenceItem]]:
        """Fetch chunks matching paper_id from retrieval service or vector store."""
        items: list[EvidenceItem] = []
        title: str | None = None
        used_chars = 0

        # Try retrieval service if query provided
        if self._retrieval_service is not None and query and query.strip():
            res = self._retrieval_service.retrieve(
                query.strip(), paper_id=paper_id_str, limit=10, mode="hybrid"
            )
            for chunk in res.results:
                # Scoping provenance check: fail closed
                if str(chunk.metadata.get("paper_id")) != paper_id_str:
                    continue
                if not title and "title" in chunk.metadata:
                    title = str(chunk.metadata["title"])
                page_num = chunk.page_number
                meta = dict(chunk.metadata)
                meta["paper_id"] = paper_id_str
                if page_num:
                    meta["page_number"] = page_num
                items.append(
                    EvidenceItem(
                        document_id=str(chunk.metadata.get("document_id", "doc-default")),
                        chunk_id=chunk.id,
                        text=chunk.text,
                        score=chunk.score,
                        metadata=meta,
                        position=len(items),
                    )
                )
            if items:
                return title, items

        # Fallback to direct vector store inspection
        if self._vector_store is not None:
            records = getattr(self._vector_store, "_records", {})
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
                    meta = dict(rec.payload)
                    meta["paper_id"] = paper_id_str
                    page_num = rec.payload.get("page_number") or rec.payload.get("page")
                    if page_num and isinstance(page_num, int) and page_num > 0:
                        meta["page_number"] = page_num

                    items.append(
                        EvidenceItem(
                            document_id=str(rec.payload.get("document_id", "doc-default")),
                            chunk_id=rec.id,
                            text=text,
                            score=1.0,
                            metadata=meta,
                            position=len(items),
                        )
                    )

        return title, items

    def _load_paper_evidence(
        self, paper_id_str: str, query: str | None = None
    ) -> tuple[str | None, list[EvidenceItem]]:
        """Load ordered evidence items and title strictly scoped to paper_id_str."""
        try:
            paper_uuid = UUID(paper_id_str)
        except (ValueError, AttributeError):
            raise InvalidRequestError(f"Invalid paper_id {paper_id_str!r}: must be a valid UUID.")

        if self._session_factory is not None:
            return self._load_evidence_from_database(paper_uuid)
        return self._load_evidence_from_vector_or_retrieval(paper_id_str, query=query)

    def compare(
        self,
        paper_ids: list[str],
        *,
        query: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> CrossPaperComparisonResult:
        """Execute cross-paper comparison across the 7 matrix dimensions."""
        if not isinstance(paper_ids, (list, tuple)):
            raise InvalidRequestError("paper_ids must be a list of strings")
        if len(paper_ids) < 2:
            raise InvalidRequestError("At least 2 paper IDs are required for cross-paper comparison.")
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

        # 1. Retrieve evidence strictly separately for each paper
        per_paper_contexts: dict[str, EvidenceContext] = {}
        paper_titles: dict[str, str | None] = {}

        for pid in cleaned_ids:
            title, items = self._load_paper_evidence(pid, query=query)
            # Strict isolation and deduplication: discard duplicates and items belonging to other papers
            seen_chunk_ids: set[str] = set()
            isolated_items: list[EvidenceItem] = []
            for i in items:
                if str(i.metadata.get("paper_id")) != pid:
                    continue
                if i.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(i.chunk_id)
                isolated_items.append(i)

            per_paper_contexts[pid] = EvidenceContext(
                query=query or f"Comparison of paper {pid}",
                items=isolated_items,
                total_candidates=len(isolated_items),
            )
            paper_titles[pid] = title

        # 2. Check if all papers have zero evidence
        all_empty = all(len(ctx.items) == 0 for ctx in per_paper_contexts.values())
        if all_empty:
            matrix: dict[str, MatrixDimensionComparison] = {}
            for dim in COMPARISON_MATRIX_DIMENSIONS:
                per_paper = {
                    pid: MatrixPaperFinding.not_reported(pid, paper_titles.get(pid))
                    for pid in cleaned_ids
                }
                matrix[dim] = MatrixDimensionComparison(
                    dimension=dim,
                    per_paper=per_paper,
                    comparison_summary="None of the compared papers provided reported evidence for this dimension.",
                    key_differences=[],
                    key_similarities=["All papers lack reported evidence for this dimension."],
                )
            return CrossPaperComparisonResult(
                paper_ids=cleaned_ids,
                paper_titles=paper_titles,
                matrix=matrix,
                conclusion=ConclusionClassification.INSUFFICIENT_EVIDENCE.value,
                conclusion_reasoning="No evidence was available for any of the compared papers.",
                comparability_status="insufficient_evidence",
                overall_summary="No evidence was available for any of the compared papers.",
                model=model or "none",
            )

        # 3. Call LLM for comparative synthesis
        messages = self._prompt_builder.build(
            cleaned_ids, per_paper_contexts, paper_titles, query=query
        )
        try:
            response = self._llm.generate(
                messages,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise ComparisonServiceError(
                f"Cross-paper comparison LLM call failed ({type(exc).__name__})"
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

        parsed_matrix = parsed.get("matrix", {})
        if not isinstance(parsed_matrix, dict):
            parsed_matrix = {}

        raw_conclusion = str(parsed.get("conclusion", ConclusionClassification.INSUFFICIENT_EVIDENCE.value))
        raw_reasoning = str(parsed.get("conclusion_reasoning", "Evidence synthesized across reported paper dimensions."))
        raw_comparability = str(parsed.get("comparability_status", "comparable"))
        overall_summary = str(parsed.get("overall_summary", "Comparative analysis across the specified research papers."))

        # 5. Build validated matrix dimensions with citation provenance and zero leakage
        constructed_matrix: dict[str, MatrixDimensionComparison] = {}

        for dim in COMPARISON_MATRIX_DIMENSIONS:
            raw_dim = parsed_matrix.get(dim) if isinstance(parsed_matrix, dict) else None
            per_paper_findings: dict[str, MatrixPaperFinding] = {}

            if isinstance(raw_dim, dict):
                comp_summary = raw_dim.get("comparison_summary", f"Comparison for {dim}.")
                if not isinstance(comp_summary, str) or not comp_summary.strip():
                    comp_summary = f"Comparison for {dim}."
                diffs = [str(d) for d in raw_dim.get("key_differences", []) if isinstance(d, str) and d.strip()]
                sims = [str(s) for s in raw_dim.get("key_similarities", []) if isinstance(s, str) and s.strip()]
                raw_per_paper = raw_dim.get("per_paper", {})
            else:
                comp_summary = f"Comparison synthesis across papers for {dim}."
                diffs = []
                sims = []
                raw_per_paper = {}

            for pid in cleaned_ids:
                p_ctx = per_paper_contexts[pid]
                p_raw = raw_per_paper.get(pid) if isinstance(raw_per_paper, dict) else None

                if isinstance(p_raw, dict):
                    summary = p_raw.get("summary", "Not reported in the provided evidence.")
                    if not isinstance(summary, str) or not summary.strip():
                        summary = "Not reported in the provided evidence."
                    status = p_raw.get("status", "reported")
                    raw_labels = p_raw.get("evidence_labels", [])
                    details = p_raw.get("details", {})
                    if not isinstance(details, dict):
                        details = {}
                else:
                    summary = "Not reported in the provided evidence."
                    status = "not_reported"
                    raw_labels = []
                    details = {}

                # Strict provenance and cross-paper isolation
                if not isinstance(raw_labels, list):
                    raw_labels = []

                if self._filter_citations:
                    filtered_text = filter_unsupplied_citations(" ".join(str(l) for l in raw_labels), len(p_ctx.items))
                    from nexora.retrieval.evidence import extract_citations
                    valid_labels = extract_citations(filtered_text)
                else:
                    valid_labels = [str(l) for l in raw_labels if isinstance(l, str)]

                sources: list[MatrixEvidenceSource] = []
                for lbl in valid_labels:
                    item = p_ctx.get_by_label(lbl)
                    if item is not None:
                        # Fail closed: verify paper_id matches expected paper
                        item_pid = str(item.metadata.get("paper_id", "")).strip()
                        if item_pid != pid:
                            continue
                        sources.append(
                            MatrixEvidenceSource(
                                document_id=item.document_id,
                                chunk_id=item.chunk_id,
                                paper_id=pid,
                                paper_title=paper_titles.get(pid),
                                page_number=item.page_number,
                                text_excerpt=item.text[:200].strip(),
                                score=item.score,
                            )
                        )

                if status not in ("reported", "not_reported", "uncertain"):
                    status = "reported" if summary != "Not reported in the provided evidence." else "not_reported"

                if status == "reported":
                    if summary == "Not reported in the provided evidence.":
                        status = "not_reported"
                    elif len(sources) == 0:
                        status = "uncertain"
                    elif dim in ("results", "metrics") and _has_unsupported_numerical_claims(summary, sources):
                        status = "uncertain"
                        details["unsupported_numerical_claims"] = True

                per_paper_findings[pid] = MatrixPaperFinding(
                    paper_id=pid,
                    paper_title=paper_titles.get(pid),
                    summary=summary,
                    status=status,
                    sources=sources,
                    details=details,
                )

            constructed_matrix[dim] = MatrixDimensionComparison(
                dimension=dim,
                per_paper=per_paper_findings,
                comparison_summary=comp_summary,
                key_differences=diffs,
                key_similarities=sims,
            )

        # 6. Apply rigorous conclusion and comparability evaluation guardrails
        final_conclusion, final_reasoning, final_comparability = evaluate_cross_paper_conclusion(
            raw_conclusion,
            raw_reasoning,
            raw_comparability,
            constructed_matrix,
            cleaned_ids,
            per_paper_contexts,
        )

        return CrossPaperComparisonResult(
            paper_ids=cleaned_ids,
            paper_titles=paper_titles,
            matrix=constructed_matrix,
            conclusion=final_conclusion,
            conclusion_reasoning=final_reasoning,
            comparability_status=final_comparability,
            overall_summary=overall_summary,
            model=response.model,
            usage=response.usage,
            raw=response.raw,
        )
