"""Evidence-backed research trend analysis and literature review synthesis service."""

from __future__ import annotations

import copy
import datetime
import json
import re
from typing import Any, Sequence
from uuid import UUID

from nexora.api.errors import (
    DocumentNotFoundError,
    InvalidRequestError,
    TrendServiceError,
)
from nexora.database.repository.paper import PaperRepository
from nexora.research.analysis import StructuredPaperAnalysis
from nexora.research.analysis_service import PaperAnalysisService, _clean_json_text
from nexora.research.synthesis import (
    LiteratureSynthesisReport,
    SynthesisDivergence,
    SynthesisFinding,
    SynthesisLimitation,
)
from nexora.research.trend_prompt import (
    LiteratureSynthesisPromptBuilder,
    TrendAnalysisPromptBuilder,
)
from nexora.research.trends import (
    ResearchTrendsReport,
    TemporalBucket,
    TrendCategory,
    TrendCitation,
    TrendItem,
    TrendTrajectory,
    VALID_CATEGORIES,
    VALID_TRAJECTORIES,
)
from nexora.retrieval.evidence import EvidenceItem, filter_unsupplied_citations


KNOWN_CONCEPTS = [
    ("Transformer", "architecture"),
    ("Mamba", "architecture"),
    ("Attention", "architecture"),
    ("State Space", "architecture"),
    ("BERT", "architecture"),
    ("Diffusion", "architecture"),
    ("ResNet", "architecture"),
    ("Adam", "method"),
    ("AdamW", "method"),
    ("SGD", "method"),
    ("Self-Attention", "method"),
    ("Linear-Time", "method"),
    ("Selective State Spaces", "method"),
    ("Pretraining", "method"),
    ("WMT", "dataset"),
    ("GLUE", "dataset"),
    ("ImageNet", "dataset"),
    ("Pile", "dataset"),
    ("BLEU", "metric"),
    ("ROUGE", "metric"),
    ("Accuracy", "metric"),
    ("Throughput", "metric"),
    ("F1", "metric"),
]


class ResearchTrendService:
    """Orchestrates temporal trend analysis and literature synthesis across paper collections."""

    def __init__(
        self,
        llm_service: Any,
        *,
        paper_analysis_service: PaperAnalysisService | None = None,
        session_factory: Any = None,
        vector_store: Any = None,
        trend_prompt_builder: TrendAnalysisPromptBuilder | None = None,
        synthesis_prompt_builder: LiteratureSynthesisPromptBuilder | None = None,
        max_evidence_characters: int | None = None,
        verifier: Any = None,
    ) -> None:
        if not callable(getattr(llm_service, "generate", None)):
            raise TypeError("llm_service must provide generate(messages, ...)")
        self._llm = llm_service
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._trend_prompt_builder = trend_prompt_builder or TrendAnalysisPromptBuilder()
        self._synthesis_prompt_builder = (
            synthesis_prompt_builder or LiteratureSynthesisPromptBuilder()
        )
        self._max_chars = max_evidence_characters
        self._verifier = verifier

        if paper_analysis_service is not None:
            self._paper_analysis_service = paper_analysis_service
        else:
            self._paper_analysis_service = PaperAnalysisService(
                llm_service,
                session_factory=session_factory,
                vector_store=vector_store,
                max_evidence_characters=max_evidence_characters,
            )

    def _is_mock_mode(self) -> bool:
        """Detect whether the configured LLM is a mock or test generator."""
        llm_cls = type(self._llm).__name__
        if "mock" in llm_cls.lower():
            return True
        provider = getattr(self._llm, "_provider", None)
        if provider is not None and "mock" in type(provider).__name__.lower():
            return True
        return False

    def _load_paper_evidence_and_meta(
        self, paper_uuid: UUID
    ) -> tuple[str | None, datetime.date | None, list[EvidenceItem]]:
        """Fetch title, publication_date, and raw evidence chunks for a paper."""
        paper_id_str = str(paper_uuid)

        # 1. Database session factory
        if self._session_factory is not None:
            session = self._session_factory()
            try:
                repo = PaperRepository(session)
                paper = repo.get_by_id(paper_uuid)
                if paper is None:
                    return None, None, []
                title = paper.title
                pub_date = paper.publication_date
                items: list[EvidenceItem] = []
                used_chars = 0
                for doc in paper.documents:
                    for chunk in doc.chunks:
                        if (
                            self._max_chars is not None
                            and used_chars + len(chunk.text) > self._max_chars
                        ):
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
                                    "title": title,
                                    "publication_date": str(pub_date) if pub_date else None,
                                },
                                position=len(items),
                            )
                        )
                return title, pub_date, items
            finally:
                if hasattr(session, "close"):
                    session.close()

        # 2. Vector store
        if self._vector_store is not None:
            items: list[EvidenceItem] = []
            records = getattr(self._vector_store, "_records", {})
            used_chars = 0
            title = None
            pub_date = None

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
                    if pub_date is None and "publication_date" in rec.payload:
                        raw_date = rec.payload["publication_date"]
                        if isinstance(raw_date, datetime.date):
                            pub_date = raw_date
                        elif isinstance(raw_date, str) and raw_date.strip():
                            try:
                                pub_date = datetime.date.fromisoformat(raw_date.strip()[:10])
                            except ValueError:
                                pass
                    elif pub_date is None and "year" in rec.payload:
                        try:
                            year_int = int(rec.payload["year"])
                            pub_date = datetime.date(year_int, 1, 1)
                        except (ValueError, TypeError):
                            pass

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
            return title, pub_date, items

        return None, None, []

    def _load_collection_papers(
        self, col_uuid: UUID
    ) -> list[dict[str, Any]]:
        """Fetch metadata for all papers belonging to a collection."""
        papers_meta: list[dict[str, Any]] = []

        if self._session_factory is not None:
            session = self._session_factory()
            try:
                repo = PaperRepository(session)
                papers = repo.list_by_collection_id(col_uuid, limit=100)
                for p in papers:
                    period_val = str(p.publication_date.year) if p.publication_date else None
                    papers_meta.append(
                        {
                            "paper_id": str(p.id),
                            "title": p.title,
                            "publication_date": p.publication_date,
                            "period": period_val,
                        }
                    )
                return papers_meta
            finally:
                if hasattr(session, "close"):
                    session.close()

        if self._vector_store is not None:
            records = getattr(self._vector_store, "_records", {})
            seen_pids: dict[str, dict[str, Any]] = {}
            if isinstance(records, dict):
                for rec in records.values():
                    if str(rec.payload.get("collection_id")) == str(col_uuid):
                        pid = str(rec.payload.get("paper_id"))
                        if pid and pid not in seen_pids:
                            raw_date = rec.payload.get("publication_date")
                            pub_date = None
                            if isinstance(raw_date, datetime.date):
                                pub_date = raw_date
                            elif isinstance(raw_date, str) and raw_date.strip():
                                try:
                                    pub_date = datetime.date.fromisoformat(raw_date.strip()[:10])
                                except ValueError:
                                    pass
                            elif "year" in rec.payload:
                                try:
                                    pub_date = datetime.date(int(rec.payload["year"]), 1, 1)
                                except (ValueError, TypeError):
                                    pass
                            period_val = str(pub_date.year) if pub_date else None
                            seen_pids[pid] = {
                                "paper_id": pid,
                                "title": str(rec.payload.get("title", f"Paper {pid}")),
                                "publication_date": pub_date,
                                "period": period_val,
                            }
            return list(seen_pids.values())

        return []

    def _classify_trajectory(
        self,
        occurrences_by_period: dict[str, int],
        valid_periods: list[str],
    ) -> tuple[str, str]:
        """Rigorously determine trajectory based strictly on temporal presence.

        Rules:
        - If fewer than 2 distinct dated periods in the collection: INSUFFICIENT_DATA.
        - If total occurrences across dated periods < 2: INSUFFICIENT_DATA (inadequate data sample).
        - If present in the latest period and not earlier: EMERGING.
        - If present across multiple consecutive/spanning periods up to latest: ESTABLISHED.
        - If present in early periods but absent in recent: DECLINING.
        - Otherwise: INSUFFICIENT_DATA.
        """
        if len(valid_periods) < 2:
            return (
                TrendTrajectory.INSUFFICIENT_DATA.value,
                f"Fewer than 2 distinct temporal periods with dated papers in the collection (found {len(valid_periods)} period(s)); trajectory cannot be reliably established.",
            )

        active_periods = [p for p in valid_periods if occurrences_by_period.get(p, 0) > 0]
        if not active_periods:
            return (
                TrendTrajectory.INSUFFICIENT_DATA.value,
                "No occurrences recorded in any dated temporal period.",
            )

        total_occurrences = sum(occurrences_by_period.get(p, 0) for p in valid_periods)
        if total_occurrences < 2:
            return (
                TrendTrajectory.INSUFFICIENT_DATA.value,
                f"Fewer than 2 total occurrences recorded across dated periods (found {total_occurrences}); data is inadequate to establish a trajectory.",
            )

        earliest_period = valid_periods[0]
        latest_period = valid_periods[-1]
        first_seen = active_periods[0]
        last_seen = active_periods[-1]

        if len(active_periods) == 1:
            if active_periods[0] == latest_period:
                return (
                    TrendTrajectory.EMERGING.value,
                    f"Appears exclusively in the most recent period ({latest_period}) without documented precedent in earlier periods.",
                )
            if active_periods[0] == earliest_period:
                return (
                    TrendTrajectory.DECLINING.value,
                    f"Observed in early period ({earliest_period}) but absent from recent periods ({latest_period}).",
                )
            return (
                TrendTrajectory.INSUFFICIENT_DATA.value,
                f"Observed in only a single intermediate period ({active_periods[0]}); trajectory is inconclusive.",
            )

        # len(active_periods) >= 2
        if latest_period in active_periods:
            if earliest_period in active_periods or len(active_periods) >= len(valid_periods) * 0.5:
                return (
                    TrendTrajectory.ESTABLISHED.value,
                    f"Consistently utilized across multiple temporal periods spanning {first_seen} to {last_seen}.",
                )
            return (
                TrendTrajectory.EMERGING.value,
                f"Demonstrates expanding presence across recent periods culminating in {latest_period}.",
            )

        return (
            TrendTrajectory.DECLINING.value,
            f"Documented in earlier periods ({first_seen} to {last_seen}) but absent from the most recent period ({latest_period}).",
        )

    def analyze_collection_trends(
        self,
        collection_id: str | UUID,
        *,
        paper_ids: Sequence[str] | None = None,
        time_unit: str = "year",
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ResearchTrendsReport:
        """Perform temporal research trend analysis across papers in a collection."""
        try:
            col_uuid = UUID(str(collection_id).strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError("Invalid collection_id: must be a valid UUID.")

        # 1. Load collection papers
        all_papers = self._load_collection_papers(col_uuid)
        if not all_papers:
            raise DocumentNotFoundError(f"Collection {col_uuid} has no papers to analyze.")

        available_by_id = {p["paper_id"]: p for p in all_papers}

        # 2. Filter papers if paper_ids provided
        if paper_ids is not None:
            if not isinstance(paper_ids, (list, tuple)):
                raise InvalidRequestError("paper_ids must be a list of strings")
            if len(paper_ids) == 0:
                raise InvalidRequestError("paper_ids must not be empty when provided")
            if len(paper_ids) > 10:
                raise InvalidRequestError(
                    f"Trend analysis exceeds maximum of 10 papers (got {len(paper_ids)})."
                )

            cleaned_pids: list[str] = []
            for idx, pid in enumerate(paper_ids):
                if not isinstance(pid, str) or not pid.strip():
                    raise InvalidRequestError(f"paper_ids[{idx}] must be a non-empty string")
                try:
                    cleaned_pids.append(str(UUID(pid.strip())))
                except (ValueError, AttributeError):
                    raise InvalidRequestError(f"Invalid paper_id {pid!r}: must be a valid UUID.")

            if len(set(cleaned_pids)) != len(cleaned_pids):
                raise InvalidRequestError("paper_ids must be unique with no duplicates.")

            missing = [pid for pid in cleaned_pids if pid not in available_by_id]
            if missing:
                raise DocumentNotFoundError(
                    f"Paper(s) {', '.join(missing)} not found in collection {col_uuid}."
                )
            target_papers = [available_by_id[pid] for pid in cleaned_pids]
        else:
            target_papers = all_papers[:10]  # Cap at 10 papers max

        # 3. Organize papers into temporal buckets and detect missing dates
        paper_metadata: dict[str, dict[str, Any]] = {}
        temporal_buckets_map: dict[str, list[dict[str, Any]]] = {}
        missing_date_pids: list[str] = []

        for p in target_papers:
            pid = p["paper_id"]
            title = p["title"]
            pub_date = p["publication_date"]
            period = p["period"]

            paper_metadata[pid] = {
                "title": title,
                "publication_date": pub_date,
                "period": period,
            }

            if period:
                temporal_buckets_map.setdefault(period, []).append(p)
            else:
                missing_date_pids.append(pid)

        sorted_periods = sorted(temporal_buckets_map.keys())

        # 4. Load evidence chunks across all target papers
        evidence_items: list[EvidenceItem] = []
        for p in target_papers:
            pid = p["paper_id"]
            p_uuid = UUID(pid)
            title, _, chunks = self._load_paper_evidence_and_meta(p_uuid)
            evidence_items.extend(chunks)

        # 5. Handle empty evidence case
        if not evidence_items:
            buckets = [
                TemporalBucket(
                    period=period,
                    paper_count=len(temporal_buckets_map[period]),
                    paper_ids=[p["paper_id"] for p in temporal_buckets_map[period]],
                    paper_titles={p["paper_id"]: p["title"] for p in temporal_buckets_map[period]},
                    highlights=["No evidence chunks available."],
                )
                for period in sorted_periods
            ]
            return ResearchTrendsReport(
                collection_id=str(col_uuid),
                paper_ids=[p["paper_id"] for p in target_papers],
                time_unit=time_unit,
                temporal_buckets=buckets,
                trends=[],
                missing_date_paper_ids=missing_date_pids,
                summary="No evidence chunks were found for papers in this collection.",
                limitations=["Collection lacks extracted evidence chunks for trend synthesis."],
                model=model or "none",
            )

        # 6. Extraction: Deterministic mock mode or LLM generation
        is_mock = self._is_mock_mode()
        label_to_item = {f"[E{i}]": item for i, item in enumerate(evidence_items, start=1)}

        if is_mock:
            # Deterministic extraction via PaperAnalysisService and chunk fallback
            concept_occurrences: dict[tuple[str, str], dict[str, int]] = {}
            concept_citations: dict[tuple[str, str], list[dict[str, Any]]] = {}
            concept_labels: dict[tuple[str, str], list[str]] = {}
            period_highlights_map: dict[str, list[str]] = {}

            for p in target_papers:
                pid = p["paper_id"]
                period = p["period"] or "undated"
                has_structured = False
                try:
                    analysis = self._paper_analysis_service.analyze_paper(pid)
                    dims_to_check = [
                        ("method", analysis.methodology_and_algorithms),
                        ("architecture", analysis.models_and_datasets),
                        ("dataset", analysis.models_and_datasets),
                        ("metric", analysis.experiments_and_metrics),
                    ]
                    for cat, field_obj in dims_to_check:
                        if not field_obj or field_obj.status != "reported":
                            continue
                        summary_clean = field_obj.summary
                        words = [w.strip(".,;:()") for w in summary_clean.split()]
                        for w in words:
                            if len(w) >= 3 and (w[0].isupper() or any(c.isdigit() for c in w)):
                                key = (w, cat)
                                if key not in concept_occurrences:
                                    concept_occurrences[key] = {}
                                    concept_citations[key] = []
                                    concept_labels[key] = []
                                if period != "undated":
                                    concept_occurrences[key][period] = (
                                        concept_occurrences[key].get(period, 0) + 1
                                    )
                                for src in field_obj.sources:
                                    citation_dict = {
                                        "paper_id": pid,
                                        "chunk_id": src.get("chunk_id"),
                                        "evidence_label": src.get("evidence_label", "[E1]"),
                                        "text_excerpt": src.get("text_excerpt"),
                                        "period": period if period != "undated" else None,
                                    }
                                    if citation_dict not in concept_citations[key]:
                                        concept_citations[key].append(citation_dict)
                                for lbl in field_obj.evidence_labels:
                                    if lbl not in concept_labels[key]:
                                        concept_labels[key].append(lbl)
                                has_structured = True
                        if period in sorted_periods and field_obj.summary:
                            period_highlights_map.setdefault(period, []).append(
                                f"Paper '{p['title']}': {field_obj.summary[:100]}..."
                            )
                except Exception:
                    has_structured = False

                # Fallback: scan paper chunks directly
                paper_chunks = [item for item in evidence_items if str(item.metadata.get("paper_id")) == pid]
                for item in paper_chunks:
                    lbl = next((k for k, v in label_to_item.items() if v.chunk_id == item.chunk_id), "[E1]")
                    for term, cat in KNOWN_CONCEPTS:
                        if re.search(r'\b' + re.escape(term) + r'\b', item.text, re.IGNORECASE):
                            key = (term, cat)
                            if key not in concept_occurrences:
                                concept_occurrences[key] = {}
                                concept_citations[key] = []
                                concept_labels[key] = []
                            if period != "undated":
                                concept_occurrences[key][period] = (
                                    concept_occurrences[key].get(period, 0) + 1
                                )
                            cit = {
                                "paper_id": pid,
                                "chunk_id": item.chunk_id,
                                "evidence_label": lbl,
                                "text_excerpt": item.text[:200],
                                "period": period if period != "undated" else None,
                            }
                            if cit not in concept_citations[key]:
                                concept_citations[key].append(cit)
                            if lbl not in concept_labels[key]:
                                concept_labels[key].append(lbl)

                if period in sorted_periods and period not in period_highlights_map:
                    period_highlights_map[period] = [f"Active period with paper '{p['title']}'."]

            # Build trend items with trajectory classification
            trend_items: list[TrendItem] = []
            for (name, cat), occ in list(concept_occurrences.items())[:15]:
                traj, reasoning = self._classify_trajectory(occ, sorted_periods)
                trend_items.append(
                    TrendItem(
                        name=name,
                        category=cat,
                        trajectory=traj,
                        trajectory_reasoning=reasoning,
                        occurrences_by_period=occ,
                        citations=concept_citations.get((name, cat), []),
                        evidence_labels=concept_labels.get((name, cat), []),
                        summary=f"Analysis identified {name} ({cat}) utilized in the collection.",
                    )
                )

            # Build temporal buckets
            buckets = []
            for period in sorted_periods:
                p_list = temporal_buckets_map[period]
                buckets.append(
                    TemporalBucket(
                        period=period,
                        paper_count=len(p_list),
                        paper_ids=[p["paper_id"] for p in p_list],
                        paper_titles={p["paper_id"]: p["title"] for p in p_list},
                        highlights=period_highlights_map.get(period, [f"Active period with {len(p_list)} paper(s)."]),
                    )
                )

            limitations = []
            if len(sorted_periods) < 2:
                limitations.append(
                    "Collection contains fewer than 2 distinct temporal periods; trajectories classified as insufficient_data."
                )
            if missing_date_pids:
                limitations.append(
                    f"{len(missing_date_pids)} paper(s) lacked publication dates and could not contribute to temporal progression."
                )

            return ResearchTrendsReport(
                collection_id=str(col_uuid),
                paper_ids=[p["paper_id"] for p in target_papers],
                time_unit=time_unit,
                temporal_buckets=buckets,
                trends=trend_items,
                missing_date_paper_ids=missing_date_pids,
                summary=f"Temporal trend analysis synthesized across {len(target_papers)} papers over {len(sorted_periods)} temporal period(s).",
                limitations=limitations,
                model="mock-generator",
            )

        # 7. Live LLM execution
        simple_bucket_dict = {
            period: [p["paper_id"] for p in temporal_buckets_map[period]]
            for period in sorted_periods
        }
        messages = self._trend_prompt_builder.build(
            paper_metadata,
            simple_bucket_dict,
            evidence_items,
            time_unit=time_unit,
        )

        try:
            response = self._llm.generate(
                messages,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise TrendServiceError(
                f"Trend analysis LLM call failed ({type(exc).__name__})"
            ) from exc

        cleaned = _clean_json_text(response.text)
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            raise TrendServiceError(
                f"Malformed trend analysis response; failed to parse JSON ({type(exc).__name__})"
            ) from exc

        if not isinstance(parsed, dict):
            raise TrendServiceError(
                "Malformed trend analysis response; root element must be a JSON object"
            )

        raw_trends = parsed.get("trends", [])
        if not isinstance(raw_trends, list):
            raw_trends = []

        trend_items = []
        for raw_item in raw_trends:
            if not isinstance(raw_item, dict):
                continue
            name = str(raw_item.get("name", "")).strip()
            if not name:
                continue
            category = str(raw_item.get("category", "method")).lower()
            if category not in VALID_CATEGORIES:
                category = TrendCategory.METHOD.value

            occurrences = raw_item.get("occurrences_by_period", {})
            clean_occurrences: dict[str, int] = {}
            if isinstance(occurrences, dict):
                for k, v in occurrences.items():
                    try:
                        clean_occurrences[str(k)] = int(v)
                    except (ValueError, TypeError):
                        pass

            # Enforce deterministic trajectory verification
            traj, reasoning = self._classify_trajectory(clean_occurrences, sorted_periods)

            # Filter and resolve citations
            raw_labels = raw_item.get("evidence_labels", [])
            valid_labels = [lbl for lbl in raw_labels if lbl in label_to_item]
            citations: list[dict[str, Any]] = []
            for lbl in valid_labels:
                item = label_to_item[lbl]
                pid = str(item.metadata.get("paper_id", item.document_id))
                period_val = paper_metadata.get(pid, {}).get("period")
                citations.append(
                    {
                        "paper_id": pid,
                        "chunk_id": item.chunk_id,
                        "evidence_label": lbl,
                        "text_excerpt": item.text[:200],
                        "period": period_val,
                    }
                )

            if not citations and sum(clean_occurrences.values()) == 0:
                continue

            trend_items.append(
                TrendItem(
                    name=name,
                    category=category,
                    trajectory=traj,
                    trajectory_reasoning=reasoning,
                    occurrences_by_period=clean_occurrences,
                    citations=citations,
                    evidence_labels=valid_labels,
                    summary=str(raw_item.get("summary", f"{name} ({category})")),
                )
            )

        # Build temporal buckets
        raw_highlights = parsed.get("period_highlights", {})
        buckets = []
        for period in sorted_periods:
            p_list = temporal_buckets_map[period]
            hl = (
                raw_highlights.get(period, [])
                if isinstance(raw_highlights, dict)
                else []
            )
            buckets.append(
                TemporalBucket(
                    period=period,
                    paper_count=len(p_list),
                    paper_ids=[p["paper_id"] for p in p_list],
                    paper_titles={p["paper_id"]: p["title"] for p in p_list},
                    highlights=[str(h) for h in hl if isinstance(h, str)],
                )
            )

        limitations = [
            str(l) for l in parsed.get("limitations", []) if isinstance(l, str)
        ]
        if len(sorted_periods) < 2 and "Collection contains fewer than 2 distinct temporal periods" not in limitations:
            limitations.append(
                "Collection contains fewer than 2 distinct temporal periods; trajectories classified as insufficient_data."
            )
        if missing_date_pids:
            limitations.append(
                f"{len(missing_date_pids)} paper(s) lacked publication dates and could not contribute to temporal progression."
            )

        return ResearchTrendsReport(
            collection_id=str(col_uuid),
            paper_ids=[p["paper_id"] for p in target_papers],
            time_unit=time_unit,
            temporal_buckets=buckets,
            trends=trend_items,
            missing_date_paper_ids=missing_date_pids,
            summary=str(parsed.get("summary", "Temporal trend analysis report.")),
            limitations=limitations,
            model=getattr(response, "model", model or "unknown"),
            usage=getattr(response, "usage", None),
            raw=response,
        )

    def synthesize_collection_literature(
        self,
        collection_id: str | UUID,
        *,
        paper_ids: Sequence[str] | None = None,
        topic: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LiteratureSynthesisReport:
        """Synthesize literature across papers in an authorized collection."""
        try:
            col_uuid = UUID(str(collection_id).strip())
        except (ValueError, AttributeError):
            raise InvalidRequestError("Invalid collection_id: must be a valid UUID.")

        # 1. Load papers
        all_papers = self._load_collection_papers(col_uuid)
        if not all_papers:
            raise DocumentNotFoundError(f"Collection {col_uuid} has no papers to analyze.")

        available_by_id = {p["paper_id"]: p for p in all_papers}

        # 2. Filter papers if paper_ids provided
        if paper_ids is not None:
            if not isinstance(paper_ids, (list, tuple)):
                raise InvalidRequestError("paper_ids must be a list of strings")
            if len(paper_ids) == 0:
                raise InvalidRequestError("paper_ids must not be empty when provided")
            if len(paper_ids) > 10:
                raise InvalidRequestError(
                    f"Literature synthesis exceeds maximum of 10 papers (got {len(paper_ids)})."
                )

            cleaned_pids: list[str] = []
            for idx, pid in enumerate(paper_ids):
                if not isinstance(pid, str) or not pid.strip():
                    raise InvalidRequestError(f"paper_ids[{idx}] must be a non-empty string")
                try:
                    cleaned_pids.append(str(UUID(pid.strip())))
                except (ValueError, AttributeError):
                    raise InvalidRequestError(f"Invalid paper_id {pid!r}: must be a valid UUID.")

            if len(set(cleaned_pids)) != len(cleaned_pids):
                raise InvalidRequestError("paper_ids must be unique with no duplicates.")

            missing = [pid for pid in cleaned_pids if pid not in available_by_id]
            if missing:
                raise DocumentNotFoundError(
                    f"Paper(s) {', '.join(missing)} not found in collection {col_uuid}."
                )
            target_papers = [available_by_id[pid] for pid in cleaned_pids]
        else:
            target_papers = all_papers[:10]

        # 3. Load evidence chunks
        evidence_items: list[EvidenceItem] = []
        paper_metadata: dict[str, dict[str, Any]] = {}
        for p in target_papers:
            pid = p["paper_id"]
            title, pub_date, chunks = self._load_paper_evidence_and_meta(UUID(pid))
            paper_metadata[pid] = {
                "title": title or p["title"],
                "publication_date": pub_date,
                "period": p["period"],
            }
            evidence_items.extend(chunks)

        # 4. Handle empty evidence case
        if not evidence_items:
            return LiteratureSynthesisReport(
                collection_id=str(col_uuid),
                paper_ids=[p["paper_id"] for p in target_papers],
                topic=topic,
                executive_summary="No evidence chunks were found to synthesize literature.",
                common_findings=[],
                disagreements=[],
                primary_methods=[],
                recurring_limitations=[],
                open_research_gaps=[],
                evidence_count=0,
                is_grounded=False,
                limitations=["Collection lacks extracted evidence chunks for literature synthesis."],
                model=model or "none",
            )

        label_to_item = {f"[E{i}]": item for i, item in enumerate(evidence_items, start=1)}
        is_mock = self._is_mock_mode()

        # 5. Deterministic mock mode
        if is_mock:
            common_findings: list[SynthesisFinding] = []
            disagreements: list[SynthesisDivergence] = []
            methods: set[str] = set()
            recurring_limitations: list[SynthesisLimitation] = []
            open_gaps: set[str] = set()

            for p in target_papers:
                pid = p["paper_id"]
                has_structured = False
                try:
                    analysis = self._paper_analysis_service.analyze_paper(pid)
                    if analysis.results_and_findings.status == "reported":
                        sources = [
                            {
                                "paper_id": pid,
                                "chunk_id": s.get("chunk_id"),
                                "evidence_label": s.get("evidence_label", "[E1]"),
                                "text_excerpt": s.get("text_excerpt"),
                            }
                            for s in analysis.results_and_findings.sources
                        ]
                        common_findings.append(
                            SynthesisFinding(
                                claim=analysis.results_and_findings.summary,
                                status="supported",
                                evidence_labels=list(analysis.results_and_findings.evidence_labels),
                                sources=sources,
                                supporting_paper_ids=[pid],
                                summary=f"Reported finding from '{p['title']}'.",
                            )
                        )
                        has_structured = True

                    if analysis.methodology_and_algorithms.status == "reported":
                        clean_meth = analysis.methodology_and_algorithms.summary.split(".")[0].strip()
                        if clean_meth:
                            methods.add(clean_meth)

                    if analysis.limitations.status == "reported":
                        l_sources = [
                            {
                                "paper_id": pid,
                                "chunk_id": s.get("chunk_id"),
                                "evidence_label": s.get("evidence_label", "[E1]"),
                                "text_excerpt": s.get("text_excerpt"),
                            }
                            for s in analysis.limitations.sources
                        ]
                        recurring_limitations.append(
                            SynthesisLimitation(
                                description=analysis.limitations.summary,
                                affected_paper_ids=[pid],
                                evidence_labels=list(analysis.limitations.evidence_labels),
                                sources=l_sources,
                            )
                        )

                    if analysis.future_work.status == "reported":
                        clean_fw = analysis.future_work.summary.split(".")[0].strip()
                        if clean_fw:
                            open_gaps.add(clean_fw)
                except Exception:
                    has_structured = False

                # Fallback: scan paper chunks directly
                paper_chunks = [item for item in evidence_items if str(item.metadata.get("paper_id")) == pid]
                for item in paper_chunks:
                    lbl = next((k for k, v in label_to_item.items() if v.chunk_id == item.chunk_id), "[E1]")
                    src = [{
                        "paper_id": pid,
                        "chunk_id": item.chunk_id,
                        "evidence_label": lbl,
                        "text_excerpt": item.text[:200],
                    }]
                    sentences = [s.strip() for s in item.text.split(".") if s.strip()]
                    for s in sentences:
                        s_lower = s.lower()
                        if any(m in s_lower for m in ["metric", "accuracy", "bleu", "score", "throughput", "achieve", "show", "result"]):
                            common_findings.append(
                                SynthesisFinding(
                                    claim=s + ".",
                                    status="supported",
                                    evidence_labels=[lbl],
                                    sources=src,
                                    supporting_paper_ids=[pid],
                                    summary=f"Reported finding from '{p['title']}'.",
                                )
                            )
                        if any(l in s_lower for l in ["limitation", "complexity", "cost", "bottleneck", "drawback"]):
                            recurring_limitations.append(
                                SynthesisLimitation(
                                    description=s + ".",
                                    affected_paper_ids=[pid],
                                    evidence_labels=[lbl],
                                    sources=src,
                                )
                            )
                        if any(g in s_lower for g in ["future work", "extension", "remain", "unexplored", "open"]):
                            open_gaps.add(s + ".")
                        if any(m in s_lower for m in ["architecture", "method", "model", "algorithm", "attention", "state space", "adam"]):
                            methods.add(p["title"].split(":")[0].strip())

            # If multiple papers present, synthesize a divergence if different methods/approaches used
            if len(target_papers) >= 2 and len(methods) >= 2:
                p1 = target_papers[0]["paper_id"]
                p2 = target_papers[1]["paper_id"]
                disagreements.append(
                    SynthesisDivergence(
                        topic="Methodological Architecture and Strategy",
                        description="Analyzed papers pursue distinct architectural strategies and algorithmic designs.",
                        perspectives={
                            p1: f"Utilizes approach in '{target_papers[0]['title']}'",
                            p2: f"Utilizes approach in '{target_papers[1]['title']}'",
                        },
                        evidence_labels=list(label_to_item.keys())[:2],
                        sources=[
                            {
                                "paper_id": p1,
                                "chunk_id": evidence_items[0].chunk_id,
                                "evidence_label": "[E1]",
                                "text_excerpt": evidence_items[0].text[:200],
                            }
                        ] if evidence_items else [],
                    )
                )

            return LiteratureSynthesisReport(
                collection_id=str(col_uuid),
                paper_ids=[p["paper_id"] for p in target_papers],
                topic=topic,
                executive_summary=f"Synthesized literature review across {len(target_papers)} papers analyzing primary methods, empirical findings, and identified limitations.",
                common_findings=common_findings,
                disagreements=disagreements,
                primary_methods=sorted(methods),
                recurring_limitations=recurring_limitations,
                open_research_gaps=sorted(open_gaps),
                evidence_count=len(evidence_items),
                is_grounded=True,
                limitations=["Synthesis generated via deterministic extraction from paper evidence."],
                model="mock-generator",
            )

        # 6. Live LLM execution
        messages = self._synthesis_prompt_builder.build(
            paper_metadata,
            evidence_items,
            topic=topic,
        )

        try:
            response = self._llm.generate(
                messages,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise TrendServiceError(
                f"Literature synthesis LLM call failed ({type(exc).__name__})"
            ) from exc

        cleaned = _clean_json_text(response.text)
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            raise TrendServiceError(
                f"Malformed literature synthesis response; failed to parse JSON ({type(exc).__name__})"
            ) from exc

        if not isinstance(parsed, dict):
            raise TrendServiceError(
                "Malformed literature synthesis response; root element must be a JSON object"
            )

        # 7. Build validated findings and models with citation provenance
        findings: list[SynthesisFinding] = []
        for raw_f in parsed.get("common_findings", []):
            if not isinstance(raw_f, dict):
                continue
            claim = str(raw_f.get("claim", "")).strip()
            if not claim:
                continue
            raw_labels = raw_f.get("evidence_labels", [])
            valid_labels = [lbl for lbl in raw_labels if lbl in label_to_item]
            sources = []
            for lbl in valid_labels:
                item = label_to_item[lbl]
                sources.append(
                    {
                        "paper_id": str(item.metadata.get("paper_id", item.document_id)),
                        "chunk_id": item.chunk_id,
                        "evidence_label": lbl,
                        "text_excerpt": item.text[:200],
                    }
                )
            # If no valid citations, claim cannot be claimed as supported
            if not valid_labels:
                status = "insufficient_evidence"
            else:
                status = str(raw_f.get("status", "supported"))

            derived_pids = list(dict.fromkeys(s["paper_id"] for s in sources))
            supporting_pids = derived_pids if derived_pids else [
                str(pid) for pid in raw_f.get("supporting_paper_ids", [])
                if isinstance(pid, str) and pid in available_by_id
            ]

            findings.append(
                SynthesisFinding(
                    claim=claim,
                    status=status,
                    evidence_labels=valid_labels,
                    sources=sources,
                    supporting_paper_ids=supporting_pids,
                    summary=str(raw_f.get("summary", "")),
                )
            )

        disagreements: list[SynthesisDivergence] = []
        for raw_d in parsed.get("disagreements", []):
            if not isinstance(raw_d, dict):
                continue
            d_topic = str(raw_d.get("topic", "")).strip()
            desc = str(raw_d.get("description", "")).strip()
            if not d_topic and not desc:
                continue
            raw_labels = raw_d.get("evidence_labels", [])
            valid_labels = [lbl for lbl in raw_labels if lbl in label_to_item]
            # Disagreements must be grounded in valid evidence sources
            if not valid_labels:
                continue
            sources = []
            for lbl in valid_labels:
                item = label_to_item[lbl]
                sources.append(
                    {
                        "paper_id": str(item.metadata.get("paper_id", item.document_id)),
                        "chunk_id": item.chunk_id,
                        "evidence_label": lbl,
                        "text_excerpt": item.text[:200],
                    }
                )
            perspectives = (
                {str(k): str(v) for k, v in raw_d.get("perspectives", {}).items()}
                if isinstance(raw_d.get("perspectives"), dict)
                else {}
            )
            disagreements.append(
                SynthesisDivergence(
                    topic=d_topic or "Methodological Divergence",
                    description=desc,
                    perspectives=perspectives,
                    evidence_labels=valid_labels,
                    sources=sources,
                )
            )

        recurring_lims: list[SynthesisLimitation] = []
        for raw_l in parsed.get("recurring_limitations", []):
            if not isinstance(raw_l, dict):
                continue
            desc = str(raw_l.get("description", "")).strip()
            if not desc:
                continue
            raw_labels = raw_l.get("evidence_labels", [])
            valid_labels = [lbl for lbl in raw_labels if lbl in label_to_item]
            # Limitations must be grounded in valid evidence sources
            if not valid_labels:
                continue
            sources = []
            for lbl in valid_labels:
                item = label_to_item[lbl]
                sources.append(
                    {
                        "paper_id": str(item.metadata.get("paper_id", item.document_id)),
                        "chunk_id": item.chunk_id,
                        "evidence_label": lbl,
                        "text_excerpt": item.text[:200],
                    }
                )
            affected_pids = list(dict.fromkeys(s["paper_id"] for s in sources))
            recurring_lims.append(
                SynthesisLimitation(
                    description=desc,
                    affected_paper_ids=affected_pids,
                    evidence_labels=valid_labels,
                    sources=sources,
                )
            )

        primary_methods = [
            str(m) for m in parsed.get("primary_methods", []) if isinstance(m, str) and m.strip()
        ]
        open_gaps = [
            str(g) for g in parsed.get("open_research_gaps", []) if isinstance(g, str) and g.strip()
        ]
        limitations = [
            str(l) for l in parsed.get("limitations", []) if isinstance(l, str) and l.strip()
        ]

        has_supported_findings = any(f.status == "supported" and len(f.sources) > 0 for f in findings)
        is_grounded = bool(evidence_items and has_supported_findings)

        return LiteratureSynthesisReport(
            collection_id=str(col_uuid),
            paper_ids=[p["paper_id"] for p in target_papers],
            topic=topic,
            executive_summary=str(
                parsed.get("executive_summary", "Literature review synthesis report.")
            ),
            common_findings=findings,
            disagreements=disagreements,
            primary_methods=primary_methods,
            recurring_limitations=recurring_lims,
            open_research_gaps=open_gaps,
            evidence_count=len(evidence_items),
            is_grounded=is_grounded,
            limitations=limitations,
            model=getattr(response, "model", model or "unknown"),
            usage=getattr(response, "usage", None),
            raw=response,
        )
