"""Research retrieval dataset schema, validation, and loader (Milestone 12C-33).

Provides a documented, validated format for research chunks, paper/document identity,
query text, relevant chunk IDs, and unanswerable queries.

Guarantees:
- Ground-truth relevance labels are strictly isolated from retrieved predictions.
- Fail-closed validation: malformed records, missing required fields, duplicate IDs,
  and references to nonexistent chunk IDs are rejected immediately with informative errors.
- Never silently drops invalid records.
- Clearly distinguishes illustrative example datasets from real-world benchmarks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence


class DatasetValidationError(ValueError):
    """Raised when a research retrieval dataset fails structural, referential, or schema validation."""


VALID_PROVENANCE_VALUES: set[str] = {
    "illustrative",
    "candidate",
    "candidate_real_world",
    "manually_reviewed_benchmark",
    "reviewed_benchmark",
    "benchmark",
    "unknown",
}

VALID_REVIEW_STATUS_VALUES: set[str] = {
    "reviewed",
    "unreviewed",
    "in_review",
    "pending",
}


@dataclass(frozen=True)
class ResearchRetrievalDataset:
    """Immutable, validated research retrieval evaluation dataset."""

    dataset_id: str
    description: str
    chunks: list[dict[str, Any]]
    queries: list[dict[str, Any]]
    is_illustrative_only: bool | None = None
    provenance: str = field(default="")
    version: str | None = None

    def __post_init__(self) -> None:
        prov = str(self.provenance).strip() if self.provenance else ""
        if not prov:
            if self.is_illustrative_only is True:
                prov = "illustrative"
            elif self.is_illustrative_only is False:
                prov = "benchmark"
            else:
                prov = "unknown"
        if prov in ("manually_reviewed_benchmark", "reviewed_benchmark"):
            if not self.is_manually_reviewed:
                raise DatasetValidationError(
                    f"Cannot classify dataset as '{prov}': "
                    f"every query (including unanswerable queries) must have explicitly confirmed review status "
                    f"with a non-empty reviewer identifier. "
                    f"Found {self.unreviewed_queries_count} unreviewed query record(s)."
                )
        object.__setattr__(self, "provenance", prov)

    @property
    def name(self) -> str:
        """Name/ID of the dataset."""
        return self.dataset_id

    @property
    def reviewed_queries_count(self) -> int:
        """Count of queries whose relevance labels are confirmed manually reviewed with non-empty reviewer."""
        return sum(
            1 for q in self.queries
            if q.get("reviewed") is True and q.get("reviewer") and str(q.get("reviewer")).strip()
        )

    @property
    def unreviewed_queries_count(self) -> int:
        """Count of queries whose relevance labels are unreviewed, pending, or lack reviewer identity."""
        return len(self.queries) - self.reviewed_queries_count

    @property
    def is_manually_reviewed(self) -> bool:
        """True if and only if every query in the dataset has explicitly confirmed manual review."""
        if not self.queries:
            return False
        return (
            self.reviewed_queries_count == len(self.queries)
            and self.unreviewed_queries_count == 0
        )

    @property
    def benchmark_tier(self) -> str:
        """Categorize dataset into illustrative, candidate_real_world, or manually_reviewed_benchmark.

        Integrity guarantee: A dataset is never classified as manually_reviewed_benchmark
        unless every query (including unanswerable) has explicitly confirmed review status with reviewer identity.
        """
        if self.is_illustrative_only is True or self.provenance == "illustrative":
            return "illustrative"
        if self.is_manually_reviewed and self.is_illustrative_only is False:
            if self.provenance in (
                "manually_reviewed_benchmark",
                "reviewed_benchmark",
                "benchmark",
                "candidate",
                "candidate_real_world",
            ):
                return "manually_reviewed_benchmark"
        if self.is_illustrative_only is False or self.provenance in ("candidate", "candidate_real_world", "benchmark"):
            return "candidate_real_world"
        return "unknown"

    @property
    def review_summary(self) -> dict[str, Any]:
        """Structured review metrics across all queries."""
        return {
            "total_queries": len(self.queries),
            "reviewed_queries": self.reviewed_queries_count,
            "unreviewed_queries": self.unreviewed_queries_count,
            "is_fully_reviewed": self.is_manually_reviewed,
            "benchmark_tier": self.benchmark_tier,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset to dictionary representation."""
        res: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "description": self.description,
            "chunks": [dict(c) for c in self.chunks],
            "queries": [dict(q) for q in self.queries],
            "is_illustrative_only": self.is_illustrative_only,
            "provenance": self.provenance,
        }
        if self.version is not None:
            res["version"] = self.version
        res["benchmark_tier"] = self.benchmark_tier
        res["review_summary"] = self.review_summary
        return res


def _validate_chunk(raw: Any, index: int, seen_ids: set[str]) -> dict[str, Any]:
    """Validate a single chunk record and check for uniqueness."""
    if not isinstance(raw, dict):
        raise DatasetValidationError(
            f"Chunk at index {index} must be a dictionary, got {type(raw).__name__}"
        )

    # 1. chunk_id validation
    chunk_id = raw.get("chunk_id")
    if chunk_id is None:
        raise DatasetValidationError(f"Chunk at index {index} is missing required field 'chunk_id'")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise DatasetValidationError(
            f"Chunk at index {index} has invalid 'chunk_id': must be a non-empty string"
        )
    chunk_id = chunk_id.strip()
    if chunk_id in seen_ids:
        raise DatasetValidationError(
            f"Duplicate chunk_id '{chunk_id}' detected at chunk index {index}"
        )
    seen_ids.add(chunk_id)

    # 2. paper_id validation
    paper_id = raw.get("paper_id")
    if paper_id is None:
        raise DatasetValidationError(
            f"Chunk '{chunk_id}' (index {index}) is missing required field 'paper_id'"
        )
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise DatasetValidationError(
            f"Chunk '{chunk_id}' (index {index}) has invalid 'paper_id': must be a non-empty string"
        )
    paper_id = paper_id.strip()

    # 3. text validation
    text = raw.get("text")
    if text is None:
        raise DatasetValidationError(
            f"Chunk '{chunk_id}' (index {index}) is missing required field 'text'"
        )
    if not isinstance(text, str) or not text.strip():
        raise DatasetValidationError(
            f"Chunk '{chunk_id}' (index {index}) has invalid 'text': must be a non-empty string"
        )

    # 4. Optional metadata fields with deterministic defaults
    document_id = raw.get("document_id")
    if document_id is not None and (not isinstance(document_id, str) or not document_id.strip()):
        raise DatasetValidationError(
            f"Chunk '{chunk_id}' has invalid 'document_id': must be a non-empty string when provided"
        )
    doc_id = str(document_id).strip() if document_id else f"doc_{paper_id}"

    title = str(raw.get("title") or "").strip()
    filename = str(raw.get("filename") or f"{paper_id}.pdf").strip()

    chunk_index = raw.get("chunk_index", index)
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise DatasetValidationError(
            f"Chunk '{chunk_id}' has invalid 'chunk_index': must be a non-negative integer"
        )

    # 5. Traceable research source identifiers (doi, url, publication_year)
    doi = raw.get("doi")
    if doi is not None:
        if not isinstance(doi, str):
            raise DatasetValidationError(
                f"Chunk '{chunk_id}' (index {index}) has invalid 'doi': must be a string, got {type(doi).__name__}"
            )
        doi = doi.strip() if doi.strip() else None

    url = raw.get("url") or raw.get("source_url")
    if url is not None:
        if not isinstance(url, str):
            raise DatasetValidationError(
                f"Chunk '{chunk_id}' (index {index}) has invalid 'url': must be a string, got {type(url).__name__}"
            )
        url = url.strip() if url.strip() else None

    pub_year = raw.get("publication_year")
    if pub_year is None and "year" in raw:
        pub_year = raw.get("year")
    if pub_year is not None:
        if isinstance(pub_year, bool) or not isinstance(pub_year, int):
            raise DatasetValidationError(
                f"Chunk '{chunk_id}' (index {index}) has invalid 'publication_year': must be an integer, got {type(pub_year).__name__}"
            )
        if pub_year < 1000 or pub_year > 3000:
            raise DatasetValidationError(
                f"Chunk '{chunk_id}' (index {index}) has invalid 'publication_year': {pub_year} out of valid range (1000-3000)"
            )

    page_number = raw.get("page_number")
    if page_number is None and "page" in raw:
        page_number = raw.get("page")
    if page_number is not None:
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number <= 0:
            raise DatasetValidationError(
                f"Chunk '{chunk_id}' (index {index}) has invalid 'page_number': must be a positive integer"
            )

    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "paper_id": paper_id,
        "title": title,
        "filename": filename,
        "chunk_index": chunk_index,
        "text": text.strip(),
        "doi": doi,
        "url": url,
        "publication_year": pub_year,
        "page_number": page_number,
    }


def _validate_query(
    raw: Any, index: int, valid_chunk_ids: set[str], seen_query_ids: set[str]
) -> dict[str, Any]:
    """Validate a single query record and check referential integrity against chunks."""
    if not isinstance(raw, dict):
        raise DatasetValidationError(
            f"Query at index {index} must be a dictionary, got {type(raw).__name__}"
        )

    # 1. query text validation
    query_text = raw.get("query") or raw.get("query_text")
    if query_text is None:
        raise DatasetValidationError(f"Query at index {index} is missing required field 'query'")
    if not isinstance(query_text, str) or not query_text.strip():
        raise DatasetValidationError(
            f"Query at index {index} has invalid 'query': must be a non-empty string"
        )
    query_text = query_text.strip()

    # 2. query_id validation (optional uniqueness)
    query_id = raw.get("query_id")
    if query_id is not None:
        if not isinstance(query_id, str) or not query_id.strip():
            raise DatasetValidationError(
                f"Query at index {index} has invalid 'query_id': must be a non-empty string"
            )
        q_id = query_id.strip()
        if q_id in seen_query_ids:
            raise DatasetValidationError(
                f"Duplicate query_id '{q_id}' detected at query index {index}"
            )
        seen_query_ids.add(q_id)
    else:
        q_id = f"q_{index + 1}"

    # 3. expected_chunk_ids validation and referential integrity
    expected_ids = raw.get("expected_chunk_ids")
    if expected_ids is None and "relevant_chunk_ids" in raw:
        expected_ids = raw.get("relevant_chunk_ids")
    if expected_ids is None:
        raise DatasetValidationError(
            f"Query '{q_id}' (index {index}) is missing required field 'expected_chunk_ids'"
        )
    if not isinstance(expected_ids, list):
        raise DatasetValidationError(
            f"Query '{q_id}' (index {index}) 'expected_chunk_ids' must be a list, got {type(expected_ids).__name__}"
        )

    sanitized_expected: list[str] = []
    seen_expected_for_query: set[str] = set()
    for cid in expected_ids:
        if not isinstance(cid, str) or not cid.strip():
            raise DatasetValidationError(
                f"Query '{q_id}' has invalid chunk reference: expected string chunk_id, got {cid!r}"
            )
        clean_cid = cid.strip()
        if clean_cid in seen_expected_for_query:
            raise DatasetValidationError(
                f"Query '{q_id}' contains duplicate expected_chunk_id '{clean_cid}'"
            )
        seen_expected_for_query.add(clean_cid)

        # Referential integrity check: every referenced chunk must exist in chunks!
        if clean_cid not in valid_chunk_ids:
            raise DatasetValidationError(
                f"Query '{q_id}' references nonexistent chunk_id '{clean_cid}' "
                f"(not present in corpus chunks)"
            )
        sanitized_expected.append(clean_cid)

    # 4. Category & Target Papers
    category = raw.get("category")
    if category is not None and not isinstance(category, str):
        raise DatasetValidationError(
            f"Query '{q_id}' has invalid 'category': must be a string"
        )
    cat_str = str(category).strip() if category else ("unanswerable_empty" if not sanitized_expected else "general")

    target_papers = raw.get("target_paper_ids")
    if target_papers is not None:
        if not isinstance(target_papers, list) or not all(isinstance(p, str) for p in target_papers):
            raise DatasetValidationError(
                f"Query '{q_id}' 'target_paper_ids' must be a list of strings or None"
            )

    # 5. Relevance label review status and audit metadata (Milestone 12C-35)
    reviewed_val = raw.get("reviewed")
    review_status_val = raw.get("review_status")
    reviewer_val = raw.get("reviewer")
    review_notes_val = raw.get("review_notes") or raw.get("notes")

    # Support nested review object if provided
    raw_review = raw.get("review")
    if raw_review is not None:
        if not isinstance(raw_review, dict):
            raise DatasetValidationError(
                f"Query '{q_id}' (index {index}) has invalid 'review': must be a dictionary, got {type(raw_review).__name__}"
            )
        if reviewed_val is None and "reviewed" in raw_review:
            reviewed_val = raw_review.get("reviewed")
        if review_status_val is None and "review_status" in raw_review:
            review_status_val = raw_review.get("review_status")
        if reviewer_val is None and "reviewer" in raw_review:
            reviewer_val = raw_review.get("reviewer")
        if review_notes_val is None:
            review_notes_val = raw_review.get("review_notes") or raw_review.get("notes")

    is_reviewed: bool | None = None
    if reviewed_val is not None:
        if not isinstance(reviewed_val, bool):
            raise DatasetValidationError(
                f"Query '{q_id}' (index {index}) has invalid 'reviewed': must be a boolean (true or false), got {type(reviewed_val).__name__}"
            )
        is_reviewed = reviewed_val

    if review_status_val is not None:
        if not isinstance(review_status_val, str):
            raise DatasetValidationError(
                f"Query '{q_id}' (index {index}) has invalid 'review_status': must be a string, got {type(review_status_val).__name__}"
            )
        clean_status = review_status_val.strip().lower()
        if clean_status not in VALID_REVIEW_STATUS_VALUES:
            raise DatasetValidationError(
                f"Query '{q_id}' (index {index}) has invalid 'review_status': '{review_status_val}'. "
                f"Must be one of: {sorted(VALID_REVIEW_STATUS_VALUES)}"
            )
        if is_reviewed is None:
            is_reviewed = (clean_status == "reviewed")

    reviewer: str | None = None
    if reviewer_val is not None:
        if not isinstance(reviewer_val, str):
            raise DatasetValidationError(
                f"Query '{q_id}' (index {index}) has invalid 'reviewer': must be a string, got {type(reviewer_val).__name__}"
            )
        reviewer = reviewer_val.strip() if reviewer_val.strip() else None

    if is_reviewed is True and not reviewer:
        raise DatasetValidationError(
            f"Query '{q_id}' (index {index}) is marked as reviewed but is missing a valid, non-empty 'reviewer' identifier"
        )

    review_notes: str | None = None
    if review_notes_val is not None:
        if not isinstance(review_notes_val, str):
            raise DatasetValidationError(
                f"Query '{q_id}' (index {index}) has invalid 'review_notes': must be a string, got {type(review_notes_val).__name__}"
            )
        review_notes = review_notes_val.strip() if review_notes_val.strip() else None

    return {
        "query": query_text,
        "expected_chunk_ids": sanitized_expected,
        "query_id": q_id,
        "category": cat_str,
        "target_paper_ids": target_papers,
        "reviewed": is_reviewed,
        "reviewer": reviewer,
        "review_notes": review_notes,
    }


def validate_dataset_dict(raw: dict[str, Any]) -> ResearchRetrievalDataset:
    """Validate a parsed dictionary containing 'chunks' and 'queries'.

    Raises DatasetValidationError on any missing field, duplicate ID, or nonexistent reference.
    Never silently discards invalid records.
    """
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"Dataset root must be a dictionary, got {type(raw).__name__}")

    raw_chunks = raw.get("chunks")
    if raw_chunks is None:
        raise DatasetValidationError("Dataset is missing required key 'chunks'")
    if not isinstance(raw_chunks, list):
        raise DatasetValidationError(f"'chunks' must be a list, got {type(raw_chunks).__name__}")
    if len(raw_chunks) == 0:
        raise DatasetValidationError("Dataset 'chunks' list must not be empty")

    raw_queries = raw.get("queries")
    if raw_queries is None and "cases" in raw:
        raw_queries = raw.get("cases")
    if raw_queries is None:
        raise DatasetValidationError("Dataset is missing required key 'queries'")
    if not isinstance(raw_queries, list):
        raise DatasetValidationError(f"'queries' must be a list, got {type(raw_queries).__name__}")
    if len(raw_queries) == 0:
        raise DatasetValidationError("Dataset 'queries' list must not be empty")

    seen_chunk_ids: set[str] = set()
    validated_chunks = [
        _validate_chunk(chunk, idx, seen_chunk_ids) for idx, chunk in enumerate(raw_chunks)
    ]

    seen_query_ids: set[str] = set()
    validated_queries = [
        _validate_query(query, idx, seen_chunk_ids, seen_query_ids)
        for idx, query in enumerate(raw_queries)
    ]

    dataset_id = str(raw.get("dataset_id") or "custom_research_dataset").strip()
    description = str(
        raw.get("description")
        or "Custom research retrieval dataset for quantitative evaluation"
    ).strip()

    # Version validation (Milestone 12C-35)
    raw_version = raw.get("version") or raw.get("dataset_version")
    if raw_version is not None:
        if not isinstance(raw_version, str) or not raw_version.strip():
            raise DatasetValidationError(
                f"Invalid 'version': must be a non-empty string when provided, got {type(raw_version).__name__}"
            )
        version = raw_version.strip()
    else:
        version = None

    # Provenance validation (Milestone 12C-35)
    raw_provenance = raw.get("provenance")
    if raw_provenance is not None:
        if not isinstance(raw_provenance, str) or not raw_provenance.strip():
            raise DatasetValidationError(
                f"Invalid 'provenance': must be a non-empty string, got {type(raw_provenance).__name__}"
            )
        prov_clean = raw_provenance.strip().lower()
        if prov_clean not in VALID_PROVENANCE_VALUES:
            raise DatasetValidationError(
                f"Invalid 'provenance': '{raw_provenance}'. Must be one of: {sorted(VALID_PROVENANCE_VALUES)}"
            )
        provenance = prov_clean
    else:
        provenance = None

    if "is_illustrative_only" in raw:
        raw_illustrative = raw["is_illustrative_only"]
        if not isinstance(raw_illustrative, bool):
            val_type = "null" if raw_illustrative is None else type(raw_illustrative).__name__
            raise DatasetValidationError(
                f"Invalid 'is_illustrative_only': must be a boolean (true or false), "
                f"got {val_type} ({raw_illustrative!r})"
            )
        is_illustrative = raw_illustrative
    else:
        is_illustrative = None

    # Check consistency between is_illustrative_only and provenance
    if provenance == "illustrative":
        if is_illustrative is False:
            raise DatasetValidationError(
                "Contradictory metadata: provenance='illustrative' cannot be combined with is_illustrative_only=False"
            )
        is_illustrative = True
    elif provenance in ("candidate", "candidate_real_world", "manually_reviewed_benchmark", "reviewed_benchmark", "benchmark"):
        if is_illustrative is True:
            raise DatasetValidationError(
                f"Contradictory metadata: provenance='{provenance}' cannot be combined with is_illustrative_only=True"
            )
        is_illustrative = False
    elif provenance is None:
        if is_illustrative is True:
            provenance = "illustrative"
        elif is_illustrative is False:
            provenance = "benchmark"
        else:
            provenance = "unknown"

    # Dataset-level review status validation
    dataset_reviewed_flag = raw.get("reviewed")
    if dataset_reviewed_flag is not None and not isinstance(dataset_reviewed_flag, bool):
        raise DatasetValidationError(
            f"Dataset-level 'reviewed' must be a boolean (true or false), got {type(dataset_reviewed_flag).__name__}"
        )

    dataset_review_status = raw.get("review_status")
    if dataset_review_status is not None:
        if not isinstance(dataset_review_status, str):
            raise DatasetValidationError(
                f"Dataset-level 'review_status' must be a string, got {type(dataset_review_status).__name__}"
            )
        if dataset_review_status.strip().lower() not in VALID_REVIEW_STATUS_VALUES:
            raise DatasetValidationError(
                f"Dataset-level 'review_status' must be one of: {sorted(VALID_REVIEW_STATUS_VALUES)}"
            )

    unreviewed_queries = [
        q.get("query_id", f"q_{idx+1}")
        for idx, q in enumerate(validated_queries)
        if q.get("reviewed") is not True or not q.get("reviewer") or not str(q.get("reviewer")).strip()
    ]

    # Fail closed: Do not label a dataset as manually reviewed unless EVERY query has explicitly confirmed review
    if provenance in ("manually_reviewed_benchmark", "reviewed_benchmark"):
        if not validated_queries:
            raise DatasetValidationError(
                f"Cannot label dataset as manually reviewed ('{provenance}'): dataset has no queries"
            )
        if unreviewed_queries:
            raise DatasetValidationError(
                f"Cannot label dataset as manually reviewed ('{provenance}'): "
                f"every query's relevance labels (including unanswerable queries) must have explicitly "
                f"confirmed review status and a non-empty reviewer identifier. "
                f"Found {len(unreviewed_queries)} unreviewed or incomplete query record(s): {unreviewed_queries}"
            )

    return ResearchRetrievalDataset(
        dataset_id=dataset_id,
        description=description,
        chunks=validated_chunks,
        queries=validated_queries,
        is_illustrative_only=is_illustrative,
        provenance=provenance,
        version=version,
    )


def load_retrieval_dataset_from_jsonl(lines: Sequence[str]) -> ResearchRetrievalDataset:
    """Parse and validate a JSONL dataset stream."""
    chunks: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    dataset_id = "custom_jsonl_dataset"
    description = "JSONL research retrieval dataset"
    is_illustrative_specified = False
    is_illustrative_val: Any = None
    metadata_provenance: Any = None
    metadata_version: Any = None
    metadata_reviewed: Any = None
    metadata_review_status: Any = None

    non_empty_lines = 0
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        non_empty_lines += 1
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"Malformed JSON on line {line_num}: {exc.msg}"
            ) from exc

        if not isinstance(record, dict):
            raise DatasetValidationError(
                f"Line {line_num} must be a JSON object, got {type(record).__name__}"
            )

        rec_type = record.get("type")
        if rec_type == "metadata" or ("dataset_id" in record and "chunks" not in record and "queries" not in record):
            dataset_id = str(record.get("dataset_id") or dataset_id)
            description = str(record.get("description") or description)
            if "is_illustrative_only" in record:
                is_illustrative_specified = True
                is_illustrative_val = record["is_illustrative_only"]
            if "provenance" in record:
                metadata_provenance = record["provenance"]
            if "version" in record:
                metadata_version = record["version"]
            elif "dataset_version" in record:
                metadata_version = record["dataset_version"]
            if "reviewed" in record:
                metadata_reviewed = record["reviewed"]
            if "review_status" in record:
                metadata_review_status = record["review_status"]
        elif rec_type == "chunk" or ("chunk_id" in record and "text" in record):
            chunks.append(record)
        elif rec_type == "query" or ("query" in record and ("expected_chunk_ids" in record or "relevant_chunk_ids" in record)):
            queries.append(record)
        else:
            raise DatasetValidationError(
                f"Line {line_num} has unrecognized record format: {record}"
            )

    if non_empty_lines == 0:
        raise DatasetValidationError("JSONL dataset is empty")

    dict_to_validate: dict[str, Any] = {
        "dataset_id": dataset_id,
        "description": description,
        "chunks": chunks,
        "queries": queries,
    }
    if is_illustrative_specified:
        dict_to_validate["is_illustrative_only"] = is_illustrative_val
    if metadata_provenance is not None:
        dict_to_validate["provenance"] = metadata_provenance
    if metadata_version is not None:
        dict_to_validate["version"] = metadata_version
    if metadata_reviewed is not None:
        dict_to_validate["reviewed"] = metadata_reviewed
    if metadata_review_status is not None:
        dict_to_validate["review_status"] = metadata_review_status

    return validate_dataset_dict(dict_to_validate)


def load_retrieval_dataset(source: str | os.PathLike | dict[str, Any]) -> ResearchRetrievalDataset:
    """Load and strictly validate a retrieval dataset from a file path or dict.

    Supports:
    - Path to JSON file (ending in .json or containing JSON object)
    - Path to JSONL file (ending in .jsonl or containing JSONL records)
    - Dictionary representation directly

    Raises DatasetValidationError on any missing field, duplicate ID, nonexistent reference,
    or invalid file syntax.
    """
    if isinstance(source, dict):
        return validate_dataset_dict(source)

    path_str = str(source).strip()
    if not os.path.exists(path_str):
        raise DatasetValidationError(f"Dataset file not found: {path_str}")

    try:
        with open(path_str, "r", encoding="utf-8") as fh:
            content = fh.read()
    except Exception as exc:
        raise DatasetValidationError(f"Could not read dataset file '{path_str}': {exc}") from exc

    if not content.strip():
        raise DatasetValidationError(f"Dataset file is empty: {path_str}")

    # Detect JSON vs JSONL
    first_char = content.lstrip()[0] if content.lstrip() else ""
    if path_str.endswith(".jsonl") or (first_char == "{" and "\n{" in content and "chunks" not in content[:200]):
        return load_retrieval_dataset_from_jsonl(content.splitlines())

    try:
        raw_json = json.loads(content)
    except json.JSONDecodeError as exc:
        # Fall back to testing JSONL lines before failing
        lines = [line for line in content.splitlines() if line.strip()]
        if len(lines) > 1:
            try:
                return load_retrieval_dataset_from_jsonl(lines)
            except DatasetValidationError:
                pass
        raise DatasetValidationError(f"Malformed JSON in dataset file '{path_str}': {exc.msg}") from exc

    if not isinstance(raw_json, dict):
        raise DatasetValidationError(f"Dataset file must contain a JSON object, got {type(raw_json).__name__}")

    return validate_dataset_dict(raw_json)


# ---------------------------------------------------------------------------
# Illustrative Example Dataset (Small-Scale, Non-Benchmark)
# ---------------------------------------------------------------------------
def get_illustrative_example_dataset() -> ResearchRetrievalDataset:
    """Return a small, validated illustrative dataset for testing and documentation.

    NOTE: This dataset is illustrative only and is NOT a validated scientific benchmark.
    """
    return ResearchRetrievalDataset(
        dataset_id="illustrative-rag-dataset-v1",
        description="Illustrative small-scale research retrieval dataset for demonstration and testing only. Not a real-world benchmark.",
        is_illustrative_only=True,
        chunks=[
            {
                "chunk_id": "chunk_kv_01",
                "document_id": "doc_kv_cache",
                "paper_id": "paper_kv_cache_2023",
                "title": "KV Cache Compression Policies",
                "filename": "kv_cache_2023.pdf",
                "chunk_index": 0,
                "text": "Key-value cache eviction using attention-score accumulation minimizes memory footprint in long-context inference.",
            },
            {
                "chunk_id": "chunk_kv_02",
                "document_id": "doc_kv_cache",
                "paper_id": "paper_kv_cache_2023",
                "title": "KV Cache Compression Policies",
                "filename": "kv_cache_2023.pdf",
                "chunk_index": 1,
                "text": "PagedAttention partitions KV cache blocks into non-contiguous physical memory pages to eradicate internal fragmentation.",
            },
            {
                "chunk_id": "chunk_spec_01",
                "document_id": "doc_speculative",
                "paper_id": "paper_speculative_2023",
                "title": "Speculative Decoding for Accelerated Generation",
                "filename": "speculative_2023.pdf",
                "chunk_index": 0,
                "text": "Speculative decoding uses a smaller draft model to generate candidate tokens which are verified in parallel by the target model.",
            },
            {
                "chunk_id": "chunk_spec_02",
                "document_id": "doc_speculative",
                "paper_id": "paper_speculative_2023",
                "title": "Speculative Decoding for Accelerated Generation",
                "filename": "speculative_2023.pdf",
                "chunk_index": 1,
                "text": "Exact acceptance criteria in speculative decoding mathematically preserve the target model's output probability distribution.",
            },
            {
                "chunk_id": "chunk_quant_01",
                "document_id": "doc_quantization",
                "paper_id": "paper_quantization_2023",
                "title": "Post-Training Quantization for Large Models",
                "filename": "quantization_2023.pdf",
                "chunk_index": 0,
                "text": "4-bit weight-only quantization using second-order Taylor expansion retains baseline perplexity while cutting DRAM bandwidth in half.",
            },
            {
                "chunk_id": "chunk_quant_02",
                "document_id": "doc_quantization",
                "paper_id": "paper_quantization_2023",
                "title": "Post-Training Quantization for Large Models",
                "filename": "quantization_2023.pdf",
                "chunk_index": 1,
                "text": "Activation outlier preservation prevents gradient divergence during low-bit linear layer inference.",
            },
        ],
        queries=[
            {
                "query": "How does attention-score accumulation guide KV cache eviction?",
                "expected_chunk_ids": ["chunk_kv_01"],
                "query_id": "q_exact_kv_cache",
                "category": "exact_term",
                "target_paper_ids": ["paper_kv_cache_2023"],
            },
            {
                "query": "Accelerating LLM token sampling by verifying parallel predictions from a small auxiliary network",
                "expected_chunk_ids": ["chunk_spec_01"],
                "query_id": "q_semantic_speculative",
                "category": "semantic",
                "target_paper_ids": ["paper_speculative_2023"],
            },
            {
                "query": "Techniques for mitigating GPU DRAM bandwidth bottlenecks through compression or lower precision",
                "expected_chunk_ids": ["chunk_kv_01", "chunk_quant_01"],
                "query_id": "q_comparative_compression",
                "category": "comparative",
                "target_paper_ids": ["paper_kv_cache_2023", "paper_quantization_2023"],
            },
            {
                "query": "Quantum flux qubit coherence times in dilution refrigerators",
                "expected_chunk_ids": [],
                "query_id": "q_unanswerable_quantum",
                "category": "unanswerable_empty",
                "target_paper_ids": None,
            },
        ],
    )


# ---------------------------------------------------------------------------
# Real-World Benchmark Dataset Template (Preparation Infrastructure, Milestone 12C-35)
# ---------------------------------------------------------------------------
def get_benchmark_template_dataset() -> ResearchRetrievalDataset:
    """Return a validated real-world benchmark dataset template.

    NOTE: This is a preparation template for structuring real research material.
    It contains schema placeholders and is not proof of real-world retrieval quality.
    """
    return ResearchRetrievalDataset(
        dataset_id="real-world-research-benchmark-template-v1",
        version="1.0.0",
        description="Template for real-world research retrieval evaluation datasets with traceable sources and reviewed relevance labels.",
        is_illustrative_only=False,
        provenance="candidate_real_world",
        chunks=[
            {
                "chunk_id": "chunk_source1_01",
                "document_id": "doc_template_alpha",
                "paper_id": "paper_template_alpha",
                "title": "Template Source Paper Alpha: Real-World Title Placeholder",
                "doi": "10.xxxx/placeholder.alpha",
                "url": "https://example.org/papers/placeholder_alpha.pdf",
                "publication_year": 2024,
                "filename": "placeholder_alpha.pdf",
                "chunk_index": 0,
                "text": "Template text placeholder for chunk 1. Replace with verbatim excerpt from genuine research paper.",
            },
            {
                "chunk_id": "chunk_source1_02",
                "document_id": "doc_template_alpha",
                "paper_id": "paper_template_alpha",
                "title": "Template Source Paper Alpha: Real-World Title Placeholder",
                "doi": "10.xxxx/placeholder.alpha",
                "url": "https://example.org/papers/placeholder_alpha.pdf",
                "publication_year": 2024,
                "filename": "placeholder_alpha.pdf",
                "chunk_index": 1,
                "text": "Template text placeholder for chunk 2. Replace with verbatim excerpt from genuine research paper.",
            },
            {
                "chunk_id": "chunk_source2_01",
                "document_id": "doc_template_beta",
                "paper_id": "paper_template_beta",
                "title": "Template Source Paper Beta: Real-World Title Placeholder",
                "doi": "10.xxxx/placeholder.beta",
                "url": "https://example.org/papers/placeholder_beta.pdf",
                "publication_year": 2023,
                "filename": "placeholder_beta.pdf",
                "chunk_index": 0,
                "text": "Template text placeholder for chunk 3. Replace with verbatim excerpt from genuine research paper.",
            },
        ],
        queries=[
            {
                "query": "Template exact-term query placeholder reflecting verified research question",
                "expected_chunk_ids": ["chunk_source1_01"],
                "query_id": "q_template_01",
                "category": "exact_term",
                "target_paper_ids": ["paper_template_alpha"],
                "reviewed": False,
                "reviewer": None,
                "review_notes": "Pending manual review. Once confirmed against source document, set reviewed=true, record reviewer ID, and update provenance to manually_reviewed_benchmark.",
            },
            {
                "query": "Template semantic query placeholder reflecting high-level conceptual search",
                "expected_chunk_ids": ["chunk_source2_01"],
                "query_id": "q_template_02",
                "category": "semantic",
                "target_paper_ids": ["paper_template_beta"],
                "reviewed": False,
                "reviewer": None,
                "review_notes": "Pending manual review. Verify that chunk_source2_01 contains direct evidential answer.",
            },
            {
                "query": "Template unanswerable query placeholder outside scope of indexed documents",
                "expected_chunk_ids": [],
                "query_id": "q_template_03",
                "category": "unanswerable_empty",
                "target_paper_ids": None,
                "reviewed": False,
                "reviewer": None,
                "review_notes": "Pending manual review. Confirmed that no indexed chunk contains supporting evidence for this topic.",
            },
        ],
    )


# ---------------------------------------------------------------------------
# Curated Pilot Real-World Benchmark (Milestone 12C-36)
# ---------------------------------------------------------------------------
def get_pilot_benchmark_dataset() -> ResearchRetrievalDataset:
    """Return a curated pilot real-world benchmark dataset with verified citations and reviewed labels.

    Pilot Corpus:
    1. Attention Is All You Need (Vaswani et al., 2017) - arXiv:1706.03762
    2. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) - arXiv:2106.09685
    3. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022) - arXiv:2205.14135

    All 7 queries (exact-term, semantic, and unanswerable out-of-domain) are manually reviewed
    against actual source text with reviewer identity recorded.
    """
    return ResearchRetrievalDataset(
        dataset_id="real-world-pilot-benchmark-v1",
        version="1.0.0",
        description="Curated pilot retrieval benchmark of 3 genuine research papers (Attention, LoRA, FlashAttention) with verified source metadata and manually reviewed relevance labels.",
        is_illustrative_only=False,
        provenance="manually_reviewed_benchmark",
        chunks=[
            {
                "chunk_id": "chunk_1706_03762_01",
                "paper_id": "arxiv_1706.03762",
                "document_id": "doc_arxiv_1706.03762",
                "title": "Attention Is All You Need",
                "doi": "10.48550/arXiv.1706.03762",
                "url": "https://arxiv.org/abs/1706.03762",
                "publication_year": 2017,
                "filename": "1706.03762.pdf",
                "chunk_index": 0,
                "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
            },
            {
                "chunk_id": "chunk_1706_03762_02",
                "paper_id": "arxiv_1706.03762",
                "document_id": "doc_arxiv_1706.03762",
                "title": "Attention Is All You Need",
                "doi": "10.48550/arXiv.1706.03762",
                "url": "https://arxiv.org/abs/1706.03762",
                "publication_year": 2017,
                "filename": "1706.03762.pdf",
                "chunk_index": 1,
                "text": "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
            },
            {
                "chunk_id": "chunk_2106_09685_01",
                "paper_id": "arxiv_2106.09685",
                "document_id": "doc_arxiv_2106.09685",
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "doi": "10.48550/arXiv.2106.09685",
                "url": "https://arxiv.org/abs/2106.09685",
                "publication_year": 2021,
                "filename": "2106.09685.pdf",
                "chunk_index": 0,
                "text": "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.",
            },
            {
                "chunk_id": "chunk_2106_09685_02",
                "paper_id": "arxiv_2106.09685",
                "document_id": "doc_arxiv_2106.09685",
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "doi": "10.48550/arXiv.2106.09685",
                "url": "https://arxiv.org/abs/2106.09685",
                "publication_year": 2021,
                "filename": "2106.09685.pdf",
                "chunk_index": 1,
                "text": "Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.",
            },
            {
                "chunk_id": "chunk_2205_14135_01",
                "paper_id": "arxiv_2205.14135",
                "document_id": "doc_arxiv_2205.14135",
                "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
                "doi": "10.48550/arXiv.2205.14135",
                "url": "https://arxiv.org/abs/2205.14135",
                "publication_year": 2022,
                "filename": "2205.14135.pdf",
                "chunk_index": 0,
                "text": "Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO-aware -- accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM.",
            },
            {
                "chunk_id": "chunk_2205_14135_02",
                "paper_id": "arxiv_2205.14135",
                "document_id": "doc_arxiv_2205.14135",
                "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
                "doi": "10.48550/arXiv.2205.14135",
                "url": "https://arxiv.org/abs/2205.14135",
                "publication_year": 2022,
                "filename": "2205.14135.pdf",
                "chunk_index": 1,
                "text": "We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3x speedup on GPT-2 (seq. length 1K), and 2.4x speedup on long-range arena (seq. length 1K-4K).",
            },
        ],
        queries=[
            {
                "query_id": "q_pilot_01",
                "query": "What BLEU score did the Transformer achieve on WMT 2014 English-to-German translation?",
                "expected_chunk_ids": ["chunk_1706_03762_02"],
                "category": "exact_term",
                "target_paper_ids": ["arxiv_1706.03762"],
                "reviewed": True,
                "reviewer": "curator_saiyed_simnan",
                "review_notes": "Verified against Vaswani et al. (2017) abstract reporting 28.4 BLEU on English-to-German translation.",
            },
            {
                "query_id": "q_pilot_02",
                "query": "How does Low-Rank Adaptation (LoRA) freeze pre-trained weights and reduce trainable parameters?",
                "expected_chunk_ids": ["chunk_2106_09685_01"],
                "category": "exact_term",
                "target_paper_ids": ["arxiv_2106.09685"],
                "reviewed": True,
                "reviewer": "curator_saiyed_simnan",
                "review_notes": "Verified against Hu et al. (2021) abstract describing freezing pre-trained model weights and injecting rank decomposition matrices.",
            },
            {
                "query_id": "q_pilot_03",
                "query": "How does IO-aware tiling reduce memory reads and writes between GPU HBM and on-chip SRAM in FlashAttention?",
                "expected_chunk_ids": ["chunk_2205_14135_01"],
                "category": "exact_term",
                "target_paper_ids": ["arxiv_2205.14135"],
                "reviewed": True,
                "reviewer": "curator_saiyed_simnan",
                "review_notes": "Verified against Dao et al. (2022) abstract describing FlashAttention tiling between GPU HBM and on-chip SRAM.",
            },
            {
                "query_id": "q_pilot_04",
                "query": "Parameter-efficient adaptation with rank decomposition matrices for large foundation models",
                "expected_chunk_ids": ["chunk_2106_09685_01"],
                "category": "semantic",
                "target_paper_ids": ["arxiv_2106.09685"],
                "reviewed": True,
                "reviewer": "curator_saiyed_simnan",
                "review_notes": "Verified conceptually against Hu et al. (2021) low-rank adaptation mechanism.",
            },
            {
                "query_id": "q_pilot_05",
                "query": "Wall-clock training speedup and IO complexity for exact attention on long sequences",
                "expected_chunk_ids": ["chunk_2205_14135_02"],
                "category": "semantic",
                "target_paper_ids": ["arxiv_2205.14135"],
                "reviewed": True,
                "reviewer": "curator_saiyed_simnan",
                "review_notes": "Verified against Dao et al. (2022) abstract describing wall-clock speedup on BERT-large, GPT-2, and long-range arena.",
            },
            {
                "query_id": "q_pilot_06",
                "query": "Dispensing with recurrence and convolutions in sequence transduction architectures",
                "expected_chunk_ids": ["chunk_1706_03762_01"],
                "category": "semantic",
                "target_paper_ids": ["arxiv_1706.03762"],
                "reviewed": True,
                "reviewer": "curator_saiyed_simnan",
                "review_notes": "Verified against Vaswani et al. (2017) abstract proposing Transformer architecture solely based on attention mechanisms.",
            },
            {
                "query_id": "q_pilot_07",
                "query": "Surface code quantum error correction threshold using topological cluster states in superconducting resonators",
                "expected_chunk_ids": [],
                "category": "unanswerable_empty",
                "target_paper_ids": None,
                "reviewed": True,
                "reviewer": "curator_saiyed_simnan",
                "review_notes": "Manually verified out-of-domain: topic is topological quantum computing and superconducting resonators, completely unrepresented in this corpus of machine learning attention and adaptation papers.",
            },
        ],
    )


