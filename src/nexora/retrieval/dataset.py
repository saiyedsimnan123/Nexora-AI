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


@dataclass(frozen=True)
class ResearchRetrievalDataset:
    """Immutable, validated research retrieval evaluation dataset."""

    dataset_id: str
    description: str
    chunks: list[dict[str, Any]]
    queries: list[dict[str, Any]]
    is_illustrative_only: bool = True

    @property
    def name(self) -> str:
        """Name/ID of the dataset."""
        return self.dataset_id

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset to dictionary representation."""
        return {
            "dataset_id": self.dataset_id,
            "description": self.description,
            "chunks": [dict(c) for c in self.chunks],
            "queries": [dict(q) for q in self.queries],
            "is_illustrative_only": self.is_illustrative_only,
        }


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

    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "paper_id": paper_id,
        "title": title,
        "filename": filename,
        "chunk_index": chunk_index,
        "text": text.strip(),
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

    return {
        "query": query_text,
        "expected_chunk_ids": sanitized_expected,
        "query_id": q_id,
        "category": cat_str,
        "target_paper_ids": target_papers,
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
    is_illustrative = bool(raw.get("is_illustrative_only", True))

    return ResearchRetrievalDataset(
        dataset_id=dataset_id,
        description=description,
        chunks=validated_chunks,
        queries=validated_queries,
        is_illustrative_only=is_illustrative,
    )


def load_retrieval_dataset_from_jsonl(lines: Sequence[str]) -> ResearchRetrievalDataset:
    """Parse and validate a JSONL dataset stream."""
    chunks: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    dataset_id = "custom_jsonl_dataset"
    description = "JSONL research retrieval dataset"
    is_illustrative = True

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
        if rec_type == "metadata":
            dataset_id = str(record.get("dataset_id") or dataset_id)
            description = str(record.get("description") or description)
            if "is_illustrative_only" in record:
                is_illustrative = bool(record["is_illustrative_only"])
        elif rec_type == "chunk" or ("chunk_id" in record and "text" in record):
            chunks.append(record)
        elif rec_type == "query" or ("query" in record and ("expected_chunk_ids" in record or "relevant_chunk_ids" in record)):
            queries.append(record)
        elif "dataset_id" in record and "chunks" not in record:
            dataset_id = str(record.get("dataset_id") or dataset_id)
            description = str(record.get("description") or description)
        else:
            raise DatasetValidationError(
                f"Line {line_num} has unrecognized record format: {record}"
            )

    if non_empty_lines == 0:
        raise DatasetValidationError("JSONL dataset is empty")

    return validate_dataset_dict({
        "dataset_id": dataset_id,
        "description": description,
        "chunks": chunks,
        "queries": queries,
        "is_illustrative_only": is_illustrative,
    })


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
