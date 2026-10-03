"""Deterministic formatting of an EvidenceContext into prompt-ready text."""

from __future__ import annotations

from nexora.retrieval.evidence import EvidenceContext

NO_EVIDENCE_TEXT = "No evidence was retrieved."


class EvidenceContextFormatter:
    """Renders evidence items as labelled blocks ([E1], [E2], ...).

    Evidence text is copied verbatim; nothing is summarized, truncated or
    reordered, and raw metadata is not included.
    """

    def format(self, context: EvidenceContext) -> str:
        if not isinstance(context, EvidenceContext):
            raise TypeError("context must be an EvidenceContext")
        if not context.items:
            return NO_EVIDENCE_TEXT
        blocks = [
            f"[E{number}]\n"
            f"Document: {item.document_id}\n"
            f"Chunk: {item.chunk_id}\n"
            f"Score: {item.score}\n"
            f"{item.text}"
            for number, item in enumerate(context.items, start=1)
        ]
        return "\n\n".join(blocks)
