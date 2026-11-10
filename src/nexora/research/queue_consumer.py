"""Vercel Queue consumer for durable research RAG query jobs.

Subscribes to 'research-query-jobs' topic events delivered by Vercel Queue.
In serverless environments, each message triggers execution of a worker
cycle via ResearchJobWorker.run_once(), preserving dialect-aware locking,
two-gate fencing tokens, lease renewal heartbeats, and atomic status transitions.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

# Ensure 'src' is on sys.path for queue consumer serverless invocations
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vercel.queue import RetryAfter, subscribe

from nexora.research.worker import ResearchJobWorker, ResearchRunOutcome

logger = logging.getLogger(__name__)

# Global worker reference for test injection and singleton caching
_global_research_worker: ResearchJobWorker | None = None


def set_global_research_worker(worker: ResearchJobWorker | None) -> None:
    """Inject a specific worker instance (useful for unit tests and local composition)."""
    global _global_research_worker
    _global_research_worker = worker


def get_research_queue_worker() -> ResearchJobWorker:
    """Obtain or initialize the ResearchJobWorker for queue processing."""
    global _global_research_worker
    if _global_research_worker is not None:
        return _global_research_worker

    # 1. Try to obtain worker from initialized production ASGI app
    try:
        from nexora.server import app

        worker = getattr(app.state, "research_worker", None)
        if worker is not None and isinstance(worker, ResearchJobWorker):
            _global_research_worker = worker
            return _global_research_worker
    except Exception as exc:
        logger.debug("Could not resolve research worker from nexora.server.app: %s", exc)

    # 2. Lazy instantiate through production factory
    try:
        from nexora.server import create_production_app

        prod_app = create_production_app()
        worker = getattr(prod_app.state, "research_worker", None)
        if worker is not None and isinstance(worker, ResearchJobWorker):
            _global_research_worker = worker
            return _global_research_worker
    except Exception as exc:
        logger.error("Failed to initialize ResearchJobWorker from production factory: %s", exc)

    raise RuntimeError(
        "No ResearchJobWorker available for queue processing. "
        "Ensure database connection and services are properly configured."
    )


def _extract_target_job_id(payload: Any) -> UUID | None:
    if not isinstance(payload, dict):
        return None
    raw_id = payload.get("job_id")
    if not raw_id:
        return None
    try:
        return UUID(str(raw_id).strip())
    except (ValueError, TypeError, AttributeError):
        logger.warning("Ignoring invalid job_id in queue payload: %r", raw_id)
        return None


def process_research_queue_trigger(payload: Any = None) -> dict[str, Any]:
    """Process a queue trigger by invoking the worker's durable run_once cycle."""
    worker = get_research_queue_worker()
    target_job_id = _extract_target_job_id(payload)
    logger.info("Research queue trigger received (target_job_id=%s), invoking worker.run_once()", target_job_id)

    if isinstance(worker, ResearchJobWorker):
        processed_count = worker.run_once(target_job_id=target_job_id)
        outcome = worker.last_run_outcome
        if isinstance(outcome, ResearchRunOutcome) and outcome.will_retry:
            delay_seconds = max(1, int(math.ceil(outcome.backoff_seconds)))
            logger.info(
                "Research job %s failed with transient error; requesting Vercel Queue RetryAfter(%ss)",
                outcome.job_id,
                delay_seconds,
            )
            raise RetryAfter(delay_seconds)
    else:
        processed_count = worker.run_once()

    logger.info("Research worker run_once completed with %d jobs processed", processed_count)
    return {"processed_jobs": processed_count}


@subscribe(topic="research-query-jobs")
def handle_research_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Process an enqueued research query job trigger in a serverless function invocation.

    Decorated with ``@subscribe(topic="research-query-jobs")`` from the official
    ``vercel-queue`` SDK.
    """
    return process_research_queue_trigger(payload)


__all__ = [
    "get_research_queue_worker",
    "handle_research_job",
    "process_research_queue_trigger",
    "set_global_research_worker",
]
