"""Vercel Queue consumer for durable knowledge graph extraction (Stage 1).

Subscribes to 'graph-extraction-jobs' topic events delivered by Vercel Queue.
In serverless environments, each message triggers execution of a worker
cycle via KnowledgeGraphWorker.run_once(), preserving dialect-aware locking,
two-gate fencing tokens, lease renewal heartbeats, and atomic status transitions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from vercel.queue import subscribe

from nexora.graph.worker import GraphExtractionWorker

logger = logging.getLogger(__name__)

# Global worker reference for test injection and singleton caching
_global_worker: GraphExtractionWorker | None = None


def set_global_worker(worker: GraphExtractionWorker | None) -> None:
    """Inject a specific worker instance (useful for unit tests and local composition)."""
    global _global_worker
    _global_worker = worker


def get_queue_worker() -> GraphExtractionWorker:
    """Obtain or initialize the GraphExtractionWorker for queue processing."""
    global _global_worker
    if _global_worker is not None:
        return _global_worker

    # 1. Try to obtain worker from initialized production ASGI app
    try:
        from nexora.server import app

        worker = getattr(app.state, "graph_worker", None)
        if worker is not None and isinstance(worker, GraphExtractionWorker):
            _global_worker = worker
            return _global_worker
    except Exception as exc:
        logger.debug("Could not resolve worker from nexora.server.app: %s", exc)

    # 2. Lazy instantiate through production factory
    try:
        from nexora.server import create_production_app

        prod_app = create_production_app()
        worker = getattr(prod_app.state, "graph_worker", None)
        if worker is not None and isinstance(worker, GraphExtractionWorker):
            _global_worker = worker
            return _global_worker
    except Exception as exc:
        logger.error("Failed to initialize GraphExtractionWorker from production factory: %s", exc)

    raise RuntimeError(
        "No GraphExtractionWorker available for queue processing. "
        "Ensure database connection and services are properly configured."
    )


def process_queue_trigger(payload: Any = None) -> dict[str, Any]:
    """Process a queue trigger by invoking the worker's durable run_once cycle."""
    worker = get_queue_worker()
    logger.info("Queue trigger received, invoking worker.run_once()")
    processed_count = worker.run_once()
    logger.info("Worker run_once completed with %d jobs processed", processed_count)
    return {"processed_jobs": processed_count}


@subscribe(topic="graph-extraction-jobs")
def handle_graph_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Process an enqueued graph extraction job trigger in a serverless function invocation.

    Decorated with ``@subscribe(topic="graph-extraction-jobs")`` from the official
    ``vercel-queue`` SDK.
    """
    return process_queue_trigger(payload)


__all__ = [
    "get_queue_worker",
    "handle_graph_job",
    "process_queue_trigger",
    "set_global_worker",
]
