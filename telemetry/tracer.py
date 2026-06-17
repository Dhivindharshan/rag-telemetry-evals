"""Telemetry utilities for measuring RAG pipeline stage latencies."""

import time
import logging
from contextlib import contextmanager
from typing import Generator

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

STAGE_INGESTION = "ingestion"
STAGE_CHUNKING = "chunking"
STAGE_EMBEDDING = "embedding"
STAGE_RETRIEVAL = "retrieval"
STAGE_GENERATION = "generation"


@contextmanager
def trace(stage: str) -> Generator[None, None, None]:
    """
    Context manager that measures and logs the wall-clock latency of a pipeline stage.

    Logs at INFO level on exit. On exception, logs at ERROR level with the
    exception message before re-raising.

    Args:
        stage: Human-readable name for the pipeline stage being timed
            (e.g. STAGE_INGESTION, STAGE_RETRIEVAL).

    Yields:
        None — the body of the ``with`` block runs inside the timer.

    Example::

        with trace(STAGE_RETRIEVAL):
            chunks = retrieve(query, top_k=3)
    """
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error("[%s] failed after %.4fs — %s", stage, elapsed, exc)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info("[%s] completed in %.4fs", stage, elapsed)


@contextmanager
def trace_ingestion() -> Generator[None, None, None]:
    """Measure and log latency for the document ingestion stage."""
    with trace(STAGE_INGESTION):
        yield


@contextmanager
def trace_chunking() -> Generator[None, None, None]:
    """Measure and log latency for the text chunking stage."""
    with trace(STAGE_CHUNKING):
        yield


@contextmanager
def trace_embedding() -> Generator[None, None, None]:
    """Measure and log latency for the embedding generation stage."""
    with trace(STAGE_EMBEDDING):
        yield


@contextmanager
def trace_retrieval() -> Generator[None, None, None]:
    """Measure and log latency for the vector store retrieval stage."""
    with trace(STAGE_RETRIEVAL):
        yield


@contextmanager
def trace_generation() -> Generator[None, None, None]:
    """Measure and log latency for the LLM answer generation stage."""
    with trace(STAGE_GENERATION):
        yield
