"""Structured span recording for RAG pipeline tracing."""

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, TypedDict

_TRACE_DIR = Path(__file__).parent.parent / "data" / "traces"


class Span(TypedDict):
    trace_id: str
    span_id: str
    stage: str
    start_time: float
    end_time: float
    duration_s: float
    status: str
    error: str | None


class Trace(TypedDict):
    trace_id: str
    query: str
    spans: list[Span]
    total_duration_s: float


class SpanRecorder:
    """
    Collects spans for a single pipeline run under one trace ID.

    All context managers on this instance share the same ``trace_id`` so
    spans can be grouped and exported together.

    Args:
        query: The user query that triggered this pipeline run.
        trace_id: Optional explicit trace ID. A UUID4 is generated if omitted.

    Example::

        recorder = SpanRecorder(query="What is MLOps?")

        with recorder.span("ingestion"):
            docs = load_documents(data_dir)

        with recorder.span("retrieval"):
            chunks = retrieve(query)

        trace = recorder.to_trace()
        recorder.save()
    """

    def __init__(self, query: str = "", trace_id: str | None = None) -> None:
        self.query = query
        self.trace_id: str = trace_id or str(uuid.uuid4())
        self._spans: list[Span] = []
        self._trace_start: float = time.perf_counter()

    @contextmanager
    def span(self, stage: str) -> Generator[None, None, None]:
        """
        Context manager that records a single named span.

        Args:
            stage: Name of the pipeline stage being timed
                (e.g. "ingestion", "retrieval").

        Yields:
            None — the body of the ``with`` block runs inside the span.
        """
        span_id = str(uuid.uuid4())
        start = time.perf_counter()
        start_wall = time.time()
        status = "ok"
        error: str | None = None

        try:
            yield
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise
        finally:
            end = time.perf_counter()
            end_wall = time.time()
            self._spans.append(
                Span(
                    trace_id=self.trace_id,
                    span_id=span_id,
                    stage=stage,
                    start_time=start_wall,
                    end_time=end_wall,
                    duration_s=round(end - start, 6),
                    status=status,
                    error=error,
                )
            )

    def to_trace(self) -> Trace:
        """
        Return a Trace TypedDict summarising all recorded spans.

        Returns:
            A Trace dict with the trace_id, query, span list, and total
            wall-clock duration from recorder construction to now.
        """
        total = round(time.perf_counter() - self._trace_start, 6)
        return Trace(
            trace_id=self.trace_id,
            query=self.query,
            spans=list(self._spans),
            total_duration_s=total,
        )

    def save(self, trace_dir: str | Path = _TRACE_DIR) -> Path:
        """
        Persist the trace to a JSON file in *trace_dir*.

        The file is named ``<trace_id>.json``. The directory is created if
        it does not already exist.

        Args:
            trace_dir: Directory where trace JSON files are written.

        Returns:
            The Path of the written file.
        """
        out_dir = Path(trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{self.trace_id}.json"
        trace = self.to_trace()
        out_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        return out_path


def load_trace(path: str | Path) -> Trace:
    """
    Load a previously saved Trace from a JSON file.

    Args:
        path: Path to the trace JSON file.

    Returns:
        A Trace TypedDict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is missing required keys.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in trace file '{p}': {exc}") from exc
    return Trace(**data)
