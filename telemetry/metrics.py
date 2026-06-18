"""Prometheus metrics registry for the RAG pipeline.

All metrics are defined as module-level singletons.
Import and observe from api/main.py.

Metric naming follows the Prometheus convention:
  <namespace>_<subsystem>_<name>_<unit>
  namespace = "rag"
"""

from prometheus_client import Counter, Gauge, Histogram

# ── HTTP request metrics ───────────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total HTTP requests received by the RAG API",
    labelnames=["method", "endpoint", "http_status"],
)

REQUESTS_IN_FLIGHT = Gauge(
    "rag_requests_in_flight",
    "Number of HTTP requests currently being processed (in-flight)",
)

REQUEST_DURATION_SECONDS = Histogram(
    "rag_request_duration_seconds",
    "Total wall-clock time for an HTTP request including FastAPI overhead",
    labelnames=["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

# ── Error metrics ──────────────────────────────────────────────────────────────

ERRORS_TOTAL = Counter(
    "rag_errors_total",
    "Total errors raised during request handling, labelled by error class",
    labelnames=["endpoint", "error_type"],
)

# ── Pipeline stage latencies ───────────────────────────────────────────────────
# Each histogram uses buckets tuned to the realistic latency range of that stage:
#   retrieval  : ChromaDB ANN search — typically 5–200 ms
#   reranking  : cross-encoder inference — typically 20–500 ms
#   generation : Gemini API round-trip — typically 1–30 s
#   pipeline   : retrieval + reranking + generation combined

RETRIEVAL_DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    "ChromaDB vector search latency per request",
    labelnames=["top_k"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

RERANKING_DURATION_SECONDS = Histogram(
    "rag_reranking_duration_seconds",
    "Cross-encoder reranking latency per request (0 when reranking is disabled)",
    labelnames=["top_k"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

GENERATION_DURATION_SECONDS = Histogram(
    "rag_generation_duration_seconds",
    "Gemini LLM generation latency per request",
    labelnames=["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

PIPELINE_DURATION_SECONDS = Histogram(
    "rag_pipeline_duration_seconds",
    "Total RAG pipeline latency: retrieval + reranking + generation",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

# ── Retrieval quality signal ───────────────────────────────────────────────────

CHUNKS_RETRIEVED = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks returned by the retriever per request",
    buckets=[1, 2, 3, 5, 8, 10, 15, 20],
)
