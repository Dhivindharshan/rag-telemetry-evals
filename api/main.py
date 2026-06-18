"""FastAPI REST API for the RAG Telemetry Evals pipeline.

Run with:
    uvicorn api.main:app --reload --reload-exclude .venv --port 8001

Interactive docs:
    http://127.0.0.1:8001/docs   (Swagger UI)
    http://127.0.0.1:8001/redoc  (ReDoc)

Metrics (Prometheus scrape endpoint):
    http://127.0.0.1:8001/metrics
"""

import os
import sys
import time
from pathlib import Path
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from google.genai import errors as genai_errors
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

# ── Environment ────────────────────────────────────────────────────────────────
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

_GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
if not _GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY is not set — POST /query will return 503.")

_PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, _PROJECT_ROOT)

# ── Metrics (import before app creation so the registry is populated) ──────────
from telemetry.metrics import (  # noqa: E402
    CHUNKS_RETRIEVED,
    ERRORS_TOTAL,
    GENERATION_DURATION_SECONDS,
    PIPELINE_DURATION_SECONDS,
    RERANKING_DURATION_SECONDS,
    REQUEST_DURATION_SECONDS,
    REQUESTS_IN_FLIGHT,
    REQUESTS_TOTAL,
    RETRIEVAL_DURATION_SECONDS,
)

# ── MLflow logger (optional — pipeline continues if unavailable) ───────────────
try:
    from telemetry.mlflow_logger import log_rag_run as _log_rag_run
except Exception:
    _log_rag_run = None  # type: ignore[assignment]

# ── Reranker — process-lifetime singleton to avoid per-request model loading ───
# CrossEncoder takes ~2 s to load; initialising once at first use is correct.
_cross_encoder = None  # lazy-loaded on first request that uses reranking

def _get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers.cross_encoder import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Telemetry Evals API",
    description=(
        "Production-ready RAG pipeline exposing retrieval, reranking, generation, "
        "telemetry, and Prometheus metrics over a REST interface. "
        "Documents are pre-indexed in ChromaDB; submit a query to retrieve "
        "relevant chunks and generate a grounded answer with Gemini."
    ),
    version="1.0.0",
    contact={"name": "RAG Telemetry Evals"},
    license_info={"name": "MIT"},
)


# ── Prometheus middleware ──────────────────────────────────────────────────────
# Wraps every request to record in-flight count, duration, and status.
# HTTPException errors are caught as non-2xx responses (not unhandled exceptions).
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next) -> Response:
    REQUESTS_IN_FLIGHT.inc()
    t_start = time.perf_counter()
    endpoint = request.url.path

    try:
        response = await call_next(request)
        duration = time.perf_counter() - t_start
        http_status = str(response.status_code)

        REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            http_status=http_status,
        ).inc()
        REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(duration)

        if response.status_code >= 400:
            ERRORS_TOTAL.labels(
                endpoint=endpoint,
                error_type=f"http_{http_status}",
            ).inc()

        return response

    except Exception as exc:
        duration = time.perf_counter() - t_start
        ERRORS_TOTAL.labels(
            endpoint=endpoint,
            error_type=type(exc).__name__,
        ).inc()
        REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            http_status="500",
        ).inc()
        REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(duration)
        raise

    finally:
        REQUESTS_IN_FLIGHT.dec()


# ── Pydantic models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The natural-language question to answer.",
        examples=["What is MLOps?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of chunks to return after retrieval (and reranking).",
    )
    use_reranker: bool = Field(
        default=True,
        description=(
            "Apply cross-encoder reranking after retrieval. "
            "Fetches top_k × 2 candidates, reranks all of them, returns top_k. "
            "Adds ~100–400 ms latency; typically improves MRR and Precision@K."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What is MLOps?", "top_k": 3, "use_reranker": True},
                {"query": "How does a reranker improve retrieval quality?", "top_k": 5, "use_reranker": False},
            ]
        }
    }


class RetrievedChunkResponse(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    source: str = Field(..., description="Source document path.")
    text: str = Field(..., description="Chunk text content.")
    distance: float = Field(..., description="L2 distance from query embedding.")
    rerank_score: float | None = Field(None, description="Cross-encoder score (present when use_reranker=true).")


class TelemetryResponse(BaseModel):
    total_seconds: float = Field(..., description="Total wall-clock time.")
    retrieval_seconds: float = Field(..., description="ChromaDB vector search time.")
    reranking_seconds: float = Field(..., description="Cross-encoder reranking time (0 when disabled).")
    generation_seconds: float = Field(..., description="Gemini generation time.")


class QueryResponse(BaseModel):
    query: str
    answer: str
    model: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunkResponse]
    telemetry: TelemetryResponse


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Ops"],
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=app.version)


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus text format metrics for scraping. Not shown in Swagger.",
    tags=["Ops"],
    include_in_schema=False,
)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Run RAG pipeline",
    description=(
        "Embed the query, retrieve the top-k most relevant chunks from ChromaDB, "
        "optionally rerank with a cross-encoder, generate a grounded answer with Gemini, "
        "and return the answer, sources, retrieved chunks, and telemetry timings."
    ),
    tags=["RAG"],
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "ChromaDB empty or API key missing."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Gemini API key rejected."},
        status.HTTP_502_BAD_GATEWAY: {"description": "Gemini server error."},
    },
)
def query_pipeline(request: QueryRequest) -> QueryResponse:
    t_total_start = time.perf_counter()

    # ── Retrieval ──────────────────────────────────────────────────────────────
    # Fetch more candidates when reranking is enabled so the reranker has a
    # wider pool to select from. Standard practice: pool = top_k × 2.
    pool_k = request.top_k * 2 if request.use_reranker else request.top_k
    t_retrieval_start = time.perf_counter()

    try:
        from retriever import retrieve
        chunks = retrieve(request.query, top_k=pool_k)
    except ValueError as exc:
        ERRORS_TOTAL.labels(endpoint="/query", error_type="ValueError").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    t_retrieval_end = time.perf_counter()
    retrieval_seconds = t_retrieval_end - t_retrieval_start

    RETRIEVAL_DURATION_SECONDS.labels(top_k=str(pool_k)).observe(retrieval_seconds)
    CHUNKS_RETRIEVED.observe(len(chunks))

    # ── Reranking (optional) ───────────────────────────────────────────────────
    reranking_seconds = 0.0
    rerank_scores: dict[str, float] = {}

    if request.use_reranker and chunks:
        t_reranking_start = time.perf_counter()
        try:
            encoder = _get_reranker()
            pairs = [(request.query, c["text"]) for c in chunks]
            scores: list[float] = encoder.predict(pairs).tolist()

            # Sort by descending reranker score, keep top_k
            scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            chunks = [c for c, _ in scored[: request.top_k]]
            rerank_scores = {c["chunk_id"]: float(s) for c, s in scored[: request.top_k]}

        except Exception as exc:
            # Reranking failure is non-fatal: fall back to retriever ordering.
            ERRORS_TOTAL.labels(endpoint="/query", error_type="RerankerError").inc()
            chunks = chunks[: request.top_k]
        finally:
            t_reranking_end = time.perf_counter()
            reranking_seconds = t_reranking_end - t_reranking_start
    else:
        chunks = chunks[: request.top_k]

    RERANKING_DURATION_SECONDS.labels(top_k=str(request.top_k)).observe(reranking_seconds)

    # ── Generation ─────────────────────────────────────────────────────────────
    t_generation_start = time.perf_counter()

    try:
        from generator import generate
        response = generate(request.query, chunks, api_key=_GEMINI_API_KEY or None)
    except ValueError as exc:
        ERRORS_TOTAL.labels(endpoint="/query", error_type="ValueError").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except genai_errors.ClientError as exc:
        ERRORS_TOTAL.labels(endpoint="/query", error_type="GeminiClientError").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Gemini API rejected the request. Check GEMINI_API_KEY. Detail: {exc}",
        ) from exc
    except genai_errors.ServerError as exc:
        ERRORS_TOTAL.labels(endpoint="/query", error_type="GeminiServerError").inc()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini server error: {exc}",
        ) from exc
    except Exception as exc:
        ERRORS_TOTAL.labels(endpoint="/query", error_type=type(exc).__name__).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed ({type(exc).__name__}): {exc}",
        ) from exc

    t_generation_end = time.perf_counter()
    generation_seconds = t_generation_end - t_generation_start
    total_seconds = t_generation_end - t_total_start

    GENERATION_DURATION_SECONDS.labels(model=response.get("model", "unknown")).observe(generation_seconds)
    PIPELINE_DURATION_SECONDS.observe(retrieval_seconds + reranking_seconds + generation_seconds)

    # ── Assemble response ──────────────────────────────────────────────────────
    telemetry = TelemetryResponse(
        total_seconds=round(total_seconds, 4),
        retrieval_seconds=round(retrieval_seconds, 4),
        reranking_seconds=round(reranking_seconds, 4),
        generation_seconds=round(generation_seconds, 4),
    )

    retrieved_chunks = [
        RetrievedChunkResponse(
            chunk_id=c["chunk_id"],
            source=c["source"],
            text=c["text"],
            distance=round(c["distance"], 6),
            rerank_score=round(rerank_scores[c["chunk_id"]], 4) if c["chunk_id"] in rerank_scores else None,
        )
        for c in chunks
    ]

    query_response = QueryResponse(
        query=response["query"],
        answer=response["answer"],
        model=response["model"],
        sources=response["sources"],
        retrieved_chunks=retrieved_chunks,
        telemetry=telemetry,
    )

    if _log_rag_run is not None:
        _log_rag_run(
            query=request.query,
            top_k=request.top_k,
            model=response["model"],
            retrieval_seconds=telemetry.retrieval_seconds,
            generation_seconds=telemetry.generation_seconds,
            total_seconds=telemetry.total_seconds,
            answer=response["answer"],
            sources=response["sources"],
            chunks=list(chunks),
        )

    return query_response


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)


if __name__ == "__main__":
    main()
