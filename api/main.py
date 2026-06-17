"""FastAPI REST API for the RAG Telemetry Evals pipeline.

Run with:
    uvicorn api.main:app --reload --reload-exclude .venv --port 8001

Interactive docs available at:
    http://127.0.0.1:8001/docs   (Swagger UI)
    http://127.0.0.1:8001/redoc  (ReDoc)
"""

import os
import sys
import time
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

# ── Environment loading ────────────────────────────────────────────────────────
# override=True ensures .env values win even if the OS already has the variable
# set to a stale or empty value (important in uvicorn --reload subprocess model)

_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

# Cache at module load time — this variable is passed explicitly to generate()
# so request handlers are not sensitive to os.environ state at call time.
_GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

print(f"[DEBUG] .env path      : {_ENV_PATH.resolve()}")
print(f"[DEBUG] .env exists    : {_ENV_PATH.exists()}")
print(f"[DEBUG] API key present: {bool(_GEMINI_API_KEY)}")
print(f"[DEBUG] API key length : {len(_GEMINI_API_KEY)}")

if not _GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY is not set — POST /query will return 503.")

_PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, _PROJECT_ROOT)

try:
    from telemetry.mlflow_logger import log_rag_run as _log_rag_run
except Exception:
    _log_rag_run = None  # type: ignore[assignment]


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Telemetry Evals API",
    description=(
        "Production-ready RAG pipeline exposing retrieval, generation, "
        "and telemetry over a REST interface. "
        "Documents are pre-indexed in ChromaDB; submit a query to retrieve "
        "relevant chunks and generate a grounded answer with Gemini."
    ),
    version="1.0.0",
    contact={"name": "RAG Telemetry Evals"},
    license_info={"name": "MIT"},
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

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
        description="Number of chunks to retrieve from the vector store.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What is MLOps?", "top_k": 3},
                {"query": "How does a reranker improve retrieval quality?", "top_k": 5},
            ]
        }
    }


class RetrievedChunkResponse(BaseModel):
    """A single chunk returned by the retriever."""

    chunk_id: str = Field(..., description="Unique identifier of the chunk.")
    source: str = Field(..., description="Absolute path of the source document.")
    text: str = Field(..., description="Raw text content of the chunk.")
    distance: float = Field(
        ..., description="L2 embedding distance to the query (lower = more relevant)."
    )


class TelemetryResponse(BaseModel):
    """Wall-clock timing data for the pipeline run."""

    total_seconds: float = Field(..., description="Total wall-clock time in seconds.")
    retrieval_seconds: float = Field(..., description="Time spent on vector retrieval in seconds.")
    generation_seconds: float = Field(..., description="Time spent on LLM generation in seconds.")


class QueryResponse(BaseModel):
    """Response body from the /query endpoint."""

    query: str = Field(..., description="The original query string.")
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved context.")
    model: str = Field(..., description="Gemini model identifier used for generation.")
    sources: list[str] = Field(
        ..., description="De-duplicated list of source file paths for the retrieved chunks."
    )
    retrieved_chunks: list[RetrievedChunkResponse] = Field(
        ..., description="Ordered list of retrieved context chunks (closest first)."
    )
    telemetry: TelemetryResponse = Field(
        ..., description="Wall-clock timing breakdown for the pipeline stages."
    )


class HealthResponse(BaseModel):
    """Response body from the /health endpoint."""

    status: str = Field(..., description="Service health status.", examples=["ok"])
    version: str = Field(..., description="API version string.")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns `ok` when the service is running.",
    tags=["Ops"],
)
def health() -> HealthResponse:
    """Confirm the API is reachable and return the current version."""
    return HealthResponse(status="ok", version=app.version)


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Run RAG pipeline",
    description=(
        "Embed the query, retrieve the top-k most relevant chunks from ChromaDB, "
        "generate a grounded answer with Gemini, and return the answer, sources, "
        "retrieved chunks, and telemetry timings."
    ),
    tags=["RAG"],
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error in request body."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "ChromaDB collection empty or key missing."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Gemini API key rejected."},
    },
)
def query_pipeline(request: QueryRequest) -> QueryResponse:
    print(">>> /query endpoint reached <<<")
    """
    Run the end-to-end RAG pipeline for the given query.

    Args:
        request: QueryRequest with the user query and optional top_k.

    Returns:
        QueryResponse with the generated answer, sources, retrieved chunks,
        and telemetry timings.
    """
    t_total_start = time.perf_counter()

    # ── Retrieval ──────────────────────────────────────────────────────────────
    t_retrieval_start = time.perf_counter()

    try:
        from retriever import retrieve
        chunks = retrieve(request.query, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    t_retrieval_end = time.perf_counter()

    # ── Generation ─────────────────────────────────────────────────────────────
    t_generation_start = time.perf_counter()

    try:
        from generator import generate
        # Pass _GEMINI_API_KEY explicitly so generate() never has to call
        # os.environ.get() itself — this is the reliable path in uvicorn
        # --reload subprocesses where os.environ mutations can be lost.
        response = generate(request.query, chunks, api_key=_GEMINI_API_KEY or None)
    except ValueError as exc:
        # Covers: empty chunks, missing key (from _resolve_api_key), or
        # genai.Client(api_key=None) raising ValueError.
        print(f"[ERROR] Generation ValueError: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except genai_errors.ClientError as exc:
        # 4xx from Gemini: invalid key, quota, permission denied
        print(f"[ERROR] Gemini ClientError: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Gemini API rejected the request. Check GEMINI_API_KEY. Detail: {exc}",
        ) from exc
    except genai_errors.ServerError as exc:
        print(f"[ERROR] Gemini ServerError: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini server error: {exc}",
        ) from exc
    except Exception as exc:
        print(f"[ERROR] Unexpected generation error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed ({type(exc).__name__}): {exc}",
        ) from exc

    t_generation_end = time.perf_counter()
    t_total_end = time.perf_counter()

    telemetry = TelemetryResponse(
        total_seconds=round(t_total_end - t_total_start, 4),
        retrieval_seconds=round(t_retrieval_end - t_retrieval_start, 4),
        generation_seconds=round(t_generation_end - t_generation_start, 4),
    )

    retrieved_chunks = [
        RetrievedChunkResponse(
            chunk_id=c["chunk_id"],
            source=c["source"],
            text=c["text"],
            distance=round(c["distance"], 6),
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
    """Start the FastAPI server with uvicorn."""
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )


if __name__ == "__main__":
    main()
