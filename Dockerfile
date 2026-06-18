# ─────────────────────────────────────────────────────────────────────────────
# RAG Telemetry Evals — production Dockerfile
#
# Key decisions:
#   - CPU-only PyTorch installed BEFORE sentence-transformers so pip never
#     pulls the CUDA wheel (~2 GB) from PyPI
#   - .dockerignore excludes .venv/ (was causing the 1.78 GB build context)
#   - uvicorn runs directly for proper signal handling in containers
#   - Non-root user for security
#   - HEALTHCHECK against the /health endpoint
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell PyTorch/CUDA to use no GPU — prevents CUDA detection at runtime
    CUDA_VISIBLE_DEVICES="" \
    # Suppresses the HuggingFace tokenizer parallelism fork warning
    TOKENIZERS_PARALLELISM=false \
    # pip global flags (avoids repeating --no-cache-dir on every RUN)
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1  : OpenMP shared memory — required by sentence-transformers / torch
# Kept to a minimum; compilers are NOT needed because all packages ship
# pre-built wheels for python:3.11-slim / manylinux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches the install step.
# Changing source files later won't invalidate this layer.
COPY requirements.txt .

# Step 1 — Upgrade pip so the backtracking resolver correctly handles
# manylinux_2_28 wheel tags (onnxruntime 1.17+, cffi 2.x, etc.).
# python:3.11-slim ships with pip 23.x which can mis-classify these wheels,
# causing exit code 2 on fresh resolves.
RUN pip install --upgrade pip

# Step 2 — CPU-only PyTorch.
# PyPI's default torch wheel bundles CUDA support and weighs ~2 GB.
# The CPU-only wheel from PyTorch's own index is ~200 MB.
# Installing it here before the rest of requirements.txt means pip sees
# torch as already satisfied and skips the CUDA variant entirely.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 3 — Remaining dependencies (sentence-transformers, chromadb, fastapi,
# mlflow, etc.).  pip reuses the CPU torch installed above.
RUN pip install -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
# .dockerignore excludes .venv/, data/, mlruns/, .env, __pycache__, etc.
COPY . .

# ── Non-root user ─────────────────────────────────────────────────────────────
# Running as root inside a container is a security risk.
# Pre-create writable runtime directories so appuser can write to them.
RUN useradd --create-home --no-log-init --shell /bin/bash appuser \
    && mkdir -p data/chroma_db data/traces data/eval_results mlruns \
    && chown -R appuser:appuser /app

USER appuser

# ── Networking ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Persistent data ───────────────────────────────────────────────────────────
# Declare the ChromaDB directory as a volume so data survives container
# restarts.  Mount with:  docker run -v rag-chroma:/app/data/chroma_db ...
VOLUME ["/app/data/chroma_db"]

# ── Health check ──────────────────────────────────────────────────────────────
# --start-period=60s   : grace period for the server to start
# --interval=30s       : check every 30 s after start-period
# --timeout=10s        : fail if no reply within 10 s
# --retries=3          : mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
# uvicorn receives OS signals (SIGTERM) correctly, unlike `python api/main.py`
# which wraps uvicorn.run() in a Python process that ignores them.
# --workers 1 : ChromaDB's SQLite writer is not multi-process safe; use 1
#               worker here and scale horizontally via multiple containers.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
