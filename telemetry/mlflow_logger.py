"""MLflow experiment logger for the RAG Telemetry Evals pipeline."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)
_EXPERIMENT_NAME = "RAG Telemetry Evals"
_experiment_initialized = False


def _init_experiment() -> None:
    """Set the MLflow experiment once (idempotent, called lazily on first run)."""
    global _experiment_initialized
    if _experiment_initialized:
        return
    import mlflow
    mlflow.set_experiment(_EXPERIMENT_NAME)
    _experiment_initialized = True


def log_rag_run(
    query: str,
    top_k: int,
    model: str,
    retrieval_seconds: float,
    generation_seconds: float,
    total_seconds: float,
    answer: str,
    sources: list[str],
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Log one RAG pipeline request as a separate MLflow run.

    Each call opens and closes its own run so requests never share state.
    All MLflow calls are wrapped in a broad except so any failure — including
    MLflow not being installed or the tracking server being unreachable — is
    logged at WARNING level and the pipeline continues unaffected.

    Args:
        query: The user query string.
        top_k: Number of retrieved chunks requested.
        model: LLM model identifier used for generation.
        retrieval_seconds: Wall-clock time for the retrieval stage.
        generation_seconds: Wall-clock time for the generation stage.
        total_seconds: Total wall-clock time for the pipeline run.
        answer: The generated answer text.
        sources: De-duplicated list of source file paths.
        chunks: Optional list of retrieved chunk dicts (chunk_id, source, text, distance).
    """
    try:
        import mlflow

        _init_experiment()

        with mlflow.start_run():
            mlflow.log_param("query", query[:500])
            mlflow.log_param("top_k", top_k)
            mlflow.log_param("model", model)

            mlflow.log_metric("retrieval_seconds", retrieval_seconds)
            mlflow.log_metric("generation_seconds", generation_seconds)
            mlflow.log_metric("total_seconds", total_seconds)

            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)

                (tmp_dir / "generated_answer.txt").write_text(answer, encoding="utf-8")
                (tmp_dir / "sources.txt").write_text("\n".join(sources), encoding="utf-8")

                if chunks is not None:
                    (tmp_dir / "retrieved_chunks.json").write_text(
                        json.dumps(chunks, indent=2), encoding="utf-8"
                    )

                mlflow.log_artifacts(str(tmp_dir))

    except Exception as exc:
        _logger.warning("MLflow logging failed (pipeline continues): %s", exc)
