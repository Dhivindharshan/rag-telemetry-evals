"""Retrieval pipeline comparison: Retriever Only vs Retriever + Reranker.

For each query in the golden dataset both pipelines are evaluated at the
same K.  The reranker pipeline retrieves a larger candidate pool
(top_k * pool_factor) so the cross-encoder has more to work with, then
narrows to top_k ranked results before metric computation.

Metrics reported:
  - Precision@K
  - Recall@K
  - MRR  (Mean Reciprocal Rank)
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - Hit Rate

Artifacts produced under <out_dir>/:
  - retrieval_metrics.json      Full per-query + aggregate data
  - retrieval_comparison.csv    Side-by-side aggregate table
  - retrieval_report.md         Human-readable Markdown report

MLflow experiment: "Retrieval Evaluation"
Run name format : retrieval_eval_<YYYYMMDD_HHMMSS>

Usage (standalone):
    python compare_retrieval.py --top-k 5
    python compare_retrieval.py --top-k 5 --pool-factor 3

Usage via run_evals.py:
    python run_evals.py --mode retrieval-compare --top-k 5
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import retrieve  # noqa: E402
from reranker import rerank    # noqa: E402
from retrieval_eval import (   # noqa: E402
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    hit_at_k,
)

_GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
_RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"

_TABLE_WIDTH = 52
_COL_METRIC  = 20
_COL_VAL     = 14


# ── TypedDicts ────────────────────────────────────────────────────────────────

class _GoldenRequired(TypedDict):
    query: str
    relevant_chunk_ids: list[str]


class GoldenSample(_GoldenRequired, total=False):
    """One labelled entry from golden_dataset.json."""

    reference: str
    expected_keywords: list[str]


class PerQueryMetrics(TypedDict):
    """All retrieval metrics for a single (query, pipeline) evaluation."""

    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_at_k: bool


class PipelineMetrics(TypedDict):
    """Macro-averaged metrics for one pipeline across the full dataset."""

    evaluation_mode: str
    top_k: int
    num_samples: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    hit_rate: float


class ComparisonResult(TypedDict):
    """Full output of one comparison run."""

    top_k: int
    pool_factor: int
    timestamp: str
    retriever_only: PipelineMetrics
    retriever_reranker: PipelineMetrics
    retriever_only_per_query: list[PerQueryMetrics]
    retriever_reranker_per_query: list[PerQueryMetrics]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_query_metrics(
    query: str,
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> PerQueryMetrics:
    """Compute all five retrieval metrics for a single query result.

    Args:
        query: The original query string.
        retrieved_ids: Ordered chunk IDs returned by the pipeline.
        relevant_ids: Ground-truth relevant chunk IDs from the golden dataset.
        k: Evaluation cut-off rank.

    Returns:
        A PerQueryMetrics dict with all metric values.
    """
    relevant = set(relevant_ids)
    return PerQueryMetrics(
        query=query,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        precision_at_k=precision_at_k(retrieved_ids, relevant),
        recall_at_k=recall_at_k(retrieved_ids, relevant),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant, k),
        hit_at_k=hit_at_k(retrieved_ids, relevant),
    )


def _aggregate(
    results: list[PerQueryMetrics],
    top_k: int,
    evaluation_mode: str,
) -> PipelineMetrics:
    """Average per-query metrics into a single PipelineMetrics dict.

    Args:
        results: Per-query metric dicts.
        top_k: K used during evaluation.
        evaluation_mode: Label string (e.g. "retriever_only").

    Returns:
        A PipelineMetrics dict with macro-averaged values.
    """
    n = len(results)
    if n == 0:
        return PipelineMetrics(
            evaluation_mode=evaluation_mode,
            top_k=top_k,
            num_samples=0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mean_mrr=0.0,
            mean_ndcg_at_k=0.0,
            hit_rate=0.0,
        )
    return PipelineMetrics(
        evaluation_mode=evaluation_mode,
        top_k=top_k,
        num_samples=n,
        mean_precision_at_k=sum(r["precision_at_k"] for r in results) / n,
        mean_recall_at_k=sum(r["recall_at_k"] for r in results) / n,
        mean_mrr=sum(r["mrr"] for r in results) / n,
        mean_ndcg_at_k=sum(r["ndcg_at_k"] for r in results) / n,
        hit_rate=sum(1 for r in results if r["hit_at_k"]) / n,
    )


# ── Pipeline evaluation ───────────────────────────────────────────────────────

def evaluate_retriever_only(
    samples: list[GoldenSample],
    top_k: int,
) -> tuple[list[PerQueryMetrics], PipelineMetrics]:
    """Evaluate the bi-encoder retriever alone.

    Retrieves exactly *top_k* chunks per query using ChromaDB L2 distance
    ordering and computes all metrics against the ground-truth chunk IDs.

    Args:
        samples: Golden dataset samples with ground-truth chunk IDs.
        top_k: Number of chunks to retrieve and evaluate at.

    Returns:
        A (per_query_results, aggregate_metrics) tuple.
    """
    print(
        f"  [Retriever Only]          "
        f"evaluating {len(samples)} queries  top_k={top_k}"
    )
    per_query: list[PerQueryMetrics] = []

    for sample in samples:
        query = sample["query"]
        relevant_ids = sample.get("relevant_chunk_ids", [])

        try:
            chunks = retrieve(query, top_k=top_k)
            retrieved_ids = [c["chunk_id"] for c in chunks]
        except Exception as exc:
            print(f"    [SKIP] {query[:50]!r} — {exc}")
            retrieved_ids = []

        per_query.append(
            _compute_query_metrics(query, retrieved_ids, relevant_ids, top_k)
        )

    return per_query, _aggregate(per_query, top_k, "retriever_only")


def evaluate_retriever_reranker(
    samples: list[GoldenSample],
    top_k: int,
    pool_factor: int = 2,
) -> tuple[list[PerQueryMetrics], PipelineMetrics]:
    """Evaluate the bi-encoder retriever followed by a cross-encoder reranker.

    Retrieves *top_k × pool_factor* candidates with the bi-encoder, scores
    each (query, passage) pair with the cross-encoder, then retains only the
    top *top_k* results by rerank score before computing metrics.  Using a
    larger initial pool means the reranker can promote highly relevant chunks
    that the bi-encoder ranked outside the final top_k.

    Args:
        samples: Golden dataset samples with ground-truth chunk IDs.
        top_k: Final number of chunks to evaluate at.
        pool_factor: Multiplier applied to *top_k* for the initial retrieval
            pool.  Defaults to 2 (retrieve 2 × top_k candidates).

    Returns:
        A (per_query_results, aggregate_metrics) tuple.
    """
    pool_k = top_k * pool_factor
    print(
        f"  [Retriever + Reranker]    "
        f"evaluating {len(samples)} queries  pool_k={pool_k} → top_k={top_k}"
    )
    per_query: list[PerQueryMetrics] = []

    for sample in samples:
        query = sample["query"]
        relevant_ids = sample.get("relevant_chunk_ids", [])

        try:
            chunks = retrieve(query, top_k=pool_k)
        except Exception as exc:
            print(f"    [SKIP] {query[:50]!r} — retrieval failed: {exc}")
            per_query.append(
                _compute_query_metrics(query, [], relevant_ids, top_k)
            )
            continue

        try:
            ranked = rerank(query, chunks, top_n=top_k)
            retrieved_ids = [c["chunk_id"] for c in ranked]
        except Exception as exc:
            print(
                f"    [WARN] {query[:50]!r} — reranking failed, "
                f"falling back to retriever order: {exc}"
            )
            retrieved_ids = [c["chunk_id"] for c in chunks[:top_k]]

        per_query.append(
            _compute_query_metrics(query, retrieved_ids, relevant_ids, top_k)
        )

    return per_query, _aggregate(per_query, top_k, "retriever_reranker")


# ── Comparison orchestration ──────────────────────────────────────────────────

def compare(
    samples: list[GoldenSample],
    top_k: int = 5,
    pool_factor: int = 2,
) -> ComparisonResult:
    """Run both pipelines on *samples* and return a ComparisonResult.

    Args:
        samples: Golden dataset samples.
        top_k: Evaluation cut-off rank (K).
        pool_factor: Initial retrieval pool multiplier for the reranker
            pipeline.

    Returns:
        A ComparisonResult containing aggregate and per-query metrics for
        both pipelines.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    retriever_per_query, retriever_agg = evaluate_retriever_only(samples, top_k)
    reranker_per_query, reranker_agg = evaluate_retriever_reranker(
        samples, top_k, pool_factor=pool_factor
    )

    return ComparisonResult(
        top_k=top_k,
        pool_factor=pool_factor,
        timestamp=ts,
        retriever_only=retriever_agg,
        retriever_reranker=reranker_agg,
        retriever_only_per_query=retriever_per_query,
        retriever_reranker_per_query=reranker_per_query,
    )


# ── Console output ────────────────────────────────────────────────────────────

def print_comparison_table(result: ComparisonResult) -> None:
    """Print a formatted side-by-side metric comparison table to stdout.

    Args:
        result: A ComparisonResult produced by :func:`compare`.
    """
    k   = result["top_k"]
    ro  = result["retriever_only"]
    rr  = result["retriever_reranker"]

    divider = "-" * _TABLE_WIDTH
    header  = (
        f"{'Metric':<{_COL_METRIC}}"
        f"{'Retriever':>{_COL_VAL}}"
        f"{'Reranker':>{_COL_VAL}}"
    )

    def row(label: str, rv: float, rrv: float) -> str:
        return (
            f"{label:<{_COL_METRIC}}"
            f"{rv:>{_COL_VAL}.4f}"
            f"{rrv:>{_COL_VAL}.4f}"
        )

    print(f"\n{divider}")
    print(header)
    print(divider)
    print(row(f"Precision@{k}",  ro["mean_precision_at_k"], rr["mean_precision_at_k"]))
    print(row(f"Recall@{k}",     ro["mean_recall_at_k"],    rr["mean_recall_at_k"]))
    print(row("MRR",             ro["mean_mrr"],             rr["mean_mrr"]))
    print(row(f"NDCG@{k}",       ro["mean_ndcg_at_k"],      rr["mean_ndcg_at_k"]))
    print(row("Hit Rate",        ro["hit_rate"],             rr["hit_rate"]))
    print(divider)
    print(f"  Samples: {ro['num_samples']}  |  K: {k}  |  pool_factor: {result['pool_factor']}\n")


# ── Artifact generation ───────────────────────────────────────────────────────

def _build_report_md(result: ComparisonResult) -> str:
    """Render a Markdown evaluation report from *result*."""
    k   = result["top_k"]
    ro  = result["retriever_only"]
    rr  = result["retriever_reranker"]

    def delta(a: float, b: float) -> str:
        d = b - a
        return f"{'+' if d >= 0 else ''}{d:.4f}"

    lines: list[str] = [
        "# Retrieval Evaluation Report",
        "",
        f"**Timestamp:** {result['timestamp']}  ",
        f"**Top-K:** {k}  ",
        f"**Pool factor (reranker):** {result['pool_factor']}  ",
        f"**Samples:** {ro['num_samples']}  ",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Retriever | Reranker | Δ (Reranker − Retriever) |",
        "|--------|:---------:|:--------:|:------------------------:|",
        f"| Precision@{k} | {ro['mean_precision_at_k']:.4f} | {rr['mean_precision_at_k']:.4f} | {delta(ro['mean_precision_at_k'], rr['mean_precision_at_k'])} |",
        f"| Recall@{k} | {ro['mean_recall_at_k']:.4f} | {rr['mean_recall_at_k']:.4f} | {delta(ro['mean_recall_at_k'], rr['mean_recall_at_k'])} |",
        f"| MRR | {ro['mean_mrr']:.4f} | {rr['mean_mrr']:.4f} | {delta(ro['mean_mrr'], rr['mean_mrr'])} |",
        f"| NDCG@{k} | {ro['mean_ndcg_at_k']:.4f} | {rr['mean_ndcg_at_k']:.4f} | {delta(ro['mean_ndcg_at_k'], rr['mean_ndcg_at_k'])} |",
        f"| Hit Rate | {ro['hit_rate']:.4f} | {rr['hit_rate']:.4f} | {delta(ro['hit_rate'], rr['hit_rate'])} |",
        "",
        "## Per-Query Results — Retriever Only",
        "",
        f"| Query | P@{k} | R@{k} | MRR | NDCG@{k} | Hit |",
        "|-------|:-----:|:-----:|:---:|:-------:|:---:|",
    ]

    for r in result["retriever_only_per_query"]:
        lines.append(
            f"| {r['query'][:55]} "
            f"| {r['precision_at_k']:.3f} "
            f"| {r['recall_at_k']:.3f} "
            f"| {r['mrr']:.3f} "
            f"| {r['ndcg_at_k']:.3f} "
            f"| {'✓' if r['hit_at_k'] else '✗'} |"
        )

    lines += [
        "",
        "## Per-Query Results — Retriever + Reranker",
        "",
        f"| Query | P@{k} | R@{k} | MRR | NDCG@{k} | Hit |",
        "|-------|:-----:|:-----:|:---:|:-------:|:---:|",
    ]

    for r in result["retriever_reranker_per_query"]:
        lines.append(
            f"| {r['query'][:55]} "
            f"| {r['precision_at_k']:.3f} "
            f"| {r['recall_at_k']:.3f} "
            f"| {r['mrr']:.3f} "
            f"| {r['ndcg_at_k']:.3f} "
            f"| {'✓' if r['hit_at_k'] else '✗'} |"
        )

    lines.append("")
    return "\n".join(lines)


def save_artifacts(
    result: ComparisonResult,
    out_dir: Path,
) -> dict[str, Path]:
    """Save all three evaluation artifacts to *out_dir*.

    Writes:
      - retrieval_metrics.json    — full per-query + aggregate JSON
      - retrieval_comparison.csv  — aggregate side-by-side CSV
      - retrieval_report.md       — human-readable Markdown report

    Args:
        result: A ComparisonResult produced by :func:`compare`.
        out_dir: Directory to write artifacts into (created if absent).

    Returns:
        A dict mapping artifact filename to its absolute Path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    k  = result["top_k"]
    ro = result["retriever_only"]
    rr = result["retriever_reranker"]

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = out_dir / "retrieval_metrics.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = out_dir / "retrieval_comparison.csv"
    csv_rows = [
        ["Metric", "Retriever", "Retriever+Reranker"],
        [f"Precision@{k}", f"{ro['mean_precision_at_k']:.4f}", f"{rr['mean_precision_at_k']:.4f}"],
        [f"Recall@{k}",    f"{ro['mean_recall_at_k']:.4f}",    f"{rr['mean_recall_at_k']:.4f}"],
        ["MRR",            f"{ro['mean_mrr']:.4f}",             f"{rr['mean_mrr']:.4f}"],
        [f"NDCG@{k}",      f"{ro['mean_ndcg_at_k']:.4f}",      f"{rr['mean_ndcg_at_k']:.4f}"],
        ["Hit Rate",       f"{ro['hit_rate']:.4f}",             f"{rr['hit_rate']:.4f}"],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)

    # ── Markdown ─────────────────────────────────────────────────────────────
    md_path = out_dir / "retrieval_report.md"
    md_path.write_text(_build_report_md(result), encoding="utf-8")

    paths: dict[str, Path] = {
        "retrieval_metrics.json":    json_path,
        "retrieval_comparison.csv":  csv_path,
        "retrieval_report.md":       md_path,
    }

    print(f"\n  Artifacts written to: {out_dir.resolve()}")
    for name in paths:
        print(f"    {name}")

    return paths


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(
    result: ComparisonResult,
    artifact_paths: dict[str, Path] | None = None,
) -> None:
    """Log comparison results to MLflow under the 'Retrieval Evaluation' experiment.

    Creates one run per call, named retrieval_eval_<timestamp>.  All MLflow
    calls are wrapped in a broad except so a missing MLflow installation or
    unreachable tracking server never aborts the evaluation.

    Params logged  : top_k, evaluation_mode, num_samples, pool_factor
    Metrics logged : precision_at_k, recall_at_k, mrr, ndcg_at_k, hit_rate
                     for both the retriever_only and reranker pipelines
    Artifacts      : all files in *artifact_paths* that exist on disk

    Args:
        result: A ComparisonResult produced by :func:`compare`.
        artifact_paths: Optional mapping from filename to Path; each
            existing file is uploaded as an MLflow artifact.
    """
    try:
        import mlflow

        mlflow.set_experiment("Retrieval Evaluation")
        run_name = f"retrieval_eval_{result['timestamp']}"

        with mlflow.start_run(run_name=run_name):
            ro = result["retriever_only"]
            rr = result["retriever_reranker"]

            # ── Params ──────────────────────────────────────────────────────
            mlflow.log_param("top_k",            result["top_k"])
            mlflow.log_param("pool_factor",       result["pool_factor"])
            mlflow.log_param("num_samples",       ro["num_samples"])
            mlflow.log_param("evaluation_mode",   "retriever_only_vs_reranker")

            # ── Retriever-only metrics ───────────────────────────────────────
            mlflow.log_metric("retriever_precision_at_k", ro["mean_precision_at_k"])
            mlflow.log_metric("retriever_recall_at_k",    ro["mean_recall_at_k"])
            mlflow.log_metric("retriever_mrr",            ro["mean_mrr"])
            mlflow.log_metric("retriever_ndcg_at_k",      ro["mean_ndcg_at_k"])
            mlflow.log_metric("retriever_hit_rate",       ro["hit_rate"])

            # ── Reranker metrics ─────────────────────────────────────────────
            mlflow.log_metric("reranker_precision_at_k",  rr["mean_precision_at_k"])
            mlflow.log_metric("reranker_recall_at_k",     rr["mean_recall_at_k"])
            mlflow.log_metric("reranker_mrr",             rr["mean_mrr"])
            mlflow.log_metric("reranker_ndcg_at_k",       rr["mean_ndcg_at_k"])
            mlflow.log_metric("reranker_hit_rate",        rr["hit_rate"])

            # ── Artifacts ────────────────────────────────────────────────────
            if artifact_paths:
                for path in artifact_paths.values():
                    if path.exists():
                        mlflow.log_artifact(str(path))

        print(f"  MLflow run '{run_name}' → experiment: 'Retrieval Evaluation'")
        print("  View at: http://127.0.0.1:5000")

    except Exception as exc:
        print(f"  [WARNING] MLflow logging failed (pipeline continues): {exc}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_golden_samples(path: Path = _GOLDEN_PATH) -> list[GoldenSample]:
    """Load golden dataset from *path*.

    Args:
        path: Path to golden_dataset.json.

    Returns:
        List of GoldenSample dicts.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the JSON structure is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in '{path}'.")

    samples: list[GoldenSample] = []
    for item in raw:
        sample = GoldenSample(
            query=item["query"],
            relevant_chunk_ids=item.get("relevant_chunk_ids", []),
        )
        if "reference" in item:
            sample["reference"] = item["reference"]
        if "expected_keywords" in item:
            sample["expected_keywords"] = item["expected_keywords"]
        samples.append(sample)

    return samples


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare retriever-only vs retriever+reranker pipelines "
            "on the golden dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Evaluation cut-off rank K.",
    )
    parser.add_argument(
        "--pool-factor",
        type=int,
        default=2,
        dest="pool_factor",
        help="Reranker initial pool = top_k * pool_factor.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        metavar="DIR",
        help=(
            "Directory for artifacts. "
            "Defaults to data/eval_results/retrieval_compare_<timestamp>."
        ),
    )
    parser.add_argument(
        "--golden",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to golden_dataset.json. Defaults to evals/golden_dataset.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load golden dataset, run comparison, save artifacts, log to MLflow."""
    args = _parse_args()

    golden_path = Path(args.golden) if args.golden else _GOLDEN_PATH

    print(f"Golden dataset : {golden_path.resolve()}")
    try:
        samples = load_golden_samples(golden_path)
        print(f"Loaded {len(samples)} sample(s).\n")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return

    print("Running retrieval comparison...\n")
    result = compare(samples, top_k=args.top_k, pool_factor=args.pool_factor)

    print_comparison_table(result)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _RESULTS_DIR / f"retrieval_compare_{result['timestamp']}"
    )

    artifact_paths = save_artifacts(result, out_dir)
    log_to_mlflow(result, artifact_paths)


if __name__ == "__main__":
    main()
