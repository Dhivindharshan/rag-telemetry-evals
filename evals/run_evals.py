"""CLI runner for RAG pipeline evaluations.

Orchestrates retrieval and generation evaluations, prints a consolidated
summary report, saves results to a JSON file, and logs metrics to MLflow
under the "RAG Evaluation" experiment.

Usage:
    python run_evals.py --mode retrieval
    python run_evals.py --mode generation
    python run_evals.py --mode all --top-k 5 --output results/run1.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import build_context, generate  # noqa: E402
from retriever import retrieve  # noqa: E402

from retrieval_eval import (  # noqa: E402
    AggregateEvalResult,
    RetrievalEvalResult,
    evaluate_dataset as eval_retrieval,
    _SAMPLE_DATASET,
)
from generation_eval import (  # noqa: E402
    AggregateGenerationResult,
    GenerationEvalResult,
    GenerationSample,
    evaluate_dataset as eval_generation,
)

_GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
_RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"
_DIVIDER = "=" * 56


class _GoldenSampleRequired(TypedDict):
    query: str
    relevant_chunk_ids: list[str]


class GoldenSample(_GoldenSampleRequired, total=False):
    """A labelled evaluation sample loaded from the golden dataset."""

    reference: str
    expected_keywords: list[str]


class EvalRunResult(TypedDict):
    """Container for the full results of one evaluation run."""

    mode: str
    top_k: int
    retrieval_aggregate: AggregateEvalResult | None
    retrieval_per_query: list[RetrievalEvalResult]
    generation_aggregate: AggregateGenerationResult | None
    generation_per_query: list[GenerationEvalResult]


def load_golden_samples(path: Path = _GOLDEN_PATH) -> list[GoldenSample]:
    """
    Load evaluation samples from *path*, falling back to built-in samples.

    Args:
        path: Path to the golden dataset JSON file.

    Returns:
        A list of GoldenSample dicts.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw:
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
            print(f"Loaded {len(samples)} sample(s) from {path.name}")
            return samples
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    fallback = [GoldenSample(**s) for s in _SAMPLE_DATASET]
    print(f"Using {len(fallback)} built-in sample(s).")
    return fallback


def run_retrieval_eval(
    samples: list[GoldenSample],
    top_k: int,
) -> tuple[list[RetrievalEvalResult], AggregateEvalResult]:
    """
    Run retrieval evaluation over *samples* and print per-query results.

    Args:
        samples: Labelled evaluation samples with ground-truth chunk IDs.
        top_k: Number of chunks to retrieve per query.

    Returns:
        A (per_query_results, aggregate) tuple.
    """
    print(f"\n{_DIVIDER}")
    print("  RETRIEVAL EVALUATION")
    print(f"{_DIVIDER}")
    print(f"  top_k={top_k}  |  samples={len(samples)}\n")

    results, aggregate = eval_retrieval(samples, top_k=top_k)  # type: ignore[arg-type]

    for r in results:
        tag = "HIT " if r["hit_at_k"] else "MISS"
        print(
            f"  [{tag}]  {r['query'][:44]!r:<48}"
            f"  P@K={r['precision_at_k']:.2f}"
            f"  R@K={r['recall_at_k']:.2f}"
            f"  MRR={r['mrr']:.2f}"
        )

    return results, aggregate


def run_generation_eval(
    samples: list[GoldenSample],
    top_k: int,
) -> tuple[list[GenerationEvalResult], AggregateGenerationResult]:
    """
    Retrieve chunks, generate answers, and evaluate them with an LLM judge.

    For each sample the function:
    1. Retrieves *top_k* chunks from ChromaDB.
    2. Generates an answer with the Gemini API.
    3. Scores the answer for faithfulness, relevance, correctness, and
       keyword coverage.

    Args:
        samples: Labelled evaluation samples.
        top_k: Number of chunks to retrieve per query.

    Returns:
        A (per_query_results, aggregate) tuple.
    """
    print(f"\n{_DIVIDER}")
    print("  GENERATION EVALUATION")
    print(f"{_DIVIDER}")
    print(f"  top_k={top_k}  |  samples={len(samples)}\n")

    gen_samples: list[GenerationSample] = []

    for sample in samples:
        query = sample["query"]
        print(f"  Generating : {query[:60]!r}")

        try:
            chunks = retrieve(query, top_k=top_k)
        except (ValueError, RuntimeError) as exc:
            print(f"    [SKIP] Retrieval failed — {exc}")
            continue

        if not chunks:
            print("    [SKIP] No chunks returned.")
            continue

        try:
            response = generate(query, chunks)
        except Exception as exc:
            print(f"    [SKIP] Generation failed — {exc}")
            continue

        gen_sample = GenerationSample(
            query=query,
            context=build_context(chunks),
            answer=response["answer"],
            reference=sample.get("reference"),
        )
        keywords = sample.get("expected_keywords")
        if keywords:
            gen_sample["expected_keywords"] = keywords

        gen_samples.append(gen_sample)

    if not gen_samples:
        print("\n  [WARNING] No samples available for generation evaluation.")
        return [], AggregateGenerationResult(
            num_samples=0,
            mean_faithfulness=0.0,
            mean_answer_relevance=0.0,
            mean_answer_correctness=None,
            mean_keyword_coverage=None,
        )

    print(f"\n  Scoring {len(gen_samples)} answer(s) with LLM judge...\n")
    results, aggregate = eval_generation(gen_samples)

    for r in results:
        corr = (
            f"{r['answer_correctness']:.2f}"
            if r["answer_correctness"] is not None
            else " n/a"
        )
        kc = r.get("keyword_coverage")
        kc_str = f"{kc:.2f}" if kc is not None else " n/a"
        print(
            f"  {r['query'][:40]!r:<44}"
            f"  Faith={r['faithfulness']:.2f}"
            f"  Rel={r['answer_relevance']:.2f}"
            f"  Corr={corr}"
            f"  KwCov={kc_str}"
        )

    return results, aggregate


def print_summary(run: EvalRunResult) -> None:
    """Print a consolidated summary report for a completed evaluation run."""
    print(f"\n{_DIVIDER}")
    print("  SUMMARY REPORT")
    print(f"{_DIVIDER}")
    print(f"  Mode   : {run['mode']}")
    print(f"  Top-K  : {run['top_k']}")

    ra = run["retrieval_aggregate"]
    if ra is not None:
        print("\n  Retrieval")
        print(f"    Samples      : {ra['num_samples']}")
        print(f"    Precision@K  : {ra['mean_precision_at_k']:.4f}")
        print(f"    Recall@K     : {ra['mean_recall_at_k']:.4f}")
        print(f"    MRR          : {ra['mean_mrr']:.4f}")
        print(f"    Hit Rate     : {ra['hit_rate']:.4f}")

    ga = run["generation_aggregate"]
    if ga is not None and ga["num_samples"] > 0:
        corr = ga["mean_answer_correctness"]
        kc = ga.get("mean_keyword_coverage")
        print("\n  Generation")
        print(f"    Samples           : {ga['num_samples']}")
        print(f"    Faithfulness      : {ga['mean_faithfulness']:.4f}")
        print(f"    Answer Relevance  : {ga['mean_answer_relevance']:.4f}")
        print(f"    Answer Correctness: {f'{corr:.4f}' if corr is not None else 'n/a'}")
        print(f"    Keyword Coverage  : {f'{kc:.4f}' if kc is not None else 'n/a'}")

    print(f"{_DIVIDER}\n")


def save_results(run: EvalRunResult, output_path: Path) -> None:
    """Serialise *run* to a JSON file at *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"  Results saved → {output_path.resolve()}")


def _log_to_mlflow(run: EvalRunResult, results_path: Path) -> None:
    """Log evaluation results to MLflow under the 'RAG Evaluation' experiment.

    All MLflow calls are wrapped in a broad except — if MLflow is unavailable
    or the tracking server is unreachable the evaluation output is unaffected.
    """
    try:
        import mlflow

        mlflow.set_experiment("RAG Evaluation")

        with mlflow.start_run():
            # ── Params ──────────────────────────────────────────────────────
            mlflow.log_param("mode", run["mode"])
            mlflow.log_param("top_k", run["top_k"])

            # ── Retrieval metrics ────────────────────────────────────────────
            ra = run["retrieval_aggregate"]
            if ra is not None:
                mlflow.log_param("retrieval_num_samples", ra["num_samples"])
                mlflow.log_metric("retrieval_precision_at_k", ra["mean_precision_at_k"])
                mlflow.log_metric("retrieval_recall_at_k", ra["mean_recall_at_k"])
                mlflow.log_metric("retrieval_mrr", ra["mean_mrr"])
                mlflow.log_metric("retrieval_hit_rate", ra["hit_rate"])

            # ── Generation metrics ───────────────────────────────────────────
            ga = run["generation_aggregate"]
            if ga is not None and ga["num_samples"] > 0:
                mlflow.log_param("generation_num_samples", ga["num_samples"])
                mlflow.log_metric("generation_faithfulness", ga["mean_faithfulness"])
                mlflow.log_metric("generation_answer_relevance", ga["mean_answer_relevance"])
                if ga["mean_answer_correctness"] is not None:
                    mlflow.log_metric(
                        "generation_answer_correctness", ga["mean_answer_correctness"]
                    )
                kc = ga.get("mean_keyword_coverage")
                if kc is not None:
                    mlflow.log_metric("generation_keyword_coverage", kc)

            # ── Artifacts ────────────────────────────────────────────────────
            mlflow.log_artifact(str(results_path))

        print(f"  MLflow run logged → experiment: 'RAG Evaluation'")
        print(f"  View at: http://127.0.0.1:5000")

    except Exception as exc:
        print(f"  [WARNING] MLflow logging failed (continuing): {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG pipeline evaluations and print a summary report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "retrieval-compare", "generation", "all"],
        default="all",
        help=(
            "Evaluation mode. "
            "'retrieval' runs basic retrieval metrics; "
            "'retrieval-compare' compares retriever-only vs retriever+reranker; "
            "'generation' runs LLM-judge scoring; "
            "'all' runs retrieval + generation."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        dest="top_k",
        help="Number of chunks to retrieve per query.",
    )
    parser.add_argument(
        "--pool-factor",
        type=int,
        default=2,
        dest="pool_factor",
        help="Reranker pool size = top_k * pool_factor (used by retrieval-compare).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to write results JSON (e.g. results/run1.json).",
    )
    return parser.parse_args()


def _run_retrieval_comparison(top_k: int, pool_factor: int) -> None:
    """Delegate to compare_retrieval for the retrieval-compare mode."""
    from compare_retrieval import (  # noqa: E402
        compare,
        load_golden_samples as _load_for_compare,
        log_to_mlflow as _mlflow_compare,
        print_comparison_table,
        save_artifacts,
    )

    samples = _load_for_compare()
    print(f"Loaded {len(samples)} sample(s) from golden_dataset.json\n")
    print("Running retrieval comparison...\n")

    result = compare(samples, top_k=top_k, pool_factor=pool_factor)
    print_comparison_table(result)

    ts = result["timestamp"]
    out_dir = _RESULTS_DIR / f"retrieval_compare_{ts}"
    artifact_paths = save_artifacts(result, out_dir)
    _mlflow_compare(result, artifact_paths)


def main() -> None:
    """
    Entry point: load samples, run selected evaluations, print summary,
    save results, and log to MLflow.
    """
    args = _parse_args()

    # retrieval-compare is a self-contained flow handled separately
    if args.mode == "retrieval-compare":
        try:
            _run_retrieval_comparison(args.top_k, args.pool_factor)
        except Exception as exc:
            print(f"\n[ERROR] Retrieval comparison failed — {exc}")
        return

    samples = load_golden_samples()

    run = EvalRunResult(
        mode=args.mode,
        top_k=args.top_k,
        retrieval_aggregate=None,
        retrieval_per_query=[],
        generation_aggregate=None,
        generation_per_query=[],
    )

    if args.mode in ("retrieval", "all"):
        try:
            per_query, aggregate = run_retrieval_eval(samples, args.top_k)
            run["retrieval_per_query"] = per_query
            run["retrieval_aggregate"] = aggregate
        except Exception as exc:
            print(f"\n[ERROR] Retrieval evaluation failed — {exc}")

    if args.mode in ("generation", "all"):
        try:
            per_query, aggregate = run_generation_eval(samples, args.top_k)
            run["generation_per_query"] = per_query
            run["generation_aggregate"] = aggregate
        except Exception as exc:
            print(f"\n[ERROR] Generation evaluation failed — {exc}")

    print_summary(run)

    # Always persist results — used as the MLflow artifact and for local review.
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _RESULTS_DIR / f"eval_{ts}.json"

    save_results(run, out_path)
    _log_to_mlflow(run, out_path)


if __name__ == "__main__":
    main()
