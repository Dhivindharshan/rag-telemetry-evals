"""Retrieval evaluation module for the RAG pipeline.

Measures Precision@K, Recall@K, MRR, NDCG@K, and Hit@K against a golden
dataset of (query, relevant_chunk_ids) pairs.
"""

import json
import math
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import retrieve  # noqa: E402

_GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"

_SAMPLE_DATASET: list[dict] = [
    {
        "query": "What is MLOps?",
        "relevant_chunk_ids": [],
    },
    {
        "query": "How does retrieval-augmented generation work?",
        "relevant_chunk_ids": [],
    },
    {
        "query": "What are the steps in a RAG pipeline?",
        "relevant_chunk_ids": [],
    },
]


class EvalSample(TypedDict):
    query: str
    relevant_chunk_ids: list[str]


class RetrievalEvalResult(TypedDict):
    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_at_k: bool


class AggregateEvalResult(TypedDict):
    num_samples: int
    top_k: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    hit_rate: float


# ── Metric functions ──────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Compute Precision@K.

    Args:
        retrieved: Ordered list of retrieved chunk IDs (length = K).
        relevant: Set of ground-truth relevant chunk IDs.

    Returns:
        Fraction of retrieved IDs that are relevant. Returns 0.0 if
        *retrieved* is empty.
    """
    if not retrieved:
        return 0.0
    hits = sum(1 for cid in retrieved if cid in relevant)
    return hits / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Compute Recall@K.

    Args:
        retrieved: Ordered list of retrieved chunk IDs (length = K).
        relevant: Set of ground-truth relevant chunk IDs.

    Returns:
        Fraction of relevant IDs that appear in *retrieved*. Returns 1.0
        when *relevant* is empty (nothing to recall).
    """
    if not relevant:
        return 1.0
    hits = sum(1 for cid in retrieved if cid in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a single query.

    Args:
        retrieved: Ordered list of retrieved chunk IDs (length = K).
        relevant: Set of ground-truth relevant chunk IDs.

    Returns:
        Reciprocal of the 1-based rank of the first relevant result, or
        0.0 if no relevant result appears in *retrieved*.
    """
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute NDCG@K (Normalized Discounted Cumulative Gain).

    Uses binary relevance: a chunk scores 1 if it is in *relevant*, 0 otherwise.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant: Set of ground-truth relevant chunk IDs.
        k: Cut-off rank for evaluation.

    Returns:
        NDCG@K in [0.0, 1.0]. Returns 1.0 when *relevant* is empty
        (no ground truth to miss) and 0.0 when *retrieved* is empty.
    """
    if not relevant:
        return 1.0
    if not retrieved:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, cid in enumerate(retrieved[:k], start=1)
        if cid in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0.0 else 0.0


def hit_at_k(retrieved: list[str], relevant: set[str]) -> bool:
    """Compute Hit@K: whether at least one relevant result was retrieved.

    Args:
        retrieved: Ordered list of retrieved chunk IDs (length = K).
        relevant: Set of ground-truth relevant chunk IDs.

    Returns:
        True if any retrieved ID appears in *relevant*.
    """
    return any(cid in relevant for cid in retrieved)


# ── Evaluation functions ──────────────────────────────────────────────────────

def evaluate_single(
    sample: EvalSample,
    top_k: int = 3,
) -> RetrievalEvalResult:
    """Run retrieval for one query and compute all metrics against ground truth.

    When *relevant_chunk_ids* is empty the sample is treated as an
    unlabelled query: metrics that require ground truth return 0.0 / False,
    but the retrieved IDs are still recorded for manual inspection.

    Args:
        sample: An EvalSample with a query and its relevant chunk IDs.
        top_k: Number of chunks to retrieve.

    Returns:
        A RetrievalEvalResult with per-query metrics.
    """
    chunks = retrieve(sample["query"], top_k=top_k)
    retrieved_ids = [c["chunk_id"] for c in chunks]
    relevant = set(sample["relevant_chunk_ids"])

    return RetrievalEvalResult(
        query=sample["query"],
        retrieved_ids=retrieved_ids,
        relevant_ids=list(relevant),
        precision_at_k=precision_at_k(retrieved_ids, relevant),
        recall_at_k=recall_at_k(retrieved_ids, relevant),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant, top_k),
        hit_at_k=hit_at_k(retrieved_ids, relevant),
    )


def evaluate_dataset(
    samples: list[EvalSample],
    top_k: int = 3,
) -> tuple[list[RetrievalEvalResult], AggregateEvalResult]:
    """Evaluate retrieval over a list of labelled samples.

    Args:
        samples: List of EvalSample dicts (query + relevant_chunk_ids).
        top_k: Number of chunks to retrieve per query.

    Returns:
        A (per_query_results, aggregate) tuple. *aggregate* contains
        macro-averaged metrics across all samples.

    Raises:
        ValueError: If *samples* is empty.
    """
    if not samples:
        raise ValueError("samples list is empty — nothing to evaluate.")

    results: list[RetrievalEvalResult] = []
    for sample in samples:
        results.append(evaluate_single(sample, top_k=top_k))

    n = len(results)
    aggregate = AggregateEvalResult(
        num_samples=n,
        top_k=top_k,
        mean_precision_at_k=sum(r["precision_at_k"] for r in results) / n,
        mean_recall_at_k=sum(r["recall_at_k"] for r in results) / n,
        mean_mrr=sum(r["mrr"] for r in results) / n,
        mean_ndcg_at_k=sum(r["ndcg_at_k"] for r in results) / n,
        hit_rate=sum(1 for r in results if r["hit_at_k"]) / n,
    )

    return results, aggregate


def load_golden_dataset(path: str | Path = _GOLDEN_PATH) -> list[EvalSample]:
    """Load a golden dataset from a JSON file.

    Args:
        path: Path to the golden dataset JSON file.

    Returns:
        A list of EvalSample dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or has the wrong structure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden dataset not found: {p}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{p}': {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in '{p}', got {type(raw).__name__}.")

    samples: list[EvalSample] = []
    for i, item in enumerate(raw):
        if "query" not in item or "relevant_chunk_ids" not in item:
            raise ValueError(
                f"Item {i} in '{p}' is missing 'query' or 'relevant_chunk_ids'."
            )
        samples.append(
            EvalSample(
                query=item["query"],
                relevant_chunk_ids=item["relevant_chunk_ids"],
            )
        )
    return samples


def _print_aggregate(aggregate: AggregateEvalResult) -> None:
    print("\n── Aggregate Metrics ──────────────────────────")
    print(f"  Samples          : {aggregate['num_samples']}")
    print(f"  Top-K            : {aggregate['top_k']}")
    print(f"  Precision@K      : {aggregate['mean_precision_at_k']:.4f}")
    print(f"  Recall@K         : {aggregate['mean_recall_at_k']:.4f}")
    print(f"  MRR              : {aggregate['mean_mrr']:.4f}")
    print(f"  NDCG@K           : {aggregate['mean_ndcg_at_k']:.4f}")
    print(f"  Hit Rate         : {aggregate['hit_rate']:.4f}")
    print("────────────────────────────────────────────────\n")


def main() -> None:
    """Evaluate retrieval against the golden dataset (or built-in samples)."""
    top_k = 3

    try:
        samples = load_golden_dataset()
        if not samples:
            raise ValueError("Empty dataset.")
        print(f"Loaded {len(samples)} sample(s) from golden_dataset.json")
    except (FileNotFoundError, ValueError):
        samples = [EvalSample(**s) for s in _SAMPLE_DATASET]
        print(f"Using {len(samples)} built-in sample query/queries.")

    print(f"Evaluating retrieval (top_k={top_k})...\n")

    try:
        results, aggregate = evaluate_dataset(samples, top_k=top_k)
    except (RuntimeError, ValueError) as e:
        print(f"[ERROR] {e}")
        return

    for r in results:
        status = "HIT" if r["hit_at_k"] else "MISS"
        print(
            f"[{status}] {r['query'][:55]!r}  "
            f"P@K={r['precision_at_k']:.2f}  "
            f"R@K={r['recall_at_k']:.2f}  "
            f"MRR={r['mrr']:.2f}  "
            f"NDCG={r['ndcg_at_k']:.2f}"
        )

    _print_aggregate(aggregate)


if __name__ == "__main__":
    main()
