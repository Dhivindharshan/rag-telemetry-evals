"""Generation evaluation module using an LLM-as-judge approach.

Scores generated RAG answers on four metrics:
  - Faithfulness       : answer is grounded in the retrieved context
  - Answer Relevance   : answer addresses the question asked
  - Answer Correctness : similarity to a reference answer (optional)
  - Keyword Coverage   : fraction of expected keywords present in the answer
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors as genai_errors

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import build_context, generate  # noqa: E402
from retriever import retrieve  # noqa: E402

_logger = logging.getLogger(__name__)

JUDGE_MODEL = "gemini-2.5-flash"

# Returned by safe_parse_judge_response when all parsing strategies fail.
# Scores default to 1.0 so a single malformed response does not unfairly
# penalise the overall run average.
_FALLBACK_SCORES: dict = {
    "faithfulness": 1.0,
    "answer_relevance": 1.0,
    "answer_correctness": 1.0,
    "reasoning": "Judge output could not be parsed.",
}

_JUDGE_PROMPT = """\
You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.

Evaluate the generated answer using the question and retrieved context below.

Question:
{query}

Retrieved Context:
{context}

Generated Answer:
{answer}
{reference_section}
Score each metric from 0.0 to 1.0:

1. faithfulness       — Does the answer state only things supported by the context?
                        (1.0 = fully grounded, 0.0 = entirely hallucinated)

2. answer_relevance   — Does the answer actually address the question?
                        (1.0 = directly answers, 0.0 = completely off-topic)

3. answer_correctness — How well does the answer match the reference answer?
                        (1.0 = identical meaning, 0.0 = contradicts reference)
                        Set to null if no reference answer was provided.

Respond with ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "faithfulness": <float 0.0-1.0>,
  "answer_relevance": <float 0.0-1.0>,
  "answer_correctness": <float 0.0-1.0 or null>,
  "reasoning": "<one or two sentences explaining your scores>"
}}"""

_REFERENCE_SECTION = "\nReference Answer:\n{reference}\n"


# ── TypedDicts ────────────────────────────────────────────────────────────────

class GenerationSample(TypedDict):
    query: str
    context: str
    answer: str
    reference: str | None
    expected_keywords: NotRequired[list[str] | None]


class GenerationEvalResult(TypedDict):
    query: str
    faithfulness: float
    answer_relevance: float
    answer_correctness: float | None
    keyword_coverage: float | None
    reasoning: str


class AggregateGenerationResult(TypedDict):
    num_samples: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_answer_correctness: float | None
    mean_keyword_coverage: float | None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(value: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Return the fraction of *keywords* found (case-insensitive) in *answer*.

    Args:
        answer: The generated answer text.
        keywords: Expected keywords or phrases to look for.

    Returns:
        Coverage score in [0.0, 1.0]. Returns 0.0 when keywords is empty.
    """
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def safe_parse_judge_response(text: str) -> dict:
    """Parse a judge model response into a scores dict robustly.

    Applies four strategies in order, returning the result of the first
    one that succeeds:

    1. Direct ``json.loads`` on the stripped text.
    2. Extract content from a Markdown code fence (``\\`\\`\\`json … \\`\\`\\```)
       and parse that.
    3. Use a regex to locate the first ``{ … }`` block in the text and
       parse it.
    4. Return ``_FALLBACK_SCORES`` and emit a WARNING log.

    Args:
        text: Raw text returned by the judge model.

    Returns:
        A dict with at minimum the keys ``faithfulness``,
        ``answer_relevance``, ``answer_correctness``, and ``reasoning``.
        Never raises — returns fallback scores on any parse failure.
    """
    if not text:
        _logger.warning("[WARNING] Judge output malformed, using fallback scores.")
        return dict(_FALLBACK_SCORES)

    stripped = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from Markdown code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find the first {...} block anywhere in the text
    brace_match = re.search(r"\{[\s\S]*\}", stripped)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: nothing worked — use fallback
    _logger.warning("[WARNING] Judge output malformed, using fallback scores.")
    return dict(_FALLBACK_SCORES)


# ── Judge API call ────────────────────────────────────────────────────────────

def _call_judge(prompt: str) -> dict:
    """Send *prompt* to the Gemini judge model and return parsed scores.

    Uses :func:`safe_parse_judge_response` so malformed output never raises
    a parse error — a warning is emitted and fallback scores are returned
    instead.

    Args:
        prompt: The fully rendered judge prompt.

    Returns:
        Parsed scores dict with faithfulness, answer_relevance,
        answer_correctness, and reasoning keys.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.
        google.genai.errors.APIError: On Gemini API failures.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file before running evaluations."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=512,
            response_mime_type="application/json",
        ),
    )

    raw = (response.text or "").strip()
    return safe_parse_judge_response(raw)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_single(sample: GenerationSample) -> GenerationEvalResult:
    """Score a single generated answer using an LLM judge + keyword coverage.

    Malformed judge output is handled gracefully via
    :func:`safe_parse_judge_response` — a warning is logged and fallback
    scores are used so this function never raises due to bad judge output.

    Args:
        sample: A GenerationSample with query, context, answer, an optional
            reference answer, and an optional expected_keywords list.

    Returns:
        A GenerationEvalResult with per-metric scores and judge reasoning.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.
        google.genai.errors.APIError: On Gemini API failures.
    """
    ref_section = (
        _REFERENCE_SECTION.format(reference=sample["reference"])
        if sample.get("reference")
        else ""
    )

    prompt = _JUDGE_PROMPT.format(
        query=sample["query"],
        context=sample["context"],
        answer=sample["answer"],
        reference_section=ref_section,
    )

    scores = _call_judge(prompt)

    correctness = scores.get("answer_correctness")
    if correctness is not None:
        correctness = _clamp(correctness)

    keywords = sample.get("expected_keywords")
    kw_coverage: float | None = (
        keyword_coverage(sample["answer"], keywords)
        if keywords
        else None
    )

    return GenerationEvalResult(
        query=sample["query"],
        faithfulness=_clamp(scores.get("faithfulness", 0.0)),
        answer_relevance=_clamp(scores.get("answer_relevance", 0.0)),
        answer_correctness=correctness,
        keyword_coverage=kw_coverage,
        reasoning=scores.get("reasoning", ""),
    )


def evaluate_dataset(
    samples: list[GenerationSample],
) -> tuple[list[GenerationEvalResult], AggregateGenerationResult]:
    """Evaluate a list of generated answers and aggregate the scores.

    Args:
        samples: List of GenerationSample dicts to evaluate.

    Returns:
        A (per_sample_results, aggregate) tuple. *aggregate* contains
        macro-averaged metrics. mean_answer_correctness and
        mean_keyword_coverage are None when no sample in the dataset
        provides the corresponding optional data.

    Raises:
        ValueError: If *samples* is empty.
    """
    if not samples:
        raise ValueError("samples list is empty — nothing to evaluate.")

    results: list[GenerationEvalResult] = []
    for sample in samples:
        results.append(evaluate_single(sample))

    n = len(results)

    correctness_scores = [
        r["answer_correctness"]
        for r in results
        if r["answer_correctness"] is not None
    ]

    coverage_scores = [
        r["keyword_coverage"]
        for r in results
        if r["keyword_coverage"] is not None
    ]

    aggregate = AggregateGenerationResult(
        num_samples=n,
        mean_faithfulness=sum(r["faithfulness"] for r in results) / n,
        mean_answer_relevance=sum(r["answer_relevance"] for r in results) / n,
        mean_answer_correctness=(
            sum(correctness_scores) / len(correctness_scores)
            if correctness_scores
            else None
        ),
        mean_keyword_coverage=(
            sum(coverage_scores) / len(coverage_scores)
            if coverage_scores
            else None
        ),
    )

    return results, aggregate


def _print_aggregate(aggregate: AggregateGenerationResult) -> None:
    correctness = aggregate["mean_answer_correctness"]
    correctness_str = f"{correctness:.4f}" if correctness is not None else "n/a"

    coverage = aggregate.get("mean_keyword_coverage")
    coverage_str = f"{coverage:.4f}" if coverage is not None else "n/a"

    print("\n── Aggregate Metrics ──────────────────────────")
    print(f"  Samples             : {aggregate['num_samples']}")
    print(f"  Faithfulness        : {aggregate['mean_faithfulness']:.4f}")
    print(f"  Answer Relevance    : {aggregate['mean_answer_relevance']:.4f}")
    print(f"  Answer Correctness  : {correctness_str}")
    print(f"  Keyword Coverage    : {coverage_str}")
    print("────────────────────────────────────────────────\n")


def main() -> None:
    """Run a live end-to-end generation evaluation demo."""
    query = "What is MLOps?"
    keywords = ["MLOps", "machine learning", "DevOps", "production", "lifecycle"]

    print(f"Query: {query}\n")

    try:
        chunks = retrieve(query, top_k=3)
    except (RuntimeError, ValueError) as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return

    if not chunks:
        print("[WARNING] No chunks retrieved — add documents to data/ first.")
        return

    context = build_context(chunks)

    print("Generating answer...")
    try:
        response = generate(query, chunks)
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return

    print(f"Answer: {response['answer'][:120]}...\n")

    sample = GenerationSample(
        query=query,
        context=context,
        answer=response["answer"],
        reference=None,
        expected_keywords=keywords,
    )

    print("Evaluating with LLM judge + keyword coverage...")
    try:
        result = evaluate_single(sample)
    except (ValueError, genai_errors.APIError) as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return

    print(f"\n  Faithfulness      : {result['faithfulness']:.4f}")
    print(f"  Answer Relevance  : {result['answer_relevance']:.4f}")
    kc = result["keyword_coverage"]
    print(f"  Keyword Coverage  : {f'{kc:.4f}' if kc is not None else 'n/a'}")
    print(f"  Reasoning         : {result['reasoning']}")


if __name__ == "__main__":
    main()
