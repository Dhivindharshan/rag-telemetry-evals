"""Reranker module for scoring retrieved chunks with a cross-encoder model."""

from pathlib import Path
from typing import TypedDict

from sentence_transformers.cross_encoder import CrossEncoder

from retriever import RetrievedChunk, retrieve

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RankedChunk(TypedDict):
    chunk_id: str
    source: str
    text: str
    distance: float
    rerank_score: float


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    model_name: str = DEFAULT_RERANK_MODEL,
    top_n: int | None = None,
) -> list[RankedChunk]:
    """
    Re-score retrieved chunks using a cross-encoder and return them sorted
    by relevance (highest score first).

    A cross-encoder jointly encodes the (query, passage) pair, giving a
    more accurate relevance signal than the bi-encoder cosine similarity
    used during initial retrieval.

    Args:
        query: The original user query.
        chunks: Candidate chunks from the retriever.
        model_name: A sentence-transformers CrossEncoder model identifier.
        top_n: If provided, return only the top *top_n* chunks after
            reranking. Defaults to returning all chunks.

    Returns:
        A list of RankedChunk dicts sorted by descending rerank_score,
        each containing all original RetrievedChunk fields plus
        'rerank_score'.

    Raises:
        ValueError: If *chunks* is empty or *query* is empty.
        OSError: If the cross-encoder model cannot be loaded.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query string must not be empty.")
    if not chunks:
        raise ValueError("chunks list is empty — nothing to rerank.")

    model = CrossEncoder(model_name)

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores: list[float] = model.predict(pairs).tolist()

    ranked: list[RankedChunk] = []
    for chunk, score in zip(chunks, scores):
        ranked.append(
            RankedChunk(
                chunk_id=chunk["chunk_id"],
                source=chunk["source"],
                text=chunk["text"],
                distance=chunk["distance"],
                rerank_score=float(score),
            )
        )

    ranked.sort(key=lambda c: c["rerank_score"], reverse=True)

    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked


def main() -> None:
    """Retrieve chunks for a sample query, rerank them, and print results."""
    query = "What is MLOps?"

    print(f"Query: {query}\n")

    try:
        chunks = retrieve(query, top_k=5)
    except (ValueError, OSError) as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return

    print(f"Retrieved {len(chunks)} chunk(s). Reranking...\n")

    try:
        ranked = rerank(query, chunks, top_n=3)
    except (ValueError, OSError) as e:
        print(f"[ERROR] Reranking failed: {e}")
        return

    print(f"Top {len(ranked)} chunk(s) after reranking:\n")
    for rank, chunk in enumerate(ranked, start=1):
        print(f"  [{rank}] chunk_id     : {chunk['chunk_id']}")
        print(f"       source       : {chunk['source']}")
        print(f"       rerank_score : {chunk['rerank_score']:.4f}")
        print(f"       distance     : {chunk['distance']:.4f}")
        preview = chunk["text"][:120].replace("\n", " ")
        print(f"       preview      : {preview}...")
        print()


if __name__ == "__main__":
    main()
