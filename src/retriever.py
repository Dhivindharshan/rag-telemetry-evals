"""Retriever module for querying the ChromaDB vector store."""

from pathlib import Path
from typing import TypedDict

from sentence_transformers import SentenceTransformer

from embeddings import DEFAULT_MODEL
from vector_store import get_collection, _DB_DIR

DEFAULT_TOP_K = 3


class RetrievedChunk(TypedDict):
    chunk_id: str
    source: str
    text: str
    distance: float


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL,
    db_dir: str | Path = _DB_DIR,
) -> list[RetrievedChunk]:
    """
    Embed a query string and return the most similar chunks from ChromaDB.

    Args:
        query: The natural-language query to search for.
        top_k: Number of top results to return (default 3).
        model_name: Sentence-transformers model used to embed the query.
                    Must match the model used during ingestion.
        db_dir: Path to the ChromaDB persistence directory.

    Returns:
        A list of RetrievedChunk dicts sorted by ascending distance
        (closest match first), each containing 'chunk_id', 'source',
        'text', and 'distance'.

    Raises:
        ValueError: If the query string is empty or the collection is empty.
        OSError: If the embedding model cannot be loaded.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query string must not be empty.")

    collection = get_collection(db_dir)

    if collection.count() == 0:
        raise ValueError(
            "The 'rag_documents' collection is empty. "
            "Run vector_store.py first to ingest documents."
        )

    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved: list[RetrievedChunk] = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                source=metadata.get("source", ""),
                text=text,
                distance=float(distance),
            )
        )

    return retrieved


def main() -> None:
    """Run a sample query against the stored collection and print results."""
    sample_query = "What is MLOps?"

    print(f"Query: {sample_query}\n")

    try:
        results = retrieve(sample_query)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    except OSError as e:
        print(f"[ERROR] Failed to load embedding model: {e}")
        return

    print(f"Top {len(results)} result(s):\n")
    for rank, chunk in enumerate(results, start=1):
        print(f"  [{rank}] chunk_id : {chunk['chunk_id']}")
        print(f"       source   : {chunk['source']}")
        print(f"       distance : {chunk['distance']:.4f}")
        preview = chunk["text"][:120].replace("\n", " ")
        print(f"       preview  : {preview}...")
        print()


if __name__ == "__main__":
    main()
