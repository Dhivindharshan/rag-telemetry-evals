"""Embedding module for generating vector representations of text chunks."""

from pathlib import Path
from typing import TypedDict

from sentence_transformers import SentenceTransformer

from chunking import Chunk, chunk_documents
from ingestion import load_documents

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddedChunk(TypedDict):
    chunk_id: str
    source: str
    text: str
    embedding: list[float]


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = DEFAULT_MODEL,
) -> list[EmbeddedChunk]:
    """
    Generate embeddings for a list of text chunks.

    Encodes all chunk texts in a single batched call for efficiency, then
    pairs each resulting vector back with its originating chunk.

    Args:
        chunks: A list of Chunk dicts produced by chunking.py.
        model_name: A sentence-transformers model identifier.

    Returns:
        A list of EmbeddedChunk dicts, each containing the original chunk
        fields plus an 'embedding' key holding a list of floats.

    Raises:
        ValueError: If the chunks list is empty.
        OSError: If the model cannot be loaded (e.g. not installed / no network).
    """
    if not chunks:
        raise ValueError("chunks list is empty — nothing to embed.")

    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    embedded: list[EmbeddedChunk] = []
    for chunk, vector in zip(chunks, vectors):
        embedded.append(
            EmbeddedChunk(
                chunk_id=chunk["chunk_id"],
                source=chunk["source"],
                text=chunk["text"],
                embedding=vector.tolist(),
            )
        )

    return embedded


def main() -> None:
    """Load, chunk, and embed documents from data/, then print a summary."""
    data_dir = Path(__file__).parent.parent / "data"

    print(f"Loading documents from: {data_dir}")

    try:
        docs = load_documents(data_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"[ERROR] {e}")
        return

    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    print(f"Chunks to embed : {len(chunks)}")

    if not chunks:
        print("[WARNING] No chunks to embed. Add documents to the data/ directory.")
        return

    try:
        embedded = embed_chunks(chunks, model_name=DEFAULT_MODEL)
    except OSError as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    dim = len(embedded[0]["embedding"])
    sample_id = embedded[0]["chunk_id"]

    print(f"Embedding dim   : {dim}")
    print(f"Sample chunk id : {sample_id}")


if __name__ == "__main__":
    main()
