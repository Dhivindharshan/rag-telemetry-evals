"""Vector store module for persisting and querying embedded chunks via ChromaDB."""

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from embeddings import EmbeddedChunk, embed_chunks
from chunking import chunk_documents
from ingestion import load_documents

COLLECTION_NAME = "rag_documents"
_DB_DIR = Path(__file__).parent.parent / "data" / "chroma_db"


def get_collection(db_dir: str | Path = _DB_DIR) -> Collection:
    """
    Return (or create) the ChromaDB collection for RAG documents.

    Args:
        db_dir: Directory where ChromaDB persists its data.

    Returns:
        A ChromaDB Collection object for 'rag_documents'.
    """
    db_path = Path(db_dir)
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def store_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
    db_dir: str | Path = _DB_DIR,
) -> Collection:
    """
    Upsert embedded chunks into the ChromaDB collection.

    Uses upsert so re-running the pipeline with the same chunk IDs is
    idempotent — existing documents are updated rather than duplicated.

    Args:
        embedded_chunks: A list of EmbeddedChunk dicts from embeddings.py.
        db_dir: Directory where ChromaDB persists its data.

    Returns:
        The ChromaDB Collection after the upsert.

    Raises:
        ValueError: If embedded_chunks is empty.
    """
    if not embedded_chunks:
        raise ValueError("embedded_chunks is empty — nothing to store.")

    collection = get_collection(db_dir)

    ids = [ec["chunk_id"] for ec in embedded_chunks]
    embeddings = [ec["embedding"] for ec in embedded_chunks]
    documents = [ec["text"] for ec in embedded_chunks]
    metadatas = [{"source": ec["source"]} for ec in embedded_chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def count_documents(db_dir: str | Path = _DB_DIR) -> int:
    """
    Return the number of documents currently stored in the collection.

    Args:
        db_dir: Directory where ChromaDB persists its data.

    Returns:
        Integer count of stored documents.
    """
    collection = get_collection(db_dir)
    return collection.count()


def main() -> None:
    """Ingest, embed, store documents from data/, then print a summary."""
    data_dir = Path(__file__).parent.parent / "data"

    print(f"Loading documents from : {data_dir}")

    try:
        docs = load_documents(data_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"[ERROR] {e}")
        return

    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

    if not chunks:
        print("[WARNING] No chunks found. Add documents to the data/ directory.")
        return

    try:
        embedded = embed_chunks(chunks)
    except OSError as e:
        print(f"[ERROR] Failed to load embedding model: {e}")
        return

    try:
        store_embedded_chunks(embedded)
    except Exception as e:
        print(f"[ERROR] Failed to store chunks: {e}")
        return

    total = count_documents()

    print(f"Collection name   : {COLLECTION_NAME}")
    print(f"Stored documents  : {total}")


if __name__ == "__main__":
    main()
