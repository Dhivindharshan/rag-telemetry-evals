"""Text chunking module for splitting documents into overlapping character chunks."""

from pathlib import Path
from typing import TypedDict

from ingestion import Document, load_documents


class Chunk(TypedDict):
    chunk_id: str
    source: str
    text: str


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Split a single document into overlapping text chunks.

    Args:
        document: A Document dict from ingestion.py.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of Chunk dicts with keys 'chunk_id', 'source', and 'text'.

    Raises:
        ValueError: If chunk_overlap >= chunk_size or either value is non-positive.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    text = document["text"]
    doc_id = document["id"]
    source = document["source"]

    if not text.strip():
        return []

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    index = 0
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_chunk_{index}",
                source=source,
                text=chunk_text,
            )
        )

        index += 1
        start += step

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Split a list of documents into overlapping text chunks.

    Args:
        documents: A list of Document dicts from ingestion.py.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A flat list of Chunk dicts across all documents.

    Raises:
        ValueError: If chunk_overlap >= chunk_size or either value is non-positive.
    """
    all_chunks: list[Chunk] = []

    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size, chunk_overlap))

    return all_chunks


def main() -> None:
    """Load documents from data/ and chunk them, then print a summary."""
    data_dir = Path(__file__).parent.parent / "data"

    print(f"Loading documents from: {data_dir}")

    try:
        docs = load_documents(data_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"[ERROR] {e}")
        return

    print(f"Input documents : {len(docs)}")

    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

    print(f"Generated chunks: {len(chunks)}\n")

    for chunk in chunks[:5]:
        preview = chunk["text"][:72].replace("\n", " ")
        print(f"  [{chunk['chunk_id']}] {preview}...")


if __name__ == "__main__":
    main()
