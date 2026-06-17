"""End-to-end RAG pipeline orchestrator."""

from pathlib import Path
from typing import TypedDict

from ingestion import load_documents
from chunking import chunk_documents
from embeddings import embed_chunks
from vector_store import store_embedded_chunks, count_documents
from retriever import RetrievedChunk, retrieve
from generator import GeneratedResponse, generate, DEFAULT_MODEL


class PipelineResult(TypedDict):
    query: str
    answer: str
    model: str
    sources: list[str]
    chunks_retrieved: list[RetrievedChunk]
    num_docs_ingested: int
    num_chunks_stored: int


def ingest(
    data_dir: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> tuple[int, int]:
    """
    Load, chunk, embed, and store documents from *data_dir*.

    Args:
        data_dir: Directory containing .txt, .md, or .pdf files.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A (num_docs, num_chunks) tuple.

    Raises:
        FileNotFoundError: If data_dir does not exist.
        ValueError: If no documents or chunks are produced.
    """
    docs = load_documents(data_dir)
    if not docs:
        raise ValueError(f"No documents found in '{data_dir}'.")

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Chunking produced no chunks.")

    embedded = embed_chunks(chunks)
    store_embedded_chunks(embedded)

    return len(docs), len(chunks)


def run_pipeline(
    query: str,
    data_dir: str | Path | None = None,
    top_k: int = 3,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    model: str = DEFAULT_MODEL,
) -> PipelineResult:
    """
    Run the full RAG pipeline for a given query.

    If *data_dir* is provided, documents are ingested (idempotent via upsert)
    before retrieval. If omitted, the existing ChromaDB collection is queried.

    Args:
        query: The user's question.
        data_dir: Optional directory of source documents to ingest first.
        top_k: Number of context chunks to retrieve.
        chunk_size: Maximum characters per chunk (used during ingestion).
        chunk_overlap: Overlap between consecutive chunks (used during ingestion).
        model: Anthropic model identifier used for generation.

    Returns:
        A PipelineResult dict with the answer, sources, retrieved chunks,
        and ingestion statistics.

    Raises:
        FileNotFoundError: If data_dir is given but does not exist.
        ValueError: If ingestion or retrieval produces no usable data.
        RuntimeError: If the ChromaDB collection is empty and no data_dir was given.
        anthropic.APIError: On Anthropic API failures.
    """
    num_docs = 0
    num_chunks = 0

    if data_dir is not None:
        num_docs, num_chunks = ingest(
            data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    chunks = retrieve(query, top_k=top_k)

    response: GeneratedResponse = generate(query, chunks, model=model)

    return PipelineResult(
        query=response["query"],
        answer=response["answer"],
        model=response["model"],
        sources=response["sources"],
        chunks_retrieved=chunks,
        num_docs_ingested=num_docs,
        num_chunks_stored=num_chunks,
    )


def main() -> None:
    """Ingest documents from data/ and answer a sample query."""
    data_dir = Path(__file__).parent.parent / "data"
    query = "What is MLOps?"

    print(f"Data directory : {data_dir}")
    print(f"Query          : {query}\n")

    try:
        result = run_pipeline(query, data_dir=data_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        return

    print(f"Docs ingested  : {result['num_docs_ingested']}")
    print(f"Chunks stored  : {result['num_chunks_stored']}")
    print(f"Model          : {result['model']}")
    print(f"Sources        : {', '.join(result['sources'])}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
