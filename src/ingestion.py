"""Document ingestion module for loading .txt, .md, and .pdf files."""

import os
from pathlib import Path
from typing import TypedDict

import pypdf


class Document(TypedDict):
    id: str
    source: str
    text: str


def load_txt(filepath: Path) -> str:
    """Read and return the text content of a .txt file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_md(filepath: Path) -> str:
    """Read and return the text content of a .md file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(filepath: Path) -> str:
    """Extract and return concatenated text from all pages of a PDF file."""
    reader = pypdf.PdfReader(str(filepath))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(data_dir: str | Path) -> list[Document]:
    """
    Load all .txt, .md, and .pdf files from a directory.

    Args:
        data_dir: Path to the directory containing documents.

    Returns:
        A list of Document dicts with keys 'id', 'source', and 'text'.

    Raises:
        FileNotFoundError: If data_dir does not exist.
        NotADirectoryError: If data_dir is not a directory.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_path}")

    loaders = {
        ".txt": load_txt,
        ".md": load_md,
        ".pdf": load_pdf,
    }

    documents: list[Document] = []

    for filepath in sorted(data_path.iterdir()):
        suffix = filepath.suffix.lower()
        if suffix not in loaders:
            continue

        try:
            text = loaders[suffix](filepath)
            documents.append(
                Document(
                    id=filepath.name,
                    source=str(filepath.resolve()),
                    text=text,
                )
            )
        except Exception as e:
            print(f"[WARNING] Skipping '{filepath.name}': {e}")

    return documents


def main() -> None:
    """Load documents from the data/ directory and print a summary."""
    data_dir = Path(__file__).parent.parent / "data"

    print(f"Loading documents from: {data_dir}")

    try:
        docs = load_documents(data_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"[ERROR] {e}")
        return

    print(f"Loaded {len(docs)} document(s).\n")

    for doc in docs:
        preview = doc["text"][:80].replace("\n", " ")
        print(f"  [{doc['id']}] {preview}{'...' if len(doc['text']) > 80 else ''}")


if __name__ == "__main__":
    main()
