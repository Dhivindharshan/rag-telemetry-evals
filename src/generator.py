"""Answer generation module for RAG using the Google Gemini API."""

import logging
import os
import traceback
from pathlib import Path
from typing import TypedDict

from google import genai
from google.genai import types

from retriever import RetrievedChunk, retrieve

_log = logging.getLogger("rag.generator")

DEFAULT_MODEL = "gemini-2.5-flash"
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "v1.txt"
_ENV_PATH = Path(__file__).parent.parent / ".env"

_FALLBACK_TEMPLATE = (
    "You are a helpful assistant answering questions based solely on the provided context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer the question using only the information in the context above. "
    "If the context does not contain enough information to answer fully, say so explicitly."
)


class GeneratedResponse(TypedDict):
    query: str
    answer: str
    model: str
    sources: list[str]


def _load_prompt_template(prompt_path: Path = _PROMPT_PATH) -> str:
    """Return the prompt template from disk, or the built-in fallback if empty/missing."""
    try:
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    return _FALLBACK_TEMPLATE


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Concatenate retrieved chunk texts into a single numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(parts)


def _resolve_api_key(api_key: str | None) -> str:
    """
    Return the Gemini API key to use.

    Precedence:
      1. Explicit *api_key* argument (set by the caller — most reliable).
      2. GEMINI_API_KEY in os.environ (set by load_dotenv at startup).
      3. Re-read the .env file directly as a last resort.

    Raises:
        ValueError: If no key can be found through any of the three sources.
    """
    if api_key:
        return api_key

    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key:
        return env_key

    # Last resort: read .env directly (covers uvicorn reload subprocess edge cases)
    try:
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                value = line.split("=", 1)[1].strip()
                if value:
                    os.environ["GEMINI_API_KEY"] = value
                    return value
    except OSError:
        pass

    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Add GEMINI_API_KEY=<your-key> to your .env file and restart the server."
    )


def generate(
    query: str,
    chunks: list[RetrievedChunk],
    model: str = DEFAULT_MODEL,
    prompt_path: Path = _PROMPT_PATH,
    api_key: str | None = None,
) -> GeneratedResponse:
    """
    Generate an answer to *query* grounded in *chunks* using the Gemini API.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks from retriever.py.
        model: Gemini model identifier (default: gemini-2.5-flash).
        prompt_path: Path to a plain-text prompt template with {context} and
            {question} placeholders. Falls back to a built-in template if the
            file is missing or empty.
        api_key: Gemini API key. If omitted, resolved from GEMINI_API_KEY env
            var or directly from .env as a fallback.

    Returns:
        A GeneratedResponse dict with keys: query, answer, model, sources.

    Raises:
        ValueError: If chunks is empty or no API key can be resolved.
        google.genai.errors.APIError: On Gemini API failures.
    """
    if not chunks:
        raise ValueError("chunks is empty — cannot generate an answer without context.")

    resolved_key = _resolve_api_key(api_key)
    _log.debug("[DEBUG] API key resolved  masked=%s…%s",
               resolved_key[:4], resolved_key[-4:] if len(resolved_key) > 8 else "****")

    template = _load_prompt_template(prompt_path)
    context = build_context(chunks)
    prompt = template.format(context=context, question=query)

    _log.debug("[DEBUG] Gemini request  model=%s  prompt_chars=%d  chunks=%d",
               model, len(prompt), len(chunks))

    client = genai.Client(api_key=resolved_key)

    try:
        raw_response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=4096,
            ),
        )
        _log.debug("[DEBUG] Gemini raw response type=%s  candidates=%d",
                   type(raw_response).__name__,
                   len(raw_response.candidates) if hasattr(raw_response, "candidates") else -1)
    except Exception as exc:
        _log.error("[DEBUG] client.models.generate_content FAILED: %s: %s\n%s",
                   type(exc).__name__, exc, traceback.format_exc())
        raise

    try:
        answer = (raw_response.text or "").strip()
        _log.debug("[DEBUG] response.text OK  answer_len=%d", len(answer))
    except Exception as exc:
        _log.error("[DEBUG] response.text FAILED: %s: %s\n%s",
                   type(exc).__name__, exc, traceback.format_exc())
        _log.error("[DEBUG] raw_response dump: candidates=%r  prompt_feedback=%r",
                   getattr(raw_response, "candidates", "N/A"),
                   getattr(raw_response, "prompt_feedback", "N/A"))
        raise

    sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

    return GeneratedResponse(
        query=query,
        answer=answer,
        model=model,
        sources=sources,
    )


def main() -> None:
    """Run the full RAG pipeline: retrieve context then generate an answer."""
    query = "What is MLOps?"

    print(f"Query: {query}\n")

    try:
        chunks = retrieve(query, top_k=3)
    except (RuntimeError, ValueError) as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return

    if not chunks:
        print("[WARNING] No chunks retrieved for the query.")
        return

    print(f"Retrieved {len(chunks)} chunk(s). Generating answer...\n")

    try:
        response = generate(query, chunks)
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return

    print(f"Model   : {response['model']}")
    print(f"Sources : {', '.join(response['sources'])}\n")
    print("Answer:")
    print(response["answer"])


if __name__ == "__main__":
    main()
