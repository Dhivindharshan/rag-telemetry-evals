"""Streamlit dashboard for the RAG Telemetry Evals pipeline.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import run_pipeline, PipelineResult  # noqa: E402
from retriever import RetrievedChunk  # noqa: E402


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Telemetry Evals",
    page_icon="🔍",
    layout="wide",
)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    top_k = st.slider(
        label="Top-K chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of context chunks passed to the LLM.",
    )

    st.markdown("---")
    st.markdown("**Model**")
    st.code("gemini-2.5-flash", language=None)

    st.markdown("**Vector store**")
    st.code("ChromaDB · rag_documents", language=None)

    st.markdown("**Embedding model**")
    st.code("all-MiniLM-L6-v2", language=None)

    st.markdown("---")
    st.markdown(
        "**Run locally**\n```bash\nstreamlit run dashboard/app.py\n```"
    )


# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🔍 RAG Telemetry Evals")
st.caption(
    "Ask a question — the pipeline retrieves relevant chunks from ChromaDB, "
    "generates a grounded answer with Claude, and displays full telemetry."
)
st.markdown("---")


# ── Query input ────────────────────────────────────────────────────────────────

query = st.text_area(
    label="Your question",
    placeholder="e.g. What is MLOps?  |  How does a reranker improve retrieval quality?",
    height=80,
    label_visibility="collapsed",
)

submitted = st.button("🚀 Submit", type="primary", use_container_width=True)


# ── Pipeline execution ─────────────────────────────────────────────────────────

if submitted:
    query = query.strip()

    if not query:
        st.warning("Please enter a question before submitting.")
        st.stop()

    with st.spinner("Running pipeline…"):

        timings: dict[str, float] = {}

        # Retrieval
        t0 = time.perf_counter()
        try:
            result: PipelineResult = run_pipeline(query, top_k=top_k)
        except ValueError as exc:
            st.error(f"**Pipeline error:** {exc}")
            st.info(
                "The ChromaDB collection may be empty. "
                "Run `python src/vector_store.py` from the project root to ingest documents."
            )
            st.stop()
        except Exception as exc:
            st.error(f"**Unexpected error:** {exc}")
            st.stop()

        timings["total"] = time.perf_counter() - t0

    st.success("Done!", icon="✅")
    st.markdown("---")

    # ── Answer ─────────────────────────────────────────────────────────────────

    st.subheader("💬 Answer")
    st.markdown(result["answer"])

    st.markdown("---")

    # ── Two-column layout: chunks + telemetry ──────────────────────────────────

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:

        # Retrieved chunks
        st.subheader("📄 Retrieved Chunks")
        chunks: list[RetrievedChunk] = result["chunks_retrieved"]

        for i, chunk in enumerate(chunks, start=1):
            source_name = Path(chunk["source"]).name
            with st.expander(
                f"Chunk {i} — {source_name}  ·  distance {chunk['distance']:.4f}",
                expanded=(i == 1),
            ):
                st.markdown(f"**Chunk ID:** `{chunk['chunk_id']}`")
                st.markdown(f"**Source:** `{chunk['source']}`")
                st.markdown(f"**Distance:** `{chunk['distance']:.6f}`")
                st.markdown("**Text:**")
                st.text(chunk["text"])

        # Sources
        st.markdown("---")
        st.subheader("🗂️ Sources")
        for src in result["sources"]:
            st.markdown(f"- `{src}`")

    with col_right:

        # Telemetry
        st.subheader("⏱️ Telemetry")

        st.metric(label="Total wall-clock time", value=f"{timings['total']:.2f} s")

        st.markdown("**Pipeline config**")
        config_rows = {
            "Query": result["query"],
            "Top-K": str(top_k),
            "Model": result["model"],
            "Chunks retrieved": str(len(result["chunks_retrieved"])),
        }
        for k, v in config_rows.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("---")
        st.markdown("**Retrieval distances**")
        dist_data = {
            f"Chunk {i+1} ({Path(c['source']).name})": round(c["distance"], 4)
            for i, c in enumerate(result["chunks_retrieved"])
        }
        st.bar_chart(dist_data)

        st.markdown("**Distance guide**")
        st.markdown(
            "| Range | Relevance |\n"
            "|---|---|\n"
            "| < 0.5 | Very high |\n"
            "| 0.5 – 1.0 | High |\n"
            "| 1.0 – 1.3 | Moderate |\n"
            "| > 1.3 | Low |"
        )


def main() -> None:
    """Entry point note — Streamlit apps are run via the CLI, not main()."""
    print("Run this dashboard with:")
    print("    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
