"""Streamlit page: FAISS index benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def render() -> None:
    st.header("Vector Index Benchmarks")
    st.caption("Compare FAISS index types: Flat (exact) vs IVF vs HNSW")

    # --- Run benchmark ---
    with st.expander("Run New Benchmark", expanded=True):
        col1, col2, col3 = st.columns(3)
        n_vectors = col1.number_input("Vectors", 500, 50000, 5000, step=500)
        dim = col2.selectbox("Dimension", [256, 512, 768], index=2)
        n_queries = col3.number_input("Queries", 10, 500, 100, step=10)

        use_real = st.checkbox(
            "Use real embeddings from ingested papers (if available)",
            value=True,
        )

        if st.button("Run Benchmark", type="primary"):
            with st.spinner("Running benchmark suite..."):
                from papermind.benchmarks.faiss_benchmark import run_benchmark

                suite = run_benchmark(
                    n_vectors=n_vectors,
                    dim=dim,
                    n_queries=n_queries,
                    use_real=use_real,
                )
                out_path = Path("data/benchmark_results.json")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                suite.save(out_path)
                st.success(
                    f"Benchmark complete! {len(suite.results)} configs tested, "
                    f"{suite.n_queries} queries each."
                )
                st.rerun()

    # --- Display results ---
    results_path = Path("data/benchmark_results.json")
    if not results_path.exists():
        st.info("No benchmark results yet. Run a benchmark above to get started.")
        return

    data = json.loads(results_path.read_text())
    results = data["results"]

    st.subheader(f"Results ({data.get('timestamp', 'unknown')[:19]})")

    # Summary table
    rows = []
    for r in results:
        label = r["index_type"]
        if r.get("params"):
            p = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            label = f"{r['index_type']} ({p})"
        rows.append({
            "Index": label,
            "Build (ms)": r["build_time_ms"],
            "Query (ms)": r["single_query_ms"],
            "Memory (KB)": round(r["memory_bytes"] / 1024),
            "Recall@1": r["recall_at_1"],
            "Recall@5": r["recall_at_5"],
            "Recall@10": r["recall_at_10"],
        })

    st.dataframe(
        rows,
        use_container_width=True,
        column_config={
            "Recall@1": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Recall@5": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Recall@10": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
        },
    )

    # --- Charts ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Query Latency vs Recall@10")
        chart_data = []
        for r in results:
            label = r["index_type"]
            if r.get("params"):
                p = ", ".join(f"{k}={v}" for k, v in r["params"].items())
                label = f"{label}({p})"
            chart_data.append({
                "config": label,
                "query_ms": r["single_query_ms"],
                "recall_10": r["recall_at_10"],
            })
        st.scatter_chart(
            chart_data,
            x="query_ms",
            y="recall_10",
            color="config",
            x_label="Query latency (ms)",
            y_label="Recall@10",
        )

    with col_right:
        st.subheader("Memory Usage")
        mem_data = []
        for r in results:
            label = r["index_type"]
            if r.get("params"):
                p = ", ".join(f"{k}={v}" for k, v in r["params"].items())
                label = f"{label}({p})"
            mem_data.append({
                "config": label,
                "memory_kb": round(r["memory_bytes"] / 1024),
            })
        st.bar_chart(mem_data, x="config", y="memory_kb", x_label="Index", y_label="Memory (KB)")

    # --- Interpretation ---
    st.divider()
    st.subheader("Index Type Guide")
    st.markdown("""
| Index | Best For | Trade-offs |
|-------|----------|------------|
| **Flat** | < 50k vectors, exact results needed | Perfect recall, but O(n) per query — slows linearly |
| **IVF** | 50k–1M vectors, tunable speed/recall | Fast with high nprobe, but requires training and recall drops if nlist is wrong |
| **HNSW** | Any scale, best recall/speed balance | Near-perfect recall at low latency, but ~15-30% more memory for the graph structure |

**For PaperMind's use case** (hundreds to low thousands of paper chunks):
- **HNSW (M=32, ef=64)** is the sweet spot — fast queries, great recall, no training needed
- **Flat** is also fine at this scale and gives exact results
- **IVF** only pays off at 50k+ vectors
""")
