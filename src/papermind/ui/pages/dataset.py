"""Streamlit page: PwC dataset builder and explorer."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

PWC_DIR = Path("data/pwc")


def render() -> None:
    st.header("Research Code Dataset")
    st.caption("Build training data from Papers with Code: paper abstracts paired with scientific Python code")

    tab_build, tab_explore = st.tabs(["Build Dataset", "Explore Dataset"])

    with tab_build:
        _build_tab()
    with tab_explore:
        _explore_tab()


def _build_tab() -> None:
    st.subheader("Build from Papers with Code")

    col1, col2 = st.columns(2)
    max_repos = col1.number_input("Max repos to clone", 10, 5000, 100, step=50)
    dedup_threshold = col2.slider("Dedup threshold (Jaccard)", 0.3, 1.0, 0.7, 0.05)

    col3, col4 = st.columns(2)
    pytorch_only = col3.checkbox("PyTorch repos only", value=False)
    official_only = col4.checkbox("Official implementations only", value=False)

    if st.button("Build Dataset", type="primary"):
        from papermind.data.pwc_dataset import build_dataset

        progress_bar = st.progress(0)
        status_text = st.empty()

        def _progress(step: str, current: int, total: int, msg: str) -> None:
            if total > 0:
                progress_bar.progress(min(current / total, 1.0))
            status_text.text(msg)

        frameworks = ("pytorch",) if pytorch_only else ("pytorch", "none")

        with st.spinner("Building dataset..."):
            stats = build_dataset(
                output_dir=str(PWC_DIR),
                max_repos=max_repos,
                dedup_threshold=dedup_threshold,
                frameworks=frameworks,
                require_official=official_only,
                progress_callback=_progress,
            )

        progress_bar.progress(1.0)
        status_text.empty()

        st.success(f"Dataset built! {stats.pairs_after_dedup} pairs saved.")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Repos Cloned", stats.repos_cloned)
        col_b.metric("Files Extracted", stats.files_extracted)
        col_c.metric("Pairs Created", stats.pairs_created)
        col_d.metric("After Dedup", stats.pairs_after_dedup)

        if stats.frameworks:
            st.write("**Frameworks:**", stats.frameworks)
        if stats.import_counts:
            st.write("**Import distribution:**", stats.import_counts)

    # Show existing stats if available
    stats_path = PWC_DIR / "stats.json"
    if stats_path.exists():
        st.divider()
        st.subheader("Last Build Stats")
        stats_data = json.loads(stats_path.read_text())

        cols = st.columns(4)
        cols[0].metric("Repos Cloned", stats_data.get("repos_cloned", 0))
        cols[1].metric("Files Extracted", stats_data.get("files_extracted", 0))
        cols[2].metric("Total Pairs", stats_data.get("pairs_created", 0))
        cols[3].metric("After Dedup", stats_data.get("pairs_after_dedup", 0))

        col_l, col_r = st.columns(2)
        with col_l:
            if stats_data.get("frameworks"):
                st.write("**Frameworks:**")
                st.json(stats_data["frameworks"])
        with col_r:
            if stats_data.get("import_counts"):
                st.write("**Import distribution:**")
                st.json(stats_data["import_counts"])


def _explore_tab() -> None:
    jsonl_path = PWC_DIR / "paper_code_pairs.jsonl"
    quality_path = PWC_DIR / "paper_code_pairs_quality.jsonl"

    if not jsonl_path.exists():
        st.info("No dataset built yet. Use the Build tab to create one.")
        return

    # Load pairs
    all_pairs = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                all_pairs.append(json.loads(line))

    quality_count = 0
    if quality_path.exists():
        with open(quality_path) as f:
            quality_count = sum(1 for line in f if line.strip())

    st.subheader(f"Dataset: {len(all_pairs)} pairs ({quality_count} with abstracts)")

    # Filters
    col1, col2, col3 = st.columns(3)

    frameworks = sorted({p["metadata"]["framework"] for p in all_pairs})
    fw_filter = col1.selectbox("Framework", ["all"] + frameworks)

    has_abstract_filter = col2.selectbox("Has abstract", ["all", "yes", "no"])

    imports_all = set()
    for p in all_pairs:
        imports_all.update(p["metadata"].get("scientific_imports", []))
    import_filter = col3.selectbox("Import", ["all"] + sorted(imports_all))

    # Apply filters
    filtered = all_pairs
    if fw_filter != "all":
        filtered = [p for p in filtered if p["metadata"]["framework"] == fw_filter]
    if has_abstract_filter == "yes":
        filtered = [p for p in filtered if p["metadata"]["has_abstract"]]
    elif has_abstract_filter == "no":
        filtered = [p for p in filtered if not p["metadata"]["has_abstract"]]
    if import_filter != "all":
        filtered = [p for p in filtered
                    if import_filter in p["metadata"].get("scientific_imports", [])]

    st.caption(f"Showing {len(filtered)} pairs after filtering")

    # Paginated display
    page_size = 10
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, total_pages, 1)
    start = (page - 1) * page_size
    page_pairs = filtered[start : start + page_size]

    for pair in page_pairs:
        meta = pair["metadata"]
        title = meta.get("title", "Untitled")
        with st.expander(
            f"**{title[:80]}** — `{meta['file_path']}` "
            f"({'official' if meta['is_official'] else 'community'}, {meta['framework']})"
        ):
            if pair.get("instruction"):
                st.markdown("**Instruction:**")
                st.text(pair["instruction"][:500])

            st.markdown("**Code:**")
            st.code(pair["output"][:2000], language="python")

            st.caption(
                f"Repo: {meta['repo_url']} | "
                f"ArXiv: {meta.get('arxiv_id', 'N/A')} | "
                f"Imports: {', '.join(meta.get('scientific_imports', []))}"
            )
