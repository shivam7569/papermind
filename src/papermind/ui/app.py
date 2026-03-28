"""PaperMind Streamlit application — main entry point."""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="PaperMind",
        page_icon="\U0001f9e0",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Sidebar navigation ---
    st.sidebar.title("PaperMind")
    st.sidebar.caption("Local AI Research System")

    page = st.sidebar.radio(
        "Navigate",
        ["Chat", "Papers", "Search", "Dataset", "Benchmarks", "System"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    # Quick stats in sidebar
    _sidebar_stats()

    # --- Page routing ---
    if page == "Chat":
        from papermind.ui.pages.chat import render
        render()
    elif page == "Papers":
        from papermind.ui.pages.papers import render
        render()
    elif page == "Search":
        from papermind.ui.pages.search import render
        render()
    elif page == "Dataset":
        from papermind.ui.pages.dataset import render
        render()
    elif page == "Benchmarks":
        from papermind.ui.pages.benchmarks import render
        render()
    elif page == "System":
        from papermind.ui.pages.system import render
        render()


def _sidebar_stats() -> None:
    """Show quick system stats in the sidebar."""
    try:
        from papermind.ui.shared import get_vector_store, get_knowledge_graph
        vs = get_vector_store()
        kg = get_knowledge_graph()
        chunk_count = vs.count if isinstance(vs.count, int) else vs.count()
        st.sidebar.metric("Chunks", chunk_count)
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Entities", kg.count_entities())
        col2.metric("Relations", kg.count_relationships())
    except Exception:
        st.sidebar.info("Services initializing...")


if __name__ == "__main__":
    main()
