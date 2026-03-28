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
    try:
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
    except Exception as e:
        st.error(f"Page error: {e}")
        import traceback
        st.code(traceback.format_exc())


def _sidebar_stats() -> None:
    """Show quick system stats in the sidebar (non-blocking)."""
    try:
        from papermind.infrastructure.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Entities", kg.count_entities())
        col2.metric("Relations", kg.count_relationships())
    except Exception:
        pass


if __name__ == "__main__":
    main()
