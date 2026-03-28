"""Search page — vector similarity search and interactive knowledge graph."""

import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


# Consistent colors for entity types
_TYPE_COLORS = {
    "method": "#4CAF50",
    "dataset": "#2196F3",
    "metric": "#FF9800",
    "author": "#9C27B0",
    "task": "#F44336",
}


def render() -> None:
    st.header("Search")

    tab_vector, tab_kg = st.tabs(["Semantic Search", "Knowledge Graph"])

    with tab_vector:
        _vector_search_tab()

    with tab_kg:
        _knowledge_graph_tab()


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

def _vector_search_tab() -> None:
    query = st.text_input(
        "Search query",
        placeholder="e.g. attention mechanism, paper-to-code generation",
    )
    col1, col2 = st.columns([1, 3])
    n_results = col1.number_input("Results", min_value=1, max_value=50, value=5)

    if query:
        with st.spinner("Searching..."):
            pipeline = _get_embedding_pipeline()
            results = pipeline.search(query, n_results=n_results)

        if not results:
            st.warning("No results found. Ingest some papers first.")
            return

        st.write(f"**{len(results)} results** for _{query}_")

        for r in results:
            score_color = (
                "green" if r.score > 0.5
                else "orange" if r.score > 0.3
                else "red"
            )
            with st.container(border=True):
                col_score, col_meta = st.columns([1, 4])
                col_score.markdown(f"### :{score_color}[{r.score:.3f}]")
                col_meta.caption(
                    f"Paper: `{r.paper_id}` | Section: {r.section_title}"
                )
                st.markdown(
                    r.text[:500] + ("..." if len(r.text) > 500 else "")
                )


# ---------------------------------------------------------------------------
# Knowledge graph with interactive visualization
# ---------------------------------------------------------------------------

def _knowledge_graph_tab() -> None:
    kg = _get_knowledge_graph()

    entity_count = kg.count_entities()
    rel_count = kg.count_relationships()

    if entity_count == 0:
        st.info("Knowledge graph is empty. Ingest papers to populate it.")
        return

    # --- Controls ---
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    query = col1.text_input(
        "Search entities", placeholder="e.g. BERT, ImageNet", key="kg_query"
    )
    entity_type = col2.selectbox(
        "Entity type",
        ["", "method", "dataset", "metric", "author", "task"],
        format_func=lambda x: "All types" if x == "" else x,
    )
    depth = col3.slider("Graph depth", 1, 3, 1, key="kg_depth")
    rel_filter = col4.selectbox(
        "Relation type",
        ["", "outperforms", "evaluated_on", "uses", "extends"],
        format_func=lambda x: "All relations" if x == "" else x,
        key="kg_rel_filter",
    )

    # --- Legend ---
    legend_cols = st.columns(len(_TYPE_COLORS))
    for col, (etype, color) in zip(legend_cols, _TYPE_COLORS.items()):
        col.markdown(
            f'<span style="color:{color}; font-weight:bold;">'
            f"\u25CF {etype}</span>",
            unsafe_allow_html=True,
        )

    # --- Fetch entities ---
    entities = kg.search_entities(query=query, entity_type=entity_type, limit=50)
    if not entities:
        st.warning("No entities found matching your filters.")
        return

    # --- Graph visualization ---
    st.subheader(f"Graph ({entity_count} entities, {rel_count} relationships)")

    entity_names = {e.id: e.name for e in entities}
    selected_id = st.selectbox(
        "Focus entity (graph will expand from here)",
        options=list(entity_names.keys()),
        format_func=lambda eid: entity_names[eid],
    )

    if selected_id:
        nx_graph = kg.get_subgraph(selected_id, depth=depth)

        # Filter edges by relation type if requested
        if rel_filter:
            edges_to_remove = [
                (u, v) for u, v, d in nx_graph.edges(data=True)
                if d.get("relation_type", "") != rel_filter
            ]
            nx_graph.remove_edges_from(edges_to_remove)
            # Remove orphaned nodes (except focus)
            orphans = [
                n for n in nx_graph.nodes()
                if n != selected_id and nx_graph.degree(n) == 0
            ]
            nx_graph.remove_nodes_from(orphans)

        _render_graph(nx_graph, selected_id)

    # --- Entity list ---
    st.subheader(f"Entities ({len(entities)})")
    for entity in entities:
        color = _TYPE_COLORS.get(entity.entity_type, "#757575")
        with st.expander(f":{color[1:]}[●] **{entity.name}** ({entity.entity_type})"):
            st.write(f"**ID:** `{entity.id}`")
            st.write(f"**Paper:** `{entity.paper_id}`")
            if entity.properties:
                st.json(entity.properties)

            neighbors = kg.get_neighbors(entity.id)
            if neighbors:
                st.write(f"**Connections ({len(neighbors)}):**")
                for rel, neighbor in neighbors:
                    arrow = "\u2192" if rel.source_id == entity.id else "\u2190"
                    st.write(
                        f"- {arrow} **{neighbor.name}** ({neighbor.entity_type}) "
                        f"— _{rel.relation_type}_"
                    )


def _render_graph(nx_graph, focus_id: str) -> None:
    """Render a networkx graph as an interactive pyvis visualization."""
    from pyvis.network import Network

    node_count = nx_graph.number_of_nodes()
    if node_count == 0:
        st.info("No nodes to display with current filters.")
        return

    # Scale height with node count for breathing room
    height = max(500, min(750, node_count * 50))

    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#0e1117",
        font_color="#fafafa",
        directed=True,
    )

    # Repulsion-based layout: strong repulsion pushes nodes far apart,
    # long springs keep connected nodes at readable distance,
    # low central gravity prevents collapse into a ball.
    net.set_options("""
    {
        "physics": {
            "repulsion": {
                "nodeDistance": 250,
                "centralGravity": 0.005,
                "springLength": 300,
                "springConstant": 0.02,
                "damping": 0.15
            },
            "solver": "repulsion",
            "stabilization": { "iterations": 200, "fit": true },
            "minVelocity": 0.75
        },
        "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
            "smooth": { "type": "curvedCW", "roundness": 0.15 },
            "color": { "opacity": 0.6 }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragNodes": true,
            "navigationButtons": true
        },
        "layout": {
            "improvedLayout": true
        }
    }
    """)

    # Deduplicate edges: if A->B has multiple relations, merge labels
    edge_labels: dict[tuple[str, str], list[str]] = {}
    for src, dst, data in nx_graph.edges(data=True):
        key = (src, dst)
        rel = data.get("relation_type", "")
        if key not in edge_labels:
            edge_labels[key] = []
        if rel and rel not in edge_labels[key]:
            edge_labels[key].append(rel)

    # Add nodes
    for node_id, data in nx_graph.nodes(data=True):
        name = data.get("name", node_id)
        etype = data.get("entity_type", "")
        color = _TYPE_COLORS.get(etype, "#757575")
        is_focus = node_id == focus_id

        # Degree-based sizing: more connections = bigger node
        degree = nx_graph.degree(node_id)
        base_size = 35 if is_focus else 20
        size = base_size + min(degree * 3, 20)

        net.add_node(
            node_id,
            label=name,
            color={
                "background": color,
                "border": "#ffffff" if is_focus else color,
                "highlight": {"background": color, "border": "#ffffff"},
            },
            shape="dot",
            size=size,
            borderWidth=4 if is_focus else 1,
            title=f"<b>{name}</b><br>Type: {etype}<br>Connections: {degree}",
            font={
                "size": 16 if is_focus else 12,
                "color": "#fafafa",
                "strokeWidth": 3,
                "strokeColor": "#0e1117",
            },
        )

    # Add deduplicated edges
    for (src, dst), labels in edge_labels.items():
        combined_label = ", ".join(labels)
        net.add_edge(
            src,
            dst,
            label=combined_label,
            title=combined_label,
            font={"size": 10, "color": "#aaaaaa", "align": "top",
                  "strokeWidth": 2, "strokeColor": "#0e1117"},
            width=2,
            color={"color": "#555555", "highlight": "#ffffff"},
        )

    # Render to temp HTML and embed
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html_path = Path(f.name)

    html_content = html_path.read_text()
    html_path.unlink(missing_ok=True)

    components.html(html_content, height=height + 20, scrolling=False)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

def _get_embedding_pipeline():
    from papermind.ui.shared import get_embedding_pipeline
    return get_embedding_pipeline()


def _get_knowledge_graph():
    from papermind.ui.shared import get_knowledge_graph
    return get_knowledge_graph()
