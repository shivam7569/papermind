"""Papers page — upload, ingest, and browse papers."""

import tempfile
from pathlib import Path

import streamlit as st


def render() -> None:
    st.header("Papers")

    tab_upload, tab_browse = st.tabs(["Upload & Ingest", "Browse"])

    with tab_upload:
        _upload_tab()

    with tab_browse:
        _browse_tab()


def _upload_tab() -> None:
    """Upload and ingest a PDF paper."""
    uploaded = st.file_uploader(
        "Upload a research paper (PDF)",
        type=["pdf"],
        help="The paper will be parsed, chunked, embedded, and entities extracted.",
    )

    if uploaded is not None:
        st.info(f"**{uploaded.name}** — {uploaded.size / 1024:.0f} KB")

        # Parser selection
        from papermind.ingestion.grobid_parser import check_grobid_health
        from papermind.ingestion.mineru_parser import check_mineru_available
        grobid_available = check_grobid_health()
        mineru_available = check_mineru_available()

        parser_options = []
        parser_descriptions = {}

        if grobid_available and mineru_available:
            parser_options.append("Hybrid (recommended)")
            parser_descriptions["Hybrid (recommended)"] = (
                "GROBID for metadata + MinerU for body with LaTeX equations"
            )
        if grobid_available:
            label = "GROBID only"
            parser_options.append(label)
            parser_descriptions[label] = "ML-based structured extraction (no LaTeX equations)"
        else:
            parser_options.append("GROBID (unavailable — start Docker)")

        parser_options.append("PyMuPDF")
        parser_descriptions["PyMuPDF"] = "Fast heuristic-based extraction (fallback)"

        parser_choice = st.radio(
            "PDF Parser",
            parser_options,
            index=0,
            horizontal=True,
            help=parser_descriptions.get(parser_options[0], ""),
        )

        if st.button("Ingest Paper", type="primary"):
            _ingest_paper(uploaded, parser=parser_choice)


def _ingest_paper(uploaded, parser: str = "Hybrid (recommended)") -> None:
    """Run the full ingestion pipeline on an uploaded PDF."""
    import re
    from papermind.ingestion.chunker import chunk_sections
    from papermind.ingestion.entity_extractor import extract_entities

    progress = st.progress(0, text="Saving file...")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(uploaded.getvalue())
        tmp_path = Path(f.name)

    try:
        # Parse — select parser
        if "Hybrid" in parser:
            from papermind.ingestion.hybrid_parser import parse_pdf_hybrid
            progress.progress(10, text="Parsing via GROBID + MinerU (hybrid)...")
            paper, sections = parse_pdf_hybrid(tmp_path)
        elif "GROBID" in parser:
            from papermind.ingestion.grobid_parser import parse_pdf
            progress.progress(10, text="Parsing via GROBID...")
            paper, sections = parse_pdf(tmp_path)
        else:
            from papermind.ingestion.pdf_parser import parse_pdf
            progress.progress(10, text="Parsing via PyMuPDF...")
            paper, sections = parse_pdf(tmp_path)

        # Count equations
        all_text = "\n".join(s.text for s in sections)
        display_eqs = len(re.findall(r'^\$\$', all_text, re.MULTILINE)) // 2
        inline_eqs = len(re.findall(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', all_text))
        has_latex = bool(re.search(r'\\frac|\\sum|\\int|\\mathrm|\\mathbf', all_text))

        # Chunk
        progress.progress(40, text="Chunking text...")
        chunks = chunk_sections(sections, paper.id)

        # Embed
        progress.progress(55, text="Embedding chunks...")
        from papermind.ui.shared import get_embedding_pipeline, get_knowledge_graph
        pipeline = get_embedding_pipeline()
        num_stored = pipeline.embed_and_store(chunks)
        paper.num_chunks = num_stored

        # Extract entities
        progress.progress(80, text="Extracting entities...")
        entities, relationships = extract_entities(sections, paper.id)
        kg = get_knowledge_graph()
        for entity in entities:
            kg.add_entity(entity)
        for rel in relationships:
            kg.add_relationship(rel)
        paper.num_entities = len(entities)

        # Store paper metadata
        if "papers" not in st.session_state:
            st.session_state.papers = {}
        st.session_state.papers[paper.id] = paper

        progress.progress(100, text="Done!")

        # Show results
        st.success(f"Ingested **{paper.title}**")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sections", len(sections))
        col2.metric("Chunks", num_stored)
        col3.metric("Entities", len(entities))
        col4.metric("Display Eqs", display_eqs)
        col5.metric("Inline Eqs", inline_eqs)

        if has_latex:
            st.caption("Equations extracted as LaTeX (via MinerU)")
        elif display_eqs or inline_eqs:
            st.caption("Equations extracted as Unicode symbols (GROBID only)")

        # Show extracted entities
        if entities:
            with st.expander(f"Extracted Entities ({len(entities)})"):
                for e in entities:
                    st.write(f"- **{e.name}** ({e.entity_type})")

        # Show sections with equation counts
        with st.expander(f"Sections ({len(sections)})"):
            for s in sections:
                sec_display = len(re.findall(r'^\$\$', s.text, re.MULTILINE)) // 2
                sec_inline = len(re.findall(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', s.text))
                eq_badge = ""
                if sec_display or sec_inline:
                    eq_badge = f" | {sec_display} display, {sec_inline} inline eqs"
                st.write(
                    f"**{s.title}** — {len(s.text)} chars{eq_badge}"
                )

        # Show sample equations if LaTeX
        if has_latex and display_eqs > 0:
            with st.expander(f"Sample Equations ({display_eqs} display)"):
                display_blocks = re.findall(
                    r'^\$\$(.*?)^\$\$', all_text, re.MULTILINE | re.DOTALL
                )
                for i, eq in enumerate(display_blocks[:8]):
                    st.latex(eq.strip())

    finally:
        tmp_path.unlink(missing_ok=True)


def _browse_tab() -> None:
    """Browse ingested papers."""
    papers = st.session_state.get("papers", {})

    if not papers:
        st.info("No papers ingested yet. Upload a PDF in the Upload tab.")
        return

    for pid, paper in papers.items():
        with st.container(border=True):
            st.subheader(paper.title or "Untitled")
            col1, col2, col3 = st.columns(3)
            col1.write(f"**ID:** `{pid}`")
            col2.write(f"**Pages:** {paper.num_pages}")
            col3.write(f"**Chunks:** {paper.num_chunks} | **Entities:** {paper.num_entities}")


