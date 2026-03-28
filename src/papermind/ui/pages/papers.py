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
    from papermind.models import make_paper_id

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(uploaded.getvalue())
        tmp_path = Path(f.name)

    # Generate deterministic paper ID from content
    content_paper_id = make_paper_id(tmp_path)

    try:
        with st.status("Ingesting paper...", expanded=True) as status:
            # Parse — select parser
            if "Hybrid" in parser:
                from papermind.ingestion.hybrid_parser import parse_pdf_hybrid
                status.write("Parsing via GROBID + MinerU hybrid (~30s)...")
                paper, sections = parse_pdf_hybrid(tmp_path)
                status.write(f"Parsed: {paper.title}")
            elif "GROBID" in parser:
                from papermind.ingestion.grobid_parser import parse_pdf
                status.write("Parsing via GROBID...")
                paper, sections = parse_pdf(tmp_path)
                status.write(f"Parsed: {paper.title}")
            else:
                from papermind.ingestion.pdf_parser import parse_pdf
                status.write("Parsing via PyMuPDF...")
                paper, sections = parse_pdf(tmp_path)
                status.write(f"Parsed: {paper.title}")

            # Override with deterministic ID from content hash
            paper.id = content_paper_id

            # Clean old data if re-ingesting the same paper
            from papermind.ui.shared import get_embedding_pipeline, get_knowledge_graph, get_paper_store
            pipeline = get_embedding_pipeline()
            kg = get_knowledge_graph()
            paper_store = get_paper_store()

            existing = paper_store.get(paper.id)
            if existing:
                status.write("Re-ingesting — clearing old data...")
                pipeline.vector_store.delete_by_paper(paper.id)
                kg.delete_by_paper(paper.id)

            # Count equations
            all_text = "\n".join(s.text for s in sections)
            display_eqs = len(re.findall(r'^\$\$', all_text, re.MULTILINE)) // 2
            inline_eqs = len(re.findall(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', all_text))
            has_latex = bool(re.search(r'\\frac|\\sum|\\int|\\mathrm|\\mathbf', all_text))

            # Chunk
            status.write(f"Chunking {len(sections)} sections...")
            chunks = chunk_sections(sections, paper.id)
            status.write(f"Created {len(chunks)} chunks")

            # Embed
            status.write("Embedding chunks (loading model on first use)...")
            num_stored = pipeline.embed_and_store(chunks)
            paper.num_chunks = num_stored
            status.write(f"Stored {num_stored} chunks")

            # Extract entities
            status.write("Extracting entities...")
            entities, relationships = extract_entities(sections, paper.id)
            for entity in entities:
                kg.add_entity(entity)
            for rel in relationships:
                kg.add_relationship(rel)
            paper.num_entities = len(entities)

            # Persist paper metadata to SQLite (survives restarts)
            paper_store.save(paper)

            status.update(label="Ingestion complete!", state="complete", expanded=False)

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
    """Browse ingested papers from persistent store."""
    from papermind.ui.shared import get_paper_store

    paper_store = get_paper_store()
    papers = paper_store.list_all()

    if not papers:
        st.info("No papers ingested yet. Upload a PDF in the Upload tab.")
        return

    st.write(f"**{len(papers)} papers** ingested")

    for paper in papers:
        with st.container(border=True):
            st.subheader(paper.title or "Untitled")
            col1, col2, col3 = st.columns(3)
            col1.write(f"**ID:** `{paper.id}`")
            col2.write(f"**Chunks:** {paper.num_chunks}")
            col3.write(f"**Entities:** {paper.num_entities}")
            if paper.authors:
                st.caption(", ".join(paper.authors[:5]))


