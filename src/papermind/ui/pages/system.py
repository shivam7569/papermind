"""System page — health, VRAM, configuration, and diagnostics."""

import streamlit as st


def render() -> None:
    st.header("System")

    tab_status, tab_config, tab_gpu = st.tabs(["Status", "Configuration", "GPU / VRAM"])

    with tab_status:
        _status_tab()

    with tab_config:
        _config_tab()

    with tab_gpu:
        _gpu_tab()


def _status_tab() -> None:
    """Show system component status."""
    st.subheader("Component Status")

    # Vector store
    try:
        from papermind.api.dependencies import _create_vector_store
        vs = _create_vector_store()
        count = vs.count if isinstance(vs.count, int) else vs.count()
        backend = type(vs).__name__
        st.success(f"Vector Store ({backend}) — {count} chunks")
    except Exception as e:
        st.error(f"Vector Store — {e}")

    # Knowledge graph
    try:
        from papermind.infrastructure.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        st.success(
            f"Knowledge Graph (SQLite) — "
            f"{kg.count_entities()} entities, {kg.count_relationships()} relationships"
        )
    except Exception as e:
        st.error(f"Knowledge Graph — {e}")

    # GROBID
    try:
        from papermind.ingestion.grobid_parser import check_grobid_health
        if check_grobid_health():
            st.success("GROBID — running at localhost:8070")
        else:
            st.warning("GROBID — not available (PDF parsing will fall back to PyMuPDF)")
    except Exception as e:
        st.warning(f"GROBID — {e}")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            st.success(f"GPU — {name} ({total:.1f} GB)")
        else:
            st.warning("GPU — CUDA not available")
    except Exception as e:
        st.error(f"GPU — {e}")

    # Local model
    model = st.session_state.get("model")
    if model is not None and model.is_loaded:
        usage = model.vram_usage()
        st.success(f"Local Model — loaded, {usage['allocated_gb']} GB VRAM")
    else:
        st.info("Local Model — not loaded (load from Chat page)")

    # Embedding model
    try:
        from papermind.infrastructure.embedding import EmbeddingService
        emb = EmbeddingService()
        dim = emb.dimension
        st.success(f"Embedding Model — {emb._model_name} ({dim}d)")
    except Exception as e:
        st.error(f"Embedding Model — {e}")


def _config_tab() -> None:
    """Show current configuration."""
    st.subheader("Current Configuration")

    from papermind.config import get_settings
    settings = get_settings()

    sections = {
        "LLM": settings.llm,
        "Embedding": settings.embedding,
        "Vector Store": settings.vector_store,
        "Knowledge Graph": settings.knowledge_graph,
        "Chunking": settings.chunking,
        "API": settings.api,
        "Ingestion": settings.ingestion,
    }

    for name, section in sections.items():
        with st.expander(name, expanded=name == "LLM"):
            st.json(section.model_dump())


def _gpu_tab() -> None:
    """Show detailed GPU and VRAM information."""
    st.subheader("GPU / VRAM Monitor")

    try:
        import torch
        if not torch.cuda.is_available():
            st.warning("CUDA not available")
            return

        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        free_gb = total_gb - reserved_gb

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", f"{total_gb:.1f} GB")
        col2.metric("Allocated", f"{allocated_gb:.1f} GB")
        col3.metric("Reserved", f"{reserved_gb:.1f} GB")
        col4.metric("Free", f"{free_gb:.1f} GB")

        # Usage bar
        usage_pct = reserved_gb / total_gb
        st.progress(min(usage_pct, 1.0), text=f"VRAM Usage: {usage_pct:.0%}")

        # Device info
        with st.expander("Device Details"):
            st.write(f"**Device:** {torch.cuda.get_device_name(0)}")
            st.write(f"**Compute Capability:** {props.major}.{props.minor}")
            st.write(f"**SM Count:** {props.multi_processor_count}")
            st.write(f"**Total VRAM:** {total_gb:.2f} GB")
            st.write(f"**PyTorch version:** {torch.__version__}")
            st.write(f"**CUDA version:** {torch.version.cuda}")

        # Memory breakdown if model is loaded
        model = st.session_state.get("model")
        if model is not None and model.is_loaded:
            st.subheader("Model Memory Breakdown")
            st.write(f"**Model:** {model.model_name}")
            st.write(f"**Quantization:** {model._quantization}")
            st.write(f"**Double quantization:** {model._double_quant}")
            st.write(f"**Compute dtype:** {model._compute_dtype}")

            # Estimate breakdown
            model_gb = allocated_gb
            kv_headroom = free_gb
            st.write(f"**Model weights:** ~{model_gb:.1f} GB")
            st.write(f"**Available for KV cache + LoRA:** ~{kv_headroom:.1f} GB")

        if st.button("Refresh"):
            st.rerun()

    except Exception as e:
        st.error(f"Error reading GPU info: {e}")
