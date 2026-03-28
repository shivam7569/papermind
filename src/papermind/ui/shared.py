"""Shared cached resources for all Streamlit pages.

All pages must import from here instead of creating their own cached instances.
This ensures the sidebar, Papers, Search, and KG tabs all see the same data.
"""

import streamlit as st


@st.cache_resource(show_spinner="Loading vector store...")
def get_vector_store():
    from papermind.api.dependencies import _create_vector_store
    return _create_vector_store()


@st.cache_resource(show_spinner="Loading knowledge graph...")
def get_knowledge_graph():
    from papermind.infrastructure.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph()


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_service():
    from papermind.infrastructure.embedding import EmbeddingService
    from papermind.config import get_settings
    settings = get_settings()
    return EmbeddingService(matryoshka_dim=settings.embedding.matryoshka_dim)


@st.cache_resource(show_spinner="Loading embedding pipeline...")
def get_embedding_pipeline():
    from papermind.ingestion.embedder import EmbeddingPipeline
    return EmbeddingPipeline(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )
