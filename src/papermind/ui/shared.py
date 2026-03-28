"""Shared resources for Streamlit pages.

Thin wrappers around the centralized service registry.
No caching here — the registry handles singleton lifecycle.
"""

from papermind.services import services


def get_vector_store():
    return services.vector_store


def get_knowledge_graph():
    return services.knowledge_graph


def get_embedding_service():
    return services.embedding_service


def get_embedding_pipeline():
    return services.embedding_pipeline


def get_paper_store():
    return services.paper_store
