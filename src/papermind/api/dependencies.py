"""FastAPI dependency injection — delegates to centralized service registry."""

from __future__ import annotations

from papermind.services import services


def get_embedding_service():
    return services.embedding_service


def get_vector_store():
    return services.vector_store


def get_knowledge_graph():
    return services.knowledge_graph


def get_llm_client():
    return services.llm_client


def get_embedding_pipeline():
    return services.embedding_pipeline


def get_paper_store():
    return services.paper_store
