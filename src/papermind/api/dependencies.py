"""FastAPI dependency injection for shared services."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from papermind.config import get_settings
from papermind.infrastructure.embedding import EmbeddingService
from papermind.infrastructure.knowledge_graph import KnowledgeGraph
from papermind.infrastructure.llm_client import LLMClient
from papermind.ingestion.embedder import EmbeddingPipeline

if TYPE_CHECKING:
    from papermind.infrastructure.faiss_store import FaissVectorStore
    from papermind.infrastructure.vector_store import VectorStore


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(matryoshka_dim=settings.embedding.matryoshka_dim)


@lru_cache
def get_vector_store() -> VectorStore | FaissVectorStore:
    """Return the configured vector store backend."""
    return _create_vector_store()


@lru_cache
def get_knowledge_graph() -> KnowledgeGraph:
    return KnowledgeGraph()


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()


@lru_cache
def get_embedding_pipeline() -> EmbeddingPipeline:
    return EmbeddingPipeline(get_embedding_service(), get_vector_store())


def _create_vector_store() -> VectorStore | FaissVectorStore:
    """Factory: build the right vector store from settings."""
    settings = get_settings()
    if settings.vector_store.backend == "faiss":
        from papermind.infrastructure.faiss_store import (
            FaissConfig,
            FaissVectorStore,
            IndexType,
        )

        emb = get_embedding_service()
        config = FaissConfig(
            index_type=IndexType(settings.vector_store.faiss_index_type),
            persist_directory=settings.vector_store.faiss_directory,
            ivf_nlist=settings.vector_store.faiss_ivf_nlist,
            ivf_nprobe=settings.vector_store.faiss_ivf_nprobe,
            hnsw_m=settings.vector_store.faiss_hnsw_m,
            hnsw_ef_construction=settings.vector_store.faiss_hnsw_ef_construction,
            hnsw_ef_search=settings.vector_store.faiss_hnsw_ef_search,
        )
        return FaissVectorStore(dimension=emb.dimension, config=config)
    else:
        from papermind.infrastructure.vector_store import VectorStore

        return VectorStore()
