"""Centralized service registry — single source of truth for all shared state.

Both the Streamlit UI and FastAPI backend import from here.
Each service is lazily initialized on first access and then reused.
No duplicate instances, no cache inconsistencies.

Usage:
    from papermind.services import services
    vs = services.vector_store
    kg = services.knowledge_graph
    pipeline = services.embedding_pipeline
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Lazy singleton registry for all shared services.

    Thread-safe: uses a lock to prevent duplicate initialization
    when Streamlit or FastAPI spawn multiple threads.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._embedding_service = None
        self._vector_store = None
        self._knowledge_graph = None
        self._embedding_pipeline = None
        self._paper_store = None
        self._llm_client = None

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            with self._lock:
                if self._embedding_service is None:
                    from papermind.config import get_settings
                    from papermind.infrastructure.embedding import EmbeddingService
                    settings = get_settings()
                    logger.info("Initializing embedding service...")
                    self._embedding_service = EmbeddingService(
                        matryoshka_dim=settings.embedding.matryoshka_dim
                    )
        return self._embedding_service

    @property
    def vector_store(self):
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    from papermind.config import get_settings
                    settings = get_settings()
                    logger.info("Initializing vector store (backend=%s)...",
                                settings.vector_store.backend)
                    if settings.vector_store.backend == "faiss":
                        from papermind.infrastructure.faiss_store import (
                            FaissConfig, FaissVectorStore, IndexType,
                        )
                        config = FaissConfig(
                            index_type=IndexType(settings.vector_store.faiss_index_type),
                            persist_directory=settings.vector_store.faiss_directory,
                            ivf_nlist=settings.vector_store.faiss_ivf_nlist,
                            ivf_nprobe=settings.vector_store.faiss_ivf_nprobe,
                            hnsw_m=settings.vector_store.faiss_hnsw_m,
                            hnsw_ef_construction=settings.vector_store.faiss_hnsw_ef_construction,
                            hnsw_ef_search=settings.vector_store.faiss_hnsw_ef_search,
                        )
                        self._vector_store = FaissVectorStore(
                            dimension=self.embedding_service.dimension, config=config,
                        )
                    else:
                        from papermind.infrastructure.vector_store import VectorStore
                        self._vector_store = VectorStore()
        return self._vector_store

    @property
    def knowledge_graph(self):
        if self._knowledge_graph is None:
            with self._lock:
                if self._knowledge_graph is None:
                    from papermind.infrastructure.knowledge_graph import KnowledgeGraph
                    logger.info("Initializing knowledge graph...")
                    self._knowledge_graph = KnowledgeGraph()
        return self._knowledge_graph

    @property
    def embedding_pipeline(self):
        if self._embedding_pipeline is None:
            with self._lock:
                if self._embedding_pipeline is None:
                    from papermind.ingestion.embedder import EmbeddingPipeline
                    logger.info("Initializing embedding pipeline...")
                    self._embedding_pipeline = EmbeddingPipeline(
                        embedding_service=self.embedding_service,
                        vector_store=self.vector_store,
                    )
        return self._embedding_pipeline

    @property
    def paper_store(self):
        if self._paper_store is None:
            with self._lock:
                if self._paper_store is None:
                    from papermind.infrastructure.paper_store import PaperStore
                    logger.info("Initializing paper store...")
                    self._paper_store = PaperStore()
        return self._paper_store

    @property
    def llm_client(self):
        if self._llm_client is None:
            with self._lock:
                if self._llm_client is None:
                    from papermind.infrastructure.llm_client import LLMClient
                    logger.info("Initializing LLM client...")
                    self._llm_client = LLMClient()
        return self._llm_client


# Global singleton — import this everywhere
services = ServiceRegistry()
