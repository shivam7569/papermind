"""Orchestrates the embedding pipeline: chunks → embeddings → vector store."""

from __future__ import annotations

from typing import Any

from papermind.infrastructure.embedding import EmbeddingService
from papermind.models import Chunk, SearchResult


class EmbeddingPipeline:
    """Embeds chunks and stores them in the vector store (ChromaDB or FAISS)."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: Any | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        if vector_store is None:
            from papermind.infrastructure.vector_store import VectorStore
            vector_store = VectorStore()
        self.vector_store = vector_store

    def embed_and_store(self, chunks: list[Chunk]) -> int:
        """Embed chunks and store them. Returns the number of chunks stored."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_documents(texts)
        self.vector_store.add_chunks(chunks, embeddings)

        # FAISS needs explicit save; ChromaDB auto-persists
        if hasattr(self.vector_store, "save"):
            self.vector_store.save()

        return len(chunks)

    def search(
        self, query: str, n_results: int = 10, paper_id: str | None = None
    ) -> list[SearchResult]:
        """Search for similar chunks by query text."""
        query_embedding = self.embedding_service.embed_query(query)
        return self.vector_store.search(query_embedding, n_results, paper_id)
