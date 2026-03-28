"""ChromaDB vector store for paper chunk retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
import numpy as np

from papermind.config import get_settings
from papermind.models import Chunk, SearchResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VectorStore:
    """ChromaDB-backed vector store for paper chunks."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ):
        settings = get_settings()
        self._persist_dir = persist_directory or settings.vector_store.persist_directory
        self._collection_name = collection_name or settings.vector_store.collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: NDArray[np.float32] | list[list[float]],
    ) -> None:
        """Add chunks with precomputed embeddings to the store."""
        if not chunks:
            return
        # ChromaDB expects list-of-lists
        emb_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=emb_list,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "paper_id": c.paper_id,
                    "section_title": c.section_title,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "token_count": c.token_count,
                }
                for c in chunks
            ],
        )

    def search(
        self,
        query_embedding: NDArray[np.float32] | list[float],
        n_results: int = 10,
        paper_id: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks. Optionally filter by paper_id."""
        emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        where = {"paper_id": paper_id} if paper_id else None
        results = self.collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                doc = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0.0
                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        text=doc,
                        score=1.0 - dist,  # cosine distance to similarity
                        paper_id=meta.get("paper_id", ""),
                        section_title=meta.get("section_title", ""),
                        metadata=meta,
                    )
                )
        return search_results

    def delete_by_paper(self, paper_id: str) -> None:
        """Delete all chunks for a given paper."""
        self.collection.delete(where={"paper_id": paper_id})

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self.collection.count()
