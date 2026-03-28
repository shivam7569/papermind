"""FAISS-backed vector store with pluggable index types.

Index types
-----------
- **Flat** (``IndexFlatIP``): Exact inner-product search.  Best recall,
  O(n) per query.  Good baseline and for corpora under ~50k vectors.
- **IVF** (``IndexIVFFlat``): Inverted-file index.  Clusters vectors into
  ``nlist`` Voronoi cells; at query time only ``nprobe`` cells are searched.
  Much faster than Flat at scale, small recall trade-off.
- **HNSW** (``IndexHNSWFlat``): Hierarchical Navigable Small World graph.
  Excellent recall-vs-speed at any scale.  Higher memory than IVF (stores
  the graph), but no training step required.

All three use **inner-product** similarity on L2-normalized vectors
(equivalent to cosine similarity).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from papermind.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"


@dataclass
class FaissConfig:
    """Configuration for the FAISS index."""

    index_type: IndexType = IndexType.HNSW
    persist_directory: str = "./data/faiss"

    # IVF parameters
    ivf_nlist: int = 100  # number of Voronoi cells
    ivf_nprobe: int = 10  # cells to visit at query time

    # HNSW parameters
    hnsw_m: int = 32  # edges per node (higher = better recall, more memory)
    hnsw_ef_construction: int = 200  # beam width during build
    hnsw_ef_search: int = 64  # beam width at query time


@dataclass
class _ChunkMeta:
    """Lightweight metadata stored alongside each vector."""

    chunk_id: str
    paper_id: str
    text: str
    section_title: str = ""
    page_start: int = 0
    page_end: int = 0
    token_count: int = 0


class FaissVectorStore:
    """FAISS vector store with Flat / IVF / HNSW index support.

    Vectors are assumed to be **L2-normalised** (unit length) so inner-product
    search is equivalent to cosine similarity.

    Persistence: the index file and a sidecar JSON metadata file are saved to
    ``persist_directory``.
    """

    def __init__(self, dimension: int, config: FaissConfig | None = None):
        self.config = config or FaissConfig()
        self.dimension = dimension
        self._persist_dir = Path(self.config.persist_directory)
        self._index: faiss.Index | None = None
        self._metadata: list[_ChunkMeta] = []

        # Try loading from disk first
        if self._index_path.exists():
            self._load()
        else:
            self._index = self._create_index()

    # ------------------------------------------------------------------
    # Index factory
    # ------------------------------------------------------------------

    def _create_index(self) -> faiss.Index:
        """Build a fresh FAISS index based on config."""
        idx_type = self.config.index_type

        if idx_type == IndexType.FLAT:
            index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created Flat (exact) index, dim=%d", self.dimension)

        elif idx_type == IndexType.IVF:
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.config.ivf_nlist, faiss.METRIC_INNER_PRODUCT
            )
            # IVF requires training before adding vectors
            logger.info(
                "Created IVF index, dim=%d, nlist=%d (needs training)",
                self.dimension,
                self.config.ivf_nlist,
            )

        elif idx_type == IndexType.HNSW:
            index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.config.hnsw_ef_construction
            index.hnsw.efSearch = self.config.hnsw_ef_search
            logger.info(
                "Created HNSW index, dim=%d, M=%d, efConstruction=%d",
                self.dimension,
                self.config.hnsw_m,
                self.config.hnsw_ef_construction,
            )

        else:
            raise ValueError(f"Unknown index type: {idx_type}")

        return index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def index(self) -> faiss.Index:
        assert self._index is not None
        return self._index

    @property
    def count(self) -> int:
        return self.index.ntotal

    @property
    def is_trained(self) -> bool:
        return self.index.is_trained

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: NDArray[np.float32],
    ) -> None:
        """Add chunks with their pre-computed embeddings.

        For IVF indexes, ``train()`` must be called first (or use
        ``add_chunks_with_training()`` which handles it automatically).
        """
        if len(chunks) == 0:
            return
        assert embeddings.shape == (len(chunks), self.dimension), (
            f"Expected ({len(chunks)}, {self.dimension}), got {embeddings.shape}"
        )

        # IVF: auto-train if needed
        if not self.index.is_trained:
            n_vectors = embeddings.shape[0]
            if n_vectors < self.config.ivf_nlist:
                logger.warning(
                    "Only %d vectors for %d clusters — reducing nlist to %d",
                    n_vectors,
                    self.config.ivf_nlist,
                    max(1, n_vectors // 4),
                )
                # Rebuild with fewer clusters
                self.config.ivf_nlist = max(1, n_vectors // 4)
                self._index = self._create_index()
            logger.info("Training IVF index on %d vectors...", n_vectors)
            self.index.train(embeddings)

        self.index.add(embeddings)

        for chunk in chunks:
            self._metadata.append(
                _ChunkMeta(
                    chunk_id=chunk.id,
                    paper_id=chunk.paper_id,
                    text=chunk.text,
                    section_title=chunk.section_title,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    token_count=chunk.token_count,
                )
            )

        logger.info("Added %d vectors (total: %d)", len(chunks), self.count)

    def search(
        self,
        query_embedding: NDArray[np.float32],
        n_results: int = 10,
        paper_id: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks by cosine similarity.

        Args:
            query_embedding: (dim,) normalised query vector.
            n_results: Number of results to return.
            paper_id: Optional filter — only return chunks from this paper.
        """
        if self.count == 0:
            return []

        query = query_embedding.reshape(1, -1)

        # If filtering by paper, over-fetch then filter
        fetch_k = n_results * 5 if paper_id else n_results

        # Set nprobe for IVF
        if self.config.index_type == IndexType.IVF:
            faiss.ParameterSpace().set_index_parameter(
                self.index, "nprobe", self.config.ivf_nprobe
            )

        scores, indices = self.index.search(query, min(fetch_k, self.count))

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS sentinel for no result
                continue
            meta = self._metadata[idx]

            if paper_id and meta.paper_id != paper_id:
                continue

            results.append(
                SearchResult(
                    chunk_id=meta.chunk_id,
                    text=meta.text,
                    score=float(score),  # inner product on unit vectors = cosine sim
                    paper_id=meta.paper_id,
                    section_title=meta.section_title,
                    metadata={
                        "page_start": meta.page_start,
                        "page_end": meta.page_end,
                        "token_count": meta.token_count,
                    },
                )
            )

            if len(results) >= n_results:
                break

        return results

    def delete_by_paper(self, paper_id: str) -> int:
        """Delete all vectors for a paper. Returns count removed.

        FAISS doesn't support deletion natively, so we rebuild the index
        without the deleted vectors.
        """
        keep_indices = [
            i for i, m in enumerate(self._metadata) if m.paper_id != paper_id
        ]
        removed = len(self._metadata) - len(keep_indices)
        if removed == 0:
            return 0

        if keep_indices:
            # Reconstruct vectors for kept items
            all_vectors = np.vstack(
                [self.index.reconstruct(i).reshape(1, -1) for i in keep_indices]
            ).astype(np.float32)
            kept_meta = [self._metadata[i] for i in keep_indices]
        else:
            all_vectors = np.empty((0, self.dimension), dtype=np.float32)
            kept_meta = []

        # Rebuild
        self._index = self._create_index()
        self._metadata = kept_meta
        if len(kept_meta) > 0:
            if not self.index.is_trained:
                self.index.train(all_vectors)
            self.index.add(all_vectors)

        logger.info("Deleted %d vectors for paper %s", removed, paper_id)
        return removed

    def get_index_stats(self) -> dict[str, Any]:
        """Return stats about the current index for benchmarking/display."""
        stats: dict[str, Any] = {
            "index_type": self.config.index_type.value,
            "dimension": self.dimension,
            "total_vectors": self.count,
            "is_trained": self.is_trained,
        }
        if self.config.index_type == IndexType.IVF:
            stats["nlist"] = self.config.ivf_nlist
            stats["nprobe"] = self.config.ivf_nprobe
        elif self.config.index_type == IndexType.HNSW:
            stats["M"] = self.config.hnsw_m
            stats["efConstruction"] = self.config.hnsw_ef_construction
            stats["efSearch"] = self.config.hnsw_ef_search
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self._persist_dir / "index.faiss"

    @property
    def _meta_path(self) -> Path:
        return self._persist_dir / "metadata.json"

    def save(self) -> None:
        """Persist index and metadata to disk."""
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path))

        meta_dicts = [
            {
                "chunk_id": m.chunk_id,
                "paper_id": m.paper_id,
                "text": m.text,
                "section_title": m.section_title,
                "page_start": m.page_start,
                "page_end": m.page_end,
                "token_count": m.token_count,
            }
            for m in self._metadata
        ]
        self._meta_path.write_text(json.dumps(meta_dicts, ensure_ascii=False))
        logger.info(
            "Saved FAISS index (%d vectors) to %s", self.count, self._persist_dir
        )

    def _load(self) -> None:
        """Load index and metadata from disk."""
        self._index = faiss.read_index(str(self._index_path))
        meta_dicts = json.loads(self._meta_path.read_text())
        self._metadata = [_ChunkMeta(**d) for d in meta_dicts]
        logger.info(
            "Loaded FAISS index (%d vectors, type=%s) from %s",
            self.count,
            self.config.index_type.value,
            self._persist_dir,
        )
