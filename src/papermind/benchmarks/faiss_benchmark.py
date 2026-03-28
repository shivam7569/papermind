"""Benchmark FAISS index types: Flat vs IVF vs HNSW.

Measures:
  - Index build time
  - Memory usage (bytes)
  - Query latency (single + batch)
  - Recall@K vs exact Flat search (ground truth)

Can run with real paper embeddings or synthetic data.

Usage:
    python -m papermind.benchmarks.faiss_benchmark [--n-vectors 5000] [--dim 768] [--use-real]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single index type."""

    index_type: str
    n_vectors: int
    dimension: int
    build_time_ms: float
    memory_bytes: int
    single_query_ms: float  # median over n_queries
    batch_query_ms: float  # total for n_queries at once
    recall_at_1: float  # vs Flat ground truth
    recall_at_5: float
    recall_at_10: float
    params: dict = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Full benchmark results across all index types."""

    results: list[BenchmarkResult]
    n_queries: int
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _generate_data(
    n_vectors: int, dim: int, n_queries: int = 100
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate random L2-normalised vectors for benchmarking."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    faiss.normalize_L2(queries)
    return vectors, queries


def _get_real_embeddings() -> NDArray[np.float32] | None:
    """Try to load real embeddings from existing FAISS index."""
    try:
        from papermind.config import get_settings

        settings = get_settings()
        index_path = Path(settings.vector_store.faiss_directory) / "index.faiss"
        if not index_path.exists():
            return None
        index = faiss.read_index(str(index_path))
        if index.ntotal == 0:
            return None
        vectors = np.vstack(
            [index.reconstruct(i).reshape(1, -1) for i in range(index.ntotal)]
        ).astype(np.float32)
        logger.info("Loaded %d real embeddings from %s", vectors.shape[0], index_path)
        return vectors
    except Exception:
        return None


def _build_flat(dim: int) -> faiss.Index:
    return faiss.IndexFlatIP(dim)


def _build_ivf(
    dim: int, nlist: int = 100, nprobe: int = 10
) -> tuple[faiss.Index, dict]:
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    params = {"nlist": nlist, "nprobe": nprobe}
    return index, params


def _build_hnsw(
    dim: int, M: int = 32, ef_construction: int = 200, ef_search: int = 64
) -> tuple[faiss.Index, dict]:
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    params = {"M": M, "efConstruction": ef_construction, "efSearch": ef_search}
    return index, params


def _measure_memory(index: faiss.Index) -> int:
    """Estimate index memory in bytes using FAISS's own accounting."""
    # Write to a temp buffer to get exact serialized size
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return writer.data.size()


def _compute_recall(
    ground_truth: NDArray[np.int64],
    predictions: NDArray[np.int64],
    k: int,
) -> float:
    """Recall@K: fraction of ground-truth top-K present in predictions top-K."""
    assert ground_truth.shape[0] == predictions.shape[0]
    n = ground_truth.shape[0]
    recall = 0.0
    for i in range(n):
        gt_set = set(ground_truth[i, :k].tolist())
        pred_set = set(predictions[i, :k].tolist())
        recall += len(gt_set & pred_set) / k
    return recall / n


def benchmark_index(
    index_type: str,
    vectors: NDArray[np.float32],
    queries: NDArray[np.float32],
    ground_truth_ids: NDArray[np.int64],
    **kwargs,
) -> BenchmarkResult:
    """Benchmark a single index type."""
    n, dim = vectors.shape
    n_queries = queries.shape[0]
    k = 10  # top-K for recall

    # Build index
    params: dict = {}
    if index_type == "flat":
        index = _build_flat(dim)
    elif index_type == "ivf":
        nlist = min(kwargs.get("nlist", 100), max(1, n // 4))
        nprobe = kwargs.get("nprobe", 10)
        index, params = _build_ivf(dim, nlist=nlist, nprobe=nprobe)
    elif index_type == "hnsw":
        index, params = _build_hnsw(
            dim,
            M=kwargs.get("M", 32),
            ef_construction=kwargs.get("ef_construction", 200),
            ef_search=kwargs.get("ef_search", 64),
        )
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Time: build (train + add)
    t0 = time.perf_counter()
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    build_ms = (time.perf_counter() - t0) * 1000

    # Set nprobe for IVF queries
    if index_type == "ivf":
        faiss.ParameterSpace().set_index_parameter(
            index, "nprobe", params.get("nprobe", 10)
        )

    # Memory
    mem_bytes = _measure_memory(index)

    # Time: single queries (measure each independently, take median)
    latencies = []
    for i in range(n_queries):
        q = queries[i : i + 1]
        t0 = time.perf_counter()
        index.search(q, k)
        latencies.append((time.perf_counter() - t0) * 1000)
    single_query_ms = float(np.median(latencies))

    # Time: batch query
    t0 = time.perf_counter()
    scores, ids = index.search(queries, k)
    batch_ms = (time.perf_counter() - t0) * 1000

    # Recall vs ground truth
    recall_1 = _compute_recall(ground_truth_ids, ids, 1)
    recall_5 = _compute_recall(ground_truth_ids, ids, 5)
    recall_10 = _compute_recall(ground_truth_ids, ids, 10)

    return BenchmarkResult(
        index_type=index_type,
        n_vectors=n,
        dimension=dim,
        build_time_ms=round(build_ms, 2),
        memory_bytes=mem_bytes,
        single_query_ms=round(single_query_ms, 4),
        batch_query_ms=round(batch_ms, 2),
        recall_at_1=round(recall_1, 4),
        recall_at_5=round(recall_5, 4),
        recall_at_10=round(recall_10, 4),
        params=params,
    )


def run_benchmark(
    n_vectors: int = 5000,
    dim: int = 768,
    n_queries: int = 100,
    use_real: bool = False,
    ivf_configs: list[dict] | None = None,
    hnsw_configs: list[dict] | None = None,
) -> BenchmarkSuite:
    """Run the full benchmark suite across all index types.

    Args:
        n_vectors: Number of vectors (ignored if use_real=True and data exists).
        dim: Embedding dimension (ignored if use_real=True).
        n_queries: Number of test queries.
        use_real: Try to load real embeddings from FAISS store.
        ivf_configs: List of IVF param dicts to test (default: one config).
        hnsw_configs: List of HNSW param dicts to test (default: one config).
    """
    from datetime import datetime, timezone

    # Load or generate data
    real = _get_real_embeddings() if use_real else None
    if real is not None:
        n_vectors = real.shape[0]
        dim = real.shape[1]
        rng = np.random.default_rng(42)
        query_idx = rng.choice(n_vectors, size=min(n_queries, n_vectors), replace=False)
        queries = real[query_idx]
        vectors = real
        logger.info("Using %d real embeddings (dim=%d)", n_vectors, dim)
    else:
        vectors, queries = _generate_data(n_vectors, dim, n_queries)
        logger.info("Using %d synthetic embeddings (dim=%d)", n_vectors, dim)

    # Ground truth: exact search with Flat
    flat_index = _build_flat(dim)
    flat_index.add(vectors)
    _, gt_ids = flat_index.search(queries, 10)

    # Default configs
    if ivf_configs is None:
        nlist = min(100, max(1, n_vectors // 4))
        ivf_configs = [
            {"nlist": nlist, "nprobe": 1},
            {"nlist": nlist, "nprobe": 10},
            {"nlist": nlist, "nprobe": 50},
        ]
    if hnsw_configs is None:
        hnsw_configs = [
            {"M": 16, "ef_construction": 100, "ef_search": 32},
            {"M": 32, "ef_construction": 200, "ef_search": 64},
            {"M": 64, "ef_construction": 400, "ef_search": 128},
        ]

    results: list[BenchmarkResult] = []

    # Flat (baseline)
    results.append(benchmark_index("flat", vectors, queries, gt_ids))
    logger.info("Flat: build=%.1fms, query=%.3fms, recall@10=%.4f",
                results[-1].build_time_ms, results[-1].single_query_ms,
                results[-1].recall_at_10)

    # IVF variants
    for cfg in ivf_configs:
        results.append(benchmark_index("ivf", vectors, queries, gt_ids, **cfg))
        r = results[-1]
        logger.info("IVF(nlist=%d,nprobe=%d): build=%.1fms, query=%.3fms, recall@10=%.4f",
                     cfg.get("nlist", 100), cfg.get("nprobe", 10),
                     r.build_time_ms, r.single_query_ms, r.recall_at_10)

    # HNSW variants
    for cfg in hnsw_configs:
        results.append(benchmark_index("hnsw", vectors, queries, gt_ids, **cfg))
        r = results[-1]
        logger.info("HNSW(M=%d,ef=%d): build=%.1fms, query=%.3fms, recall@10=%.4f",
                     cfg.get("M", 32), cfg.get("ef_search", 64),
                     r.build_time_ms, r.single_query_ms, r.recall_at_10)

    suite = BenchmarkSuite(
        results=results,
        n_queries=len(queries),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return suite


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Benchmark FAISS index types")
    parser.add_argument("--n-vectors", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--use-real", action="store_true")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json")
    args = parser.parse_args()

    suite = run_benchmark(
        n_vectors=args.n_vectors,
        dim=args.dim,
        n_queries=args.n_queries,
        use_real=args.use_real,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    suite.save(args.output)

    # Print summary table
    print(f"\n{'Index':<25} {'Build(ms)':>10} {'Query(ms)':>10} "
          f"{'Mem(KB)':>10} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 90)
    for r in suite.results:
        label = r.index_type
        if r.params:
            p = ",".join(f"{k}={v}" for k, v in r.params.items())
            label = f"{r.index_type}({p})"
        print(f"{label:<25} {r.build_time_ms:>10.1f} {r.single_query_ms:>10.3f} "
              f"{r.memory_bytes / 1024:>10.0f} {r.recall_at_1:>8.4f} "
              f"{r.recall_at_5:>8.4f} {r.recall_at_10:>8.4f}")


if __name__ == "__main__":
    main()
