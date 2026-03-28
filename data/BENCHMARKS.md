# PaperMind Benchmark Report

**Generated:** 2026-03-28T20:07:10
**System:** NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227 MiB, 580.119.02, 12.0 | Python 3.12.3 | Linux 6.17.9-76061709-generic

---

## 1. Model Loading & Generation (Qwen2.5-Coder-7B, NF4)

| Metric | Value |
|--------|-------|
| **Model** | Qwen/Qwen2.5-Coder-7B-Instruct |
| **Quantization** | NF4 (bitsandbytes, double quant, bfloat16 compute) |
| **Load time** | 9.3s |
| **VRAM allocated** | 5.18 GB |
| **VRAM free** | 5.57 GB / 11.5 GB total |

### Generation Performance

| Task | Time | Output | Speed |
|------|------|--------|-------|
| Simple Q&A | 2.1s | 64 words | ~30 words/s |
| Code generation | 5.1s | 129 words (typed Python + docstrings) | ~25 words/s |
| Streaming | 2.1s | 101 tokens | **48.9 tok/s** |

---

## 2. Vector Store Benchmark

**Corpus:** 4102 vectors × 768d from 50 AI papers
**Queries:** 200 (real embeddings + Gaussian noise σ=0.05)
**Embedding model:** nomic-ai/nomic-embed-text-v1.5 (768d, L2-normalized)

| Index | Build | Memory | Latency | QPS | R@1 | R@5 | R@10 |
|-------|-------|--------|---------|-----|-----|-----|------|
| Flat (exact) | 0.004s | 12.0MB | 0.289ms | 3,770 | 1.0000 | 1.0000 | 1.0000 |
| IVF(nlist=64,nprobe=8) | 0.136s | 12.2MB | 0.044ms | 44,181 | 0.9950 | 0.9560 | 0.9360 |
| IVF(nlist=128,nprobe=16) | 0.194s | 12.4MB | 0.049ms | 104,544 | 1.0000 | 0.9720 | 0.9600 |
| IVF(nlist=256,nprobe=32) | 0.351s | 12.8MB | 0.057ms | 113,806 | 1.0000 | 0.9870 | 0.9780 |
| HNSW(M=16,ef_c=100,ef_s=32) | 0.043s | 12.6MB | 0.029ms | 24,925 | 0.3750 | 0.9190 | 0.9640 |
| HNSW(M=32,ef_c=200,ef_s=64) | 0.06s | 13.1MB | 0.061ms | 192,047 | 0.3750 | 0.9230 | 0.9750 |
| HNSW(M=64,ef_c=400,ef_s=128) | 0.093s | 14.1MB | 0.125ms | 38,757 | 0.3850 | 0.9260 | 0.9810 |
| ChromaDB (HNSW/cosine) | 0.755s | 17.5MB | 0.588ms | 1,854 | N/A | N/A | N/A |

### Recommendations

- **Current scale (<5K vectors):** Use **Flat** (exact search). Perfect recall, sub-millisecond latency. No reason for approximate indices.
- **Medium scale (50K-500K):** Switch to **IVF(nlist=256, nprobe=32)**. 113K QPS with 97.8% R@10.
- **ChromaDB** is 2x slower than Flat FAISS but provides persistence + metadata filtering out of the box.
- **HNSW** shows low R@1 (0.38) on our clustered academic vectors. Better suited for diverse corpora.

---

## 3. Matryoshka Dimension Impact

nomic-embed-text-v1.5 supports Matryoshka representation learning — truncating embeddings to lower dimensions while maintaining quality.

| Dimension | Memory | Latency | QPS | Recall vs 768d | Savings |
|-----------|--------|---------|-----|----------------|---------|
| **768** (full) | 12.0 MB | 0.252ms | 4,334 | baseline | 0% |
| 512 | 8.0 MB | 0.173ms | 6,234 | 53.9% | 33% |
| 256 | 4.0 MB | 0.071ms | 27,101 | 54.4% | 67% |

**Recommendation:** Stay with **768d**. Truncation to 512d or 256d drops recall to ~54% — unacceptable for RAG quality. The memory savings (4-8MB) are negligible at our scale.

---

## 4. RAG Pipeline

| Stage | Component | VRAM | Time |
|-------|-----------|------|------|
| Retrieval | nomic-embed-text-v1.5 (bi-encoder) | 0.09 GB | <1ms |
| Reranking | BAAI/bge-reranker-v2-m3 (cross-encoder, 568M, FP32) | 2.3 GB | ~2s |
| Context | Lost-in-middle ordering + dedup + compression | 0 | <1ms |
| Generation | Qwen2.5-Coder-7B (NF4) | 5.2 GB | 2-5s |

**VRAM management:** Reranker unloads before LLM loads. Peak VRAM: 5.2 GB. Headroom: 6.3 GB.

---

## 5. PDF Parsing

**Pipeline:** GROBID (metadata) + MinerU (body text + LaTeX equations)
**Test corpus:** 50 landmark AI papers

| Metric | GROBID Only | Hybrid (GROBID + MinerU) |
|--------|------------|--------------------------|
| Papers parsed | 50/50 | 50/50 |
| Display equations | 0 | **549** |
| Inline equations | 728 (Unicode) | **9,598** (LaTeX) |
| Papers with LaTeX | 0 | **49/50** |
| Avg parse time | 2.0s | 35.4s |

---

## 6. VRAM Budget (12 GB RTX 5070 Ti)

| Component | VRAM | When |
|-----------|------|------|
| Qwen2.5-Coder-7B (NF4) | 5.2 GB | Generation time |
| BGE-Reranker-v2-m3 (FP32) | 2.3 GB | Reranking time (unloaded before LLM) |
| nomic-embed-text-v1.5 | 0.09 GB | Always loaded |
| MinerU (pipeline models) | ~6 GB | Ingestion time only |
| **Peak usage** | **5.3 GB** | Query time (embed + LLM) |
| **Available** | **11.5 GB** | |
| **Headroom** | **6.2 GB** | |
