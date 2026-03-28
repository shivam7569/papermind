# PaperMind — Complete Project Context

> This file is the single source of truth for the entire project.
> It is loaded into every Claude Code conversation. When context compacts,
> this file ensures no knowledge is lost.

## What This Project Is

PaperMind is a **local AI research system** that runs entirely on a single NVIDIA RTX 5070 Ti (12GB VRAM). It ingests research papers (PDF), extracts structured content including LaTeX equations, builds a knowledge graph, embeds everything into a vector store, and provides RAG-grounded chat powered by a locally-running Qwen2.5-Coder-7B model.

No cloud APIs. No data leaves the machine.

**Owner:** Shivam (GitHub: shivam7569)
**Repo:** https://github.com/shivam7569/papermind

---

## Current State (as of 2026-03-28)

### What's Built and Working
- Hybrid PDF parsing (GROBID metadata + MinerU LaTeX equations)
- Parent-child chunking with section-aware splitting
- ChromaDB vector store with nomic-embed-text-v1.5 (768d)
- SQLite knowledge graph (entities + relationships)
- Full RAG pipeline: retrieve → rerank (BGE-v2-m3) → lost-in-middle → compress → generate
- Local LLM: Qwen2.5-Coder-7B NF4 quantized (5.2GB VRAM, 48.9 tok/s)
- Streamlit UI (Chat, Papers, Search, KG, Benchmarks, System)
- FastAPI backend with all endpoints
- Centralized ServiceRegistry (thread-safe singletons)
- Deterministic paper IDs (SHA-256 of PDF content)
- Persistent paper metadata (SQLite paper_store)
- 170 exhaustive tests, all passing
- PwC dataset builder (GitHub API, rate-limit aware)
- Comprehensive benchmarks (FAISS Flat/IVF/HNSW + ChromaDB + Matryoshka)

### What's NOT Built Yet
- QLoRA fine-tuning on PwC dataset
- LLM-based entity extraction (currently regex heuristics)
- Multi-agent orchestration (code gen + sandbox execution)
- Code sandbox (Docker + Jupyter kernels)
- Production-quality frontend (attempted React, reverted to Streamlit)
- Graph visualization in the UI (PyVis exists but basic)
- Embedding caching
- WebSocket streaming for LLM tokens
- MCP (Model Context Protocol) server integration

---

## Architecture At A Glance

```
PDF → [Hybrid Parser] → Paper + Sections → [Chunker] → Chunks
                                                ↓
                                    [Embedder] → ChromaDB
                                    [Entity Extractor] → SQLite KG
                                    [Paper Store] → SQLite

Query → [Hybrid Retriever (Vector + KG)] → [Cross-Encoder Reranker]
    → [Context Assembly (dedup + compress + lost-in-middle)]
    → [Qwen2.5-Coder-7B NF4] → Grounded Answer + Sources
```

---

## Key Files & What They Do

### Core
- `src/papermind/models.py` — Pydantic models: Paper, Section, Chunk, Entity, Relationship, SearchResult, LatexEquation. `make_paper_id(pdf_path)` generates deterministic SHA-256 IDs from file content.
- `src/papermind/config.py` — Pydantic Settings. Loads from `config/settings.yaml` + env vars (prefix `PAPERMIND_`, delimiter `__`). Cached via `@lru_cache`.
- `src/papermind/services.py` — `ServiceRegistry` singleton with lazy-loaded properties. Uses `threading.RLock` (NOT Lock — RLock prevents deadlocks when embedding_pipeline accesses embedding_service internally). Global instance: `from papermind.services import services`.
- `src/papermind/cli.py` — Click CLI: `papermind ingest`, `papermind search`, `papermind serve`, `papermind generate`.

### Infrastructure
- `infrastructure/embedding.py` — `EmbeddingService` wrapping sentence-transformers. Default model: `nomic-ai/nomic-embed-text-v1.5` (768d). Uses asymmetric prefixes: `"search_document: "` for docs, `"search_query: "` for queries. L2-normalized. Matryoshka support (but 768d recommended — truncation drops recall to 54%).
- `infrastructure/vector_store.py` — ChromaDB wrapper. `add_chunks()`, `search()`, `delete_by_paper()`, `count()`, `get_stored_papers()`. Cosine similarity via HNSW. Score = 1.0 - cosine_distance.
- `infrastructure/faiss_store.py` — Alternative FAISS store with Flat/IVF/HNSW index types. `count()` is a METHOD (not property — was changed for consistency with ChromaDB). Not used by default (config: `vector_store.backend = "chroma"`).
- `infrastructure/knowledge_graph.py` — SQLite + networkx. `check_same_thread=False`. Entities table + relationships table. `get_subgraph(entity_id, depth)` returns networkx DiGraph. `search_entities()` uses LIKE query.
- `infrastructure/paper_store.py` — SQLite paper metadata. `save()` upserts. `list_all()` orders by created_at DESC. Authors stored as JSON. Path: `./data/papers.db`.
- `infrastructure/llm_client.py` — Routes to `LocalModel` (backend="local") or Ollama HTTP API (backend="ollama"). Async methods: `generate()`, `chat()`, `generate_stream()`.
- `infrastructure/local_model.py` — Loads Qwen2.5-Coder-7B-Instruct with bitsandbytes NF4 quantization. 5.18GB VRAM. Chat template: ChatML (`<|im_start|>`). `generate_stream()` uses threading. `unload()` frees GPU. `vram_usage()` returns dict.

### Ingestion
- `ingestion/hybrid_parser.py` — **Primary parser**. Stage 1: GROBID for metadata. Stage 2: MinerU for body text with LaTeX. Stage 3: Reconcile (prefer MinerU body, GROBID abstract if longer). Stage 4: Validate metadata (trim >30 authors — reference pollution).
- `ingestion/grobid_parser.py` — Calls GROBID `/api/processFulltextDocument`. Parses TEI XML. Good metadata but NO LaTeX (outputs Unicode math symbols).
- `ingestion/mineru_parser.py` — Runs MinerU in **isolated virtualenv** (`.venv-mineru`) via subprocess because MinerU needs transformers 4.x but main project uses 5.x. Monkey-patches `torch.load` for `weights_only=False` compatibility. Outputs Markdown with `$...$` and `$$...$$` LaTeX. Takes ~30s per paper.
- `ingestion/pdf_parser.py` — PyMuPDF fallback. Font-size heuristics for title/sections.
- `ingestion/chunker.py` — Parent-child strategy. Parent = full section (capped 2048 tokens). Children = 512-token chunks with 64-token overlap. Uses tiktoken `cl100k_base`. Respects paragraph/sentence boundaries.
- `ingestion/embedder.py` — `EmbeddingPipeline` orchestrates: extract texts → embed via EmbeddingService → store in vector_store. Calls `save()` if FAISS.
- `ingestion/entity_extractor.py` — Regex patterns for methods ("we propose X"), datasets ("evaluated on X"), metrics ("achieves 95.2%"). Heuristic — NOT LLM-based yet.
- `ingestion/latex_extractor.py` — Extracts `$$`, `\[\]`, `\(\)`, `\begin{equation}` patterns with ±150 char context.

### RAG
- `rag/retriever.py` — `hybrid_retrieve()`: vector search (2x n_results) + KG entity expansion + RRF fusion (k=60). `kg_search()` finds entities matching query, gets neighbors, collects paper_ids, then vector-searches within those papers with enriched query.
- `rag/reranker.py` — `Reranker` class loads `BAAI/bge-reranker-v2-m3` (568M, FP32, 2.3GB VRAM). `score_pairs()` returns sigmoid-normalized [0,1] scores. `unload()` frees GPU. **CRITICAL: must unload before LLM loads** — both can't fit on 12GB simultaneously.
- `rag/context.py` — `assemble_context()` pipeline: deduplicate (Jaccard word overlap >0.7) → compress (greedy token budget, default 4096) → lost-in-middle order (best at start+end). Formats with `[Source N]` headers.
- `rag/pipeline.py` — `RAGPipeline` orchestrator. `DEFAULT_SYSTEM_PROMPT` instructs model to ONLY answer from context, cite sources, never fabricate. `query()` is async. `query_stream()` yields (type, data) tuples.

### API
- `api/app.py` — FastAPI factory with CORS (localhost:3000, :5173). Includes routers: health, papers, search, chat.
- `api/routes/papers.py` — `POST /papers/ingest` uses SSE (Server-Sent Events) for streaming progress. Returns `text/event-stream` with JSON events: save, parse, clean, chunk, embed, entities, done/error.
- `api/routes/chat.py` — `POST /chat/rag` runs full RAG. Unloads reranker before LLM generation.
- `api/routes/health.py` — `GET /health/detailed` returns GPU info via nvidia-smi subprocess.

### UI (Streamlit)
- `ui/app.py` — Page routing with error display. `_sidebar_stats()` reads KG counts (lightweight SQLite, NOT embedding model — to avoid blocking on model load).
- `ui/shared.py` — Thin wrappers around `services` registry (no `@st.cache_resource` — registry handles singletons).
- `ui/pages/chat.py` — RAG mode + Direct mode. `_render_latex()` converts `\(...\)` → `$...$` and `\[...\]` → `$$...$$` for Streamlit rendering. Right sidebar with sliders (retrieve count, rerank top-k, token budget) and toggles (reranking, KG). **Unloads reranker before LLM generation.**
- `ui/pages/papers.py` — `st.status()` for ingestion progress (NOT `st.progress()` which doesn't update during blocking calls). Parser selection: Hybrid/GROBID/PyMuPDF. Deterministic paper IDs. Re-ingestion cleans old data first.

### Data
- `data/pwc_dataset_fast.py` — GitHub API-based PwC dataset builder. `RateLimiter` class reads `X-RateLimit-Remaining` headers. Separate semaphores: `api_workers=5` (rate-limited) and `raw_workers=20` (unlimited — raw.githubusercontent.com). Streams pairs to disk via raw JSONL.

---

## Critical Gotchas & Lessons Learned

### VRAM Management
- Reranker (2.3GB) + LLM (5.2GB) = 7.5GB — fits on 12GB but with margin. **Always unload reranker before LLM.**
- Embedding model (nomic-embed) runs on **CPU** by default (config `embedding.device: "cpu"`). This is intentional — it's only 90MB and doesn't need GPU for our batch sizes.
- MinerU uses ~6GB VRAM but only during ingestion (separate subprocess). Never conflicts with query-time models.

### Threading & Concurrency
- `ServiceRegistry` uses `threading.RLock` — NOT `threading.Lock`. Changed from Lock to RLock to fix deadlock where `embedding_pipeline` property acquires lock, then internally accesses `embedding_service` which tries to acquire the same lock.
- Streamlit runs callbacks in different threads. SQLite connections use `check_same_thread=False`.
- `asyncio.run()` inside Streamlit fails because Streamlit has its own event loop. The chat page uses `concurrent.futures.ThreadPoolExecutor` as fallback.

### Persistence
- Paper IDs are **deterministic** via `make_paper_id(pdf_path)` — SHA-256 of file content, first 12 hex chars. Same PDF always gets same ID regardless of filename or machine.
- Paper metadata persists in `data/papers.db` (SQLite). Chunks persist in `data/chroma/` (ChromaDB). KG persists in `data/kg.sqlite`. All survive app restarts.
- Old `st.session_state` approach was removed — caused data loss on restart.

### PDF Parsing
- GROBID Docker image has a JDK cgroups v2 bug on newer kernels. Fixed with custom Dockerfile using Eclipse Temurin JRE 21 + mounting cgroup filesystem.
- MinerU runs in isolated `.venv-mineru/` because it needs transformers 4.x (main project uses 5.x for Qwen). Invoked via subprocess with `torch.load` monkey-patch for PyTorch 2.6+ `weights_only=True` compatibility.
- GROBID extracts equations as Unicode symbols (α, √, ∑) — NOT LaTeX. MinerU extracts proper LaTeX. That's why hybrid parser exists.

### Config
- Default vector store backend is `"chroma"` (changed from `"faiss"` after FAISS caused startup hangs).
- `faiss_store.py` has `count()` as a **method** (not property) — was `@property` originally, changed for consistency with ChromaDB's `count()` method.

---

## Commands Reference

```bash
# Install
uv sync && uv sync --extra dev --extra streamlit

# Run Streamlit UI
uv run streamlit run src/papermind/ui/app.py --server.port 8501

# Run FastAPI backend
uv run uvicorn papermind.api.app:app --host 0.0.0.0 --port 8000

# Run tests (170 tests, ~18s)
uv run pytest tests/ -v

# Run tests without slow ones (skip model loading)
uv run pytest tests/ -v -m "not slow"

# Start GROBID
docker start grobid  # or docker run with cgroup flags

# Check GROBID health
curl http://localhost:8070/api/isalive

# Build PwC dataset
export $(grep -v '^#' .env | xargs)
uv run python -m papermind.data.pwc_dataset_fast --output data/pwc/full --api-workers 5

# Run FAISS benchmarks
uv run python -m papermind.benchmarks.faiss_benchmark --use-real --output data/benchmark_results.json
```

---

## Benchmark Results Summary

**Model:** Qwen2.5-Coder-7B NF4 — loads in 9.3s, 5.18GB VRAM, 48.9 tok/s streaming.

**Vector Store (4,102 chunks × 768d):**
- Flat: 0.29ms, perfect recall — **best for <10K vectors**
- IVF(256,32): 113K QPS, 97.8% R@10 — best for 50K+
- ChromaDB: 0.59ms, 1,854 QPS — slower but has persistence + metadata filtering
- Matryoshka 768→256d: drops recall to 54% — **stay with 768d**

**PDF Parsing (50 papers):**
- Hybrid: 549 display + 9,598 inline LaTeX equations (vs 0 from GROBID alone)
- 49/50 papers with LaTeX. Avg 35.4s/paper.

---

## Test Coverage

170 tests across 15 files. All pass in ~18s without external services.

| File | Tests | Covers |
|------|-------|--------|
| test_models.py | 30 | All data models, make_paper_id determinism |
| test_rag_context.py | 17 | Lost-in-middle, dedup, compress, assemble |
| test_config.py | 14 | Settings defaults, YAML, env vars |
| test_latex_extractor.py | 12 | All equation patterns |
| test_embedding.py | 11 | Shape, normalization, asymmetry |
| test_paper_store.py | 11 | CRUD, upsert, JSON roundtrip |
| test_entity_extractor.py | 11 | Method/dataset/metric extraction |
| test_rag_retriever.py | 9 | RRF fusion, hybrid retrieve |
| test_rag_reranker.py | 8 | Cross-encoder scoring, thresholds |
| test_api.py | 8 | All FastAPI endpoints |
| test_services.py | 7 | Singleton, RLock, thread safety |
| test_knowledge_graph.py | 7 | Entity/relationship CRUD |
| test_chunker.py | 6 | Parent-child, boundaries |
| test_vector_store.py | 4 | ChromaDB operations |
| test_pdf_parser.py | 2 | PDF extraction |

---

## Data Files

| Path | Purpose | Git-tracked? |
|------|---------|-------------|
| `config/settings.yaml` | All configuration | Yes |
| `data/BENCHMARKS.md` | Benchmark report | Yes |
| `data/benchmark_report.json` | Structured benchmarks | Yes |
| `data/chroma/` | ChromaDB vector store | No |
| `data/kg.sqlite` | Knowledge graph | No |
| `data/papers.db` | Paper metadata registry | No |
| `data/pwc/` | PwC dataset build output | No |
| `docs/*.pdf` | 50 test papers | No |
| `.env` | Tokens (GITHUB_TOKEN, HF_TOKEN) | No |
| `.venv-mineru/` | Isolated MinerU virtualenv | No |

---

## Conventions

- Python 3.12+, src layout, type hints everywhere
- Config: `config/settings.yaml` + `PAPERMIND_` env var overrides
- Tests: pytest (no external services required)
- All shared state via `papermind.services.services` registry
- Deterministic paper IDs from content hash
- ChromaDB as default vector store (not FAISS)
- Hybrid parser as default PDF parser (not GROBID-only)
- Reranker always unloaded before LLM loads
- Streamlit UI (React attempt abandoned — needs proper component library)
