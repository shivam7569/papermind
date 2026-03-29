## PERSISTENT CONTEXT MAINTENANCE — STANDING INSTRUCTION

This is a standing instruction that applies for the entire duration of this project.
You must follow it without being reminded. Do not wait for me to ask.

---

### OVERVIEW

You are working on **Papermind** — an AI research paper intelligence system built
as a modular, interconnected set of sub-projects, each covering a distinct set of
LLM ecosystem topics (RAG, GraphRAG, Agentic systems, Fine-tuning/PEFT, RLHF,
Evaluation, Inference Optimization). The sub-projects are architecturally
interdependent — not isolated. Every decision made in one module may affect others.

Two files track everything about this project. You are responsible for keeping
both of them accurate, current, and useful at all times:

1. `CLAUDE.md` — located at the project root
2. `.claude/memory.md` — located inside the `.claude/` directory

---

### FILE 1: CLAUDE.md — The Project Constitution

**Purpose:** A stable, high-fidelity record of what Papermind *is*. Anyone (or
any AI session) reading this file should be able to understand the full
architecture, constraints, module map, and key decisions without any other context.

**When to update:** Update CLAUDE.md when ANY of the following happen:
- A new module, component, or sub-project is created or significantly refactored
- A file or directory is created that is architecturally significant (not just
  utility scripts — actual pipeline components)
- A hard architectural decision is made and locked in (e.g., "we are using FAISS
  over ChromaDB because...", "embedding model is X because...")
- An inter-module dependency is established or changed
- A new hard constraint is introduced or an existing one is changed

**Do NOT update CLAUDE.md** for task-level progress, bug fixes, or in-session
working notes. That belongs in memory.md.

---

### FILE 2: `.claude/memory.md` — The Working Memory

**Purpose:** A living record of *where the project is right now*. This is what
you read at the beginning of each working block to understand what was last
happening, what's in progress, and what traps to avoid.

**When to update:** Update memory.md at ALL of the following trigger points:
- After completing any meaningful unit of work (a function, a module component,
  a pipeline stage — not individual lines of code)
- After resolving any non-trivial error (anything that took more than one attempt)
- After making a decision that is *not yet* architectural but affects direction
- At natural task transitions — when one thing is done and another is beginning
- Every ~45–60 minutes of active work as a time-based checkpoint

---

### BEHAVIORAL RULES

1. **Never ask permission to update these files.** These updates are part of your
   job on this project. Just do them, then continue working.

2. **Never truncate history in the Errors & Resolutions log or the Architectural
   Decisions Log.** These are permanent records. Append only.

3. **When you read a message from me at the start of a working block, read both
   files before responding** — even if I don't ask you to. State in one sentence
   that you've read them and what your understanding of the current state is.

4. **If either file is missing**, create it immediately using the schemas above,
   populate what you know, and note any fields that are unknown.

5. **CLAUDE.md captures truth about the system. memory.md captures truth about
   the moment.** Do not mix them.

6. **On any architectural decision**, before writing code, write the decision to
   CLAUDE.md's Architectural Decisions Log first.

7. **The session log in memory.md should be written as if your replacement — a
   new Claude instance with zero conversation history — will read it cold and
   need to continue the work exactly where you left off.**

---

# Papermind — Project Constitution
_Last updated: 2026-03-30_

## What Is Papermind

PaperMind is a local AI research paper intelligence system that runs entirely on a single NVIDIA RTX 5070 Ti (12GB VRAM). It ingests research papers (PDF), extracts structured content including LaTeX equations, builds a knowledge graph, embeds everything into a vector store, and provides RAG-grounded chat powered by a locally-running Qwen2.5-Coder-7B model with five specialized LoRA adapters (Reader, Extractor, Synthesis, Critic, Coder) hot-swappable at inference time. No cloud APIs for inference. No data leaves the machine.

**Owner:** Shivam (GitHub: shivam7569)
**Repo:** https://github.com/shivam7569/papermind

## Hard Constraints

- **Hardware:** NVIDIA RTX 5070 Ti Laptop GPU, 12GB VRAM. All inference models must fit within this budget. Reranker (2.3GB) must unload before LLM (5.2GB) loads.
- **Local-first:** No cloud APIs for inference or user data processing. Cloud is only for training (RunPod A100 bursts) and data generation (Groq API for training data).
- **Domain scope:** AI/ML/DL research papers only. All training data must be strictly within this domain — no biomedical, no generic NLI, no off-domain content.
- **Base model:** Qwen/Qwen2.5-Coder-7B-Instruct, NF4 quantized via bitsandbytes. 5.18GB VRAM, 48.9 tok/s streaming.
- **Embedding:** nomic-ai/nomic-embed-text-v1.5 (768d) on CPU. Do NOT truncate to lower dimensions (recall drops to 54%).
- **Vector store:** ChromaDB (not FAISS). FAISS is available as alternative but ChromaDB is default for persistence + metadata filtering.
- **PDF parsing:** Hybrid GROBID (metadata) + MinerU (LaTeX equations). MinerU runs in isolated `.venv-mineru/` (needs transformers 4.x, main uses 5.x).
- **Training data quality:** Always validate 20% sample with Claude Opus audit before scaling to full dataset. Never generate full dataset without intermediate validation.
- **No shortcuts:** Always verify script versions match approved versions before launching. Always update generation scripts immediately after prompt approval.

## System Architecture

Papers enter the system through the **Hybrid Parser** (GROBID for metadata + MinerU for body text with LaTeX equations), producing structured Paper and Section objects. The **Chunker** splits sections into parent-child chunks (parent = full section capped at 2048 tokens, children = 512-token chunks with 64-token overlap). The **EmbeddingPipeline** embeds chunks via nomic-embed-text-v1.5 and stores them in **ChromaDB**. Simultaneously, the **Entity Extractor** (currently regex heuristics, planned: LLM-based via Extractor adapter) extracts entities and relationships into a **SQLite Knowledge Graph**.

At query time, the **Hybrid Retriever** performs vector search (2x n_results) and KG entity expansion, fusing results via Reciprocal Rank Fusion (k=60). The **Reranker** (BAAI/bge-reranker-v2-m3, 568M params, FP32) re-scores the candidates. The reranker then **unloads from GPU** before the **LLM** (Qwen2.5-Coder-7B NF4) loads. **Context Assembly** deduplicates (Jaccard >0.7), compresses (token budget), and applies lost-in-middle ordering. The LLM generates a grounded answer with source citations.

Four **Reasoning Frameworks** wrap the generation: Direct, Chain-of-Thought, Self-Consistency (N=5 majority vote), and ReAct (Thought→Action→Observation with tools: search, lookup_entity, read_section).

Five **LoRA adapters** (all sharing lora_r=16, alpha=32, dropout=0.05, DoRA=true, targeting all linear layers, ~20-34MB each) will specialize the base model: Reader (summarize/QA), Extractor (entity-relation extraction for KG), Synthesis (multi-paper comparison), Critic (claim verification), Coder (paper-to-code). Hot-swappable via PEFT `set_adapter()` in <1ms.

All shared state flows through the **ServiceRegistry** singleton (RLock-based, prevents deadlocks in nested property access). The **Streamlit UI** provides Chat, Papers, Search, Training, Dataset, Benchmarks, and System pages. A **FastAPI backend** exposes health, papers, search, and chat endpoints.

## Module Map

### Ingestion (`src/papermind/ingestion/`)
- **Purpose:** Parse PDFs, chunk text, embed, extract entities, store everything
- **Covers:** Document parsing, LaTeX extraction, chunking strategies, embedding pipelines
- **Entry point:** `hybrid_parser.py` (primary), `chunker.py`, `embedder.py`
- **Key dependencies:** GROBID Docker container, MinerU .venv-mineru/, nomic-embed model
- **Produces:** Chunks in ChromaDB, entities in SQLite KG, paper metadata in SQLite paper_store
- **Status:** Complete

### RAG (`src/papermind/rag/`)
- **Purpose:** Retrieve, rerank, assemble context, generate grounded answers
- **Covers:** Hybrid retrieval (vector + KG), cross-encoder reranking, lost-in-middle ordering, context compression
- **Entry point:** `pipeline.py` (orchestrator), `retriever.py`, `reranker.py`, `context.py`
- **Key dependencies:** ChromaDB, KnowledgeGraph, EmbeddingService, LLMClient
- **Produces:** Grounded answers with source citations
- **Status:** Complete

### Infrastructure (`src/papermind/infrastructure/`)
- **Purpose:** Core services — embedding, vector store, KG, LLM, paper store
- **Covers:** Model loading, quantization, VRAM management, persistence
- **Entry point:** `local_model.py` (Qwen NF4), `embedding.py`, `vector_store.py`, `knowledge_graph.py`
- **Key dependencies:** transformers, bitsandbytes, sentence-transformers, ChromaDB, SQLite
- **Produces:** Singleton services consumed by all other modules
- **Status:** Complete

### Reasoning (`src/papermind/reasoning/`)
- **Purpose:** Wrap LLM generation with structured reasoning strategies
- **Covers:** CoT prompting, Self-Consistency (majority vote), ReAct (tool-use loop)
- **Entry point:** `frameworks.py`
- **Key dependencies:** RAG pipeline, LLMClient
- **Produces:** ReasoningResult(answer, reasoning_trace, sources, metadata)
- **Status:** Complete

### Training (`src/papermind/training/`)
- **Purpose:** QLoRA/DoRA fine-tuning pipeline for all 5 adapters
- **Covers:** PEFT/LoRA, bitsandbytes quantization, SFT training, DPO/ORPO alignment
- **Entry point:** `trainer.py` (CLI), `config.py` (hyperparameters), `data.py` (data loading)
- **Key dependencies:** transformers, peft, trl, bitsandbytes
- **Produces:** LoRA adapter weights (~20-34MB each) in `data/adapters/`
- **Status:** Pipeline complete, training data in progress (Phase 5 of 10)

### Data Generation (`scripts/generate_*.py`)
- **Purpose:** Generate training data for adapters using Groq API + gold datasets
- **Covers:** S2ORC paper fetching, Groq-based gold labeling, prompt engineering
- **Entry point:** `generate_reader_data.py`, `generate_extractor_data.py`, `generate_critic_data.py`
- **Key dependencies:** Groq API (GROQ_API_KEY), S2 API (S2_API_KEY), Anthropic API (quality audits)
- **Produces:** JSONL training files in `data/{adapter}_adapter/`
- **Status:** In progress — scripts updated with Opus-approved prompts, awaiting S2ORC re-fetch

### UI (`src/papermind/ui/`)
- **Purpose:** Streamlit-based interface for all system functions
- **Covers:** Chat (RAG + reasoning modes), Papers (upload/ingest), Search (vector + KG graph), Training (adapter config + live logs), Dataset, Benchmarks, System
- **Entry point:** `app.py`
- **Key dependencies:** ServiceRegistry, all backend modules
- **Produces:** Web UI at localhost:8501
- **Status:** Complete (React attempt abandoned — needs proper component library)

### API (`src/papermind/api/`)
- **Purpose:** FastAPI REST backend
- **Covers:** Health checks, paper CRUD, search, chat endpoints
- **Entry point:** `app.py`
- **Key dependencies:** ServiceRegistry
- **Produces:** REST API at localhost:8000
- **Status:** Complete

## Architectural Decisions Log

| Date | Decision | Rationale | Alternatives Rejected |
|------|----------|-----------|----------------------|
| 2026-03-27 | ChromaDB over FAISS as default vector store | FAISS caused startup hangs, ChromaDB has built-in persistence + metadata filtering | FAISS (faster but no persistence, caused bugs) |
| 2026-03-27 | Hybrid GROBID + MinerU parser | GROBID has no LaTeX (outputs Unicode), MinerU has LaTeX but poor metadata. Hybrid gets both. | GROBID-only (no LaTeX), MinerU-only (poor metadata), PyMuPDF (no structure) |
| 2026-03-27 | Custom GROBID Docker with Eclipse Temurin JRE 21 | Stock GROBID image has JDK cgroups v2 bug on kernel 6.17+ | Stock image (crashes), older GROBID versions (same bug) |
| 2026-03-27 | MinerU in isolated .venv-mineru/ | MinerU needs transformers 4.x, main project uses 5.x for Qwen | Same venv (version conflict), Docker (overhead) |
| 2026-03-28 | ServiceRegistry with RLock (not Lock) | Lock caused deadlock when embedding_pipeline accessed embedding_service internally | threading.Lock (deadlocked), no lock (race conditions) |
| 2026-03-28 | Reranker unloads before LLM loads | Both can't fit on 12GB simultaneously (2.3 + 5.2 = 7.5GB + overhead) | Keep both loaded (OOM), CPU reranker (too slow) |
| 2026-03-28 | 768d embeddings, no Matryoshka truncation | Truncation to 256d drops recall to 54% — unacceptable for RAG | 256d (fast but bad recall), 512d (still 54% recall) |
| 2026-03-28 | Streamlit over React for UI | React attempt produced bare-bones UI that was worse than Streamlit. Streamlit works, has data, is proven. | React + shadcn/ui (needs proper frontend effort), Next.js (overkill for now) |
| 2026-03-28 | DoRA over standard LoRA for all adapters | DoRA decomposes weight updates into magnitude + direction, consistently outperforms LoRA at low ranks (r=16) | Standard LoRA (lower quality at r=16), full fine-tuning (won't fit on 12GB) |
| 2026-03-29 | Groq model selection per adapter task (benchmarked with Sonnet judge) | Different models excel at different tasks. gpt-oss-120b best for TLDR/verification, llama-3.3-70b best for QA, kimi-k2 best for extraction, qwen3-32b best for claim-gen | Single model for all (lower quality) |
| 2026-03-29 | 20% sample validation before full generation | First attempt generated 15K+ without validation → catastrophic quality issues (truncated JSON, 87% dupes, hallucinations, off-domain) | Generate full then validate (too costly when issues found) |
| 2026-03-29 | Gold + augmented hybrid for Reader and Critic | Human-annotated data (SciTLDR, QASPER, SciFact) as quality anchor + LLM-generated for volume. Pure LLM lacks human quality ceiling. | LLM-only (no quality anchor), Gold-only (too few examples) |
| 2026-03-29 | SciFact: skip cross-validation folds | Loading all 5 folds created 5x duplicates (67% dupes in training set) | Load all folds (massive duplication) |
| 2026-03-29 | Coder: dedup by instruction (paper), not code | Same paper abstract → multiple code files produced 87% instruction dupes. Keep 1 best file per paper. | Dedup by code only (instruction dupes remain), keep all files (87% dupes) |
| 2026-03-29 | Groq reasoning parameters per model family | gpt-oss: include_reasoning + reasoning_effort. qwen3: reasoning_format (parsed). llama/kimi: no reasoning support. | Same params for all (400 errors on unsupported models) |

## File Structure

```
papermind/
├── CLAUDE.md                           # Project constitution (this file)
├── .claude/memory.md                   # Working memory (session state)
├── config/settings.yaml                # All configuration
├── pyproject.toml                      # Python project config (uv)
├── docker/
│   ├── Dockerfile.grobid               # Custom GROBID with JRE 21 fix
│   └── Dockerfile.sandbox              # Code execution sandbox
├── src/papermind/
│   ├── models.py                       # Pydantic data models
│   ├── config.py                       # Settings loader
│   ├── services.py                     # ServiceRegistry singleton (RLock)
│   ├── cli.py                          # Click CLI
│   ├── infrastructure/
│   │   ├── embedding.py                # nomic-embed-text-v1.5 wrapper
│   │   ├── vector_store.py             # ChromaDB wrapper
│   │   ├── faiss_store.py              # FAISS alternative
│   │   ├── knowledge_graph.py          # SQLite + networkx KG
│   │   ├── paper_store.py              # SQLite paper metadata
│   │   ├── llm_client.py               # LLM routing (local/ollama)
│   │   └── local_model.py              # Qwen NF4 loader + generation
│   ├── ingestion/
│   │   ├── hybrid_parser.py            # GROBID + MinerU pipeline
│   │   ├── grobid_parser.py            # GROBID TEI XML parser
│   │   ├── mineru_parser.py            # MinerU subprocess wrapper
│   │   ├── pdf_parser.py               # PyMuPDF fallback
│   │   ├── chunker.py                  # Parent-child chunking
│   │   ├── embedder.py                 # Embed + store pipeline
│   │   ├── entity_extractor.py         # Regex entity extraction
│   │   └── latex_extractor.py          # LaTeX equation extraction
│   ├── rag/
│   │   ├── retriever.py                # Hybrid vector + KG retrieval
│   │   ├── reranker.py                 # BGE cross-encoder reranker
│   │   ├── context.py                  # Dedup + compress + lost-in-middle
│   │   └── pipeline.py                 # RAG orchestrator
│   ├── reasoning/
│   │   └── frameworks.py               # CoT, Self-Consistency, ReAct
│   ├── training/
│   │   ├── config.py                   # TrainingConfig + ADAPTER_CONFIGS
│   │   ├── data.py                     # ChatML formatter
│   │   └── trainer.py                  # QLoRA/DoRA SFTTrainer
│   ├── ui/
│   │   ├── app.py                      # Streamlit main + routing
│   │   ├── shared.py                   # Service getters
│   │   └── pages/                      # Chat, Papers, Search, Training, etc.
│   ├── api/
│   │   ├── app.py                      # FastAPI factory
│   │   └── routes/                     # health, papers, search, chat
│   ├── data/
│   │   ├── pwc_dataset_fast.py         # PwC GitHub API builder
│   │   └── validate_dataset.py         # Dataset quality validator
│   └── benchmarks/
│       └── faiss_benchmark.py          # Vector store benchmarks
├── scripts/
│   ├── generate_reader_data.py         # Groq Reader augmentation
│   ├── generate_extractor_data.py      # Groq Extractor labeling
│   ├── generate_critic_data.py         # Groq Critic claim generation
│   ├── evaluate_reader.py              # Base model evaluation
│   ├── test_grobid_parsing.py          # GROBID test on 50 papers
│   └── validate_parsing_accuracy.py    # Ground truth validation
├── tests/                              # 170 pytest tests
├── data/
│   ├── generation_prompts.json         # Opus-validated generation prompts
│   ├── BENCHMARKS.md                   # Benchmark report
│   ├── benchmark_report.json           # Structured benchmarks
│   ├── reader_adapter/                 # Reader gold + augmented data
│   ├── extractor_adapter/              # Extractor training data
│   ├── synthesis_adapter/              # Multi-XScience (22K, done)
│   ├── critic_adapter/                 # Critic gold + augmented data
│   ├── pwc/full/                       # Coder dataset (72K deduped)
│   ├── chroma/                         # ChromaDB persistence
│   ├── kg.sqlite                       # Knowledge graph
│   └── papers.db                       # Paper metadata
├── docs/                               # 50 test PDFs
└── .env                                # API keys (not tracked)
```
