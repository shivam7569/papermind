# PaperMind

Local AI research system for paper understanding, knowledge graphs, and code generation.
Runs on RTX 5070 Ti (12GB VRAM).

## Quick Start

```bash
uv sync              # Install dependencies
uv sync --extra dev  # Include dev tools
papermind ingest <pdf_path>   # Ingest a paper
papermind search "query"      # Search ingested papers
papermind serve               # Start API server
```

## Architecture

- **LLM**: Qwen2.5-Coder-7B via Ollama/llama.cpp
- **Backend**: FastAPI (REST + MCP on same server)
- **Vector Store**: ChromaDB with all-MiniLM-L6-v2 embeddings
- **Knowledge Graph**: SQLite + networkx
- **Code Sandbox**: Docker + Jupyter kernels

## Conventions

- Python 3.12+, src layout, type hints everywhere
- Config via `config/settings.yaml` + env var overrides (PAPERMIND_ prefix)
- Tests with pytest + hypothesis for property-based testing
- Format/lint with ruff, type check with mypy
