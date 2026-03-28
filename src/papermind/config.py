"""Configuration system using Pydantic Settings with YAML file + env overrides."""

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


class LLMSettings(BaseModel):
    # Ollama remote settings
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:7b-instruct-q5_K_M"
    timeout: int = 120

    # Local transformers settings
    local_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    backend: str = "local"  # "local" (transformers) or "ollama"
    quantization: str = "nf4"  # "nf4", "int8", or "none"
    double_quant: bool = True  # double quantization for NF4
    compute_dtype: str = "bfloat16"  # compute dtype for quantized matmul
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.95
    repetition_penalty: float = 1.05


class EmbeddingSettings(BaseModel):
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    device: str = "cpu"
    batch_size: int = 64
    matryoshka_dim: int | None = None  # None = full 768d; set 256/512 for speed


class VectorStoreSettings(BaseModel):
    backend: str = "faiss"  # "faiss" or "chroma"
    # ChromaDB settings
    persist_directory: str = "./data/chroma"
    collection_name: str = "paper_chunks"
    # FAISS settings
    faiss_directory: str = "./data/faiss"
    faiss_index_type: str = "hnsw"  # "flat", "ivf", "hnsw"
    faiss_ivf_nlist: int = 100
    faiss_ivf_nprobe: int = 10
    faiss_hnsw_m: int = 32
    faiss_hnsw_ef_construction: int = 200
    faiss_hnsw_ef_search: int = 64


class KnowledgeGraphSettings(BaseModel):
    db_path: str = "./data/kg.sqlite"


class ChunkingSettings(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class GrobidSettings(BaseModel):
    url: str = "http://localhost:8070"
    timeout: int = 120


class IngestionSettings(BaseModel):
    papers_directory: str = "./data/papers"
    pdf_parser: str = "grobid"  # "grobid" or "pymupdf"


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    knowledge_graph: KnowledgeGraphSettings = KnowledgeGraphSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    grobid: GrobidSettings = GrobidSettings()
    api: APISettings = APISettings()
    ingestion: IngestionSettings = IngestionSettings()

    model_config = {"env_prefix": "PAPERMIND_", "env_nested_delimiter": "__"}


def _load_yaml_config() -> dict:
    """Load settings from YAML config file if it exists."""
    config_path = CONFIG_DIR / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


@lru_cache
def get_settings() -> Settings:
    """Get application settings. Cached after first call.

    Priority: env vars > YAML config > defaults.
    """
    yaml_config = _load_yaml_config()
    return Settings(**yaml_config)
