"""Tests for the configuration system."""

import os
from unittest.mock import patch

import pytest

from papermind.config import (
    Settings,
    LLMSettings,
    EmbeddingSettings,
    VectorStoreSettings,
    KnowledgeGraphSettings,
    ChunkingSettings,
    APISettings,
    GrobidSettings,
    IngestionSettings,
    _load_yaml_config,
)


class TestDefaultSettings:
    def test_default_settings_load(self):
        s = Settings()
        assert isinstance(s.llm, LLMSettings)
        assert isinstance(s.embedding, EmbeddingSettings)
        assert isinstance(s.vector_store, VectorStoreSettings)

    def test_vector_store_backend_default(self):
        s = Settings()
        assert s.vector_store.backend == "chroma"

    def test_llm_defaults(self):
        s = Settings()
        assert s.llm.base_url == "http://localhost:11434"
        assert s.llm.timeout == 120
        assert s.llm.backend == "local"

    def test_embedding_defaults(self):
        s = Settings()
        assert s.embedding.model_name == "nomic-ai/nomic-embed-text-v1.5"
        assert s.embedding.device == "cpu"
        assert s.embedding.batch_size == 64
        assert s.embedding.matryoshka_dim is None

    def test_chunking_defaults(self):
        s = Settings()
        assert s.chunking.chunk_size == 512
        assert s.chunking.chunk_overlap == 64
        assert s.chunking.min_chunk_size == 50

    def test_api_defaults(self):
        s = Settings()
        assert s.api.host == "0.0.0.0"
        assert s.api.port == 8000

    def test_grobid_defaults(self):
        s = Settings()
        assert s.grobid.url == "http://localhost:8070"

    def test_ingestion_defaults(self):
        s = Settings()
        assert s.ingestion.pdf_parser in ("grobid", "pymupdf")

    def test_kg_defaults(self):
        s = Settings()
        assert s.knowledge_graph.db_path == "./data/kg.sqlite"

    def test_nested_settings_accessible(self):
        s = Settings()
        assert s.vector_store.faiss_hnsw_m == 32
        assert s.llm.repetition_penalty == 1.05


class TestYamlOverride:
    def test_yaml_config_loads(self):
        # _load_yaml_config should return a dict (possibly empty)
        result = _load_yaml_config()
        assert isinstance(result, dict)

    def test_yaml_override_applies(self, tmp_dir):
        yaml_content = "llm:\n  timeout: 999\n"
        config_file = tmp_dir / "settings.yaml"
        config_file.write_text(yaml_content)
        with patch("papermind.config.CONFIG_DIR", tmp_dir):
            result = _load_yaml_config()
        assert result["llm"]["timeout"] == 999


class TestEnvVarOverride:
    def test_env_var_override_simple(self):
        with patch.dict(os.environ, {"PAPERMIND_API__PORT": "9999"}):
            s = Settings()
            assert s.api.port == 9999

    def test_env_var_override_nested(self):
        with patch.dict(os.environ, {"PAPERMIND_LLM__TIMEOUT": "300"}):
            s = Settings()
            assert s.llm.timeout == 300

    def test_env_var_override_string(self):
        with patch.dict(os.environ, {"PAPERMIND_EMBEDDING__DEVICE": "cuda"}):
            s = Settings()
            assert s.embedding.device == "cuda"
