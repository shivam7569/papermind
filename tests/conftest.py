"""Shared test fixtures."""

import tempfile
from pathlib import Path

import pytest

from papermind.infrastructure.knowledge_graph import KnowledgeGraph
from papermind.infrastructure.vector_store import VectorStore


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def kg(tmp_dir):
    """Provide a fresh in-memory knowledge graph."""
    graph = KnowledgeGraph(db_path=str(tmp_dir / "test_kg.sqlite"))
    yield graph
    graph.close()


@pytest.fixture
def vector_store(tmp_dir):
    """Provide a fresh ChromaDB instance in a temp directory."""
    return VectorStore(
        persist_directory=str(tmp_dir / "chroma"),
        collection_name="test_chunks",
    )
