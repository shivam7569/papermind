"""Shared test fixtures."""

import tempfile
from pathlib import Path

import pytest

from papermind.infrastructure.knowledge_graph import KnowledgeGraph
from papermind.infrastructure.paper_store import PaperStore
from papermind.infrastructure.vector_store import VectorStore
from papermind.models import Chunk, Entity, Section


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


@pytest.fixture
def paper_store(tmp_dir):
    """Provide a fresh PaperStore backed by a temp SQLite DB."""
    return PaperStore(db_path=tmp_dir / "papers.db")


@pytest.fixture(scope="session")
def embedding_service():
    """Load the embedding model once per test session (cached)."""
    from papermind.infrastructure.embedding import EmbeddingService
    return EmbeddingService(device="cpu")


@pytest.fixture
def sample_sections():
    """Three sections with realistic academic text."""
    return [
        Section(
            title="Introduction",
            text=(
                "Transformer models have revolutionized natural language processing. "
                "The attention mechanism allows the model to weigh the importance of "
                "different parts of the input sequence. We propose TransformerNet, "
                "a novel architecture that extends the standard transformer with "
                "hierarchical attention for long document understanding."
            ),
            page_start=0,
            page_end=1,
        ),
        Section(
            title="Methods",
            text=(
                "Our method builds on the self-attention mechanism introduced by "
                "Vaswani et al. We use multi-head attention with 8 heads and a "
                "hidden dimension of 512. The model is trained on the ImageNet dataset "
                "using the Adam optimizer with a learning rate of 1e-4. We achieve "
                "95.2% accuracy on the validation set, outperforming ResNet by 2.3%."
            ),
            page_start=2,
            page_end=4,
        ),
        Section(
            title="Results",
            text=(
                "We evaluated our approach on three benchmarks: SQuAD, GLUE, and "
                "SuperGLUE. TransformerNet achieves state-of-the-art results on all "
                "three datasets, with an F1 score of 93.1 on SQuAD and an accuracy "
                "of 89.7 on GLUE. Compared to BERT-large, our model uses 40% fewer "
                "parameters while maintaining comparable performance."
            ),
            page_start=5,
            page_end=6,
        ),
    ]


@pytest.fixture
def sample_chunks(sample_sections):
    """Chunks derived from sample_sections."""
    chunks = []
    for section in sample_sections:
        parent = Chunk(
            text=section.text,
            paper_id="test_paper",
            section_title=section.title,
            page_start=section.page_start,
            page_end=section.page_end,
            chunk_type="parent",
            token_count=len(section.text.split()),
        )
        child = Chunk(
            text=section.text[:150],
            paper_id="test_paper",
            section_title=section.title,
            page_start=section.page_start,
            page_end=section.page_end,
            chunk_type="child",
            parent_id=parent.id,
            token_count=len(section.text[:150].split()),
        )
        chunks.extend([parent, child])
    return chunks


@pytest.fixture
def sample_entities():
    """A method and a dataset entity for testing."""
    return [
        Entity(name="TransformerNet", entity_type="method", paper_id="test_paper"),
        Entity(name="ImageNet", entity_type="dataset", paper_id="test_paper"),
    ]
