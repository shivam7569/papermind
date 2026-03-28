"""Tests for the ChromaDB vector store."""

from papermind.models import Chunk


def test_add_and_search(vector_store):
    chunks = [
        Chunk(text="Attention is all you need", paper_id="p1", section_title="Intro"),
        Chunk(text="Convolutional neural networks for image classification", paper_id="p1", section_title="Methods"),
        Chunk(text="Reinforcement learning from human feedback", paper_id="p2", section_title="Intro"),
    ]
    # Use simple mock embeddings (3 dimensions for testing)
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    vector_store.add_chunks(chunks, embeddings)
    assert vector_store.count() == 3

    # Search with a query embedding close to the first chunk
    results = vector_store.search([0.9, 0.1, 0.0], n_results=2)
    assert len(results) == 2
    assert results[0].text == "Attention is all you need"


def test_filter_by_paper_id(vector_store):
    chunks = [
        Chunk(text="Paper 1 content", paper_id="p1"),
        Chunk(text="Paper 2 content", paper_id="p2"),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    vector_store.add_chunks(chunks, embeddings)

    results = vector_store.search([1.0, 0.0], n_results=10, paper_id="p1")
    assert len(results) == 1
    assert results[0].paper_id == "p1"


def test_delete_by_paper(vector_store):
    chunks = [
        Chunk(text="A", paper_id="p1"),
        Chunk(text="B", paper_id="p2"),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    vector_store.add_chunks(chunks, embeddings)
    assert vector_store.count() == 2

    vector_store.delete_by_paper("p1")
    assert vector_store.count() == 1


def test_empty_search(vector_store):
    results = vector_store.search([1.0, 0.0], n_results=5)
    assert results == []
