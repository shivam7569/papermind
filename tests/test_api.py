"""Tests for FastAPI endpoints using TestClient.

All external services (embedding, LLM, GPU) are mocked.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from papermind.models import Paper, Entity, SearchResult


@pytest.fixture
def mock_services():
    """Create a mock services registry with all dependencies stubbed."""
    svc = MagicMock()

    # Paper store
    svc.paper_store.list_all.return_value = [
        Paper(id="p1", title="Test Paper", authors=["Alice"], num_pages=10, num_chunks=50),
    ]
    svc.paper_store.get.return_value = Paper(id="p1", title="Test Paper")
    svc.paper_store.count.return_value = 1

    # Vector store
    svc.vector_store.count.return_value = 100

    # Knowledge graph
    svc.knowledge_graph.count_entities.return_value = 25
    svc.knowledge_graph.count_relationships.return_value = 40
    svc.knowledge_graph.search_entities.return_value = [
        Entity(id="e1", name="BERT", entity_type="method", paper_id="p1"),
    ]

    # Embedding pipeline
    svc.embedding_pipeline.search.return_value = [
        SearchResult(chunk_id="c1", text="Found result", score=0.9, paper_id="p1"),
    ]

    return svc


@pytest.fixture
def client(mock_services):
    """TestClient with mocked services."""
    with patch("papermind.api.routes.health.services", mock_services), \
         patch("papermind.api.routes.papers.services", mock_services), \
         patch("papermind.api.routes.search.services", mock_services):
        from papermind.api.app import create_app
        app = create_app()
        yield TestClient(app)


class TestHealthEndpoints:
    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_detailed_health(self, client):
        resp = client.get("/api/health/detailed")
        assert resp.status_code == 200
        data = resp.json()
        assert "vector_store_count" in data
        assert "kg_entities" in data
        assert "kg_relations" in data
        assert "papers_count" in data
        assert data["vector_store_count"] == 100
        assert data["kg_entities"] == 25


class TestPapersEndpoints:
    def test_list_papers(self, client):
        resp = client.get("/api/papers")
        assert resp.status_code == 200
        data = resp.json()
        assert "papers" in data
        assert len(data["papers"]) == 1
        assert data["papers"][0]["title"] == "Test Paper"
        assert data["papers"][0]["id"] == "p1"


class TestSearchEndpoints:
    def test_vector_search(self, client):
        resp = client.post("/api/search", json={"query": "attention", "n_results": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "Found result"

    def test_search_entities(self, client):
        resp = client.get("/api/kg/entities", params={"query": "BERT"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "BERT"


class TestChatEndpoint:
    def test_rag_chat(self, mock_services):
        """Test the RAG chat endpoint with fully mocked pipeline."""
        mock_retrieve = MagicMock(return_value=[
            SearchResult(chunk_id="c1", text="Relevant context", score=0.9, paper_id="p1", section_title="Intro"),
        ])
        mock_rerank_fn = MagicMock(side_effect=lambda **kwargs: kwargs["results"])
        mock_reranker = MagicMock()

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Test answer based on context")

        with patch("papermind.api.routes.health.services", mock_services), \
             patch("papermind.api.routes.papers.services", mock_services), \
             patch("papermind.api.routes.search.services", mock_services), \
             patch("papermind.rag.retriever.hybrid_retrieve", mock_retrieve), \
             patch("papermind.rag.reranker.rerank", mock_rerank_fn), \
             patch("papermind.rag.reranker.get_reranker", return_value=mock_reranker), \
             patch("papermind.infrastructure.llm_client.LLMClient", return_value=mock_llm):
            from papermind.api.app import create_app
            app = create_app()
            client = TestClient(app)
            resp = client.post("/api/chat/rag", json={"query": "What is attention?", "use_rerank": False})

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data

    def test_rag_chat_no_results(self, mock_services):
        """Test RAG chat when no documents are found."""
        mock_retrieve = MagicMock(return_value=[])

        with patch("papermind.api.routes.health.services", mock_services), \
             patch("papermind.api.routes.papers.services", mock_services), \
             patch("papermind.api.routes.search.services", mock_services), \
             patch("papermind.rag.retriever.hybrid_retrieve", mock_retrieve):
            from papermind.api.app import create_app
            app = create_app()
            client = TestClient(app)
            resp = client.post("/api/chat/rag", json={"query": "unknown topic"})

        assert resp.status_code == 200
        data = resp.json()
        assert "No relevant" in data["answer"]
        assert data["retrieval_count"] == 0
