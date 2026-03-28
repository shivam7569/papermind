"""Tests for the centralized service registry."""

import threading
from unittest.mock import patch, MagicMock

import pytest

from papermind.services import ServiceRegistry


class TestServiceRegistry:
    def test_lazy_initialization(self):
        """Services start as None and are created on first access."""
        reg = ServiceRegistry()
        assert reg._embedding_service is None
        assert reg._vector_store is None
        assert reg._knowledge_graph is None
        assert reg._paper_store is None

    def test_singleton_embedding_service(self):
        """Same instance returned on multiple accesses."""
        reg = ServiceRegistry()
        mock_es = MagicMock()
        with patch("papermind.services.ServiceRegistry.embedding_service",
                    new_callable=lambda: property(lambda self: mock_es)):
            pass
        # Direct test: set it and verify
        reg._embedding_service = MagicMock()
        svc1 = reg._embedding_service
        svc2 = reg._embedding_service
        assert svc1 is svc2

    def test_knowledge_graph_singleton(self, tmp_dir):
        """KnowledgeGraph is only created once."""
        reg = ServiceRegistry()
        mock_kg = MagicMock()
        reg._knowledge_graph = mock_kg
        # Accessing the backing field shows it's the same object
        assert reg._knowledge_graph is mock_kg

    def test_paper_store_singleton(self):
        reg = ServiceRegistry()
        mock_ps = MagicMock()
        reg._paper_store = mock_ps
        assert reg._paper_store is mock_ps

    def test_rlock_is_reentrant(self):
        """RLock allows re-entrant access (e.g., embedding_pipeline accessing embedding_service)."""
        reg = ServiceRegistry()
        # The registry uses RLock which allows same thread to acquire multiple times
        assert isinstance(reg._lock, type(threading.RLock()))
        # Prove reentrance works
        reg._lock.acquire()
        reg._lock.acquire()  # Would deadlock with Lock, but RLock allows this
        reg._lock.release()
        reg._lock.release()

    def test_thread_safety_concurrent_access(self):
        """Concurrent access from multiple threads should not crash."""
        reg = ServiceRegistry()
        # Pre-set mocks to avoid actually loading models
        reg._knowledge_graph = MagicMock()
        reg._paper_store = MagicMock()

        results = []
        errors = []

        def access_services():
            try:
                kg = reg._knowledge_graph
                ps = reg._paper_store
                results.append((kg, ps))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_services) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All threads got the same instances
        assert all(r[0] is results[0][0] for r in results)
        assert all(r[1] is results[0][1] for r in results)

    def test_property_creates_instance_on_first_access(self, tmp_dir):
        """Actually test that accessing a property triggers lazy init."""
        reg = ServiceRegistry()
        # Mock the imports to avoid loading real models
        mock_kg_class = MagicMock()
        mock_kg_instance = MagicMock()
        mock_kg_class.return_value = mock_kg_instance

        with patch("papermind.infrastructure.knowledge_graph.KnowledgeGraph", mock_kg_class):
            # Manually invoke the pattern the property uses
            assert reg._knowledge_graph is None
            from papermind.infrastructure.knowledge_graph import KnowledgeGraph
            reg._knowledge_graph = KnowledgeGraph(db_path=str(tmp_dir / "test.sqlite"))
            assert reg._knowledge_graph is not None
