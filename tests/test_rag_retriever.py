"""Tests for the hybrid retriever and reciprocal rank fusion."""

from unittest.mock import patch, MagicMock

import pytest

from papermind.models import SearchResult
from papermind.rag.retriever import (
    reciprocal_rank_fusion,
    vector_search,
    hybrid_retrieve,
)


def _sr(text: str, score: float = 0.5, chunk_id: str = "") -> SearchResult:
    return SearchResult(
        text=text,
        score=score,
        chunk_id=chunk_id or text[:12],
        paper_id="p1",
    )


class TestReciprocalRankFusion:
    def test_single_list_passthrough(self):
        results = [_sr("a", 0.9), _sr("b", 0.8), _sr("c", 0.7)]
        fused = reciprocal_rank_fusion(results)
        assert len(fused) == 3
        # Order preserved from single list (rank order)
        assert fused[0].text == "a"

    def test_two_lists_merged(self):
        list1 = [_sr("a", 0.9, "id_a"), _sr("b", 0.8, "id_b")]
        list2 = [_sr("b", 0.7, "id_b"), _sr("c", 0.6, "id_c")]
        fused = reciprocal_rank_fusion(list1, list2)
        # "b" appears in both lists, should get higher RRF score
        assert len(fused) == 3
        ids = [r.chunk_id for r in fused]
        assert "id_b" in ids
        # "b" should rank higher due to appearing in both lists
        assert fused[0].chunk_id == "id_b"

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_one_empty_one_full(self):
        list1 = [_sr("a", 0.9)]
        fused = reciprocal_rank_fusion(list1, [])
        assert len(fused) == 1

    def test_deduplication_by_chunk_id(self):
        list1 = [_sr("text a", 0.9, "same_id")]
        list2 = [_sr("text a", 0.8, "same_id")]
        fused = reciprocal_rank_fusion(list1, list2)
        assert len(fused) == 1  # deduplicated

    def test_rrf_scores_in_metadata(self):
        results = [_sr("a", 0.9)]
        fused = reciprocal_rank_fusion(results)
        assert "rrf_score" in fused[0].metadata


class TestVectorSearch:
    def test_empty_store_returns_empty(self):
        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = []
        mock_services = MagicMock()
        mock_services.embedding_pipeline = mock_pipeline

        with patch("papermind.services.services", mock_services):
            results = vector_search("test query")
        assert results == []


class TestHybridRetrieve:
    def test_use_kg_false_vector_only(self):
        mock_pipeline = MagicMock()
        mock_results = [_sr("result1", 0.9), _sr("result2", 0.8)]
        mock_pipeline.search.return_value = mock_results
        mock_services = MagicMock()
        mock_services.embedding_pipeline = mock_pipeline

        with patch("papermind.services.services", mock_services):
            results = hybrid_retrieve("test query", n_results=5, use_kg=False)

        assert len(results) <= 5
        # Should have "vector" source tag
        for r in results:
            assert r.metadata.get("source") == "vector"

    def test_hybrid_with_no_kg_results(self):
        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = [_sr("vec result")]
        mock_kg = MagicMock()
        mock_kg.search_entities.return_value = []  # no entities found
        mock_services = MagicMock()
        mock_services.embedding_pipeline = mock_pipeline
        mock_services.knowledge_graph = mock_kg

        with patch("papermind.services.services", mock_services):
            results = hybrid_retrieve("test query", n_results=5, use_kg=True)
        # Falls back to vector-only
        assert len(results) >= 1
