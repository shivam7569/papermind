"""Tests for the cross-encoder reranker.

These tests load BAAI/bge-reranker-v2-m3 (~2.3GB), so they are marked slow.
Run with: pytest -m slow tests/test_rag_reranker.py
"""

import pytest

from papermind.models import SearchResult
from papermind.rag.reranker import Reranker, rerank


def _sr(text: str, score: float = 0.5, chunk_id: str = "") -> SearchResult:
    return SearchResult(
        text=text,
        score=score,
        chunk_id=chunk_id or text[:12],
        paper_id="p1",
        section_title="Test",
    )


@pytest.fixture(scope="module")
def reranker():
    """Load the reranker model once for all tests in this module."""
    r = Reranker(device="cpu")
    yield r
    r.unload()


@pytest.mark.slow
class TestReranker:
    def test_empty_results(self):
        results = rerank(query="test", results=[], top_k=5)
        assert results == []

    def test_score_pairs_basic(self, reranker):
        scores = reranker.score_pairs("What is attention?", [
            "Attention mechanism allows the model to focus on relevant parts",
            "Cooking pasta requires boiling water",
        ])
        assert len(scores) == 2
        # Relevant passage should score higher
        assert scores[0] > scores[1]

    def test_scores_in_0_1_range(self, reranker):
        scores = reranker.score_pairs("deep learning", [
            "Neural networks for classification",
            "Random unrelated text",
        ])
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_rerank_filters_below_threshold(self, reranker):
        results = [
            _sr("Attention is a mechanism in transformer models"),
            _sr("Completely irrelevant text about gardening and plants"),
        ]
        reranked = rerank("What is attention in transformers?", results, score_threshold=0.5)
        # High threshold should filter some results
        for r in reranked:
            assert r.score >= 0.5

    def test_rerank_respects_top_k(self, reranker):
        results = [_sr(f"Result {i} about neural networks") for i in range(10)]
        reranked = rerank("neural networks", results, top_k=3, score_threshold=0.0)
        assert len(reranked) <= 3

    def test_metadata_preserved_with_scores(self, reranker):
        results = [_sr("Attention mechanisms in deep learning", score=0.8)]
        reranked = rerank("attention", results, score_threshold=0.0)
        assert len(reranked) == 1
        assert "retrieval_score" in reranked[0].metadata
        assert "reranker_score" in reranked[0].metadata
        assert reranked[0].metadata["retrieval_score"] == 0.8

    def test_rerank_preserves_paper_id(self, reranker):
        results = [_sr("Some text", chunk_id="c1")]
        results[0].paper_id = "paper123"
        reranked = rerank("query", results, score_threshold=0.0)
        assert reranked[0].paper_id == "paper123"

    def test_empty_passages(self, reranker):
        scores = reranker.score_pairs("test", [])
        assert scores == []
