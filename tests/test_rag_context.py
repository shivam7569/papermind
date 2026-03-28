"""Tests for RAG context assembly: ordering, deduplication, compression."""

import pytest

from papermind.rag.context import (
    count_tokens,
    lost_in_middle_order,
    deduplicate,
    compress_context,
    assemble_context,
)
from papermind.models import SearchResult


def _sr(text: str, score: float = 0.5, **kwargs) -> SearchResult:
    """Helper to create a SearchResult."""
    return SearchResult(text=text, score=score, chunk_id=text[:10], **kwargs)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_known_text(self):
        # "Hello world" should be a small number of tokens
        n = count_tokens("Hello world")
        assert n > 0
        assert n < 10

    def test_longer_text(self):
        short = count_tokens("hello")
        long = count_tokens("hello " * 100)
        assert long > short


class TestLostInMiddleOrder:
    def test_empty(self):
        assert lost_in_middle_order([]) == []

    def test_single_item(self):
        r = [_sr("only one")]
        result = lost_in_middle_order(r)
        assert len(result) == 1
        assert result[0].text == "only one"

    def test_two_items(self):
        r = [_sr("first"), _sr("second")]
        result = lost_in_middle_order(r)
        assert len(result) == 2
        # With 2 items, returned as-is
        assert result[0].text == "first"
        assert result[1].text == "second"

    def test_six_items_pattern(self):
        items = [_sr(f"item{i}") for i in range(6)]
        result = lost_in_middle_order(items)
        assert len(result) == 6
        # Top half: [0,1,2], bottom half reversed: [5,4,3]
        # Result: [0,1,2,5,4,3]
        assert result[0].text == "item0"
        assert result[1].text == "item1"
        assert result[2].text == "item2"
        assert result[3].text == "item5"
        assert result[4].text == "item4"
        assert result[5].text == "item3"

    def test_preserves_best_at_start(self):
        items = [_sr(f"item{i}", score=1.0 - i * 0.1) for i in range(4)]
        result = lost_in_middle_order(items)
        # Best item (highest score) should be first
        assert result[0].text == "item0"


class TestDeduplicate:
    def test_exact_duplicates_removed(self):
        results = [
            _sr("This is a test sentence about transformers"),
            _sr("This is a test sentence about transformers"),
            _sr("Completely different text"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 2

    def test_near_duplicates_removed(self):
        results = [
            _sr("Transformer models use attention mechanisms for sequence processing"),
            _sr("Transformer models use attention mechanisms for sequence processing tasks"),
            _sr("Convolutional neural networks for image classification"),
        ]
        deduped = deduplicate(results, overlap_threshold=0.7)
        assert len(deduped) == 2

    def test_unique_kept(self):
        results = [
            _sr("First unique text about machine learning"),
            _sr("Second completely different text about cooking"),
            _sr("Third text about quantum physics concepts"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 3

    def test_empty_input(self):
        assert deduplicate([]) == []

    def test_empty_text_filtered(self):
        results = [_sr(""), _sr("  "), _sr("actual content")]
        deduped = deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].text == "actual content"

    def test_substring_overlap(self):
        results = [
            _sr("The full parent chunk with lots of detailed content about transformers"),
            _sr("lots of detailed content about transformers"),  # substring of above
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 1


class TestCompressContext:
    def test_fits_within_budget(self):
        results = [_sr("Word " * 100) for _ in range(5)]
        compressed = compress_context(results, token_budget=200)
        total_tokens = sum(count_tokens(r.text) for r in compressed)
        assert total_tokens <= 200

    def test_skips_short_chunks(self):
        results = [
            _sr("hi"),  # too short (< 20 tokens default)
            _sr("This is a sufficiently long chunk that should be included in the final "
                "results because it contains enough tokens to pass the minimum threshold "
                "for useful context in the retrieval augmented generation pipeline"),
        ]
        compressed = compress_context(results, token_budget=10000)
        # Only the long chunk should be included
        assert len(compressed) == 1
        assert "sufficiently" in compressed[0].text

    def test_empty_results(self):
        compressed = compress_context([], token_budget=4096)
        assert compressed == []

    def test_single_large_chunk_truncated_behavior(self):
        """If no chunk fits the budget, the first qualifying chunk is still included."""
        results = [_sr("Word " * 1000)]  # very large
        compressed = compress_context(results, token_budget=10)
        # Should still include it since nothing else was selected
        assert len(compressed) == 1


class TestAssembleContext:
    def test_full_pipeline_produces_string(self):
        results = [
            _sr("Deep learning uses neural networks for pattern recognition in computer vision "
                "and natural language processing tasks across many different application domains",
                section_title="Intro", paper_id="p1"),
            _sr("Convolutional networks are effective for image classification tasks and have "
                "been shown to achieve state of the art results on many benchmark datasets",
                section_title="Methods", paper_id="p1"),
            _sr("Recurrent networks handle sequential data processing effectively and are widely "
                "used for time series prediction and natural language understanding applications",
                section_title="Background", paper_id="p2"),
        ]
        context_str, selected = assemble_context(results, token_budget=4096)
        assert isinstance(context_str, str)
        assert len(context_str) > 0
        assert "[Source" in context_str
        assert len(selected) > 0

    def test_source_headers_present(self):
        results = [
            _sr("The attention mechanism allows the model to focus on relevant parts of the "
                "input sequence when generating each element of the output sequence in neural "
                "machine translation and other sequence to sequence tasks",
                section_title="Attention", paper_id="p1"),
        ]
        context_str, _ = assemble_context(results, token_budget=4096)
        assert "[Source 1]" in context_str
        assert "Attention" in context_str
        assert "p1" in context_str

    def test_empty_results(self):
        context_str, selected = assemble_context([], token_budget=4096)
        assert context_str == ""
        assert selected == []
