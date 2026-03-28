"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Initial bi-encoder retrieval is fast but approximate — it embeds query and
documents independently. A cross-encoder jointly encodes each query-document
pair, producing much more accurate relevance scores.

BGE-Reranker-v2-m3:
  - 568M params, XLMRoberta-based, multilingual
  - ~2.3 GB VRAM at FP32
  - +6 NDCG@10 over ms-marco-MiniLM on BEIR benchmarks
  - Supports up to 8192 tokens (good for long paper chunks)
  - Standard cross-encoder: works with sentence-transformers or raw transformers

Pipeline: query + N candidates → cross-encoder → top-K reranked results.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from papermind.models import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Cross-encoder reranker for post-retrieval relevance scoring."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model on first use."""
        if self._model is not None:
            return
        logger.info("Loading reranker: %s (FP32, device=%s)", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info("Reranker loaded on %s", self.device)

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Score query-passage pairs. Returns sigmoid-normalized scores in [0, 1]."""
        self._ensure_loaded()
        if not passages:
            return []

        pairs = [[query, p] for p in passages]
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            logits = self._model(**inputs).logits.view(-1).float()
            scores = torch.sigmoid(logits).tolist()
        return scores

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Reranker unloaded")


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Get the singleton reranker instance."""
    return Reranker()


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int | None = None,
    score_threshold: float = 0.01,
) -> list[SearchResult]:
    """Rerank search results using the cross-encoder.

    Args:
        query: The user's search query.
        results: Initial retrieval results from bi-encoder search.
        top_k: Return only top K after reranking. None = all above threshold.
        score_threshold: Drop results below this relevance score.

    Returns:
        Reranked SearchResults with updated scores.
    """
    if not results:
        return []

    reranker = get_reranker()
    passages = [r.text for r in results]
    scores = reranker.score_pairs(query, passages)

    reranked = []
    for result, score in zip(results, scores):
        if score < score_threshold:
            continue
        reranked.append(SearchResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=float(score),
            paper_id=result.paper_id,
            section_title=result.section_title,
            metadata={
                **result.metadata,
                "retrieval_score": result.score,
                "reranker_score": float(score),
            },
        ))

    reranked.sort(key=lambda r: r.score, reverse=True)

    if top_k is not None:
        reranked = reranked[:top_k]

    logger.info(
        "Reranked %d → %d results (top: %.3f)",
        len(results), len(reranked),
        reranked[0].score if reranked else 0,
    )
    return reranked
