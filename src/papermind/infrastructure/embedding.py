"""Embedding model wrapper using sentence-transformers.

Supports nomic-embed-text-v1.5 (default, 768d) with search_document:/search_query:
task prefixes, and falls back gracefully to models without prefixes (e.g. all-MiniLM-L6-v2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from papermind.config import get_settings

# Models that use task-specific prefixes for asymmetric retrieval.
_PREFIX_MODELS: dict[str, tuple[str, str]] = {
    "nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1": ("search_document: ", "search_query: "),
}


class EmbeddingService:
    """Wraps a sentence-transformer model for text embedding.

    Default model: nomic-ai/nomic-embed-text-v1.5
      - 768 dimensions (or truncated via Matryoshka to 256/512)
      - Trained with contrastive + distillation objectives
      - Uses task prefixes: ``search_document:`` for indexing,
        ``search_query:`` for retrieval
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        matryoshka_dim: int | None = None,
    ):
        settings = get_settings()
        self._model_name = model_name or settings.embedding.model_name
        self._device = device or settings.embedding.device
        self._matryoshka_dim = matryoshka_dim
        self._model: SentenceTransformer | None = None

        # Resolve task prefixes
        prefixes = _PREFIX_MODELS.get(self._model_name)
        self._doc_prefix = prefixes[0] if prefixes else ""
        self._query_prefix = prefixes[1] if prefixes else ""

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            kwargs: dict = {"trust_remote_code": True}
            if self._matryoshka_dim:
                kwargs["truncate_dim"] = self._matryoshka_dim
            self._model = SentenceTransformer(
                self._model_name, device=self._device, **kwargs
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Effective embedding dimension (respects Matryoshka truncation)."""
        if self._matryoshka_dim:
            return self._matryoshka_dim
        return self.model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed_texts(
        self,
        texts: list[str],
        is_query: bool = False,
    ) -> NDArray[np.float32]:
        """Embed a batch of texts. Returns (N, dim) float32 array.

        Args:
            texts: Input texts to embed.
            is_query: If True, prepend query prefix (for retrieval).
                      If False, prepend document prefix (for indexing).
        """
        settings = get_settings()
        prefix = self._query_prefix if is_query else self._doc_prefix
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts

        embeddings = self.model.encode(
            prefixed,
            batch_size=settings.embedding.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts for indexing (document prefix)."""
        return self.embed_texts(texts, is_query=False)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query text (query prefix). Returns (dim,) array."""
        return self.embed_texts([text], is_query=True)[0]

    def embed_texts_list(self, texts: list[str]) -> list[list[float]]:
        """Legacy helper: returns list-of-lists for ChromaDB compatibility."""
        return self.embed_texts(texts, is_query=False).tolist()

    def embed_query_list(self, text: str) -> list[float]:
        """Legacy helper: returns list for ChromaDB compatibility."""
        return self.embed_query(text).tolist()
