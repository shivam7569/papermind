"""Tests for the embedding service.

These tests load the actual sentence-transformers model (nomic-embed-text-v1.5).
The model is ~270MB and cached after first download.
"""

import numpy as np
import pytest

from papermind.infrastructure.embedding import EmbeddingService


@pytest.fixture(scope="module")
def embedding_svc():
    """Load the embedding model once for all tests in this module."""
    return EmbeddingService(device="cpu")


class TestEmbeddingService:
    def test_model_loads(self, embedding_svc):
        assert embedding_svc.model is not None

    def test_embed_documents_shape(self, embedding_svc):
        texts = ["Hello world", "Attention is all you need", "Deep learning"]
        result = embedding_svc.embed_documents(texts)
        assert result.shape[0] == 3
        assert result.shape[1] == embedding_svc.dimension

    def test_embed_query_shape(self, embedding_svc):
        result = embedding_svc.embed_query("What is attention?")
        assert result.ndim == 1
        assert result.shape[0] == embedding_svc.dimension

    def test_embeddings_l2_normalized(self, embedding_svc):
        texts = ["Test normalization", "Another sentence", "Third one here"]
        result = embedding_svc.embed_documents(texts)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_query_embedding_l2_normalized(self, embedding_svc):
        result = embedding_svc.embed_query("test query")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_empty_text_list(self, embedding_svc):
        result = embedding_svc.embed_documents([])
        assert result.shape[0] == 0

    def test_document_vs_query_prefix_different(self, embedding_svc):
        """Document and query embeddings should differ due to task prefixes."""
        text = "transformer architecture for sequence modeling"
        doc_emb = embedding_svc.embed_documents([text])[0]
        query_emb = embedding_svc.embed_query(text)
        # They should be close but not identical due to different prefixes
        cosine_sim = np.dot(doc_emb, query_emb)
        assert cosine_sim < 1.0  # not identical
        assert cosine_sim > 0.5  # but still similar

    def test_dimension_property(self, embedding_svc):
        dim = embedding_svc.dimension
        assert isinstance(dim, int)
        assert dim > 0
        # nomic-embed-text-v1.5 is 768d
        assert dim == 768

    def test_embed_texts_list_returns_lists(self, embedding_svc):
        result = embedding_svc.embed_texts_list(["hello"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)

    def test_embed_query_list_returns_list(self, embedding_svc):
        result = embedding_svc.embed_query_list("hello")
        assert isinstance(result, list)
        assert isinstance(result[0], float)

    def test_similar_texts_have_high_similarity(self, embedding_svc):
        e1 = embedding_svc.embed_query("neural network architecture")
        e2 = embedding_svc.embed_query("deep learning model design")
        e3 = embedding_svc.embed_query("cooking pasta recipe")
        sim_related = np.dot(e1, e2)
        sim_unrelated = np.dot(e1, e3)
        assert sim_related > sim_unrelated
