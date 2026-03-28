"""Tests for heuristic-based entity and relationship extraction."""

import pytest

from papermind.ingestion.entity_extractor import extract_entities, _clean_name
from papermind.models import Section


def _make_sections(text: str) -> list[Section]:
    return [Section(title="Test", text=text, page_start=0, page_end=0)]


class TestExtractMethods:
    def test_propose_pattern(self):
        sections = _make_sections("We propose a novel TransformerNet for sequence modeling.")
        entities, _ = extract_entities(sections, "p1")
        names = {e.name for e in entities}
        assert any("Transformer" in n for n in names)

    def test_model_suffix_pattern(self):
        sections = _make_sections("We compare against ResNet and EfficientNet baselines.")
        entities, _ = extract_entities(sections, "p1")
        names = {e.name for e in entities}
        assert "ResNet" in names
        assert "EfficientNet" in names

    def test_bert_suffix_pattern(self):
        """The regex matches names ending with known suffixes like BERT, GAN, etc."""
        sections = _make_sections("We compare against RoBERTa and CodeBERT on this task.")
        entities, _ = extract_entities(sections, "p1")
        names = {e.name for e in entities}
        assert "CodeBERT" in names


class TestExtractDatasets:
    def test_evaluated_on_pattern(self):
        sections = _make_sections("We evaluated on the ImageNet dataset for image classification.")
        entities, _ = extract_entities(sections, "p1")
        names = {e.name for e in entities}
        assert "ImageNet" in names

    def test_benchmark_pattern(self):
        sections = _make_sections("datasets such as CIFAR10, MNIST are commonly used")
        entities, _ = extract_entities(sections, "p1")
        names = {e.name for e in entities}
        # Should extract at least one dataset
        dataset_entities = [e for e in entities if e.entity_type == "dataset"]
        assert len(dataset_entities) >= 1


class TestExtractRelationships:
    def test_outperforms_relationship(self):
        text = (
            "We propose a novel SuperNet for image classification. "
            "SuperNet outperforms ResNet on all benchmarks."
        )
        sections = _make_sections(text)
        entities, relationships = extract_entities(sections, "p1")
        outperforms = [r for r in relationships if r.relation_type == "outperforms"]
        assert len(outperforms) >= 1

    def test_evaluated_on_relationship(self):
        text = (
            "We propose DeepNet for classification. "
            "DeepNet evaluated on the ImageNet dataset achieves good results."
        )
        sections = _make_sections(text)
        entities, relationships = extract_entities(sections, "p1")
        eval_rels = [r for r in relationships if r.relation_type == "evaluated_on"]
        assert len(eval_rels) >= 1


class TestEdgeCases:
    def test_no_entities_from_empty_sections(self):
        sections = [Section(title="Empty", text="", page_start=0, page_end=0)]
        entities, relationships = extract_entities(sections, "p1")
        assert len(entities) == 0
        assert len(relationships) == 0

    def test_deduplication(self):
        text = "ResNet is great. ResNet outperforms everything. ResNet is the best."
        sections = _make_sections(text)
        entities, _ = extract_entities(sections, "p1")
        resnet_entities = [e for e in entities if e.name == "ResNet"]
        assert len(resnet_entities) == 1

    def test_short_names_filtered(self):
        """Names with 2 or fewer characters should be filtered."""
        text = "We propose an AI model."
        sections = _make_sections(text)
        entities, _ = extract_entities(sections, "p1")
        for e in entities:
            assert len(e.name) > 2

    def test_stop_words_filtered(self):
        """Common stop words should not appear as entity names."""
        from papermind.ingestion.entity_extractor import _STOP_WORDS
        text = "The proposed method, called TransformerGAN, works well."
        sections = _make_sections(text)
        entities, _ = extract_entities(sections, "p1")
        for e in entities:
            assert e.name not in _STOP_WORDS

    def test_multiple_sections(self):
        sections = [
            Section(title="Intro", text="We propose SuperBERT for NLP tasks.", page_start=0, page_end=0),
            Section(title="Exp", text="Evaluated on the SQuAD dataset, SuperBERT outperforms BERT.", page_start=1, page_end=1),
        ]
        entities, relationships = extract_entities(sections, "p1")
        assert len(entities) >= 2
        names = {e.name for e in entities}
        assert "SuperBERT" in names


class TestCleanName:
    def test_strips_whitespace(self):
        assert _clean_name("  Hello  ") == "Hello"

    def test_strips_punctuation(self):
        assert _clean_name("Method,") == "Method"
        assert _clean_name("(Model)") == "Model"

    def test_removes_trailing_common_words(self):
        assert _clean_name("Model that") == "Model"
        assert _clean_name("Method is") == "Method"
