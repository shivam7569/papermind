"""Tests for all data models in papermind.models."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from papermind.models import (
    Chunk,
    Entity,
    LatexEquation,
    Paper,
    Relationship,
    SearchResult,
    Section,
    make_paper_id,
)


# --- Paper ---


class TestPaper:
    def test_creation_with_defaults(self):
        p = Paper()
        assert p.id  # non-empty auto-generated
        assert p.title == ""
        assert p.authors == []
        assert p.abstract == ""
        assert p.source_path == ""
        assert p.num_pages == 0
        assert p.num_chunks == 0
        assert p.num_entities == 0
        assert isinstance(p.created_at, datetime)

    def test_creation_with_values(self):
        p = Paper(
            id="abc123",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer"],
            abstract="We propose a new architecture...",
            source_path="/tmp/paper.pdf",
            num_pages=15,
        )
        assert p.id == "abc123"
        assert p.title == "Attention Is All You Need"
        assert len(p.authors) == 2
        assert p.num_pages == 15

    def test_unique_ids(self):
        p1 = Paper()
        p2 = Paper()
        assert p1.id != p2.id

    def test_created_at_is_utc(self):
        p = Paper()
        assert p.created_at.tzinfo is not None

    def test_authors_serialization_roundtrip(self):
        p = Paper(authors=["Alice", "Bob", "Charlie"])
        data = p.model_dump()
        p2 = Paper(**data)
        assert p2.authors == ["Alice", "Bob", "Charlie"]


# --- make_paper_id ---


class TestMakePaperId:
    def test_deterministic_same_file(self, tmp_dir):
        pdf_path = tmp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content for test")
        id1 = make_paper_id(pdf_path)
        id2 = make_paper_id(pdf_path)
        assert id1 == id2

    def test_different_files_different_ids(self, tmp_dir):
        f1 = tmp_dir / "a.pdf"
        f2 = tmp_dir / "b.pdf"
        f1.write_bytes(b"%PDF-1.4 content A")
        f2.write_bytes(b"%PDF-1.4 content B")
        assert make_paper_id(f1) != make_paper_id(f2)

    def test_id_is_12_chars_hex(self, tmp_dir):
        f = tmp_dir / "test.pdf"
        f.write_bytes(b"some bytes")
        pid = make_paper_id(f)
        assert len(pid) == 12
        int(pid, 16)  # should not raise

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            make_paper_id("/nonexistent/fake.pdf")

    def test_same_content_different_path(self, tmp_dir):
        content = b"%PDF-1.4 identical content"
        f1 = tmp_dir / "dir1" / "paper.pdf"
        f2 = tmp_dir / "dir2" / "paper.pdf"
        f1.parent.mkdir(parents=True)
        f2.parent.mkdir(parents=True)
        f1.write_bytes(content)
        f2.write_bytes(content)
        assert make_paper_id(f1) == make_paper_id(f2)


# --- Section ---


class TestSection:
    def test_valid_creation(self):
        s = Section(title="Introduction", text="Some text here.", page_start=0, page_end=2)
        assert s.title == "Introduction"
        assert s.level == 1  # default

    def test_empty_title(self):
        s = Section(title="", text="content", page_start=0, page_end=0)
        assert s.title == ""

    def test_empty_text(self):
        s = Section(title="Empty", text="", page_start=0, page_end=0)
        assert s.text == ""

    def test_page_end_before_start_allowed(self):
        # Pydantic doesn't enforce page_end >= page_start by default
        s = Section(title="X", text="y", page_start=5, page_end=3)
        assert s.page_start == 5
        assert s.page_end == 3

    def test_custom_level(self):
        s = Section(title="Sub", text="t", page_start=0, page_end=0, level=2)
        assert s.level == 2


# --- Chunk ---


class TestChunk:
    def test_creation_defaults(self):
        c = Chunk(text="hello", paper_id="p1")
        assert c.id  # auto-generated
        assert c.chunk_type == "child"
        assert c.parent_id == ""
        assert c.token_count == 0

    def test_parent_type(self):
        c = Chunk(text="parent text", paper_id="p1", chunk_type="parent")
        assert c.chunk_type == "parent"

    def test_child_with_parent_id(self):
        c = Chunk(text="child text", paper_id="p1", parent_id="parent123", chunk_type="child")
        assert c.parent_id == "parent123"

    def test_token_count(self):
        c = Chunk(text="some text", paper_id="p1", token_count=42)
        assert c.token_count == 42


# --- LatexEquation ---


class TestLatexEquation:
    def test_display_equation(self):
        eq = LatexEquation(latex="E = mc^2", display=True, paper_id="p1")
        assert eq.display is True

    def test_inline_equation(self):
        eq = LatexEquation(latex="x^2", display=False, paper_id="p1")
        assert eq.display is False

    def test_defaults(self):
        eq = LatexEquation(latex="y = mx + b")
        assert eq.display is False
        assert eq.context == ""
        assert eq.paper_id == ""

    def test_with_context(self):
        eq = LatexEquation(latex="\\sum_i x_i", context="We define the sum as")
        assert "sum" in eq.context


# --- Entity ---


class TestEntity:
    def test_all_entity_types(self):
        for etype in ["method", "dataset", "metric", "author", "task"]:
            e = Entity(name=f"Test_{etype}", entity_type=etype, paper_id="p1")
            assert e.entity_type == etype

    def test_properties_default_empty(self):
        e = Entity(name="BERT", entity_type="method")
        assert e.properties == {}

    def test_properties_custom(self):
        e = Entity(name="BERT", entity_type="method", properties={"params": "110M"})
        assert e.properties["params"] == "110M"

    def test_unique_ids(self):
        e1 = Entity(name="A", entity_type="method")
        e2 = Entity(name="B", entity_type="method")
        assert e1.id != e2.id

    def test_created_at(self):
        e = Entity(name="X", entity_type="task")
        assert isinstance(e.created_at, datetime)


# --- Relationship ---


class TestRelationship:
    def test_valid_relation_types(self):
        for rtype in ["uses", "outperforms", "extends", "trained_on", "authored_by", "evaluated_on"]:
            r = Relationship(source_id="s1", target_id="t1", relation_type=rtype, paper_id="p1")
            assert r.relation_type == rtype

    def test_defaults(self):
        r = Relationship(source_id="s", target_id="t", relation_type="uses")
        assert r.properties == {}
        assert r.paper_id == ""


# --- SearchResult ---


class TestSearchResult:
    def test_defaults(self):
        sr = SearchResult()
        assert sr.score == 0.0
        assert sr.text == ""
        assert sr.metadata == {}

    def test_score_boundaries(self):
        sr_low = SearchResult(score=0.0)
        sr_high = SearchResult(score=1.0)
        sr_neg = SearchResult(score=-0.5)
        assert sr_low.score == 0.0
        assert sr_high.score == 1.0
        assert sr_neg.score == -0.5  # no validation constraint

    def test_with_metadata(self):
        sr = SearchResult(
            chunk_id="c1",
            text="hello",
            score=0.95,
            paper_id="p1",
            section_title="Intro",
            metadata={"source": "vector"},
        )
        assert sr.metadata["source"] == "vector"
