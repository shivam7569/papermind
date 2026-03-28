"""Tests for the text chunker with parent-child strategy."""

from papermind.ingestion.chunker import chunk_sections
from papermind.models import Section


def test_basic_chunking():
    sections = [
        Section(title="Intro", text="This is a test. " * 100, page_start=0, page_end=0)
    ]
    chunks = chunk_sections(sections, paper_id="p1")
    assert len(chunks) > 0
    assert all(c.paper_id == "p1" for c in chunks)
    assert all(c.section_title == "Intro" for c in chunks)
    assert all(c.token_count > 0 for c in chunks)


def test_parent_child_structure():
    sections = [
        Section(title="Methods", text="We propose a method. " * 80, page_start=0, page_end=1)
    ]
    chunks = chunk_sections(sections, paper_id="p1")

    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    assert len(parents) == 1
    assert len(children) >= 1

    # Every child must link to the parent
    for child in children:
        assert child.parent_id == parents[0].id


def test_respects_section_boundaries():
    sections = [
        Section(title="A", text="Content A. " * 50, page_start=0, page_end=0),
        Section(title="B", text="Content B. " * 50, page_start=1, page_end=1),
    ]
    chunks = chunk_sections(sections, paper_id="p1")
    # No chunk should mix section titles
    for c in chunks:
        assert c.section_title in ("A", "B")

    # Should have 2 parents (one per section)
    parents = [c for c in chunks if c.chunk_type == "parent"]
    assert len(parents) == 2


def test_short_text_produces_parent_and_child():
    sections = [
        Section(title="Short", text="Just a short section with enough text to be useful for the system.", page_start=0, page_end=0)
    ]
    chunks = chunk_sections(sections, paper_id="p1")
    parents = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    # Short section: 1 parent + 1 child (same text, child links to parent)
    assert len(parents) == 1
    assert len(children) == 1
    assert children[0].parent_id == parents[0].id


def test_empty_section():
    sections = [
        Section(title="Empty", text="", page_start=0, page_end=0)
    ]
    chunks = chunk_sections(sections, paper_id="p1")
    assert len(chunks) == 0


def test_no_orphan_children():
    """Every child chunk must have a parent_id pointing to an existing parent."""
    sections = [
        Section(title="Big", text="Deep learning is great. " * 200, page_start=0, page_end=3),
    ]
    chunks = chunk_sections(sections, paper_id="p1")
    parents = {c.id for c in chunks if c.chunk_type == "parent"}
    children = [c for c in chunks if c.chunk_type == "child"]

    assert len(parents) >= 1
    assert len(children) >= 2  # big section should split into multiple children

    for child in children:
        assert child.parent_id in parents, f"Orphan child: {child.id}"
