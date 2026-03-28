"""Tests for PDF parser using the project's own design doc."""

from pathlib import Path

import pytest

from papermind.ingestion.pdf_parser import parse_pdf

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DESIGN_PDF = DOCS_DIR / "Unified AI Research System on 12GB VRAM_ A Complete Design Guide.pdf"


@pytest.mark.skipif(not DESIGN_PDF.exists(), reason="Design doc PDF not found")
def test_parse_design_doc():
    paper, sections = parse_pdf(DESIGN_PDF)

    assert paper.title  # should extract a title
    assert paper.num_pages > 0
    assert len(sections) > 0

    # Check that sections have text
    total_text = sum(len(s.text) for s in sections)
    assert total_text > 1000  # design doc has substantial text


@pytest.mark.skipif(not DESIGN_PDF.exists(), reason="Design doc PDF not found")
def test_sections_have_valid_pages():
    _, sections = parse_pdf(DESIGN_PDF)

    for s in sections:
        assert s.page_start >= 0
        assert s.page_end >= s.page_start
