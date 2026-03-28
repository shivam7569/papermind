"""PDF to structured text extraction using PyMuPDF."""

import re
from pathlib import Path

import pymupdf

from papermind.models import Paper, Section


# Heuristic patterns for section headers in academic papers
_SECTION_PATTERN = re.compile(
    r"^(?:"
    r"(?:\d+\.?\s+)"           # "1 Introduction" or "1. Introduction"
    r"|(?:[A-Z]\.?\s+)"        # "A. Related Work"
    r"|(?:Abstract|Introduction|Related Work|Background|Methods?|Methodology|"
    r"Experiments?|Results?|Discussion|Conclusion|References|Appendix|Acknowledgments)"
    r")",
    re.IGNORECASE,
)


def parse_pdf(pdf_path: str | Path) -> tuple[Paper, list[Section]]:
    """Parse a PDF file into a Paper and list of Sections.

    Uses font-size heuristics and regex patterns to detect section boundaries.
    """
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(str(pdf_path))

    # Extract title from first page (largest font text)
    title = _extract_title(doc)

    # Extract text page by page, detecting sections
    sections = _extract_sections(doc)

    from papermind.models import make_paper_id

    paper = Paper(
        id=make_paper_id(pdf_path),
        title=title,
        source_path=str(pdf_path),
        num_pages=len(doc),
    )

    doc.close()
    return paper, sections


def _extract_title(doc: pymupdf.Document) -> str:
    """Extract the paper title from the first page using font size heuristics."""
    if len(doc) == 0:
        return ""

    page = doc[0]
    blocks = page.get_text("dict")["blocks"]

    max_size = 0.0
    title_lines: list[str] = []

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                size = span["size"]
                text = span["text"].strip()
                if not text:
                    continue
                if size > max_size + 1:  # new largest font
                    max_size = size
                    title_lines = [text]
                elif abs(size - max_size) <= 1:  # same size (tolerance)
                    title_lines.append(text)

    return " ".join(title_lines).strip()


def _extract_sections(doc: pymupdf.Document) -> list[Section]:
    """Extract sections from the document using header detection."""
    sections: list[Section] = []
    current_title = "Preamble"
    current_text: list[str] = []
    current_page_start = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text:
                    continue

                # Check if this line is a section header
                is_header = _is_section_header(line, page)
                if is_header and line_text:
                    # Save previous section
                    if current_text:
                        sections.append(
                            Section(
                                title=current_title,
                                text="\n".join(current_text).strip(),
                                page_start=current_page_start,
                                page_end=page_num,
                            )
                        )
                    current_title = line_text
                    current_text = []
                    current_page_start = page_num
                else:
                    current_text.append(line_text)

    # Don't forget the last section
    if current_text:
        sections.append(
            Section(
                title=current_title,
                text="\n".join(current_text).strip(),
                page_start=current_page_start,
                page_end=len(doc) - 1,
            )
        )

    return sections


def _is_section_header(line: dict, page: pymupdf.Page) -> bool:
    """Determine if a text line is a section header."""
    spans = line.get("spans", [])
    if not spans:
        return False

    text = "".join(s["text"] for s in spans).strip()
    if len(text) < 3 or len(text) > 120:
        return False

    # Check for bold font
    is_bold = any("Bold" in (s.get("font", "") or "") for s in spans)

    # Check for larger-than-body font size
    avg_size = sum(s["size"] for s in spans) / len(spans)
    body_size = _estimate_body_font_size(page)
    is_larger = avg_size > body_size + 1.0

    # Check for section-like pattern
    matches_pattern = bool(_SECTION_PATTERN.match(text))

    # A header if it matches the pattern, or if it's both bold and larger
    return matches_pattern or (is_bold and is_larger)


def _estimate_body_font_size(page: pymupdf.Page) -> float:
    """Estimate the most common (body) font size on a page."""
    size_counts: dict[float, int] = {}
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                size = round(span["size"], 1)
                text = span["text"].strip()
                if text:
                    size_counts[size] = size_counts.get(size, 0) + len(text)

    if not size_counts:
        return 10.0
    return max(size_counts, key=size_counts.get)  # type: ignore[arg-type]
