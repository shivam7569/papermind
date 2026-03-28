"""PDF parsing via GROBID — structured extraction of research papers.

GROBID (GeneRation Of BIbliographic Data) uses ML models trained on millions
of papers to extract: title, authors, abstract, sections with headings,
references, figures, equations, and affiliations.

Requires a running GROBID service (see docker/Dockerfile.grobid).

The parser returns a Paper + hierarchical list of Sections with parent-child
relationships (e.g. "3.1 Data Augmentation" is a child of "3 Methods").
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

from papermind.config import get_settings
from papermind.models import Paper, Section

logger = logging.getLogger(__name__)

# TEI XML namespace
TEI_NS = "http://www.tei-c.org/ns/1.0"


def parse_pdf(
    pdf_path: str | Path,
    grobid_url: str | None = None,
) -> tuple[Paper, list[Section]]:
    """Parse a PDF using GROBID's fulltext endpoint.

    Args:
        pdf_path: Path to the PDF file.
        grobid_url: GROBID service URL. Defaults to config value.

    Returns:
        (Paper, list[Section]) with structured extraction.
    """
    pdf_path = Path(pdf_path)
    settings = get_settings()
    url = grobid_url or settings.grobid.url

    # Call GROBID fulltext endpoint
    tei_xml = _call_grobid(pdf_path, url)

    # Parse TEI XML into structured data
    soup = BeautifulSoup(tei_xml, "xml")

    paper = _extract_paper(soup, pdf_path)
    sections = _extract_sections(soup)

    # If GROBID returned very few sections, it may not have recognized
    # the document structure well. Log a warning.
    if len(sections) <= 1:
        logger.warning(
            "GROBID extracted only %d section(s) from %s — "
            "document may not be a standard research paper",
            len(sections), pdf_path.name,
        )

    paper.num_pages = _estimate_page_count(soup)

    return paper, sections


def _call_grobid(pdf_path: Path, grobid_url: str) -> str:
    """Send PDF to GROBID and return TEI XML response."""
    endpoint = f"{grobid_url.rstrip('/')}/api/processFulltextDocument"

    with open(pdf_path, "rb") as f:
        response = httpx.post(
            endpoint,
            files={"input": (pdf_path.name, f, "application/pdf")},
            data={
                "consolidateHeader": "1",
                "consolidateCitations": "0",
                "includeRawAffiliations": "0",
                "segmentSentences": "1",
            },
            timeout=120.0,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"GROBID returned {response.status_code}: {response.text[:500]}"
        )

    return response.text


def _extract_paper(soup: BeautifulSoup, pdf_path: Path) -> Paper:
    """Extract paper metadata from TEI header."""
    # Title
    title_tag = soup.find("title", attrs={"type": "main"})
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Authors — only from the header's sourceDesc, not from references
    authors: list[str] = []
    header = soup.find("teiHeader")
    if header:
        source_desc = header.find("sourceDesc")
        if source_desc:
            for author_tag in source_desc.find_all("author"):
                persname = author_tag.find("persName")
                if persname:
                    forename = persname.find("forename")
                    surname = persname.find("surname")
                    parts = []
                    if forename:
                        parts.append(forename.get_text(strip=True))
                    if surname:
                        parts.append(surname.get_text(strip=True))
                    if parts:
                        authors.append(" ".join(parts))

    # Abstract
    abstract_tag = soup.find("abstract")
    abstract = ""
    if abstract_tag:
        paragraphs = abstract_tag.find_all("p")
        if paragraphs:
            abstract = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
        else:
            abstract = abstract_tag.get_text(strip=True)

    return Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        source_path=str(pdf_path),
    )


def _extract_sections(soup: BeautifulSoup) -> list[Section]:
    """Extract sections from TEI body with hierarchical structure.

    GROBID TEI uses nested <div> elements with <head> for section titles.
    Section numbering (e.g. "3.1") indicates hierarchy.
    """
    body = soup.find("body")
    if not body:
        logger.warning("No <body> found in TEI XML")
        return []

    sections: list[Section] = []

    # First, extract abstract as a section
    abstract_tag = soup.find("abstract")
    if abstract_tag:
        abstract_text = _tag_to_text(abstract_tag)
        if abstract_text.strip():
            sections.append(Section(
                title="Abstract",
                text=abstract_text,
                page_start=0,
                page_end=0,
                level=1,
            ))

    # Process body divs
    _process_div(body, sections, level=1)

    # Extract references section if present
    back = soup.find("back")
    if back:
        ref_list = back.find("listBibl")
        if ref_list:
            ref_texts = []
            for bibl in ref_list.find_all("biblStruct"):
                ref_text = _format_reference(bibl)
                if ref_text:
                    ref_texts.append(ref_text)
            if ref_texts:
                sections.append(Section(
                    title="References",
                    text="\n\n".join(ref_texts),
                    page_start=0,
                    page_end=0,
                    level=1,
                ))

    return sections


def _process_div(element: Tag, sections: list[Section], level: int) -> None:
    """Recursively process a TEI div element into sections."""
    for child in element.children:
        if not isinstance(child, Tag):
            continue

        if child.name == "div":
            head = child.find("head", recursive=False)

            if head:
                title = head.get_text(strip=True)
                # Determine heading level from numbering or the n attribute
                n_attr = head.get("n", "")
                detected_level = _detect_heading_level(title, n_attr, level)

                # Collect text from paragraphs in this div (not nested divs)
                text_parts: list[str] = []
                for sub in child.children:
                    if not isinstance(sub, Tag):
                        continue
                    if sub.name == "p":
                        text_parts.append(_tag_to_text(sub))
                    elif sub.name == "formula":
                        text_parts.append(_format_formula(sub))
                    elif sub.name == "figure":
                        caption = sub.find("figDesc")
                        if caption:
                            fig_head = sub.find("head")
                            prefix = fig_head.get_text(strip=True) + ": " if fig_head else ""
                            text_parts.append(f"[{prefix}{caption.get_text(strip=True)}]")

                section_text = "\n\n".join(t for t in text_parts if t.strip())

                if section_text.strip():
                    sections.append(Section(
                        title=title,
                        text=section_text,
                        page_start=0,
                        page_end=0,
                        level=detected_level,
                    ))

                # Recurse into nested divs (subsections)
                _process_div(child, sections, level=detected_level + 1)

            else:
                # Div without a head — just recurse
                _process_div(child, sections, level=level)


def _detect_heading_level(title: str, n_attr: str, parent_level: int) -> int:
    """Detect heading level from numbering pattern.

    Examples:
        "1" or "1." → level 1
        "3.1" or "3.1." → level 2
        "3.1.2" → level 3
    """
    # Use GROBID's n attribute first
    if n_attr:
        dots = n_attr.strip(".").count(".")
        return dots + 1

    # Fallback: detect from title text
    match = re.match(r"^(\d+(?:\.\d+)*)\s", title)
    if match:
        dots = match.group(1).count(".")
        return dots + 1

    return parent_level


def _tag_to_text(tag: Tag) -> str:
    """Convert a TEI tag to clean text, preserving paragraph structure."""
    parts: list[str] = []

    for child in tag.children:
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, Tag):
            if child.name == "ref":
                # Inline reference — keep the text
                parts.append(child.get_text())
            elif child.name == "formula":
                parts.append(_format_formula(child))
            elif child.name == "s":
                # Sentence tag from segmentSentences=1
                parts.append(child.get_text())
            else:
                parts.append(child.get_text())

    return " ".join("".join(parts).split())


def _format_formula(formula_tag: Tag) -> str:
    """Format a LaTeX formula from TEI."""
    text = formula_tag.get_text(strip=True)
    if not text:
        return ""
    # Check if it's display or inline math
    formula_type = formula_tag.get("type", "")
    if formula_type == "display":
        return f"\n$${text}$$\n"
    return f"${text}$"


def _format_reference(bibl: Tag) -> str:
    """Format a bibliography entry from TEI."""
    parts: list[str] = []

    # Authors
    authors = []
    for author in bibl.find_all("author"):
        persname = author.find("persName")
        if persname:
            surname = persname.find("surname")
            forename = persname.find("forename")
            name_parts = []
            if forename:
                name_parts.append(forename.get_text(strip=True))
            if surname:
                name_parts.append(surname.get_text(strip=True))
            if name_parts:
                authors.append(" ".join(name_parts))

    if authors:
        parts.append(", ".join(authors[:3]))
        if len(authors) > 3:
            parts.append("et al.")

    # Title
    title = bibl.find("title")
    if title:
        parts.append(f'"{title.get_text(strip=True)}"')

    # Year
    date = bibl.find("date")
    if date:
        year = date.get("when", "") or date.get_text(strip=True)
        if year:
            parts.append(f"({year[:4]})")

    return " ".join(parts)


def _estimate_page_count(soup: BeautifulSoup) -> int:
    """Estimate page count from TEI (GROBID doesn't always include this)."""
    # Look for page references in the XML
    pages = soup.find_all(attrs={"coords": True})
    max_page = 0
    for elem in pages:
        coords = elem.get("coords", "")
        # coords format: "page,x1,y1,x2,y2;page,x1,y1,x2,y2"
        for segment in coords.split(";"):
            parts = segment.split(",")
            if parts and parts[0].isdigit():
                max_page = max(max_page, int(parts[0]))
    return max_page + 1 if max_page > 0 else 0


def check_grobid_health(grobid_url: str | None = None) -> bool:
    """Check if GROBID service is running and healthy."""
    settings = get_settings()
    url = grobid_url or settings.grobid.url
    try:
        resp = httpx.get(f"{url.rstrip('/')}/api/isalive", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False
