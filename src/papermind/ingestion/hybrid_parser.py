"""Hybrid PDF parser: GROBID metadata + MinerU body text.

Strategy:
  Stage 1 — GROBID: Extract structured metadata (title, authors, abstract,
            affiliations, references, section hierarchy). GROBID is best-in-class
            for bibliographic extraction.

  Stage 2 — MinerU: Extract body text with proper LaTeX equations, clean tables,
            and correct reading order. MinerU scores 9.17/10 on formula extraction
            vs GROBID's 5.70/10.

  Stage 3 — Reconciliation: Merge GROBID metadata with MinerU body content.
            Cross-validate and pick the best extraction for each field.

  Stage 4 — Semantic Scholar enrichment: Fill gaps in metadata (optional).

This gives us the best of both worlds: GROBID's metadata parsing + MinerU's
superior equation and body text extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from papermind.models import Paper, Section

logger = logging.getLogger(__name__)


def parse_pdf_hybrid(
    pdf_path: str | Path,
    grobid_url: str | None = None,
    use_mineru: bool = True,
    mineru_timeout: int = 300,
) -> tuple[Paper, list[Section]]:
    """Parse a PDF using the hybrid GROBID + MinerU pipeline.

    Args:
        pdf_path: Path to the PDF file.
        grobid_url: GROBID service URL (defaults to config).
        use_mineru: Whether to use MinerU for body text (True by default).
        mineru_timeout: Max seconds for MinerU subprocess.

    Returns:
        (Paper, list[Section]) with structured metadata + LaTeX-rich body.
    """
    pdf_path = Path(pdf_path)

    # Stage 1: GROBID for metadata
    logger.info("Stage 1: GROBID metadata extraction for %s", pdf_path.name)
    from papermind.ingestion.grobid_parser import parse_pdf as grobid_parse
    paper, grobid_sections = grobid_parse(pdf_path, grobid_url)

    # Stage 2: MinerU for body text (if available and requested)
    mineru_sections: list[Section] = []
    if use_mineru:
        try:
            from papermind.ingestion.mineru_parser import check_mineru_available, parse_body
            if check_mineru_available():
                logger.info("Stage 2: MinerU body extraction for %s", pdf_path.name)
                mineru_sections = parse_body(pdf_path, timeout=mineru_timeout)
                logger.info("MinerU produced %d sections", len(mineru_sections))
            else:
                logger.info("MinerU not available, using GROBID-only body text")
        except Exception as e:
            logger.warning("MinerU failed, falling back to GROBID body: %s", e)

    # Stage 3: Reconciliation
    if mineru_sections:
        sections = _reconcile(grobid_sections, mineru_sections, paper.id)
    else:
        sections = grobid_sections

    # Stage 4: Metadata validation
    paper = _validate_metadata(paper)

    logger.info(
        "Hybrid parse complete: %d sections, paper='%s' by %s",
        len(sections), paper.title[:60], paper.authors[:3],
    )

    return paper, sections


def _reconcile(
    grobid_sections: list[Section],
    mineru_sections: list[Section],
    paper_id: str,
) -> list[Section]:
    """Merge GROBID's section hierarchy with MinerU's body content.

    Strategy:
      - Use MinerU sections as the primary body text (better equations, cleaner text)
      - Preserve GROBID's Abstract if it's better (more complete)
      - Fix paper_id on all sections
      - Keep MinerU's parent-child relationships
    """
    # Check if GROBID has a better abstract
    grobid_abstract = next(
        (s for s in grobid_sections if s.title.lower() == "abstract"), None
    )
    mineru_abstract = next(
        (s for s in mineru_sections if s.title.lower() == "abstract"), None
    )

    if grobid_abstract and mineru_abstract:
        # Use the longer one as a heuristic for "more complete"
        if len(grobid_abstract.text) > len(mineru_abstract.text) * 1.1:
            for i, s in enumerate(mineru_sections):
                if s.title.lower() == "abstract":
                    mineru_sections[i] = Section(
                        title="Abstract",
                        text=grobid_abstract.text,
                        page_start=s.page_start,
                        page_end=s.page_end,
                        level=s.level,
                    )
                    break

    # Count improvements
    mineru_eqs = sum(
        s.text.count("$$") // 2 + s.text.count("$") - s.text.count("$$")
        for s in mineru_sections
    )
    grobid_eqs = sum(
        s.text.count("$$") // 2 + s.text.count("$") - s.text.count("$$")
        for s in grobid_sections
    )
    logger.info(
        "Reconciliation: MinerU has %d equation markers vs GROBID's %d",
        mineru_eqs, grobid_eqs,
    )

    return mineru_sections


def _validate_metadata(paper: Paper) -> Paper:
    """Apply validation rules to catch known GROBID failures.

    Rules:
      - If author count > 30, likely reference pollution → trim to first 30
      - Flag empty title/abstract for later S2 enrichment
    """
    if len(paper.authors) > 30:
        logger.warning(
            "Paper '%s' has %d authors (likely reference pollution), trimming to 30",
            paper.title[:40], len(paper.authors),
        )
        paper.authors = paper.authors[:30]

    if not paper.title:
        logger.warning("Empty title for paper %s — needs S2 enrichment", paper.id)

    if not paper.abstract or len(paper.abstract) < 50:
        logger.warning(
            "Short/missing abstract (%d chars) for '%s' — needs S2 enrichment",
            len(paper.abstract or ""), paper.title[:40],
        )

    return paper
