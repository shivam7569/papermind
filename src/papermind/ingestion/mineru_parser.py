"""PDF body-text extraction via MinerU (magic-pdf).

MinerU excels at:
  - Equation recognition → proper LaTeX (UniMERNet, 9.17/10 benchmark)
  - Layout analysis → removes headers/footers/page numbers
  - Table extraction → HTML tables
  - Reading order detection

MinerU runs in an isolated virtualenv (.venv-mineru) because it requires
transformers 4.x while the main project uses transformers 5.x. We invoke
it via subprocess and parse its Markdown + JSON output.

Usage:
    from papermind.ingestion.mineru_parser import parse_body
    body_sections = parse_body("paper.pdf")
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from papermind.models import Section

logger = logging.getLogger(__name__)

# Path to the isolated MinerU virtualenv
MINERU_VENV = Path(__file__).resolve().parents[3] / ".venv-mineru"
MINERU_PYTHON = MINERU_VENV / "bin" / "python"

# Wrapper script that patches torch.load for PyTorch 2.6+ compatibility
_MINERU_WRAPPER = """\
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

import sys
sys.argv = ['mineru', '-p', '{pdf_path}', '-o', '{output_dir}', '-b', 'pipeline']
from mineru.cli.client import main
main()
"""


def check_mineru_available() -> bool:
    """Check if the MinerU virtualenv and models are available."""
    return MINERU_PYTHON.exists()


def parse_body(
    pdf_path: str | Path,
    timeout: int = 300,
) -> list[Section]:
    """Extract body text with LaTeX equations using MinerU.

    Runs MinerU in a subprocess (isolated venv), parses the Markdown output,
    and returns a list of Section objects with LaTeX equations preserved.

    Args:
        pdf_path: Path to the PDF file.
        timeout: Max seconds to wait for MinerU. Default 300 (5 min).

    Returns:
        List of Section objects with body text and LaTeX equations.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not check_mineru_available():
        raise RuntimeError(
            "MinerU virtualenv not found at .venv-mineru. "
            "Run: uv venv .venv-mineru && uv pip install --python .venv-mineru mineru"
        )

    with tempfile.TemporaryDirectory(prefix="mineru_") as tmpdir:
        # Run MinerU via subprocess
        md_content = _run_mineru(pdf_path, tmpdir, timeout)
        if not md_content:
            return []

        # Parse Markdown into sections
        sections = _markdown_to_sections(md_content)

    return sections


def _run_mineru(pdf_path: Path, output_dir: str, timeout: int) -> str | None:
    """Execute MinerU in the isolated virtualenv."""
    script = _MINERU_WRAPPER.format(
        pdf_path=str(pdf_path).replace("'", "\\'"),
        output_dir=output_dir,
    )

    try:
        result = subprocess.run(
            [str(MINERU_PYTHON), "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(pdf_path.parent),
        )

        if result.returncode != 0:
            logger.warning("MinerU failed (exit %d): %s", result.returncode, result.stderr[-500:])
            return None

    except subprocess.TimeoutExpired:
        logger.warning("MinerU timed out after %ds for %s", timeout, pdf_path.name)
        return None

    # Find the output markdown
    stem = pdf_path.stem
    md_path = Path(output_dir) / stem / "auto" / f"{stem}.md"
    if not md_path.exists():
        # Try finding any .md file
        md_files = list(Path(output_dir).rglob("*.md"))
        if md_files:
            md_path = md_files[0]
        else:
            logger.warning("MinerU produced no markdown output for %s", pdf_path.name)
            return None

    content = md_path.read_text(encoding="utf-8")
    logger.info("MinerU extracted %d chars from %s", len(content), pdf_path.name)
    return content


def _markdown_to_sections(md_content: str) -> list[Section]:
    """Parse MinerU's Markdown output into Section objects.

    MinerU outputs Markdown with:
      - # headings at various levels
      - $inline math$
      - $$ display math $$ (on separate lines)
      - ![](images/...) for figures
      - HTML tables
    """
    sections: list[Section] = []
    current_heading = "Abstract"
    current_level = 1
    current_lines: list[str] = []
    section_order = 0

    for line in md_content.splitlines():
        # Detect headings
        heading_match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if heading_match:
            # Save previous section
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append(Section(
                        title=current_heading,
                        text=text,
                        page_start=0,
                        page_end=0,
                        level=current_level,
                    ))
                    section_order += 1

            level = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            current_level = level
            current_lines = []
            continue

        # Skip image references (we don't need them for text)
        if re.match(r'^\s*!\[.*?\]\(.*?\)\s*$', line):
            # Keep figure captions that follow
            continue

        # Skip the "Figure N:" caption lines? No, keep them — useful context
        current_lines.append(line)

    # Save last section
    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(Section(
                title=current_heading,
                text=text,
                page_start=0,
                page_end=0,
                level=current_level,
            ))

    logger.info(
        "Parsed %d sections, %d display equations, %d inline equations",
        len(sections),
        sum(1 for s in sections for _ in re.finditer(r'^\$\$', s.text, re.MULTILINE)),
        sum(1 for s in sections for _ in re.finditer(r'(?<!\$)\$(?!\$)', s.text)),
    )

    return sections
