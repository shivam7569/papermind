"""Extract LaTeX equations from parsed text."""

import re

from papermind.models import LatexEquation

# Patterns for LaTeX math environments, ordered from most specific to least
_PATTERNS = [
    # Display math environments
    (re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}(.+?)\\end\{\1\}", re.DOTALL), True),
    (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), True),
    (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), True),
    # Inline math
    (re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)"), False),
    (re.compile(r"\\\((.+?)\\\)"), False),
]

# Context window: characters around the equation to capture
_CONTEXT_CHARS = 150


def extract_equations(text: str, paper_id: str = "") -> list[LatexEquation]:
    """Extract all LaTeX equations from text.

    Returns a list of LatexEquation objects with surrounding context.
    """
    equations: list[LatexEquation] = []
    seen_positions: set[tuple[int, int]] = set()  # avoid duplicates from overlapping patterns

    for pattern, is_display in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            # Skip if this region was already captured by a more specific pattern
            if any(s <= start and end <= e for s, e in seen_positions):
                continue
            seen_positions.add((start, end))

            # Get the captured group (last group for environment patterns)
            latex = match.group(match.lastindex or 0).strip()
            if not latex:
                continue

            # Extract surrounding context
            ctx_start = max(0, start - _CONTEXT_CHARS)
            ctx_end = min(len(text), end + _CONTEXT_CHARS)
            context = text[ctx_start:ctx_end].strip()

            equations.append(
                LatexEquation(
                    latex=latex,
                    display=is_display,
                    context=context,
                    paper_id=paper_id,
                )
            )

    return equations
