"""Tests for LaTeX equation extraction."""

import pytest

from papermind.ingestion.latex_extractor import extract_equations


class TestDisplayEquations:
    def test_double_dollar(self):
        text = "Consider the equation $$E = mc^2$$ which describes energy."
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 1
        assert eqs[0].display is True
        assert "E = mc^2" in eqs[0].latex

    def test_bracket_display(self):
        text = r"The loss is \[L = \sum_i (y_i - \hat{y}_i)^2\] for regression."
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 1
        assert eqs[0].display is True

    def test_equation_environment(self):
        text = r"\begin{equation}f(x) = ax^2 + bx + c\end{equation}"
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 1
        assert eqs[0].display is True

    def test_align_environment(self):
        text = r"\begin{align}a &= b + c \\ d &= e + f\end{align}"
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 1
        assert eqs[0].display is True


class TestInlineEquations:
    def test_single_dollar(self):
        text = "The variable $x$ represents input."
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 1
        assert eqs[0].display is False
        assert eqs[0].latex == "x"

    def test_paren_inline(self):
        text = r"We define \(f(x) = x^2\) as the activation."
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 1
        assert eqs[0].display is False


class TestEdgeCases:
    def test_no_equations(self):
        text = "This is plain text with no math at all."
        eqs = extract_equations(text, "p1")
        assert eqs == []

    def test_empty_text(self):
        eqs = extract_equations("", "p1")
        assert eqs == []

    def test_context_extraction(self):
        prefix = "A" * 200
        text = f"{prefix} $$E = mc^2$$ some trailing text"
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 1
        # Context should be extracted (up to 150 chars on each side)
        assert len(eqs[0].context) > 0
        assert "E = mc^2" in eqs[0].context

    def test_multiple_equations(self):
        text = "Given $x = 1$ and $y = 2$, then $$z = x + y$$."
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 3
        display_eqs = [e for e in eqs if e.display]
        inline_eqs = [e for e in eqs if not e.display]
        assert len(display_eqs) == 1
        assert len(inline_eqs) == 2

    def test_equation_at_start(self):
        text = "$x = 1$ is the solution."
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 1

    def test_equation_at_end(self):
        text = "The answer is $x = 42$"
        eqs = extract_equations(text, "p1")
        assert len(eqs) == 1
        assert "42" in eqs[0].latex

    def test_paper_id_propagated(self):
        eqs = extract_equations("$x$", paper_id="my_paper")
        assert eqs[0].paper_id == "my_paper"

    def test_empty_equation_skipped(self):
        # An empty $$ pair should produce a display equation container
        # but the inner content is empty, so it should be skipped
        text = "$$$$ not real"
        eqs = extract_equations(text, "p1")
        # Implementation skips empty latex content after strip
        for eq in eqs:
            assert eq.latex.strip() != ""
