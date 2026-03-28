"""Tests for reasoning frameworks: CoT, Self-Consistency, ReAct."""

import pytest
from papermind.reasoning.cot import (
    build_cot_prompt, get_system_prompt, extract_final_answer,
    ZERO_SHOT_SYSTEM, STRUCTURED_SYSTEM, DECOMPOSE_SYSTEM,
)
from papermind.reasoning.self_consistency import (
    _extract_answer, _normalize_answer, ConsistencyResult,
)
from papermind.reasoning.react import (
    _parse_response, ReActStep, ReActResult,
)


# ============================================================
# Chain-of-Thought
# ============================================================


class TestCoTSystemPrompts:
    def test_zero_shot_prompt(self):
        p = get_system_prompt("zero_shot")
        assert p == ZERO_SHOT_SYSTEM
        assert "step by step" in p.lower()

    def test_structured_prompt(self):
        p = get_system_prompt("structured")
        assert "## Reasoning" in p
        assert "## Evidence" in p
        assert "## Answer" in p

    def test_decompose_prompt(self):
        p = get_system_prompt("decompose")
        assert "Sub-questions" in p
        assert "Synthesis" in p

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown CoT mode"):
            get_system_prompt("invalid")


class TestCoTPromptBuilding:
    def test_zero_shot(self):
        p = build_cot_prompt("What is X?", "Context", mode="zero_shot")
        assert "What is X?" in p
        assert "Context" in p
        assert "step by step" in p.lower()

    def test_structured(self):
        p = build_cot_prompt("What is X?", "Context", mode="structured")
        assert "Reasoning" in p
        assert "Context" in p

    def test_decompose(self):
        p = build_cot_prompt("What is X?", "Context", mode="decompose")
        assert "sub-questions" in p.lower()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            build_cot_prompt("Q", "C", mode="invalid")

    def test_context_included(self):
        p = build_cot_prompt("Q", "My special context", mode="zero_shot")
        assert "My special context" in p


class TestCoTExtractAnswer:
    def test_extract_answer_section(self):
        response = "## Reasoning\nSome reasoning.\n## Evidence\nFacts.\n## Answer\nThe answer is 42."
        assert extract_final_answer(response) == "The answer is 42."

    def test_extract_synthesis_section(self):
        response = "## Sub-questions\n1. Q1\n## Sub-answers\nA1\n## Synthesis\nCombined answer here."
        assert extract_final_answer(response) == "Combined answer here."

    def test_no_structure_returns_full(self):
        response = "Just a plain response without headers"
        assert extract_final_answer(response) == response

    def test_multiline_answer(self):
        response = "## Reasoning\nBlah.\n## Answer\nLine one.\nLine two.\nLine three."
        result = extract_final_answer(response)
        assert "Line one" in result
        assert "Line two" in result

    def test_empty_answer_section(self):
        response = "## Reasoning\nBlah.\n## Answer\n"
        result = extract_final_answer(response)
        # Should return something (fallback)
        assert isinstance(result, str)


# ============================================================
# Self-Consistency
# ============================================================


class TestSCExtractAnswer:
    def test_final_answer_marker(self):
        assert _extract_answer("Blah blah\nFINAL ANSWER: The answer is 42") == "The answer is 42"

    def test_final_answer_case_insensitive(self):
        assert _extract_answer("blah\nfinal answer: yes") == "yes"

    def test_no_marker_returns_last_line(self):
        result = _extract_answer("Line 1\nLine 2\nThis is the actual answer")
        assert "actual answer" in result

    def test_empty_response(self):
        assert _extract_answer("") == ""

    def test_skips_short_source_lines(self):
        result = _extract_answer("The actual answer is here\n[Source 1] ref")
        assert "actual answer" in result.lower()


class TestSCNormalize:
    def test_lowercase(self):
        assert _normalize_answer("THE ANSWER") == "the answer"

    def test_strip_whitespace(self):
        assert _normalize_answer("  answer  ") == "answer"

    def test_strip_period(self):
        assert _normalize_answer("answer.") == "answer"

    def test_collapse_spaces(self):
        assert _normalize_answer("the   answer   is") == "the answer is"

    def test_numbers_preserved(self):
        assert "42" in _normalize_answer("42%")

    def test_same_content_same_normalized(self):
        a = _normalize_answer("The answer is 42.")
        b = _normalize_answer("the answer is 42")
        assert a == b


class TestConsistencyResult:
    def test_default_values(self):
        r = ConsistencyResult()
        assert r.answer == ""
        assert r.confidence == 0.0
        assert r.n_samples == 0
        assert r.all_answers == []

    def test_with_values(self):
        r = ConsistencyResult(
            answer="42", confidence=0.8, n_samples=5,
            n_unique=2, all_answers=["42", "42", "42", "42", "43"],
            vote_distribution={"42": 4, "43": 1},
        )
        assert r.confidence == 0.8
        assert r.n_unique == 2


# ============================================================
# ReAct
# ============================================================


class TestReActParsing:
    def test_thought_and_action(self):
        t, a, ai, ans = _parse_response(
            'Thought: I need to search for attention\nAction: search("attention mechanism")'
        )
        assert "search for attention" in t
        assert a == "search"
        assert ai == "attention mechanism"
        assert ans == ""

    def test_thought_and_answer(self):
        t, a, ai, ans = _parse_response(
            'Thought: I have enough info\nAnswer: Multi-head attention uses Q, K, V projections.'
        )
        assert "enough info" in t
        assert a == ""
        assert "Multi-head" in ans

    def test_answer_takes_priority(self):
        t, a, ai, ans = _parse_response(
            'Thought: Done\nAction: search("x")\nAnswer: The final answer.'
        )
        assert ans == "The final answer."
        assert a == ""  # Answer takes priority, action ignored

    def test_lookup_entity(self):
        t, a, ai, ans = _parse_response(
            'Thought: Check KG\nAction: lookup_entity("ResNet")'
        )
        assert a == "lookup_entity"
        assert ai == "ResNet"

    def test_single_quotes(self):
        t, a, ai, ans = _parse_response(
            "Thought: Search\nAction: search('attention')"
        )
        assert a == "search"
        assert ai == "attention"

    def test_no_thought(self):
        t, a, ai, ans = _parse_response('Action: search("query")')
        assert t == ""
        assert a == "search"

    def test_no_action_no_answer(self):
        t, a, ai, ans = _parse_response("Just some random text")
        assert a == ""
        assert ans == ""

    def test_multiline_answer(self):
        t, a, ai, ans = _parse_response(
            "Thought: Done\nAnswer: Line 1.\nLine 2.\nLine 3."
        )
        assert "Line 1" in ans
        assert "Line 3" in ans


class TestReActStep:
    def test_step_creation(self):
        step = ReActStep(
            thought="I need to search",
            action="search",
            action_input="attention",
            observation="Found 5 results",
        )
        assert step.thought == "I need to search"
        assert step.action == "search"


class TestReActResult:
    def test_trajectory_formatting(self):
        result = ReActResult(
            answer="The final answer",
            question="What is attention?",
            steps=[
                ReActStep(thought="Need to search", action="search", action_input="attention", observation="Found results"),
                ReActStep(thought="Now I know", action="", action_input="", observation=""),
            ],
            iterations=2,
            finished=True,
        )
        traj = result.trajectory
        assert "Step 1" in traj
        assert "Step 2" in traj
        assert "search(attention)" in traj
        assert "Found results" in traj
        assert "Final Answer" in traj

    def test_empty_result(self):
        result = ReActResult()
        assert result.trajectory == ""
        assert result.answer == ""
        assert not result.finished
