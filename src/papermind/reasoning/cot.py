"""Chain-of-Thought (CoT) prompting framework.

CoT instructs the model to reason step-by-step before producing a final answer.
This dramatically improves accuracy on complex questions — especially those
requiring multi-hop reasoning across multiple paper sections, mathematical
derivations, or comparing approaches from different papers.

Three CoT modes:
  1. zero_shot: "Let's think step by step" suffix (Wei et al., 2022)
  2. structured: Forces a Reasoning → Evidence → Answer format
  3. decompose: Breaks the question into sub-questions, answers each, then synthesizes

Usage:
    from papermind.reasoning.cot import build_cot_prompt, extract_final_answer

    prompt = build_cot_prompt(question, context, mode="structured")
    raw_response = await llm.generate(prompt, system=COT_SYSTEM_PROMPT)
    answer = extract_final_answer(raw_response)
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts for each CoT mode
# ---------------------------------------------------------------------------

ZERO_SHOT_SYSTEM = """\
You are a research assistant that answers questions from provided paper context.
Think through your reasoning step by step before giving your final answer.
Always cite source numbers. Only use information from the context provided."""

STRUCTURED_SYSTEM = """\
You are a research assistant that answers questions from provided paper context.
You MUST structure your response in exactly this format:

## Reasoning
Walk through your thought process step by step. For each step:
- State what you're looking for
- Quote or reference the relevant source
- Draw a conclusion from that evidence

## Evidence
List the key facts from the sources that support your answer.
Format: [Source N] — fact

## Answer
Give your final, concise answer based on the reasoning above.

Rules:
- ONLY use information from the provided context
- Cite sources as [Source N]
- If the context doesn't contain the answer, say so in the Answer section
- Never fabricate information"""

DECOMPOSE_SYSTEM = """\
You are a research assistant that answers complex questions by breaking them down.
You MUST structure your response in exactly this format:

## Sub-questions
Break the original question into 2-4 simpler sub-questions that, when answered together, fully answer the original question.

## Sub-answers
Answer each sub-question using ONLY the provided context. Cite sources as [Source N].
If a sub-question can't be answered from the context, say so.

## Synthesis
Combine the sub-answers into a complete, coherent final answer.

Rules:
- ONLY use information from the provided context
- Never fabricate information
- If the context is insufficient, acknowledge the gaps"""


def get_system_prompt(mode: str = "structured") -> str:
    """Get the system prompt for a CoT mode."""
    prompts = {
        "zero_shot": ZERO_SHOT_SYSTEM,
        "structured": STRUCTURED_SYSTEM,
        "decompose": DECOMPOSE_SYSTEM,
    }
    if mode not in prompts:
        raise ValueError(f"Unknown CoT mode: {mode}. Choose from: {list(prompts.keys())}")
    return prompts[mode]


def build_cot_prompt(
    question: str,
    context: str,
    mode: str = "structured",
) -> str:
    """Build a CoT-augmented prompt.

    Args:
        question: The user's question.
        context: Assembled context from retrieval (with [Source N] headers).
        mode: CoT mode — "zero_shot", "structured", or "decompose".

    Returns:
        Formatted prompt string.
    """
    base = (
        f"Context from research papers:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

    if mode == "zero_shot":
        return base + "\n\nLet's think step by step."
    elif mode == "structured":
        return base + (
            "\n\nAnswer using the Reasoning → Evidence → Answer format. "
            "Work through your reasoning step by step before concluding."
        )
    elif mode == "decompose":
        return base + (
            "\n\nBreak this question into sub-questions, answer each from the "
            "context, then synthesize a complete answer."
        )
    else:
        raise ValueError(f"Unknown CoT mode: {mode}")


def extract_final_answer(response: str) -> str:
    """Extract the final answer from a CoT-structured response.

    Looks for the ## Answer or ## Synthesis section. If not found,
    returns the full response (graceful degradation).
    """
    # Try ## Answer section
    match = re.search(
        r'##\s*Answer\s*\n(.*?)(?:\n##|\Z)',
        response, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Try ## Synthesis section (decompose mode)
    match = re.search(
        r'##\s*Synthesis\s*\n(.*?)(?:\n##|\Z)',
        response, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: return everything after the last "---" or the full response
    parts = response.rsplit("---", 1)
    if len(parts) > 1 and len(parts[1].strip()) > 20:
        return parts[1].strip()

    return response
