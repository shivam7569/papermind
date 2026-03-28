"""Self-Consistency reasoning (Wang et al., 2022).

Instead of generating one answer with greedy decoding, we:
  1. Sample N diverse responses (higher temperature)
  2. Extract the final answer from each
  3. Cluster/vote on the most common answer
  4. Return the majority answer with confidence score

This improves accuracy on questions with definitive answers — especially
factual questions about paper claims, numerical results, or methodology
comparisons. Less useful for open-ended explanations.

Usage:
    from papermind.reasoning.self_consistency import self_consistent_answer

    result = await self_consistent_answer(
        question="What learning rate does Adam use by default?",
        context=context_str,
        llm=llm_client,
        n_samples=5,
    )
    print(result.answer)       # "0.001"
    print(result.confidence)   # 0.8 (4/5 agreed)
    print(result.all_answers)  # ["0.001", "0.001", "0.01", "0.001", "0.001"]
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    """Result of self-consistency voting."""

    answer: str = ""
    confidence: float = 0.0
    n_samples: int = 0
    n_unique: int = 0
    all_answers: list[str] = field(default_factory=list)
    all_responses: list[str] = field(default_factory=list)
    vote_distribution: dict[str, int] = field(default_factory=dict)


EXTRACTION_SYSTEM = """\
You are a research assistant answering from provided paper context.
Think step by step, then give a clear, concise final answer.
ONLY use information from the context. Cite sources as [Source N].

IMPORTANT: End your response with a line starting with "FINAL ANSWER:" \
followed by your concise answer (1-2 sentences max)."""


def _build_sc_prompt(question: str, context: str) -> str:
    """Build a prompt that encourages diverse reasoning paths."""
    return (
        f"Context from research papers:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Think step by step, then state your FINAL ANSWER on the last line."
    )


def _extract_answer(response: str) -> str:
    """Extract the final answer from a response.

    Looks for "FINAL ANSWER:" marker. Falls back to last non-empty line.
    """
    # Look for explicit marker
    match = re.search(
        r'FINAL\s*ANSWER\s*:\s*(.+?)(?:\n|$)',
        response, re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: last substantive line
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    if lines:
        # Skip citation-only lines
        for line in reversed(lines):
            if len(line) > 10 and not line.startswith("[Source"):
                return line
        return lines[-1]

    return response.strip()


def _normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison (lowercase, strip punctuation)."""
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s\d.,%-]', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    # Remove trailing period
    answer = answer.rstrip('.')
    return answer


async def self_consistent_answer(
    question: str,
    context: str,
    n_samples: int = 5,
    temperature: float = 0.7,
    system: str = "",
) -> ConsistencyResult:
    """Generate N diverse answers and return the majority vote.

    Args:
        question: The user's question.
        context: Retrieved context with [Source N] headers.
        n_samples: Number of diverse samples to generate (default 5).
        temperature: Sampling temperature (higher = more diverse). Default 0.7.
        system: Override system prompt.

    Returns:
        ConsistencyResult with majority answer and confidence.
    """
    from papermind.infrastructure.llm_client import LLMClient

    client = LLMClient()
    prompt = _build_sc_prompt(question, context)
    sys_prompt = system or EXTRACTION_SYSTEM

    # Generate N samples concurrently
    # For local model, we run sequentially (single GPU)
    # For Ollama, we could parallelize but keep it simple
    logger.info("Self-consistency: generating %d samples (temp=%.2f)", n_samples, temperature)

    responses = []
    for i in range(n_samples):
        try:
            if client.backend == "local":
                local = client._get_local_model()
                resp = await asyncio.to_thread(
                    local.generate, prompt, sys_prompt,
                    max_new_tokens=512, temperature=temperature,
                )
            else:
                # Ollama with temperature override
                import httpx
                payload = {
                    "model": client.model,
                    "prompt": prompt,
                    "system": sys_prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                }
                async with httpx.AsyncClient(timeout=client.timeout) as http:
                    r = await http.post(f"{client.base_url}/api/generate", json=payload)
                    r.raise_for_status()
                    resp = r.json()["response"]

            responses.append(resp)
            logger.debug("Sample %d/%d: %s", i + 1, n_samples, resp[:100])
        except Exception as e:
            logger.warning("Sample %d/%d failed: %s", i + 1, n_samples, e)

    if not responses:
        return ConsistencyResult(
            answer="Failed to generate any responses.",
            n_samples=n_samples,
        )

    # Extract answers
    raw_answers = [_extract_answer(r) for r in responses]
    normalized = [_normalize_answer(a) for a in raw_answers]

    # Vote
    vote_counts = Counter(normalized)
    majority_normalized, majority_count = vote_counts.most_common(1)[0]

    # Find the best raw answer that matches the majority
    majority_answer = ""
    for raw, norm in zip(raw_answers, normalized):
        if norm == majority_normalized:
            # Prefer the longest matching raw answer (most detailed)
            if len(raw) > len(majority_answer):
                majority_answer = raw

    confidence = majority_count / len(responses)

    result = ConsistencyResult(
        answer=majority_answer,
        confidence=confidence,
        n_samples=len(responses),
        n_unique=len(vote_counts),
        all_answers=raw_answers,
        all_responses=responses,
        vote_distribution={k: v for k, v in vote_counts.most_common()},
    )

    logger.info(
        "Self-consistency: %d/%d agreed (%.0f%% confidence), %d unique answers",
        majority_count, len(responses), confidence * 100, len(vote_counts),
    )

    return result
