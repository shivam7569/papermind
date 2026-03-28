"""Context assembly: lost-in-middle ordering + compression.

After retrieval and reranking, we need to assemble the context window for
the LLM. Two key techniques:

1. Lost-in-Middle Ordering (Liu et al., 2023):
   LLMs attend more to the beginning and end of their context window.
   Information in the middle gets "lost." Solution: place the most relevant
   chunks at the start and end, less relevant in the middle.

2. Context Compression:
   - Deduplicate overlapping chunks (parent-child overlap)
   - Filter by token budget (fit within LLM context window)
   - Remove low-information chunks (too short, boilerplate)
"""

from __future__ import annotations

import logging

import tiktoken

from papermind.models import SearchResult

logger = logging.getLogger(__name__)

# Default token budget leaves room for system prompt + generation
DEFAULT_TOKEN_BUDGET = 4096
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_ENCODER.encode(text))


def lost_in_middle_order(results: list[SearchResult]) -> list[SearchResult]:
    """Reorder results so the most relevant are at the start and end.

    Given results ranked by relevance [1, 2, 3, 4, 5, 6]:
      → Reordered: [1, 3, 5, 6, 4, 2]

    The best result goes first, then alternating between start and end,
    so the strongest signals bookend the context window.
    """
    if len(results) <= 2:
        return results

    # Split into two halves
    mid = len(results) // 2
    top_half = results[:mid]     # higher relevance
    bottom_half = results[mid:]  # lower relevance

    # Top half stays at the start, bottom half goes in reverse at the end
    # This places rank 1 first, rank 2 last, rank 3 second, rank 4 second-to-last...
    reordered = []
    reordered.extend(top_half)
    reordered.extend(reversed(bottom_half))

    return reordered


def deduplicate(results: list[SearchResult], overlap_threshold: float = 0.7) -> list[SearchResult]:
    """Remove near-duplicate chunks based on text overlap.

    Parent and child chunks often overlap significantly. We keep the
    higher-scored version and drop duplicates.
    """
    if not results:
        return []

    seen_texts: list[str] = []
    deduped: list[SearchResult] = []

    for result in results:
        text = result.text.strip()
        if not text:
            continue

        # Check overlap with already-seen texts
        is_dup = False
        for seen in seen_texts:
            # Simple substring overlap check
            shorter, longer = (text, seen) if len(text) <= len(seen) else (seen, text)
            if shorter in longer:
                is_dup = True
                break
            # Jaccard on words for fuzzy overlap
            words_a = set(text.lower().split())
            words_b = set(seen.lower().split())
            if words_a and words_b:
                jaccard = len(words_a & words_b) / len(words_a | words_b)
                if jaccard > overlap_threshold:
                    is_dup = True
                    break

        if not is_dup:
            deduped.append(result)
            seen_texts.append(text)

    if len(deduped) < len(results):
        logger.info("Deduplication: %d → %d chunks", len(results), len(deduped))

    return deduped


def compress_context(
    results: list[SearchResult],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    min_chunk_tokens: int = 20,
) -> list[SearchResult]:
    """Select chunks that fit within the token budget.

    Greedy selection: take chunks in order (already ranked by relevance)
    until the budget is exhausted. Skip chunks that are too short to be
    useful (boilerplate, headers, etc.).

    Args:
        results: Ranked, deduplicated results.
        token_budget: Maximum tokens for the assembled context.
        min_chunk_tokens: Skip chunks shorter than this.

    Returns:
        Subset of results fitting within the token budget.
    """
    selected: list[SearchResult] = []
    tokens_used = 0

    for result in results:
        chunk_tokens = count_tokens(result.text)

        if chunk_tokens < min_chunk_tokens:
            continue

        if tokens_used + chunk_tokens > token_budget:
            # If we haven't selected anything yet, take a truncated version
            if not selected:
                selected.append(result)
            break

        selected.append(result)
        tokens_used += chunk_tokens

    logger.info(
        "Context compression: %d → %d chunks, %d tokens (budget: %d)",
        len(results), len(selected), tokens_used, token_budget,
    )
    return selected


def assemble_context(
    results: list[SearchResult],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> tuple[str, list[SearchResult]]:
    """Full context assembly pipeline: dedup → compress → lost-in-middle order.

    Args:
        results: Reranked search results.
        token_budget: Max tokens for the context block.

    Returns:
        (formatted_context_string, selected_results)
    """
    # Step 1: Deduplicate overlapping chunks
    results = deduplicate(results)

    # Step 2: Compress to fit token budget
    results = compress_context(results, token_budget=token_budget)

    # Step 3: Lost-in-middle ordering
    results = lost_in_middle_order(results)

    # Step 4: Format into a context string
    context_parts = []
    for i, result in enumerate(results):
        header = f"[Source {i + 1}]"
        if result.section_title:
            header += f" Section: {result.section_title}"
        if result.paper_id:
            header += f" | Paper: {result.paper_id}"
        context_parts.append(f"{header}\n{result.text}")

    context_str = "\n\n---\n\n".join(context_parts)

    return context_str, results
