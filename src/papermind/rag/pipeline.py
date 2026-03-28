"""RAG pipeline orchestrator.

Wires together the full retrieval-augmented generation pipeline:

  Query → Hybrid Retrieve → Rerank → Context Assembly → LLM Generation

Each stage is independently configurable and can be bypassed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from papermind.models import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Retrieval
    n_retrieve: int = 30           # initial retrieval count (over-fetch for reranking)
    use_kg: bool = True            # include knowledge graph in hybrid retrieval
    paper_id: str | None = None    # filter to a specific paper

    # Reranking
    rerank: bool = True            # enable cross-encoder reranking
    rerank_top_k: int = 10         # keep top K after reranking
    rerank_threshold: float = 0.01 # minimum reranker score

    # Context assembly
    token_budget: int = 4096       # max tokens for context window
    lost_in_middle: bool = True    # apply lost-in-middle ordering

    # Generation
    system_prompt: str = ""        # override system prompt
    temperature: float = 0.1       # LLM temperature
    max_tokens: int = 2048         # max generation tokens

    # Reasoning framework
    reasoning: str = "direct"      # "direct", "cot", "self_consistency", "react"
    cot_mode: str = "structured"   # "zero_shot", "structured", "decompose"
    sc_samples: int = 5            # self-consistency: number of samples


@dataclass
class RAGResult:
    """Complete result from the RAG pipeline."""

    answer: str = ""
    query: str = ""
    context: str = ""
    sources: list[SearchResult] = field(default_factory=list)
    retrieval_count: int = 0
    reranked_count: int = 0
    context_count: int = 0
    context_tokens: int = 0
    reasoning_mode: str = "direct"
    reasoning_trace: str = ""      # full CoT reasoning or ReAct trajectory
    confidence: float = 1.0        # self-consistency confidence (1.0 for others)


DEFAULT_SYSTEM_PROMPT = """\
You are a research assistant that answers questions ONLY from the provided context. \
Rules you MUST follow:
1. ONLY use information from the provided context. Do NOT use your own knowledge.
2. If the context does not contain the answer, say: "The ingested papers do not contain information about this topic."
3. Cite which source number you are drawing from (e.g. "[Source 1]").
4. If the context contains equations, include them in your explanation.
5. If multiple papers are referenced, compare and contrast their approaches.
6. Never fabricate information that is not in the context."""


class RAGPipeline:
    """Full RAG pipeline: retrieve → rerank → assemble context → generate."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()

    async def query(self, question: str, config: RAGConfig | None = None) -> RAGResult:
        """Run the full RAG pipeline.

        Args:
            question: The user's question.
            config: Optional per-query config override.

        Returns:
            RAGResult with answer, context, and source chunks.
        """
        cfg = config or self.config
        result = RAGResult(query=question)

        # Stage 1: Hybrid retrieval (vector + knowledge graph)
        logger.info("RAG Stage 1: Retrieving (n=%d, kg=%s)", cfg.n_retrieve, cfg.use_kg)
        from papermind.rag.retriever import hybrid_retrieve

        retrieved = hybrid_retrieve(
            query=question,
            n_results=cfg.n_retrieve,
            paper_id=cfg.paper_id,
            use_kg=cfg.use_kg,
        )
        result.retrieval_count = len(retrieved)

        if not retrieved:
            result.answer = (
                "I couldn't find any relevant information in the ingested papers. "
                "Try ingesting more papers or rephrasing your question."
            )
            return result

        # Stage 2: Cross-encoder reranking
        if cfg.rerank and len(retrieved) > 1:
            logger.info("RAG Stage 2: Reranking %d results", len(retrieved))
            from papermind.rag.reranker import rerank

            retrieved = rerank(
                query=question,
                results=retrieved,
                top_k=cfg.rerank_top_k,
                score_threshold=cfg.rerank_threshold,
            )
        result.reranked_count = len(retrieved)

        # Stage 3: Context assembly (dedup → compress → lost-in-middle)
        logger.info("RAG Stage 3: Assembling context (budget=%d tokens)", cfg.token_budget)
        from papermind.rag.context import assemble_context, count_tokens

        context_str, context_results = assemble_context(
            results=retrieved,
            token_budget=cfg.token_budget,
        )
        result.context = context_str
        result.sources = context_results
        result.context_count = len(context_results)
        result.context_tokens = count_tokens(context_str)

        # Stage 4: LLM generation with reasoning framework
        result.reasoning_mode = cfg.reasoning
        logger.info(
            "RAG Stage 4: Generating (mode=%s, context=%d tokens, %d sources)",
            cfg.reasoning, result.context_tokens, result.context_count,
        )

        try:
            if cfg.reasoning == "react":
                # ReAct bypasses normal retrieval — it does its own search
                await self._generate_react(question, result, cfg)
            elif cfg.reasoning == "self_consistency":
                await self._generate_self_consistency(question, context_str, result, cfg)
            elif cfg.reasoning == "cot":
                await self._generate_cot(question, context_str, result, cfg)
            else:
                # Direct generation (default)
                await self._generate_direct(question, context_str, result, cfg)
        except Exception as e:
            logger.error("Generation failed: %s", e)
            result.answer = (
                f"I found {result.context_count} relevant sources but couldn't generate "
                f"an answer (LLM error: {e}). Here are the top sources:\n\n"
                + "\n\n".join(
                    f"**{r.section_title}** (score: {r.score:.2f}): {r.text[:200]}..."
                    for r in context_results[:3]
                )
            )

        logger.info("RAG complete: %d retrieved → %d reranked → %d in context",
                     result.retrieval_count, result.reranked_count, result.context_count)

        return result

    async def _generate_direct(
        self, question: str, context: str, result: RAGResult, cfg: RAGConfig,
    ) -> None:
        """Standard direct generation."""
        from papermind.infrastructure.llm_client import LLMClient

        system = cfg.system_prompt or DEFAULT_SYSTEM_PROMPT
        prompt = self._build_prompt(question, context)
        client = LLMClient()
        result.answer = await client.generate(prompt, system=system)

    async def _generate_cot(
        self, question: str, context: str, result: RAGResult, cfg: RAGConfig,
    ) -> None:
        """Chain-of-Thought generation."""
        from papermind.reasoning.cot import build_cot_prompt, get_system_prompt, extract_final_answer
        from papermind.infrastructure.llm_client import LLMClient

        system = cfg.system_prompt or get_system_prompt(cfg.cot_mode)
        prompt = build_cot_prompt(question, context, mode=cfg.cot_mode)
        client = LLMClient()
        raw_response = await client.generate(prompt, system=system)
        result.reasoning_trace = raw_response
        result.answer = extract_final_answer(raw_response)

    async def _generate_self_consistency(
        self, question: str, context: str, result: RAGResult, cfg: RAGConfig,
    ) -> None:
        """Self-Consistency generation with majority voting."""
        from papermind.reasoning.self_consistency import self_consistent_answer

        sc_result = await self_consistent_answer(
            question=question,
            context=context,
            n_samples=cfg.sc_samples,
            temperature=0.7,
            system=cfg.system_prompt,
        )
        result.answer = sc_result.answer
        result.confidence = sc_result.confidence
        result.reasoning_trace = (
            f"Self-Consistency: {sc_result.n_samples} samples, "
            f"{sc_result.n_unique} unique answers, "
            f"{sc_result.confidence:.0%} confidence\n\n"
            f"Vote distribution:\n"
            + "\n".join(
                f"  {count}x: {ans[:100]}"
                for ans, count in sc_result.vote_distribution.items()
            )
        )

    async def _generate_react(
        self, question: str, result: RAGResult, cfg: RAGConfig,
    ) -> None:
        """ReAct reasoning loop — model uses tools to gather information."""
        from papermind.reasoning.react import react_answer

        react_result = await react_answer(
            question=question,
            max_iterations=5,
            system=cfg.system_prompt,
        )
        result.answer = react_result.answer
        result.reasoning_trace = react_result.trajectory
        result.reasoning_mode = "react"

    async def query_stream(self, question: str, config: RAGConfig | None = None):
        """Stream the RAG pipeline — yields (event_type, data) tuples.

        Event types:
          - ("status", str): Pipeline stage status messages
          - ("source", SearchResult): A source chunk being used
          - ("token", str): A generated token from the LLM
          - ("done", RAGResult): Final result
        """
        cfg = config or self.config

        yield ("status", "Searching papers...")

        from papermind.rag.retriever import hybrid_retrieve
        retrieved = hybrid_retrieve(
            query=question, n_results=cfg.n_retrieve,
            paper_id=cfg.paper_id, use_kg=cfg.use_kg,
        )

        if not retrieved:
            yield ("token", "No relevant information found in ingested papers.")
            yield ("done", RAGResult(query=question))
            return

        if cfg.rerank and len(retrieved) > 1:
            yield ("status", f"Reranking {len(retrieved)} results...")
            from papermind.rag.reranker import rerank
            retrieved = rerank(
                query=question, results=retrieved,
                top_k=cfg.rerank_top_k, score_threshold=cfg.rerank_threshold,
            )

        yield ("status", "Assembling context...")
        from papermind.rag.context import assemble_context, count_tokens
        context_str, context_results = assemble_context(
            results=retrieved, token_budget=cfg.token_budget,
        )

        for source in context_results:
            yield ("source", source)

        yield ("status", f"Generating answer from {len(context_results)} sources...")

        system = cfg.system_prompt or DEFAULT_SYSTEM_PROMPT
        prompt = self._build_prompt(question, context_str)

        from papermind.infrastructure.llm_client import LLMClient
        client = LLMClient()

        full_answer = []
        try:
            async for token in client.generate_stream(prompt, system=system):
                full_answer.append(token)
                yield ("token", token)
        except Exception as e:
            yield ("token", f"\n\n[Generation error: {e}]")

        yield ("done", RAGResult(
            query=question,
            answer="".join(full_answer),
            context=context_str,
            sources=context_results,
            retrieval_count=len(retrieved),
            reranked_count=len(retrieved),
            context_count=len(context_results),
            context_tokens=count_tokens(context_str),
        ))

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the generation prompt with context and question."""
        return (
            f"Context from research papers:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the context above. Be specific and cite the sources."
        )
