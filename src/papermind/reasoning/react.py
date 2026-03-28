"""ReAct reasoning loop (Yao et al., 2022).

ReAct interleaves reasoning (Thought) with actions (Action) and their
results (Observation) in a loop until the model has enough information
to produce a final answer.

Available actions for PaperMind:
  - search(query): Semantic search across ingested papers
  - lookup_entity(name): Look up an entity in the knowledge graph
  - get_section(paper_id, section_title): Get full text of a specific section
  - calculate(expression): Evaluate a math expression

The loop:
  1. Model produces: Thought: I need to find information about X
  2. Model produces: Action: search("attention mechanism")
  3. System executes the action, returns: Observation: [results]
  4. Repeat until model produces: Answer: [final answer]

Max iterations prevent infinite loops. Each iteration adds to context.

Usage:
    from papermind.reasoning.react import react_answer

    result = await react_answer(
        question="How does the learning rate in Adam compare to SGD?",
        max_iterations=5,
    )
    print(result.answer)
    print(result.trajectory)  # Full Thought/Action/Observation trace
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""

    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""


@dataclass
class ReActResult:
    """Result of the ReAct reasoning loop."""

    answer: str = ""
    question: str = ""
    steps: list[ReActStep] = field(default_factory=list)
    iterations: int = 0
    finished: bool = False

    @property
    def trajectory(self) -> str:
        """Format the full reasoning trajectory as readable text."""
        parts = []
        for i, step in enumerate(self.steps, 1):
            parts.append(f"--- Step {i} ---")
            if step.thought:
                parts.append(f"Thought: {step.thought}")
            if step.action:
                parts.append(f"Action: {step.action}({step.action_input})")
            if step.observation:
                obs_preview = step.observation[:500]
                if len(step.observation) > 500:
                    obs_preview += "..."
                parts.append(f"Observation: {obs_preview}")
            parts.append("")
        if self.answer:
            parts.append(f"Final Answer: {self.answer}")
        return "\n".join(parts)


REACT_SYSTEM = """\
You are a research assistant with access to tools for answering questions about research papers.

You MUST follow the ReAct format. On each turn, produce EXACTLY one of:

1. A Thought + Action pair:
   Thought: [your reasoning about what to do next]
   Action: [tool_name]("[argument]")

2. A final answer (when you have enough information):
   Thought: I now have enough information to answer.
   Answer: [your complete answer citing sources]

Available tools:
- search("query") — Search across all ingested papers by semantic similarity. Returns relevant text chunks with scores.
- lookup_entity("name") — Look up an entity (method, dataset, metric) in the knowledge graph. Returns the entity and its connections.
- get_neighbors("entity_id") — Get all entities connected to a given entity in the knowledge graph.

Rules:
- Use ONLY information from tool observations. Never fabricate.
- Cite sources in your final answer.
- If tools return no useful information, say so honestly.
- Stop after at most {max_iter} iterations.
- Always end with Answer: on the last turn."""


def _build_react_prompt(
    question: str,
    history: list[ReActStep],
    max_iter: int = MAX_ITERATIONS,
) -> str:
    """Build the prompt including the full reasoning history."""
    parts = [f"Question: {question}\n"]

    for step in history:
        if step.thought:
            parts.append(f"Thought: {step.thought}")
        if step.action:
            parts.append(f"Action: {step.action}(\"{step.action_input}\")")
        if step.observation:
            parts.append(f"Observation: {step.observation}\n")

    remaining = max_iter - len(history)
    if remaining <= 1:
        parts.append(
            "You have 1 step remaining. You MUST provide your final Answer now."
        )

    return "\n".join(parts)


def _parse_response(response: str) -> tuple[str, str, str, str]:
    """Parse the model's response into (thought, action, action_input, answer).

    Returns whichever fields are present.
    """
    thought = ""
    action = ""
    action_input = ""
    answer = ""

    # Extract Thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Answer)|$)', response, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract Answer (takes priority — ends the loop)
    answer_match = re.search(r'Answer:\s*(.+)', response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        return thought, "", "", answer

    # Extract Action
    action_match = re.search(r'Action:\s*(\w+)\(\s*["\'](.+?)["\']\s*\)', response)
    if action_match:
        action = action_match.group(1).strip()
        action_input = action_match.group(2).strip()

    return thought, action, action_input, answer


async def _execute_action(action: str, action_input: str) -> str:
    """Execute a ReAct action and return the observation."""
    try:
        if action == "search":
            return await _action_search(action_input)
        elif action == "lookup_entity":
            return await _action_lookup_entity(action_input)
        elif action == "get_neighbors":
            return await _action_get_neighbors(action_input)
        else:
            return f"Unknown action: {action}. Available: search, lookup_entity, get_neighbors"
    except Exception as e:
        return f"Action failed: {e}"


async def _action_search(query: str) -> str:
    """Execute semantic search and format results."""
    from papermind.rag.retriever import vector_search

    results = vector_search(query, n_results=5)
    if not results:
        return "No results found for this query."

    parts = []
    for i, r in enumerate(results, 1):
        score = f"{r.score:.3f}"
        section = r.section_title or "Unknown"
        text = r.text[:300].replace("\n", " ")
        parts.append(f"[Result {i}] (score: {score}, section: {section})\n{text}")

    return "\n\n".join(parts)


async def _action_lookup_entity(name: str) -> str:
    """Look up an entity in the knowledge graph."""
    from papermind.services import services

    kg = services.knowledge_graph
    entities = kg.search_entities(query=name, limit=3)
    if not entities:
        return f"No entity found matching '{name}'."

    parts = []
    for e in entities:
        info = f"Entity: {e.name} (type: {e.entity_type}, paper: {e.paper_id})"
        if e.properties:
            info += f"\nProperties: {e.properties}"

        neighbors = kg.get_neighbors(e.id)
        if neighbors:
            connections = []
            for rel, neighbor in neighbors[:5]:
                direction = "→" if rel.source_id == e.id else "←"
                connections.append(
                    f"  {direction} {neighbor.name} ({neighbor.entity_type}) "
                    f"[{rel.relation_type}]"
                )
            info += "\nConnections:\n" + "\n".join(connections)

        parts.append(info)

    return "\n\n".join(parts)


async def _action_get_neighbors(entity_id: str) -> str:
    """Get all neighbors of an entity."""
    from papermind.services import services

    kg = services.knowledge_graph
    neighbors = kg.get_neighbors(entity_id)
    if not neighbors:
        return f"No connections found for entity '{entity_id}'."

    parts = []
    for rel, entity in neighbors:
        direction = "→" if rel.source_id == entity_id else "←"
        parts.append(
            f"{direction} {entity.name} ({entity.entity_type}) "
            f"— {rel.relation_type}"
        )

    return "\n".join(parts)


async def react_answer(
    question: str,
    max_iterations: int = MAX_ITERATIONS,
    system: str = "",
) -> ReActResult:
    """Run the ReAct reasoning loop.

    Args:
        question: The user's question.
        max_iterations: Maximum Thought→Action→Observation cycles.
        system: Override system prompt.

    Returns:
        ReActResult with answer and full reasoning trajectory.
    """
    from papermind.infrastructure.llm_client import LLMClient

    client = LLMClient()
    sys_prompt = system or REACT_SYSTEM.format(max_iter=max_iterations)
    result = ReActResult(question=question)

    for iteration in range(max_iterations):
        # Build prompt with history
        prompt = _build_react_prompt(question, result.steps, max_iterations)

        # Get model response
        try:
            response = await client.generate(prompt, system=sys_prompt)
        except Exception as e:
            logger.error("ReAct iteration %d failed: %s", iteration + 1, e)
            result.answer = f"Reasoning failed at step {iteration + 1}: {e}"
            break

        # Parse response
        thought, action, action_input, answer = _parse_response(response)

        step = ReActStep(thought=thought, action=action, action_input=action_input)

        # Check if model produced a final answer
        if answer:
            step.thought = thought
            result.steps.append(step)
            result.answer = answer
            result.finished = True
            result.iterations = iteration + 1
            logger.info("ReAct finished in %d iterations", iteration + 1)
            break

        # Execute action if provided
        if action:
            logger.info(
                "ReAct step %d: %s(\"%s\")",
                iteration + 1, action, action_input[:50],
            )
            observation = await _execute_action(action, action_input)
            step.observation = observation
        else:
            # No action and no answer — model is confused
            step.observation = (
                "You must either provide an Action to gather more information, "
                "or provide your final Answer."
            )

        result.steps.append(step)

    # If we exhausted iterations without an answer
    if not result.finished:
        result.iterations = max_iterations
        if result.steps:
            # Use the last thought as a best-effort answer
            last_thoughts = " ".join(
                s.thought for s in result.steps if s.thought
            )
            result.answer = (
                f"I was unable to reach a definitive answer after {max_iterations} "
                f"reasoning steps. Based on my investigation: {last_thoughts}"
            )

    return result
