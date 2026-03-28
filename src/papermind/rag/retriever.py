"""Hybrid retriever: vector search + knowledge graph fusion.

Combines two complementary retrieval strategies:
  1. Vector search (semantic similarity via embeddings)
  2. Knowledge graph traversal (structured entity/relationship paths)

The KG retriever expands the query by finding relevant entities and pulling
chunks from papers that mention those entities. This catches results that
are topically related but semantically distant from the query text.

Fusion uses Reciprocal Rank Fusion (RRF) to merge ranked lists without
needing score calibration between the two sources.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from papermind.models import SearchResult

logger = logging.getLogger(__name__)


def vector_search(
    query: str,
    n_results: int = 20,
    paper_id: str | None = None,
) -> list[SearchResult]:
    """Retrieve chunks via bi-encoder vector similarity."""
    from papermind.services import services

    results = services.embedding_pipeline.search(query, n_results=n_results, paper_id=paper_id)
    return results


def kg_search(
    query: str,
    n_results: int = 10,
    paper_id: str | None = None,
) -> list[SearchResult]:
    """Retrieve context via knowledge graph entity lookup.

    Strategy:
      1. Search KG for entities matching the query terms
      2. For each entity, get its neighbors (related entities)
      3. Collect paper_ids from these entities
      4. Retrieve chunks from those papers via vector search, biased by
         the entity names as additional query context
    """
    from papermind.services import services

    kg = services.knowledge_graph
    entities = kg.search_entities(query=query, limit=5)

    if not entities:
        return []

    # Collect entity context: names + relationship info
    entity_context = []
    related_paper_ids = set()

    for entity in entities:
        entity_context.append(entity.name)
        if entity.paper_id:
            related_paper_ids.add(entity.paper_id)

        neighbors = kg.get_neighbors(entity.id)
        for rel, neighbor in neighbors[:3]:  # top 3 neighbors per entity
            entity_context.append(neighbor.name)
            if neighbor.paper_id:
                related_paper_ids.add(neighbor.paper_id)

    if not related_paper_ids:
        return []

    # Build an enriched query with entity context
    enriched_query = f"{query} {' '.join(entity_context[:10])}"

    # Search within the papers found via KG
    all_results = []
    for pid in list(related_paper_ids)[:5]:
        if paper_id and pid != paper_id:
            continue
        results = vector_search(enriched_query, n_results=n_results // 2, paper_id=pid)
        all_results.extend(results)

    # Mark these as KG-sourced
    for r in all_results:
        r.metadata["source"] = "knowledge_graph"
        r.metadata["kg_entities"] = entity_context[:5]

    logger.info(
        "KG search found %d entities → %d papers → %d chunks",
        len(entities), len(related_paper_ids), len(all_results),
    )
    return all_results


def reciprocal_rank_fusion(
    *result_lists: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each list where the document appears.
    This is robust to different score scales between retrieval methods.

    Args:
        *result_lists: Multiple ranked lists of SearchResult.
        k: RRF constant (default 60, standard value from the original paper).

    Returns:
        Merged and re-ranked list of SearchResult.
    """
    # Aggregate RRF scores by chunk_id
    rrf_scores: dict[str, float] = defaultdict(float)
    result_map: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            key = result.chunk_id or result.text[:100]
            rrf_scores[key] += 1.0 / (k + rank + 1)
            # Keep the result with the highest original score
            if key not in result_map or result.score > result_map[key].score:
                result_map[key] = result

    # Build fused results
    fused = []
    for key, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        result = result_map[key]
        fused.append(SearchResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=rrf_score,
            paper_id=result.paper_id,
            section_title=result.section_title,
            metadata={
                **result.metadata,
                "rrf_score": rrf_score,
            },
        ))

    logger.info(
        "RRF fused %d lists → %d unique results",
        len(result_lists), len(fused),
    )
    return fused


def hybrid_retrieve(
    query: str,
    n_results: int = 20,
    paper_id: str | None = None,
    use_kg: bool = True,
) -> list[SearchResult]:
    """Retrieve using both vector search and knowledge graph, fused via RRF.

    Args:
        query: User's search query.
        n_results: Total number of results to return.
        paper_id: Optional filter to a specific paper.
        use_kg: Whether to include KG-based retrieval.

    Returns:
        Fused and ranked list of SearchResult.
    """
    # Vector search — primary retrieval
    vector_results = vector_search(query, n_results=n_results * 2, paper_id=paper_id)
    for r in vector_results:
        r.metadata["source"] = "vector"

    if not use_kg:
        return vector_results[:n_results]

    # Knowledge graph search — complementary retrieval
    kg_results = kg_search(query, n_results=n_results, paper_id=paper_id)

    if not kg_results:
        return vector_results[:n_results]

    # Fuse via RRF
    fused = reciprocal_rank_fusion(vector_results, kg_results)
    return fused[:n_results]
