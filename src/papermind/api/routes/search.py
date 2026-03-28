"""Search endpoints for vector and knowledge graph queries."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from papermind.api.dependencies import get_embedding_pipeline, get_knowledge_graph
from papermind.ingestion.embedder import EmbeddingPipeline
from papermind.infrastructure.knowledge_graph import KnowledgeGraph

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    n_results: int = 10
    paper_id: str | None = None


@router.post("/search")
async def vector_search(
    req: SearchRequest,
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline),
) -> list[dict]:
    """Search paper chunks by semantic similarity."""
    results = pipeline.search(req.query, req.n_results, req.paper_id)
    return [r.model_dump() for r in results]


@router.get("/kg/entities")
async def search_entities(
    q: str = Query("", description="Search query for entity name"),
    entity_type: str = Query("", description="Filter by entity type"),
    paper_id: str = Query("", description="Filter by paper ID"),
    limit: int = Query(50, le=200),
    kg: KnowledgeGraph = Depends(get_knowledge_graph),
) -> list[dict]:
    """Search entities in the knowledge graph."""
    entities = kg.search_entities(q, entity_type, paper_id, limit)
    return [e.model_dump() for e in entities]


@router.get("/kg/entity/{entity_id}/neighbors")
async def get_entity_neighbors(
    entity_id: str,
    relation_type: str = Query("", description="Filter by relation type"),
    kg: KnowledgeGraph = Depends(get_knowledge_graph),
) -> list[dict]:
    """Get entities connected to the given entity."""
    neighbors = kg.get_neighbors(entity_id, relation_type)
    return [
        {"relationship": rel.model_dump(), "entity": entity.model_dump()}
        for rel, entity in neighbors
    ]
