"""Search endpoints for vector and knowledge graph queries."""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from papermind.services import services

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    n_results: int = 10
    paper_id: str | None = None


@router.post("/search")
async def vector_search(req: SearchRequest) -> dict:
    """Search paper chunks by semantic similarity."""
    pipeline = services.embedding_pipeline
    results = pipeline.search(req.query, req.n_results, req.paper_id)
    return {"results": [r.model_dump() for r in results]}


@router.get("/kg/entities")
async def search_entities(
    query: str = Query("", alias="query"),
    entity_type: str = Query(""),
    limit: int = Query(50, le=200),
) -> list[dict]:
    """Search entities in the knowledge graph."""
    kg = services.knowledge_graph
    entities = kg.search_entities(query, entity_type, limit=limit)
    return [e.model_dump() for e in entities]


@router.get("/kg/entity/{entity_id}/neighbors")
async def get_entity_neighbors(entity_id: str) -> list[dict]:
    """Get entities connected to the given entity."""
    kg = services.knowledge_graph
    neighbors = kg.get_neighbors(entity_id)
    return [
        {"relation": rel.model_dump(), "entity": entity.model_dump()}
        for rel, entity in neighbors
    ]
