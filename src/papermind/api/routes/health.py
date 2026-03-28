"""Health check endpoints."""

from fastapi import APIRouter, Depends

from papermind.api.dependencies import get_knowledge_graph, get_llm_client, get_vector_store
from papermind.infrastructure.knowledge_graph import KnowledgeGraph
from papermind.infrastructure.llm_client import LLMClient
from papermind.infrastructure.vector_store import VectorStore

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(
    vs: VectorStore = Depends(get_vector_store),
    kg: KnowledgeGraph = Depends(get_knowledge_graph),
    llm: LLMClient = Depends(get_llm_client),
) -> dict:
    """Check system health: vector store, knowledge graph, and LLM availability."""
    llm_ok = await llm.is_available()
    return {
        "status": "ok",
        "vector_store": {"chunks": vs.count()},
        "knowledge_graph": {
            "entities": kg.count_entities(),
            "relationships": kg.count_relationships(),
        },
        "llm": {"available": llm_ok, "model": llm.model},
    }
