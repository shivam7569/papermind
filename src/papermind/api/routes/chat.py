"""RAG chat endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    n_retrieve: int = 30
    rerank_top_k: int = 10
    token_budget: int = 4096
    use_kg: bool = True
    use_rerank: bool = True
    paper_id: str | None = None


@router.post("/chat/rag")
async def rag_chat(req: ChatRequest) -> dict:
    """Run the full RAG pipeline and return answer + sources."""
    from papermind.rag.retriever import hybrid_retrieve
    from papermind.rag.reranker import rerank, get_reranker
    from papermind.rag.context import assemble_context
    from papermind.rag.pipeline import DEFAULT_SYSTEM_PROMPT
    from papermind.infrastructure.llm_client import LLMClient

    # Stage 1: Retrieve
    retrieved = hybrid_retrieve(
        query=req.query,
        n_results=req.n_retrieve,
        paper_id=req.paper_id,
        use_kg=req.use_kg,
    )

    if not retrieved:
        return {
            "answer": "No relevant information found in ingested papers.",
            "sources": [],
            "retrieval_count": 0,
            "reranked_count": 0,
            "context_count": 0,
            "context_tokens": 0,
        }

    # Stage 2: Rerank
    if req.use_rerank and len(retrieved) > 1:
        retrieved = rerank(
            query=req.query,
            results=retrieved,
            top_k=req.rerank_top_k,
        )

    # Free reranker GPU memory before LLM
    if req.use_rerank:
        get_reranker().unload()

    # Stage 3: Context assembly
    context_str, context_results = assemble_context(
        results=retrieved,
        token_budget=req.token_budget,
    )

    # Stage 4: Generate
    prompt = (
        f"Context from research papers:\n\n{context_str}\n\n---\n\n"
        f"Question: {req.query}\n\n"
        f"Answer based on the context above. Be specific and cite the sources."
    )

    client = LLMClient()
    try:
        answer = await client.generate(prompt, system=DEFAULT_SYSTEM_PROMPT)
    except Exception as e:
        answer = (
            f"Found {len(context_results)} relevant sources but LLM generation failed: {e}\n\n"
            + "\n\n".join(
                f"**{r.section_title}** (score: {r.score:.2f}): {r.text[:200]}..."
                for r in context_results[:3]
            )
        )

    return {
        "answer": answer,
        "sources": [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "paper_id": r.paper_id,
                "section_title": r.section_title,
            }
            for r in context_results
        ],
        "retrieval_count": len(retrieved),
        "reranked_count": len(retrieved),
        "context_count": len(context_results),
        "context_tokens": len(context_str.split()),
    }
