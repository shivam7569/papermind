"""Health check endpoints."""

from fastapi import APIRouter

from papermind.services import services

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@router.get("/health/detailed")
async def detailed_health() -> dict:
    """Detailed system health with GPU, services, and data store info."""
    kg = services.knowledge_graph
    vs = services.vector_store
    ps = services.paper_store

    result = {
        "vector_store_count": vs.count(),
        "kg_entities": kg.count_entities(),
        "kg_relations": kg.count_relationships(),
        "papers_count": ps.count(),
        "grobid": False,
        "mineru": False,
    }

    # Check GROBID
    try:
        from papermind.ingestion.grobid_parser import check_grobid_health
        result["grobid"] = check_grobid_health()
    except Exception:
        pass

    # Check MinerU
    try:
        from papermind.ingestion.mineru_parser import check_mineru_available
        result["mineru"] = check_mineru_available()
    except Exception:
        pass

    # GPU info
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            parts = out.stdout.strip().split(", ")
            if len(parts) == 4:
                result["gpu"] = {
                    "name": parts[0],
                    "total_mb": int(parts[1]),
                    "free_mb": int(parts[2]),
                    "used_mb": int(parts[3]),
                }
    except Exception:
        pass

    return result
