"""Paper upload and management endpoints."""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from papermind.api.dependencies import get_embedding_pipeline, get_knowledge_graph
from papermind.config import get_settings
from papermind.ingestion.chunker import chunk_sections
from papermind.ingestion.embedder import EmbeddingPipeline
from papermind.ingestion.entity_extractor import extract_entities
from papermind.ingestion.pdf_parser import parse_pdf
from papermind.infrastructure.knowledge_graph import KnowledgeGraph
from papermind.models import Paper

router = APIRouter(prefix="/papers", tags=["papers"])

# In-memory paper index (will move to SQLite in a later phase)
_papers: dict[str, Paper] = {}


@router.post("/upload")
async def upload_paper(
    file: UploadFile,
    pipeline: EmbeddingPipeline = Depends(get_embedding_pipeline),
    kg: KnowledgeGraph = Depends(get_knowledge_graph),
) -> dict:
    """Upload a PDF paper, parse, chunk, embed, and extract entities."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    settings = get_settings()
    papers_dir = Path(settings.ingestion.papers_directory)
    papers_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    dest = papers_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse PDF
    paper, sections = parse_pdf(dest)

    # Chunk and embed
    chunks = chunk_sections(sections, paper.id)
    num_stored = pipeline.embed_and_store(chunks)
    paper.num_chunks = num_stored

    # Extract entities
    entities, relationships = extract_entities(sections, paper.id)
    for entity in entities:
        kg.add_entity(entity)
    for rel in relationships:
        kg.add_relationship(rel)
    paper.num_entities = len(entities)

    _papers[paper.id] = paper
    return {"paper": paper.model_dump(), "chunks": num_stored, "entities": len(entities)}


@router.get("/")
async def list_papers() -> list[dict]:
    """List all ingested papers."""
    return [p.model_dump() for p in _papers.values()]


@router.get("/{paper_id}")
async def get_paper(paper_id: str) -> dict:
    """Get details for a specific paper."""
    paper = _papers.get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper.model_dump()
