"""Paper management endpoints: list, ingest with SSE progress."""

import json
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from papermind.models import make_paper_id
from papermind.services import services

router = APIRouter(tags=["papers"])


@router.get("/papers")
async def list_papers() -> dict:
    """List all ingested papers."""
    papers = services.paper_store.list_all()
    return {
        "papers": [
            {
                "id": p.id,
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "num_pages": p.num_pages,
                "num_chunks": p.num_chunks,
                "num_entities": p.num_entities,
                "created_at": p.created_at.isoformat(),
            }
            for p in papers
        ]
    }


@router.post("/papers/ingest")
async def ingest_paper(
    file: UploadFile = File(...),
    parser: str = Form("hybrid"),
):
    """Ingest a PDF paper with SSE progress streaming."""

    async def stream():
        def emit(step: str, detail: str, **extra):
            event = {"step": step, "detail": detail, **extra}
            return f"data: {json.dumps(event)}\n\n"

        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            content = await file.read()
            f.write(content)
            tmp_path = Path(f.name)

        try:
            paper_id = make_paper_id(tmp_path)
            yield emit("save", "File saved")

            # Parse
            yield emit("parse", f"Parsing via {parser}...")
            if parser == "hybrid":
                from papermind.ingestion.hybrid_parser import parse_pdf_hybrid
                paper, sections = parse_pdf_hybrid(tmp_path)
            elif parser == "grobid":
                from papermind.ingestion.grobid_parser import parse_pdf
                paper, sections = parse_pdf(tmp_path)
            else:
                from papermind.ingestion.pdf_parser import parse_pdf
                paper, sections = parse_pdf(tmp_path)

            paper.id = paper_id
            yield emit("parse", f"Parsed: {paper.title}", title=paper.title)

            # Clean old data if re-ingesting
            pipeline = services.embedding_pipeline
            kg = services.knowledge_graph
            ps = services.paper_store

            if ps.get(paper.id):
                yield emit("clean", "Re-ingesting — clearing old data...")
                pipeline.vector_store.delete_by_paper(paper.id)
                kg.delete_by_paper(paper.id)

            # Count equations
            all_text = "\n".join(s.text for s in sections)
            display_eqs = len(re.findall(r'^\$\$', all_text, re.MULTILINE)) // 2
            inline_eqs = len(re.findall(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', all_text))
            has_latex = bool(re.search(r'\\frac|\\sum|\\int|\\mathrm|\\mathbf', all_text))

            # Chunk
            yield emit("chunk", f"Chunking {len(sections)} sections...")
            from papermind.ingestion.chunker import chunk_sections
            chunks = chunk_sections(sections, paper.id)
            yield emit("chunk", f"Created {len(chunks)} chunks")

            # Embed
            yield emit("embed", "Embedding chunks...")
            num_stored = pipeline.embed_and_store(chunks)
            paper.num_chunks = num_stored
            yield emit("embed", f"Stored {num_stored} chunks")

            # Extract entities
            yield emit("entities", "Extracting entities...")
            from papermind.ingestion.entity_extractor import extract_entities
            entities, relationships = extract_entities(sections, paper.id)
            for e in entities:
                kg.add_entity(e)
            for r in relationships:
                kg.add_relationship(r)
            paper.num_entities = len(entities)
            yield emit("entities", f"{len(entities)} entities, {len(relationships)} relations")

            # Save
            ps.save(paper)

            yield emit("done", "Ingestion complete!", done=True, paper={
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "num_chunks": num_stored,
                "num_entities": len(entities),
            }, stats={
                "sections": len(sections),
                "chunks": num_stored,
                "entities": len(entities),
                "display_equations": display_eqs,
                "inline_equations": inline_eqs,
                "has_latex": has_latex,
            })

        except Exception as e:
            yield emit("error", str(e), error=str(e))
        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(stream(), media_type="text/event-stream")
