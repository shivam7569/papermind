"""CLI entry points for PaperMind."""

from pathlib import Path

import click


@click.group()
def cli() -> None:
    """PaperMind: Local AI research system."""


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
def ingest(pdf_path: Path) -> None:
    """Ingest a PDF paper: parse, chunk, embed, and extract entities."""
    from papermind.config import get_settings
    from papermind.ingestion.chunker import chunk_sections
    from papermind.ingestion.embedder import EmbeddingPipeline
    from papermind.ingestion.entity_extractor import extract_entities
    from papermind.infrastructure.knowledge_graph import KnowledgeGraph

    settings = get_settings()

    # Use GROBID if configured and available, else fall back to PyMuPDF
    if settings.ingestion.pdf_parser == "grobid":
        from papermind.ingestion.grobid_parser import check_grobid_health, parse_pdf
        if check_grobid_health():
            click.echo(f"Parsing {pdf_path.name} via GROBID...")
        else:
            click.echo("GROBID not available, falling back to PyMuPDF...")
            from papermind.ingestion.pdf_parser import parse_pdf
    else:
        from papermind.ingestion.pdf_parser import parse_pdf
        click.echo(f"Parsing {pdf_path.name} via PyMuPDF...")

    paper, sections = parse_pdf(pdf_path)
    click.echo(f"  Title: {paper.title}")
    click.echo(f"  Authors: {', '.join(paper.authors[:3]) or 'N/A'}")
    click.echo(f"  Pages: {paper.num_pages}")
    click.echo(f"  Sections: {len(sections)}")
    for s in sections[:8]:
        indent = "  " * s.level
        click.echo(f"    {indent}{'─' * s.level} {s.title} ({len(s.text)} chars)")

    click.echo("Chunking...")
    chunks = chunk_sections(sections, paper.id)
    click.echo(f"  Chunks: {len(chunks)}")

    click.echo("Embedding and storing...")
    pipeline = EmbeddingPipeline()
    num_stored = pipeline.embed_and_store(chunks)
    click.echo(f"  Stored: {num_stored} chunks in ChromaDB")

    click.echo("Extracting entities...")
    entities, relationships = extract_entities(sections, paper.id)
    kg = KnowledgeGraph()
    for entity in entities:
        kg.add_entity(entity)
    for rel in relationships:
        kg.add_relationship(rel)
    kg.close()
    click.echo(f"  Entities: {len(entities)}, Relationships: {len(relationships)}")

    click.echo(f"\nDone! Paper '{paper.title}' ingested as {paper.id}")


@cli.command()
@click.argument("query")
@click.option("-n", "--n-results", default=5, help="Number of results")
@click.option("--paper-id", default=None, help="Filter by paper ID")
def search(query: str, n_results: int, paper_id: str | None) -> None:
    """Search ingested papers by semantic similarity."""
    from papermind.ingestion.embedder import EmbeddingPipeline

    pipeline = EmbeddingPipeline()
    results = pipeline.search(query, n_results, paper_id)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        click.echo(f"\n--- Result {i} (score: {r.score:.3f}) ---")
        click.echo(f"Paper: {r.paper_id} | Section: {r.section_title}")
        click.echo(r.text[:300] + ("..." if len(r.text) > 300 else ""))


@cli.command()
@click.argument("prompt")
@click.option("-s", "--system", default="", help="System prompt")
@click.option("--stream/--no-stream", default=True, help="Stream output")
@click.option("--max-tokens", default=None, type=int, help="Max tokens to generate")
def generate(prompt: str, system: str, stream: bool, max_tokens: int | None) -> None:
    """Generate text using the local Qwen2.5-Coder model."""
    from papermind.infrastructure.local_model import LocalModel

    model = LocalModel()
    model.load()

    if stream:
        for token in model.generate_stream(prompt, system, max_new_tokens=max_tokens):
            click.echo(token, nl=False)
        click.echo()
    else:
        response = model.generate(prompt, system, max_new_tokens=max_tokens)
        click.echo(response)

    click.echo(f"\n--- {model.vram_usage()} ---")


@cli.command()
def model_info() -> None:
    """Show local model info and VRAM usage."""
    from papermind.infrastructure.local_model import LocalModel

    model = LocalModel()
    click.echo(f"Model: {model.model_name}")
    click.echo(f"Quantization: {model._quantization}")
    click.echo(f"Double quant: {model._double_quant}")
    click.echo(f"Compute dtype: {model._compute_dtype}")
    click.echo("\nLoading model to check VRAM...")
    model.load()
    usage = model.vram_usage()
    click.echo(f"  Allocated: {usage['allocated_gb']} GB")
    click.echo(f"  Reserved:  {usage['reserved_gb']} GB")
    click.echo(f"  Total:     {usage['total_gb']} GB")
    click.echo(f"  Free:      {usage['free_gb']} GB")


@cli.command()
@click.option("--port", default=8501, type=int, help="Streamlit port")
def ui(port: int) -> None:
    """Launch the Streamlit web UI."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.port", str(port), "--server.headless", "true"],
    )


@cli.command("build-dataset")
@click.option("--output", default="data/pwc", help="Output directory")
@click.option("--max-repos", default=None, type=int, help="Max repos (default: all)")
@click.option("--dedup-threshold", default=0.7, type=float, help="MinHash dedup threshold")
@click.option("--pytorch-only", is_flag=True, help="Only PyTorch repos")
@click.option("--official-only", is_flag=True, help="Only official implementations")
def build_dataset(
    output: str, max_repos: int | None, dedup_threshold: float,
    pytorch_only: bool, official_only: bool,
) -> None:
    """Build research code training dataset from Papers with Code."""
    from papermind.data.pwc_dataset import build_dataset as _build

    frameworks = ("pytorch",) if pytorch_only else None
    stats = _build(
        output_dir=output,
        max_repos=max_repos,
        dedup_threshold=dedup_threshold,
        frameworks=frameworks,
        require_official=official_only,
        quality_only=True,
    )
    click.echo(f"\nDone! {stats.pairs_after_dedup} pairs saved to {output}/")
    click.echo(f"  Repos cloned: {stats.repos_cloned}/{stats.repos_attempted}")
    click.echo(f"  With abstract: {stats.pairs_with_abstract}")
    click.echo(f"  Frameworks: {stats.frameworks}")


@cli.command()
@click.option("--host", default=None, help="Bind host")
@click.option("--port", default=None, type=int, help="Bind port")
def serve(host: str | None, port: int | None) -> None:
    """Start the FastAPI server."""
    import uvicorn

    from papermind.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "papermind.api.app:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=True,
    )
