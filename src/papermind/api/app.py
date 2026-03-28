"""FastAPI application factory."""

from fastapi import FastAPI

from papermind.api.routes import health, papers, search


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PaperMind",
        description="Local AI research system: RAG, knowledge graphs, and paper-to-code",
        version="0.1.0",
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(papers.router, prefix="/api")
    app.include_router(search.router, prefix="/api")

    return app


app = create_app()
