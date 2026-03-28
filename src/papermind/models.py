"""Shared data models for PaperMind."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex[:12]


def make_paper_id(pdf_path: str | Path) -> str:
    """Generate a deterministic paper ID from PDF file content.

    Uses SHA-256 of the raw PDF bytes, so the same file always produces
    the same ID regardless of filename, path, or which machine uploads it.
    """
    pdf_bytes = Path(pdf_path).read_bytes()
    return hashlib.sha256(pdf_bytes).hexdigest()[:12]


class Paper(BaseModel):
    """A research paper ingested into the system."""

    id: str = Field(default_factory=_new_id)
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    source_path: str = ""
    num_pages: int = 0
    num_chunks: int = 0
    num_entities: int = 0
    created_at: datetime = Field(default_factory=_utcnow)


class Section(BaseModel):
    """A section extracted from a paper."""

    title: str
    text: str
    page_start: int
    page_end: int
    level: int = 1  # heading level


class Chunk(BaseModel):
    """A text chunk ready for embedding."""

    id: str = Field(default_factory=_new_id)
    text: str
    paper_id: str
    section_title: str = ""
    page_start: int = 0
    page_end: int = 0
    token_count: int = 0
    parent_id: str = ""  # ID of parent chunk (for parent-child retrieval)
    chunk_type: str = "child"  # "parent" (full section summary) or "child" (granular)


class LatexEquation(BaseModel):
    """A LaTeX equation extracted from a paper."""

    id: str = Field(default_factory=_new_id)
    latex: str
    display: bool = False  # True for display math, False for inline
    context: str = ""  # surrounding text
    paper_id: str = ""


class Entity(BaseModel):
    """An entity in the knowledge graph."""

    id: str = Field(default_factory=_new_id)
    name: str
    entity_type: str  # method, dataset, metric, author, task
    properties: dict = Field(default_factory=dict)
    paper_id: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class Relationship(BaseModel):
    """A relationship between two entities in the knowledge graph."""

    id: str = Field(default_factory=_new_id)
    source_id: str
    target_id: str
    relation_type: str  # uses, outperforms, extends, trained_on, authored_by
    properties: dict = Field(default_factory=dict)
    paper_id: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class SearchResult(BaseModel):
    """A search result from vector or knowledge graph query."""

    chunk_id: str = ""
    text: str = ""
    score: float = 0.0
    paper_id: str = ""
    section_title: str = ""
    metadata: dict = Field(default_factory=dict)
