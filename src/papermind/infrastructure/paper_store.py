"""Persistent paper metadata store using SQLite.

Papers are registered here during ingestion and survive app restarts.
This solves the session-state problem where paper metadata was lost
when Streamlit restarted, even though chunks remained in ChromaDB.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from papermind.models import Paper

logger = logging.getLogger(__name__)


class PaperStore:
    """SQLite-backed paper metadata registry."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or Path("./data/papers.db"))
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                authors TEXT NOT NULL DEFAULT '[]',
                abstract TEXT NOT NULL DEFAULT '',
                source_path TEXT NOT NULL DEFAULT '',
                num_pages INTEGER DEFAULT 0,
                num_chunks INTEGER DEFAULT 0,
                num_entities INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()

    def save(self, paper: Paper) -> None:
        """Save or update a paper's metadata."""
        self.conn.execute("""
            INSERT OR REPLACE INTO papers
                (id, title, authors, abstract, source_path, num_pages, num_chunks, num_entities, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper.id,
            paper.title,
            json.dumps(paper.authors),
            paper.abstract,
            paper.source_path,
            paper.num_pages,
            paper.num_chunks,
            paper.num_entities,
            paper.created_at.isoformat(),
        ))
        self.conn.commit()
        logger.info("Saved paper '%s' (id=%s)", paper.title[:40], paper.id)

    def get(self, paper_id: str) -> Paper | None:
        """Get a paper by ID."""
        row = self.conn.execute(
            "SELECT * FROM papers WHERE id = ?", (paper_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_paper(row)

    def list_all(self) -> list[Paper]:
        """Get all papers ordered by creation time (newest first)."""
        rows = self.conn.execute(
            "SELECT * FROM papers ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def get_paper_map(self) -> dict[str, Paper]:
        """Get {paper_id: Paper} for all stored papers."""
        return {p.id: p for p in self.list_all()}

    def delete(self, paper_id: str) -> None:
        """Delete a paper by ID."""
        self.conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        self.conn.commit()

    def count(self) -> int:
        """Return total number of papers."""
        row = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()
        return row[0]

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        return Paper(
            id=row["id"],
            title=row["title"],
            authors=json.loads(row["authors"]),
            abstract=row["abstract"],
            source_path=row["source_path"],
            num_pages=row["num_pages"],
            num_chunks=row["num_chunks"],
            num_entities=row["num_entities"],
        )
