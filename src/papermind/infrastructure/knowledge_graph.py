"""SQLite + networkx knowledge graph store."""

import json
import sqlite3
from pathlib import Path

import networkx as nx

from papermind.config import get_settings
from papermind.models import Entity, Relationship


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    properties TEXT DEFAULT '{}',
    paper_id TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES entities(id),
    target_id TEXT NOT NULL REFERENCES entities(id),
    relation_type TEXT NOT NULL,
    properties TEXT DEFAULT '{}',
    paper_id TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_paper ON entities(paper_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);
"""


class KnowledgeGraph:
    """SQLite-backed knowledge graph with networkx query support."""

    def __init__(self, db_path: str | None = None):
        settings = get_settings()
        self._db_path = db_path or settings.knowledge_graph.db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def add_entity(self, entity: Entity) -> Entity:
        """Insert an entity. Returns the entity with its id."""
        self.conn.execute(
            "INSERT OR REPLACE INTO entities (id, name, entity_type, properties, paper_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                entity.id,
                entity.name,
                entity.entity_type,
                json.dumps(entity.properties),
                entity.paper_id,
            ),
        )
        self.conn.commit()
        return entity

    def add_relationship(self, rel: Relationship) -> Relationship:
        """Insert a relationship between two entities."""
        self.conn.execute(
            "INSERT OR REPLACE INTO relationships "
            "(id, source_id, target_id, relation_type, properties, paper_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                rel.id,
                rel.source_id,
                rel.target_id,
                rel.relation_type,
                json.dumps(rel.properties),
                rel.paper_id,
            ),
        )
        self.conn.commit()
        return rel

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by id."""
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_entity(row)

    def search_entities(
        self,
        query: str = "",
        entity_type: str = "",
        paper_id: str = "",
        limit: int = 50,
    ) -> list[Entity]:
        """Search entities by name, type, or paper."""
        conditions = []
        params: list[str | int] = []
        if query:
            conditions.append("name LIKE ?")
            params.append(f"%{query}%")
        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        if paper_id:
            conditions.append("paper_id = ?")
            params.append(paper_id)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM entities {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def get_neighbors(
        self, entity_id: str, relation_type: str = ""
    ) -> list[tuple[Relationship, Entity]]:
        """Get all entities connected to the given entity."""
        conditions = "(r.source_id = ? OR r.target_id = ?)"
        params: list[str] = [entity_id, entity_id]
        if relation_type:
            conditions += " AND r.relation_type = ?"
            params.append(relation_type)

        rows = self.conn.execute(
            f"""
            SELECT r.*, e.id as eid, e.name, e.entity_type, e.properties as eprops,
                   e.paper_id as epaper
            FROM relationships r
            JOIN entities e ON (
                CASE WHEN r.source_id = ? THEN r.target_id ELSE r.source_id END = e.id
            )
            WHERE {conditions}
            """,
            [entity_id, *params],
        ).fetchall()
        results = []
        for row in rows:
            rel = Relationship(
                id=row["id"],
                source_id=row["source_id"],
                target_id=row["target_id"],
                relation_type=row["relation_type"],
                properties=json.loads(row["properties"] or "{}"),
                paper_id=row["paper_id"] or "",
            )
            entity = Entity(
                id=row["eid"],
                name=row["name"],
                entity_type=row["entity_type"],
                properties=json.loads(row["eprops"] or "{}"),
                paper_id=row["epaper"] or "",
            )
            results.append((rel, entity))
        return results

    def get_subgraph(self, entity_id: str, depth: int = 2) -> nx.DiGraph:
        """Build a networkx subgraph around an entity up to given depth."""
        graph = nx.DiGraph()
        visited: set[str] = set()
        queue = [(entity_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)
            if current_id in visited or current_depth > depth:
                continue
            visited.add(current_id)

            entity = self.get_entity(current_id)
            if not entity:
                continue
            graph.add_node(
                entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                **entity.properties,
            )
            for rel, neighbor in self.get_neighbors(current_id):
                graph.add_node(
                    neighbor.id,
                    name=neighbor.name,
                    entity_type=neighbor.entity_type,
                    **neighbor.properties,
                )
                graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_type=rel.relation_type,
                    **rel.properties,
                )
                if current_depth < depth:
                    queue.append((neighbor.id, current_depth + 1))

        return graph

    def delete_by_paper(self, paper_id: str) -> None:
        """Delete all entities and relationships for a paper."""
        self.conn.execute("DELETE FROM relationships WHERE paper_id = ?", (paper_id,))
        self.conn.execute("DELETE FROM entities WHERE paper_id = ?", (paper_id,))
        self.conn.commit()

    def count_entities(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return row[0]

    def count_relationships(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM relationships").fetchone()
        return row[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            properties=json.loads(row["properties"] or "{}"),
            paper_id=row["paper_id"] or "",
        )
