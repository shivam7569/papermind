"""Tests for the SQLite paper metadata store."""

import pytest

from papermind.infrastructure.paper_store import PaperStore
from papermind.models import Paper


@pytest.fixture
def paper_store(tmp_dir):
    """Provide a fresh PaperStore backed by a temp SQLite DB."""
    return PaperStore(db_path=tmp_dir / "papers.db")


class TestPaperStore:
    def test_save_and_retrieve(self, paper_store):
        paper = Paper(id="p1", title="Test Paper", authors=["Alice"])
        paper_store.save(paper)
        result = paper_store.get("p1")
        assert result is not None
        assert result.title == "Test Paper"
        assert result.authors == ["Alice"]

    def test_list_all_ordered_by_newest(self, paper_store):
        import time
        p1 = Paper(id="p1", title="First")
        p2 = Paper(id="p2", title="Second")
        paper_store.save(p1)
        time.sleep(0.01)
        paper_store.save(p2)
        papers = paper_store.list_all()
        assert len(papers) == 2
        # Newest first
        assert papers[0].id == "p2"
        assert papers[1].id == "p1"

    def test_get_paper_map(self, paper_store):
        paper_store.save(Paper(id="p1", title="A"))
        paper_store.save(Paper(id="p2", title="B"))
        pmap = paper_store.get_paper_map()
        assert "p1" in pmap
        assert "p2" in pmap
        assert pmap["p1"].title == "A"

    def test_delete(self, paper_store):
        paper_store.save(Paper(id="p1", title="To Delete"))
        assert paper_store.count() == 1
        paper_store.delete("p1")
        assert paper_store.count() == 0
        assert paper_store.get("p1") is None

    def test_count(self, paper_store):
        assert paper_store.count() == 0
        paper_store.save(Paper(id="p1", title="A"))
        assert paper_store.count() == 1
        paper_store.save(Paper(id="p2", title="B"))
        assert paper_store.count() == 2

    def test_upsert_updates_existing(self, paper_store):
        paper_store.save(Paper(id="p1", title="Original", num_chunks=5))
        paper_store.save(Paper(id="p1", title="Updated", num_chunks=10))
        assert paper_store.count() == 1
        result = paper_store.get("p1")
        assert result.title == "Updated"
        assert result.num_chunks == 10

    def test_get_nonexistent_returns_none(self, paper_store):
        assert paper_store.get("nonexistent") is None

    def test_empty_store_returns_empty_list(self, paper_store):
        assert paper_store.list_all() == []
        assert paper_store.get_paper_map() == {}

    def test_authors_json_roundtrip(self, paper_store):
        paper = Paper(id="p1", title="Multi-Author", authors=["Alice", "Bob", "Charlie"])
        paper_store.save(paper)
        result = paper_store.get("p1")
        assert result.authors == ["Alice", "Bob", "Charlie"]

    def test_empty_authors(self, paper_store):
        paper = Paper(id="p1", title="No Authors", authors=[])
        paper_store.save(paper)
        result = paper_store.get("p1")
        assert result.authors == []

    def test_delete_nonexistent_no_error(self, paper_store):
        # Should not raise
        paper_store.delete("nonexistent")
        assert paper_store.count() == 0
