"""Tests for the SQLite + networkx knowledge graph."""

from papermind.models import Entity, Relationship


def test_add_and_get_entity(kg):
    entity = Entity(name="Transformer", entity_type="method", paper_id="p1")
    kg.add_entity(entity)

    result = kg.get_entity(entity.id)
    assert result is not None
    assert result.name == "Transformer"
    assert result.entity_type == "method"


def test_search_entities_by_type(kg):
    kg.add_entity(Entity(name="BERT", entity_type="method", paper_id="p1"))
    kg.add_entity(Entity(name="ImageNet", entity_type="dataset", paper_id="p1"))
    kg.add_entity(Entity(name="GPT", entity_type="method", paper_id="p2"))

    methods = kg.search_entities(entity_type="method")
    assert len(methods) == 2
    assert {e.name for e in methods} == {"BERT", "GPT"}


def test_search_entities_by_name(kg):
    kg.add_entity(Entity(name="BERT-base", entity_type="method", paper_id="p1"))
    kg.add_entity(Entity(name="BERT-large", entity_type="method", paper_id="p1"))
    kg.add_entity(Entity(name="GPT-4", entity_type="method", paper_id="p2"))

    results = kg.search_entities(query="BERT")
    assert len(results) == 2


def test_add_relationship_and_get_neighbors(kg):
    e1 = Entity(name="ResNet", entity_type="method", paper_id="p1")
    e2 = Entity(name="ImageNet", entity_type="dataset", paper_id="p1")
    kg.add_entity(e1)
    kg.add_entity(e2)

    rel = Relationship(
        source_id=e1.id, target_id=e2.id,
        relation_type="evaluated_on", paper_id="p1",
    )
    kg.add_relationship(rel)

    neighbors = kg.get_neighbors(e1.id)
    assert len(neighbors) == 1
    r, e = neighbors[0]
    assert e.name == "ImageNet"
    assert r.relation_type == "evaluated_on"


def test_get_subgraph(kg):
    e1 = Entity(name="BERT", entity_type="method", paper_id="p1")
    e2 = Entity(name="SQuAD", entity_type="dataset", paper_id="p1")
    e3 = Entity(name="NLP", entity_type="task", paper_id="p1")
    for e in [e1, e2, e3]:
        kg.add_entity(e)

    kg.add_relationship(Relationship(
        source_id=e1.id, target_id=e2.id, relation_type="evaluated_on", paper_id="p1",
    ))
    kg.add_relationship(Relationship(
        source_id=e1.id, target_id=e3.id, relation_type="used_for", paper_id="p1",
    ))

    graph = kg.get_subgraph(e1.id, depth=1)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2


def test_delete_by_paper(kg):
    kg.add_entity(Entity(name="A", entity_type="method", paper_id="p1"))
    kg.add_entity(Entity(name="B", entity_type="method", paper_id="p2"))
    assert kg.count_entities() == 2

    kg.delete_by_paper("p1")
    assert kg.count_entities() == 1


def test_count(kg):
    assert kg.count_entities() == 0
    kg.add_entity(Entity(name="X", entity_type="method", paper_id="p1"))
    assert kg.count_entities() == 1
