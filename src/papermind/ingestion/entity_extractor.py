"""Heuristic-based entity and relationship extraction from paper text.

This is a foundation implementation using regex patterns.
Will be upgraded to LLM-based extraction once Ollama is available.
"""

import re

from papermind.models import Entity, Relationship, Section

# Patterns for extracting entities from academic text
_METHOD_PATTERNS = [
    re.compile(r"(?:we\s+)?propos(?:e|ed|es|ing)\s+(?:a\s+)?(?:novel\s+)?([A-Z][A-Za-z0-9\-\s]{2,40})", re.IGNORECASE),
    re.compile(r"(?:our|the)\s+(?:proposed\s+)?(?:method|approach|model|framework|system|algorithm),?\s+(?:called\s+|named\s+|dubbed\s+)?([A-Z][A-Za-z0-9\-]{1,30})", re.IGNORECASE),
    re.compile(r"([A-Z][A-Za-z0-9\-]+(?:Net|GAN|BERT|GPT|ViT|Former|LLM|Coder))\b"),
]

_DATASET_PATTERNS = [
    re.compile(r"(?:on|using|evaluated?\s+on)\s+(?:the\s+)?([A-Z][A-Za-z0-9\-]+(?:\s*\d*[Kk]?)?)\s+(?:dataset|benchmark|corpus)", re.IGNORECASE),
    re.compile(r"(?:dataset|benchmark)s?\s+(?:such\s+as\s+|including\s+)?([A-Z][A-Za-z0-9\-]+(?:,\s*[A-Z][A-Za-z0-9\-]+)*)", re.IGNORECASE),
]

_METRIC_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s*%?\s+(?:on\s+)?(?:accuracy|F1|BLEU|ROUGE|perplexity|AUC|mAP|top-\d)", re.IGNORECASE),
    re.compile(r"(?:accuracy|F1|BLEU|ROUGE|perplexity|AUC|mAP|HumanEval)\s*(?:of|:)?\s*(\d+\.?\d*)\s*%?", re.IGNORECASE),
]

_COMPARISON_PATTERNS = [
    re.compile(r"outperform(?:s|ed|ing)?\s+([A-Z][A-Za-z0-9\-]+)", re.IGNORECASE),
    re.compile(r"(?:compared?\s+(?:to|with)|versus|vs\.?)\s+([A-Z][A-Za-z0-9\-]+)", re.IGNORECASE),
    re.compile(r"(?:surpass(?:es|ed|ing)?|exceed(?:s|ed|ing)?)\s+([A-Z][A-Za-z0-9\-]+)", re.IGNORECASE),
]

# Words to exclude from entity names
_STOP_WORDS = {
    "The", "This", "That", "These", "Those", "Our", "We", "In", "On", "For",
    "With", "From", "Using", "Based", "Table", "Figure", "Section", "Chapter",
    "Note", "However", "Moreover", "Furthermore", "Additionally", "Specifically",
}


def extract_entities(
    sections: list[Section], paper_id: str
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from paper sections.

    Returns a tuple of (entities, relationships).
    """
    entities: dict[str, Entity] = {}  # name -> Entity (dedup by name)
    relationships: list[Relationship] = []

    full_text = "\n\n".join(s.text for s in sections)

    # Extract methods
    for pattern in _METHOD_PATTERNS:
        for match in pattern.finditer(full_text):
            name = _clean_name(match.group(1))
            if name and name not in _STOP_WORDS and len(name) > 2:
                if name not in entities:
                    entities[name] = Entity(
                        name=name, entity_type="method", paper_id=paper_id
                    )

    # Extract datasets
    for pattern in _DATASET_PATTERNS:
        for match in pattern.finditer(full_text):
            raw = match.group(1)
            # Handle comma-separated lists
            for name_raw in raw.split(","):
                name = _clean_name(name_raw)
                if name and name not in _STOP_WORDS and len(name) > 2:
                    if name not in entities:
                        entities[name] = Entity(
                            name=name, entity_type="dataset", paper_id=paper_id
                        )

    # Extract comparison relationships
    method_entities = [e for e in entities.values() if e.entity_type == "method"]
    for pattern in _COMPARISON_PATTERNS:
        for match in pattern.finditer(full_text):
            target_name = _clean_name(match.group(1))
            if not target_name or target_name in _STOP_WORDS:
                continue
            # Ensure target exists as an entity
            if target_name not in entities:
                entities[target_name] = Entity(
                    name=target_name, entity_type="method", paper_id=paper_id
                )
            # Create outperforms relationships from all known methods
            for method in method_entities:
                if method.name != target_name:
                    relationships.append(
                        Relationship(
                            source_id=method.id,
                            target_id=entities[target_name].id,
                            relation_type="outperforms",
                            paper_id=paper_id,
                        )
                    )

    # Link methods to datasets (if method "evaluated on" dataset found nearby)
    dataset_entities = {e.name: e for e in entities.values() if e.entity_type == "dataset"}
    for method in method_entities:
        for ds_name, ds_entity in dataset_entities.items():
            # Simple co-occurrence heuristic
            if method.name in full_text and ds_name in full_text:
                relationships.append(
                    Relationship(
                        source_id=method.id,
                        target_id=ds_entity.id,
                        relation_type="evaluated_on",
                        paper_id=paper_id,
                    )
                )

    return list(entities.values()), relationships


def _clean_name(raw: str) -> str:
    """Clean up an extracted entity name."""
    name = raw.strip().strip(".,;:()[]")
    # Remove trailing common words
    name = re.sub(r"\s+(is|are|was|were|has|have|that|which|with|and|or|the|a|an)$", "", name, flags=re.IGNORECASE)
    return name.strip()
