"""Section-aware text chunking with parent-child strategy.

Parent-child chunking creates two layers:
  - **Parent chunks**: One per section — the full section text (or a truncated
    version if the section is very long). Used for broad context retrieval.
  - **Child chunks**: Granular splits within each section at ~512 tokens.
    Each child stores its parent_id so retrieval can "zoom out" to get the
    full section context when a child matches.

This enables a retrieval pattern where:
  1. Search returns the best child chunks (precise)
  2. For each hit, optionally fetch the parent (broad context)
  3. LLM gets both: the specific passage + surrounding section
"""

import tiktoken

from papermind.config import get_settings
from papermind.models import Chunk, Section


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def chunk_sections(sections: list[Section], paper_id: str) -> list[Chunk]:
    """Chunk sections using parent-child strategy.

    For each section:
      1. Create a parent chunk (full section, capped at max_parent_tokens)
      2. Create child chunks (granular splits at chunk_size tokens)
      3. Link children to parent via parent_id
    """
    settings = get_settings()
    chunk_size = settings.chunking.chunk_size
    chunk_overlap = settings.chunking.chunk_overlap
    min_chunk_size = settings.chunking.min_chunk_size
    encoder = _get_encoder()

    all_chunks: list[Chunk] = []

    for section in sections:
        if not section.text.strip():
            continue

        section_tokens = len(encoder.encode(section.text))

        # --- Parent chunk ---
        # Cap parent text to avoid huge embeddings (max 2048 tokens)
        max_parent_tokens = 2048
        if section_tokens <= max_parent_tokens:
            parent_text = section.text
        else:
            # Truncate to max_parent_tokens, breaking at sentence boundary
            parent_text = _truncate_to_tokens(
                section.text, max_parent_tokens, encoder
            )

        parent_chunk = Chunk(
            text=parent_text,
            paper_id=paper_id,
            section_title=section.title,
            page_start=section.page_start,
            page_end=section.page_end,
            token_count=len(encoder.encode(parent_text)),
            chunk_type="parent",
        )
        all_chunks.append(parent_chunk)

        # --- Child chunks ---
        # If section is small enough to be a single child, just use the parent
        if section_tokens <= chunk_size:
            child = Chunk(
                text=section.text,
                paper_id=paper_id,
                section_title=section.title,
                page_start=section.page_start,
                page_end=section.page_end,
                token_count=section_tokens,
                parent_id=parent_chunk.id,
                chunk_type="child",
            )
            all_chunks.append(child)
        else:
            # Split into multiple children
            child_texts = _split_text(
                text=section.text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size,
                encoder=encoder,
            )
            for text in child_texts:
                child = Chunk(
                    text=text,
                    paper_id=paper_id,
                    section_title=section.title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    token_count=len(encoder.encode(text)),
                    parent_id=parent_chunk.id,
                    chunk_type="child",
                )
                all_chunks.append(child)

    return all_chunks


def _truncate_to_tokens(
    text: str, max_tokens: int, encoder: tiktoken.Encoding
) -> str:
    """Truncate text to max_tokens, breaking at a sentence boundary."""
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Decode truncated tokens, then find last sentence boundary
    truncated = encoder.decode(tokens[:max_tokens])
    # Find last period/question mark/exclamation
    for end_char in [". ", "? ", "! "]:
        last_idx = truncated.rfind(end_char)
        if last_idx > len(truncated) // 2:  # Don't cut more than half
            return truncated[: last_idx + 1]

    return truncated


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    encoder: tiktoken.Encoding,
) -> list[str]:
    """Split text into chunks by token count, breaking at paragraph boundaries."""
    if not text.strip():
        return []

    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(encoder.encode(para))

        # If a single paragraph exceeds chunk_size, split it by sentences
        if para_tokens > chunk_size:
            # Flush current buffer first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            # Split long paragraph
            chunks.extend(
                _split_long_paragraph(para, chunk_size, chunk_overlap, encoder)
            )
            continue

        # Would adding this paragraph exceed the limit?
        if current_tokens + para_tokens > chunk_size and current_parts:
            chunks.append("\n\n".join(current_parts))
            # Keep overlap: retain last part(s) up to chunk_overlap tokens
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for part in reversed(current_parts):
                pt = len(encoder.encode(part))
                if overlap_tokens + pt > chunk_overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_tokens += pt
            current_parts = overlap_parts
            current_tokens = overlap_tokens

        current_parts.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_parts:
        text_out = "\n\n".join(current_parts)
        if len(encoder.encode(text_out)) >= min_chunk_size:
            chunks.append(text_out)

    return chunks


def _split_long_paragraph(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoder: tiktoken.Encoding,
) -> list[str]:
    """Split a single long paragraph into chunks by sentence boundaries."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        st = len(encoder.encode(sent))
        if current_tokens + st > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap
            overlap: list[str] = []
            ot = 0
            for s in reversed(current):
                t = len(encoder.encode(s))
                if ot + t > chunk_overlap:
                    break
                overlap.insert(0, s)
                ot += t
            current = overlap
            current_tokens = ot
        current.append(sent)
        current_tokens += st

    if current:
        chunks.append(" ".join(current))

    return chunks
