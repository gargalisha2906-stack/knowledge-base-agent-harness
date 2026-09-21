"""Text chunking.

Retrieval works on chunks, not whole documents: smaller units embed more
precisely and give the answerer tighter, citable evidence. Chunks overlap
so a fact that spans a boundary is still captured whole in at least one
chunk. The step size is ``chunk_size - overlap``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """One retrievable unit of text, traceable back to its source."""

    id: str          # e.g. "refund_policy.md#2"
    doc_id: str      # source document name
    position: int    # index of this chunk inside its document
    text: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping windows of ``chunk_size`` characters.

    Character windows are used (instead of tokens) so the chunker has zero
    dependencies and is easy to reason about. For knowledge-base prose this
    works well; swap in a token-aware splitter behind the same interface if
    you need tighter size control.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        window = text[start : start + chunk_size]
        chunks.append(window)
        if start + chunk_size >= len(text):
            break
    return chunks


def chunk_document(doc_id: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk one document and tag every chunk with its origin."""
    return [
        Chunk(id=f"{doc_id}#{i}", doc_id=doc_id, position=i, text=window)
        for i, window in enumerate(chunk_text(text, chunk_size, overlap))
    ]
