"""Ingestion: knowledge-base files on disk -> chunks in the vector store."""

from __future__ import annotations

from pathlib import Path

from .chunking import chunk_document
from .config import Config
from .store import VectorStore


def load_documents(kb_dir: str) -> dict[str, str]:
    """Read every .md/.txt document under ``kb_dir`` as {doc_id: text}."""
    docs = {}
    for path in sorted(Path(kb_dir).glob("**/*")):
        if path.suffix in {".md", ".txt"}:
            docs[path.name] = path.read_text()
    if not docs:
        raise FileNotFoundError(f"no .md/.txt documents found in {kb_dir}")
    return docs


def ingest(config: Config, store: VectorStore) -> int:
    """Chunk every document and add it to the store. Returns chunk count."""
    count = 0
    for doc_id, text in load_documents(config.kb_dir).items():
        for chunk in chunk_document(doc_id, text, config.chunk_size, config.chunk_overlap):
            store.add(chunk)
            count += 1
    return count
