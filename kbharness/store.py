"""A tiny local vector store.

Chunks and their vectors live in memory and persist to one JSON file. That
is the right scale for a knowledge base of hundreds of chunks; the seam to
a real vector database is the ``VectorStore`` interface (add / search).
"""

from __future__ import annotations

import json
from pathlib import Path

from .chunking import Chunk
from .embeddings import Embedder, cosine_similarity
from dataclasses import dataclass


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    """In-memory cosine-similarity index with JSON persistence."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunk: Chunk) -> None:
        self._chunks.append(chunk)
        self._vectors.append(self.embedder.embed(chunk.text))

    def search(self, query: str, top_k: int, doc_ids: set[str] | None = None) -> list[ScoredChunk]:
        """Return the ``top_k`` closest chunks, optionally restricted to
        ``doc_ids``. Filtering before ranking keeps tenants/domains apart."""
        if not self._chunks:
            return []
        qvec = self.embedder.embed(query)
        scored = []
        for chunk, vec in zip(self._chunks, self._vectors):
            if doc_ids is not None and chunk.doc_id not in doc_ids:
                continue
            scored.append(ScoredChunk(chunk=chunk, score=cosine_similarity(qvec, vec)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)

    # --- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "chunks": [c.__dict__ for c in self._chunks],
            "vectors": self._vectors,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload))

    def load(self, path: str) -> None:
        payload = json.loads(Path(path).read_text())
        self._chunks = [Chunk(**c) for c in payload["chunks"]]
        self._vectors = payload["vectors"]
