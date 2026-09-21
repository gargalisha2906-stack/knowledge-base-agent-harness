"""Embedding: turning text into vectors so meaning can be compared with
cosine similarity.

The default ``HashingEmbedder`` is pure Python and needs no API key, which
keeps the demo and tests fully offline. It is a *lexical* embedder: it maps
tokens (and token bigrams) into a fixed-size vector by hashing, then
L2-normalises. Two texts that share vocabulary land close together.

Semantic quality is lower than a hosted embedding model, so the interface is
the seam: implement ``Embedder`` against your provider of choice and the
rest of the harness is unchanged.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very common words carry no retrieval signal but dominate raw counts, so we
# drop them. Keep this list short and obvious.
STOPWORDS = frozenset(
    "a an the is are was were be been of to in on for with and or not no "
    "how do does did i you we they my your our their me it this that what "
    "when where which who why can could should would will if at by from as".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed; used everywhere."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    """Anything that can turn a string into a fixed-size float vector."""

    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic bag-of-words(+bigrams) vectors via feature hashing."""

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def _bump(self, vec: list[float], feature: str, weight: float) -> None:
        digest = hashlib.md5(feature.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign * weight

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        vec = [0.0] * self.dim
        for token in tokens:
            self._bump(vec, token, 1.0)
            # Character trigrams give sub-word matching ("refunding" still
            # shares most trigrams with "refund"), which makes the vectors
            # robust to inflections without any stemming library.
            padded = f"#{token}#"
            for i in range(len(padded) - 2):
                self._bump(vec, f"tri:{padded[i:i+3]}", 0.5)
        for a, b in zip(tokens, tokens[1:]):
            self._bump(vec, f"{a} {b}", 1.0)  # bigrams catch short phrases
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (both are already L2-normalised)."""
    return sum(x * y for x, y in zip(a, b))
