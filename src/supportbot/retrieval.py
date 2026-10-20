"""Retrieval: TF-IDF vector search over chunks, stdlib-only.

A deliberate default: zero heavy dependencies, deterministic, fast enough
for support corpora (thousands of chunks). The `Embedder` protocol lets a
deployment swap in dense embeddings (e.g. sentence-transformers or an API
embedder) without touching the bot or API layers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from .models import Chunk, RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small built-in stopword list: keeps the index lean without NLTK.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can did do does for from had has have
    he her his how i if in into is it its me my not of on or our out over
    said she so that the their them then there these they this to was we
    were what when where which who will with would you your
    """.split()
)


class Embedder(Protocol):
    """Bring-your-own dense embedder contract."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens minus stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class TfidfIndex:
    """In-memory TF-IDF index with cosine top-k search.

    IDF uses add-one smoothing: idf(t) = log((N + 1) / (df(t) + 1)) + 1,
    so unseen query terms get a finite (low) weight instead of zeroing out.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._doc_vectors: list[dict[str, float]] = []
        self._doc_norms: list[float] = []
        self._idf: dict[str, float] = {}
        self._vocab: set[str] = set()

    def __len__(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks; call `finalize()` once after all adds."""
        for chunk in chunks:
            tokens = tokenize(chunk.text + " " + chunk.title + " " + chunk.section)
            self.chunks.append(chunk)
            self._doc_vectors.append(dict(Counter(tokens)))
        self._vocab.update(t for vec in self._doc_vectors for t in vec)

    def finalize(self) -> None:
        """Compute IDF weights and TF-IDF vectors + norms."""
        n = len(self.chunks)
        if n == 0:
            return
        df: Counter[str] = Counter()
        for vec in self._doc_vectors:
            df.update(vec.keys())
        self._idf = {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in self._vocab}
        for i, vec in enumerate(self._doc_vectors):
            weighted = {t: (1.0 + math.log(c)) * self._idf[t] for t, c in vec.items()}
            self._doc_vectors[i] = weighted
            self._doc_norms.append(math.sqrt(sum(w * w for w in weighted.values())))

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Return the top-k chunks by cosine similarity (k clamped)."""
        if not self.chunks or not self._idf:
            return []
        if k <= 0:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_vec = Counter(q_tokens)
        n = len(self.chunks)
        q_weighted = {
            t: (1.0 + math.log(c)) * self._idf.get(t, math.log(n + 1) + 1.0)
            for t, c in q_vec.items()
        }
        q_norm = math.sqrt(sum(w * w for w in q_weighted.values()))
        if q_norm == 0.0:
            return []
        scored: list[tuple[float, int]] = []
        for i, doc_vec in enumerate(self._doc_vectors):
            dot = sum(q_weighted.get(t, 0.0) * w for t, w in doc_vec.items())
            denom = q_norm * self._doc_norms[i]
            score = dot / denom if denom else 0.0
            if score > 0.0:
                scored.append((score, i))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [
            RetrievalResult(chunk=self.chunks[i], score=score)
            for score, i in scored[:k]
        ]


class Retriever:
    """Convenience wrapper: build from chunks in one call."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.index = TfidfIndex()
        if chunks:
            self.index.add_chunks(chunks)
            self.index.finalize()

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return self.index.search(query, k)
