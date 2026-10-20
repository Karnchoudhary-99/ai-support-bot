"""Corpus store: documents + chunks + index with JSON persistence.

One object owns the whole retrieval corpus so the bot, API, and CLI all
share a single load path. `save`/`load` round-trip through JSON; the
TF-IDF index is rebuilt on load (fast, and avoids versioning a binary
format).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .ingest import chunk_document, ingest_directory
from .models import Chunk, Document
from .rerank import mmr
from .retrieval import RetrievalResult, Retriever

STORE_VERSION = 1


class CorpusStore:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.retriever = Retriever()

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def from_documents(cls, documents: list[Document]) -> CorpusStore:
        store = cls()
        store.documents = list(documents)
        store.chunks = [c for doc in documents for c in chunk_document(doc)]
        store.retriever = Retriever(store.chunks)
        return store

    @classmethod
    def from_directory(cls, path: str | Path) -> CorpusStore:
        return cls.from_documents(ingest_directory(path))

    def search(self, query: str, k: int = 5, diversify: bool = True) -> list[RetrievalResult]:
        """Top-k retrieval; MMR diversification on by default."""
        results = self.retriever.search(query, k=max(k * 3, k))
        if diversify and len(results) > k:
            return mmr(results, k)
        return results[:k]

    def to_dict(self) -> dict:
        return {
            "version": STORE_VERSION,
            "documents": [asdict(d) for d in self.documents],
            "chunks": [asdict(c) for c in self.chunks],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CorpusStore:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("version") != STORE_VERSION:
            raise ValueError(f"unsupported store version: {raw.get('version')}")
        store = cls()
        store.documents = [Document(**d) for d in raw["documents"]]
        store.chunks = [Chunk(**c) for c in raw["chunks"]]
        store.retriever = Retriever(store.chunks)
        return store
