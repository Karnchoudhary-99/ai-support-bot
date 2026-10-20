"""Tests for MMR reranking and the CorpusStore."""

import pytest

from supportbot.ingest import ingest_markdown
from supportbot.models import Chunk, RetrievalResult
from supportbot.rerank import mmr
from supportbot.store import STORE_VERSION, CorpusStore


def _res(cid: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=cid, doc_id="d", title="T", source="s", text=text),
        score=score,
    )


def test_mmr_lambda_one_is_pure_relevance_order():
    results = [_res("a", "alpha one", 0.9), _res("b", "beta two", 0.8)]
    assert [r.chunk.chunk_id for r in mmr(results, 2, lambda_=1.0)] == ["a", "b"]


def test_mmr_prefers_diverse_second_pick():
    near_dup_a = _res("a", "reset your password with the email link now", 0.95)
    near_dup_b = _res("b", "reset your password with the email link today", 0.94)
    different = _res("c", "billing invoices arrive monthly by card", 0.70)
    ranked = mmr([near_dup_a, near_dup_b, different], 2, lambda_=0.5)
    assert [r.chunk.chunk_id for r in ranked] == ["a", "c"]


def test_mmr_edge_cases():
    assert mmr([], 3) == []
    assert mmr([_res("a", "x", 0.5)], 0) == []
    with pytest.raises(ValueError):
        mmr([_res("a", "x", 0.5)], 1, lambda_=1.5)


def test_store_build_search_and_roundtrip(tmp_path):
    docs = [
        ingest_markdown("billing", "# Billing\ninvoices are charged monthly"),
        ingest_markdown("deploy", "# Deploy\npush to main to deploy your app"),
    ]
    store = CorpusStore.from_documents(docs)
    assert len(store) == 2
    results = store.search("monthly invoices", k=2)
    assert results[0].chunk.doc_id == "billing"

    path = tmp_path / "store.json"
    store.save(path)
    loaded = CorpusStore.load(path)
    assert len(loaded) == 2
    assert loaded.search("deploy app", k=1)[0].chunk.doc_id == "deploy"


def test_store_load_rejects_bad_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"version": 999, "documents": [], "chunks": []}')
    with pytest.raises(ValueError):
        CorpusStore.load(path)


def test_store_search_empty():
    assert CorpusStore().search("anything") == []
