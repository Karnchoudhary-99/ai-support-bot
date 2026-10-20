"""Tests for TF-IDF retrieval: ranking quality, edge cases, protocol."""

from supportbot.models import Chunk
from supportbot.retrieval import Retriever, TfidfIndex, tokenize


def _chunk(cid: str, text: str, title: str = "T") -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", title=title, source="s", text=text)


def test_tokenize_lowercases_and_drops_stopwords():
    tokens = tokenize("The QUICK brown Fox, and the dog!")
    assert "quick" in tokens and "fox" in tokens
    assert "the" not in tokens and "and" not in tokens


def test_search_ranks_relevant_chunk_first():
    chunks = [
        _chunk("c1", "how to reset your password via email link"),
        _chunk("c2", "billing invoices are sent on the first of the month"),
        _chunk("c3", "password strength requirements and 2fa setup"),
    ]
    results = Retriever(chunks).search("reset my password", k=3)
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score > results[1].score


def test_search_title_terms_boost_ranking():
    chunks = [
        _chunk("c1", "some generic body text here", title="Password Reset Guide"),
        _chunk("c2", "password password password everywhere", title="Other"),
    ]
    results = Retriever(chunks).search("password reset guide", k=2)
    assert results[0].chunk.chunk_id == "c1"


def test_search_empty_index_and_query():
    assert Retriever().search("anything") == []
    assert Retriever([_chunk("c1", "text")]).search("") == []
    assert Retriever([_chunk("c1", "text")]).search("the and or") == []


def test_search_k_clamped():
    chunks = [_chunk(f"c{i}", f"unique term alpha{i}") for i in range(5)]
    assert len(Retriever(chunks).search("unique term", k=99)) == 5
    assert Retriever(chunks).search("unique term", k=0) == []


def test_index_finalize_empty_ok():
    idx = TfidfIndex()
    idx.finalize()
    assert idx.search("x") == []


def test_scores_are_cosine_bounded():
    chunks = [_chunk("c1", "deploy app to production region"), _chunk("c2", "cats")]
    for r in Retriever(chunks).search("deploy production", k=2):
        assert 0.0 < r.score <= 1.0 + 1e-9
