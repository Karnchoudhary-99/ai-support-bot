"""Tests for conversation memory: budgeting, pruning, sessions."""

import pytest

from supportbot.memory import ConversationStore, estimate_tokens
from supportbot.models import ChatMessage


def _msg(role: str, words: int) -> ChatMessage:
    return ChatMessage(role=role, content=" ".join(["w"] * words))


def test_estimate_tokens_scales_with_words():
    assert estimate_tokens("a b c d") > estimate_tokens("a b")


def test_history_round_trip():
    store = ConversationStore()
    store.add_message("s1", _msg("user", 5))
    history = store.get_history("s1")
    assert len(history) == 1 and history[0].role == "user"


def test_prune_keeps_newest_message():
    store = ConversationStore(max_tokens=10)
    for i in range(6):
        store.add_message("s1", _msg("user", 10))
    history = store.get_history("s1")
    assert store.token_count("s1") <= 10 or len(history) == 1
    assert history[-1].role == "user"


def test_prune_prefers_dropping_non_system():
    store = ConversationStore(max_tokens=14)
    store.add_message("s1", ChatMessage(role="system", content="w " * 4))
    for _ in range(4):
        store.add_message("s1", _msg("user", 6))
    roles = [m.role for m in store.get_history("s1")]
    assert "system" in roles  # system prompt survives pruning
    assert store.token_count("s1") <= 14


def test_clear_and_session_ids():
    store = ConversationStore()
    store.add_message("a", _msg("user", 1))
    store.add_message("b", _msg("user", 1))
    assert set(store.session_ids()) == {"a", "b"}
    store.clear("a")
    assert store.get_history("a") == []
    assert store.session_ids() == ["b"]


def test_invalid_budget():
    with pytest.raises(ValueError):
        ConversationStore(max_tokens=0)
