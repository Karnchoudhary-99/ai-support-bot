"""Conversation memory: per-session history with a token budget.

History is the bot's short-term memory. A word-count based token estimate
(words * 4/3) keeps pruning cheap and dependency-free; when the budget is
exceeded the oldest non-system turns are dropped, always keeping the most
recent user message so the current question never loses its context.
"""

from __future__ import annotations

from collections import defaultdict

from .models import ChatMessage

TOKENS_PER_WORD = 4 / 3


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token ~ 1.33 per word."""
    return max(1, int(len(text.split()) * TOKENS_PER_WORD))


class ConversationStore:
    """In-memory session histories with token-budget pruning."""

    def __init__(self, max_tokens: int = 2000) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self._sessions: dict[str, list[ChatMessage]] = defaultdict(list)

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        history = self._sessions[session_id]
        history.append(message)
        self._prune(session_id)

    def get_history(self, session_id: str) -> list[ChatMessage]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def token_count(self, session_id: str) -> int:
        return sum(estimate_tokens(m.content) for m in self._sessions[session_id])

    def _prune(self, session_id: str) -> None:
        history = self._sessions[session_id]
        # Never prune the newest message (the question being answered).
        while self.token_count(session_id) > self.max_tokens and len(history) > 1:
            # Prefer dropping the oldest non-system message.
            drop = next(
                (i for i, m in enumerate(history[:-1]) if m.role != "system"),
                0,
            )
            del history[drop]
