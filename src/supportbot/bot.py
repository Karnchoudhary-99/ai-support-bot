"""Answer composer: extractive RAG with citations and escalation.

No LLM required for the core loop: the bot retrieves grounded chunks, pulls
the most query-relevant sentence out of each, and composes a cited answer.
Confidence comes straight from retrieval scores; when nothing relevant is
found the bot escalates to a human instead of hallucinating — the single
most important behavior of a support bot.
"""

from __future__ import annotations

import re

from .memory import ConversationStore
from .models import (
    ChatMessage,
    ChatResponse,
    Citation,
    RetrievalResult,
)
from .retrieval import tokenize
from .store import CorpusStore

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
EXCERPT_CHARS = 160

# Below this top-1 cosine score the bot admits it doesn't know.
ESCALATION_THRESHOLD = 0.15
MAX_CITED_CHUNKS = 3


def _best_sentence(chunk_text: str, query_terms: set[str]) -> str:
    """Pick the sentence with the most query-term overlap."""
    sentences = [s.strip() for s in SENTENCE_RE.findall(chunk_text)]
    if not sentences:
        return chunk_text[:EXCERPT_CHARS].strip()
    scored = sorted(
        sentences,
        key=lambda s: len(query_terms & set(tokenize(s))),
        reverse=True,
    )
    return scored[0] if scored[0] else sentences[0]


def _excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS].rstrip() + "…"


class SupportBot:
    """Stateless answering engine over a CorpusStore + ConversationStore."""

    def __init__(
        self,
        store: CorpusStore,
        memory: ConversationStore | None = None,
        escalation_threshold: float = ESCALATION_THRESHOLD,
    ) -> None:
        self.store = store
        self.memory = memory or ConversationStore()
        self.escalation_threshold = escalation_threshold

    def ask(self, session_id: str, message: str, top_k: int = 5) -> ChatResponse:
        """Answer one user message, recording the turn in session memory."""
        query_terms = set(tokenize(message))
        results = self.store.search(message, k=top_k)

        self.memory.add_message(session_id, ChatMessage(role="user", content=message))

        if not results or results[0].score < self.escalation_threshold:
            answer = (
                "I couldn't find a confident answer in our help docs. "
                "I've flagged this for a human support agent — they'll "
                "follow up in this chat shortly."
            )
            response = ChatResponse(
                session_id=session_id,
                answer=answer,
                citations=[],
                confidence=0.0,
                escalated=True,
                retrieved=0,
            )
        else:
            response = self._compose(session_id, message, query_terms, results)

        self.memory.add_message(session_id, ChatMessage(role="assistant", content=response.answer))
        return response

    def _compose(
        self,
        session_id: str,
        message: str,
        query_terms: set[str],
        results: list[RetrievalResult],
    ) -> ChatResponse:
        cited = results[:MAX_CITED_CHUNKS]
        lines = ["Here's what I found in our help docs:"]
        citations: list[Citation] = []
        for i, result in enumerate(cited, start=1):
            sentence = _best_sentence(result.chunk.text, query_terms)
            lines.append(f"[{i}] {sentence}")
            citations.append(
                Citation(
                    doc_id=result.chunk.doc_id,
                    title=result.chunk.title,
                    section=result.chunk.section,
                    excerpt=_excerpt(result.chunk.text),
                    score=round(result.score, 4),
                )
            )
        where = ", ".join(sorted({c.title for c in citations}))
        lines.append(f"\nSources: {where}. Ask a follow-up if you need more detail!")
        confidence = round(min(1.0, sum(r.score for r in cited) / len(cited)), 4)
        return ChatResponse(
            session_id=session_id,
            answer="\n".join(lines),
            citations=citations,
            confidence=confidence,
            escalated=False,
            retrieved=len(cited),
        )
