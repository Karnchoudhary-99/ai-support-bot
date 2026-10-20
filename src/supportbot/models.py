"""Domain models for supportbot.

Plain dataclasses shared by the ingestion, retrieval, conversation, and
service layers. Kept free of any framework dependency so the core engine is
trivially testable and embeddable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utcnow() -> datetime:
    """Timezone-aware now, used as the default timestamp factory."""
    return datetime.now(timezone.utc)


@dataclass
class Document:
    """A raw source document before chunking."""

    doc_id: str
    title: str
    source: str  # file path or URL the document came from
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit of text cut from a Document."""

    chunk_id: str  # "{doc_id}#{index}"
    doc_id: str
    title: str
    source: str
    text: str
    section: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """One chunk returned for a query, with its similarity score."""

    chunk: Chunk
    score: float


@dataclass
class Citation:
    """A source citation attached to a bot answer."""

    doc_id: str
    title: str
    section: str
    excerpt: str
    score: float


@dataclass
class ChatMessage:
    """A single turn in a conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class ChatRequest:
    session_id: str
    message: str
    top_k: int = 5


@dataclass
class ChatResponse:
    session_id: str
    answer: str
    citations: list[Citation]
    confidence: float  # 0.0 - 1.0, retrieval-grounded confidence
    escalated: bool  # True when the bot defers to a human
    retrieved: int  # number of chunks the answer was grounded in


@dataclass
class FeedbackEvent:
    """A thumbs up/down (plus optional note) on a bot answer."""

    session_id: str
    helpful: bool
    message_id: str | None = None
    note: str = ""
    created_at: datetime = field(default_factory=utcnow)
