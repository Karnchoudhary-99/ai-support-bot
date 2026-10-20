"""FastAPI service: chat (JSON + SSE streaming), feedback, stats.

Run locally with::

    uvicorn supportbot.api:app --reload

The corpus directory is taken from the ``SUPPORTBOT_CORPUS`` environment
variable, falling back to the bundled ``examples/corpus`` sample docs.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .bot import SupportBot
from .memory import ConversationStore
from .models import ChatResponse, FeedbackEvent
from .store import CorpusStore

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent.parent / "examples" / "corpus"
QUERY_LOG_SIZE = 200  # ring buffer: last N queries, for /v1/logs
QUERY_PREVIEW_CHARS = 120  # truncate query text in the log


class ChatBody(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class FeedbackBody(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    helpful: bool
    note: str = Field(default="", max_length=2000)


class ServiceState:
    def __init__(self, corpus_dir: str | Path | None = None) -> None:
        root = Path(corpus_dir or os.environ.get("SUPPORTBOT_CORPUS", DEFAULT_CORPUS))
        self.store = CorpusStore.from_directory(root)
        self.bot = SupportBot(self.store, ConversationStore())
        self.feedback: list[FeedbackEvent] = []
        self.queries = 0
        self.escalations = 0
        self.query_log: list[dict] = []  # newest last, capped at QUERY_LOG_SIZE

    def log_query(
        self, session_id: str, message: str, response: ChatResponse
    ) -> None:
        preview = message if len(message) <= QUERY_PREVIEW_CHARS else message[
            :QUERY_PREVIEW_CHARS
        ].rstrip() + "…"
        self.query_log.append(
            {
                "session_id": session_id,
                "query": preview,
                "escalated": response.escalated,
                "confidence": response.confidence,
                "citations": len(response.citations),
            }
        )
        if len(self.query_log) > QUERY_LOG_SIZE:
            del self.query_log[: len(self.query_log) - QUERY_LOG_SIZE]


def _response_to_dict(response: ChatResponse) -> dict:
    return {
        "session_id": response.session_id,
        "answer": response.answer,
        "citations": [
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "section": c.section,
                "excerpt": c.excerpt,
                "score": c.score,
            }
            for c in response.citations
        ],
        "confidence": response.confidence,
        "escalated": response.escalated,
        "retrieved": response.retrieved,
    }


def create_app(corpus_dir: str | Path | None = None) -> FastAPI:
    state = ServiceState(corpus_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(title="SupportBot API", version="0.1.0", lifespan=lifespan)
    app.state.service = state

    @app.get("/v1/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(state.store)}

    @app.post("/v1/chat")
    def chat(body: ChatBody) -> dict:
        state.queries += 1
        response = state.bot.ask(body.session_id, body.message, top_k=body.top_k)
        if response.escalated:
            state.escalations += 1
        state.log_query(body.session_id, body.message, response)
        return _response_to_dict(response)

    @app.post("/v1/chat/stream")
    def chat_stream(body: ChatBody) -> StreamingResponse:
        state.queries += 1
        response = state.bot.ask(body.session_id, body.message, top_k=body.top_k)
        if response.escalated:
            state.escalations += 1
        state.log_query(body.session_id, body.message, response)

        async def events() -> AsyncIterator[str]:
            for word in response.answer.split():
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            yield f"data: {json.dumps({'done': True, **_response_to_dict(response)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/feedback")
    def feedback(body: FeedbackBody) -> dict:
        if body.session_id not in state.bot.memory.session_ids():
            raise HTTPException(status_code=404, detail="unknown session_id")
        state.feedback.append(
            FeedbackEvent(session_id=body.session_id, helpful=body.helpful, note=body.note)
        )
        return {"recorded": True, "total": len(state.feedback)}

    @app.get("/v1/stats")
    def stats() -> dict:
        helpful = sum(1 for f in state.feedback if f.helpful)
        noted = sum(1 for f in state.feedback if f.note)
        escalation_rate = round(state.escalations / max(1, state.queries), 4)
        return {
            "queries": state.queries,
            "escalations": state.escalations,
            "escalation_rate": escalation_rate,
            "deflection_rate": round(1.0 - escalation_rate, 4),
            "feedback_total": len(state.feedback),
            "feedback_helpful_rate": round(helpful / max(1, len(state.feedback)), 4),
            "feedback_with_notes": noted,
            "active_sessions": len(state.bot.memory.session_ids()),
            "chunks": len(state.store),
        }

    @app.get("/v1/logs")
    def logs(limit: int = 20) -> dict:
        """Recent queries for admin review (newest first)."""
        limit = max(1, min(limit, QUERY_LOG_SIZE))
        return {"total": len(state.query_log), "logs": list(reversed(state.query_log[-limit:]))}

    return app


app = create_app()
