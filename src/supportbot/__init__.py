"""supportbot: RAG-powered customer support bot."""

from .ingest import chunk_document, ingest_directory, ingest_markdown
from .models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Chunk,
    Citation,
    Document,
    FeedbackEvent,
    RetrievalResult,
)
from .rerank import mmr
from .retrieval import Embedder, Retriever, TfidfIndex, tokenize
from .store import CorpusStore

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Chunk",
    "Citation",
    "CorpusStore",
    "Document",
    "Embedder",
    "FeedbackEvent",
    "RetrievalResult",
    "Retriever",
    "TfidfIndex",
    "chunk_document",
    "ingest_directory",
    "ingest_markdown",
    "mmr",
    "tokenize",
]

__version__ = "0.1.0"
