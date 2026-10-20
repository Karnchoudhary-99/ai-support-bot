# AI Customer Support Bot

A production-style support bot in Python: **RAG over your docs** with a
FastAPI service, streaming answers, conversation memory, an eval harness,
and admin analytics — built so every layer is observable and testable.

## Roadmap

- [x] Project scaffold (layout, README, CI)
- [ ] Document ingestion: markdown/PDF/web -> chunked corpus with metadata
- [ ] Embeddings + vector retrieval (bring your own embedder)
- [ ] FastAPI service: `POST /v1/chat` (streaming SSE), sessions, citations
- [ ] Conversation memory with token-budgeted history
- [ ] Eval harness: groundedness + faithfulness scoring on a gold set
- [ ] Admin analytics: query logs, deflection rate, thumbs feedback
- [ ] Dockerfile + docker-compose for one-command local demo

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Project layout

- `src/supportbot/` — bot package (ingest, retrieval, service, memory, eval)
- `tests/` — pytest suite
- `examples/` — sample corpora and demo scripts

## License

MIT — see [LICENSE](LICENSE).
