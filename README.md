# AI Customer Support Bot

A production-style support bot in Python: **RAG over your docs** with a
FastAPI service, streaming answers, conversation memory, an eval harness,
and admin analytics — built so every layer is observable and testable.

## Roadmap

- [x] Project scaffold (layout, README, CI)
- [x] Document ingestion: markdown/text -> chunked corpus with metadata
- [x] TF-IDF retrieval + MMR rerank (bring-your-own `Embedder` protocol)
- [x] FastAPI service: `POST /v1/chat` (JSON + SSE stream), sessions, citations
- [x] Conversation memory with token-budgeted history
- [ ] Eval harness: groundedness + faithfulness scoring on a gold set
- [ ] Admin analytics: query logs, deflection rate, thumbs feedback
- [ ] Dockerfile + docker-compose for one-command local demo

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest                                    # 46 tests
python examples/ask.py                    # offline Q&A demo over sample docs
python -m supportbot.cli                  # interactive REPL demo
uvicorn supportbot.api:app                # API on http://127.0.0.1:8000
```

Set `SUPPORTBOT_CORPUS=/path/to/docs` to serve your own markdown corpus.

## API

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/v1/health` | Liveness + indexed chunk count |
| POST | `/v1/chat` | Answer with citations, confidence, escalation flag |
| POST | `/v1/chat/stream` | Same, streamed as SSE `token` events + final `done` |
| POST | `/v1/feedback` | Thumbs up/down on a session's answer |
| GET | `/v1/stats` | Queries, escalation rate, feedback, active sessions |

## Project layout

- `src/supportbot/` — bot package (ingest, retrieval, service, memory, eval)
- `tests/` — pytest suite
- `examples/` — sample corpora and demo scripts

## License

MIT — see [LICENSE](LICENSE).
