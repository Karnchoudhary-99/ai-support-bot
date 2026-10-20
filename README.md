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
- [x] Eval harness: groundedness + faithfulness scoring on a gold set
- [x] Admin analytics: query logs, deflection rate, thumbs feedback
- [x] Dockerfile + docker-compose for one-command local demo

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest                                    # 57 tests
python -m supportbot.eval                 # groundedness eval on bundled gold set
python examples/ask.py                    # offline Q&A demo over sample docs
python -m supportbot.cli                  # interactive REPL demo
uvicorn supportbot.api:app                # API on http://127.0.0.1:8000
```

Docker, one command:

```bash
docker compose up --build                 # API on http://127.0.0.1:8000
docker compose up --build -v ./my-docs:/app/corpus:ro   # your own docs
```

Set `SUPPORTBOT_CORPUS=/path/to/docs` to serve your own markdown corpus.

## API

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/v1/health` | Liveness + indexed chunk count |
| POST | `/v1/chat` | Answer with citations, confidence, escalation flag |
| POST | `/v1/chat/stream` | Same, streamed as SSE `token` events + final `done` |
| POST | `/v1/feedback` | Thumbs up/down on a session's answer |
| GET | `/v1/stats` | Queries, escalation/deflection rates, feedback, sessions |
| GET | `/v1/logs?limit=` | Recent queries (newest first) for admin review |

### Eval harness

The bot is extractive, so groundedness is scored as checkable properties on a
gold question set (`examples/eval/gold.json`, 12 cases over the bundled corpus
including 2 out-of-corpus questions that must escalate):

```bash
python -m supportbot.eval                    # bundled gold set + sample corpus
python -m supportbot.eval path/to/gold.json --corpus path/to/docs
```

Metrics: `groundedness` (headline: citation precision blended with escalation
accuracy), `citation_precision` (cited docs all expected?), `retrieval_recall`
(expected docs surfaced?), `escalation_accuracy`, `answer_rate`. Exits non-zero
if groundedness drops below 0.5 — suitable for CI quality gates. Latest run:
**groundedness 0.778, escalation accuracy 1.0**.

## Project layout

- `src/supportbot/` — bot package (ingest, retrieval, rerank, memory, bot, eval, service)
- `tests/` — pytest suite
- `examples/` — sample corpora, `ask.py` demo, eval gold set
- `Dockerfile`, `docker-compose.yml` — containerized demo

## License

MIT — see [LICENSE](LICENSE).
