# Production-style image: deps first (cached), then the app source.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SUPPORTBOT_CORPUS=/app/corpus

WORKDIR /app

# System deps for any doc parsing extensions; keep the layer thin.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY src/ ./src/
COPY examples/corpus/ ./corpus/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/v1/health || exit 1

CMD ["uvicorn", "supportbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
