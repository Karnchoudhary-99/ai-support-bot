"""Tests for the FastAPI service (in-process TestClient, no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from supportbot.api import create_app


@pytest.fixture()
def client(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "billing.md").write_text("# Billing\nInvoices are charged monthly.")
    (corpus / "deploy.md").write_text("# Deploy\nPush to main to deploy.")
    return TestClient(create_app(corpus))


def test_health(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 2


def test_chat_answers_with_citations(client):
    resp = client.post(
        "/v1/chat",
        json={"session_id": "s1", "message": "when are invoices charged?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s1"
    assert not body["escalated"]
    assert body["citations"], "expected grounded citations"
    assert body["citations"][0]["doc_id"] == "billing"


def test_chat_escalates_unknown(client):
    body = client.post("/v1/chat", json={"session_id": "s2", "message": "train my dragon?"}).json()
    assert body["escalated"] is True
    assert body["citations"] == []


def test_chat_validation(client):
    assert client.post("/v1/chat", json={"session_id": "s", "message": ""}).status_code == 422
    assert (
        client.post("/v1/chat", json={"session_id": "s", "message": "hi", "top_k": 99}).status_code
        == 422
    )


def test_chat_stream_sse(client):
    with client.stream(
        "POST", "/v1/chat/stream", json={"session_id": "s3", "message": "how to deploy?"}
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    assert any("token" in e for e in events)
    final = next(e for e in events if e.get("done"))
    assert final["session_id"] == "s3"
    assert "citations" in final


def test_feedback_flow_and_stats(client):
    client.post("/v1/chat", json={"session_id": "s1", "message": "deploy help"})
    fb = client.post(
        "/v1/feedback", json={"session_id": "s1", "helpful": True, "note": "great"}
    ).json()
    assert fb["recorded"] is True
    stats = client.get("/v1/stats").json()
    assert stats["queries"] == 1
    assert stats["feedback_total"] == 1
    assert stats["feedback_helpful_rate"] == 1.0
    assert stats["active_sessions"] == 1


def test_feedback_unknown_session_404(client):
    resp = client.post("/v1/feedback", json={"session_id": "nope", "helpful": False})
    assert resp.status_code == 404


def test_stats_escalation_rate(client):
    client.post("/v1/chat", json={"session_id": "a", "message": "train my dragon?"})
    client.post("/v1/chat", json={"session_id": "b", "message": "deploy help"})
    stats = client.get("/v1/stats").json()
    assert stats["queries"] == 2
    assert stats["escalations"] == 1
    assert stats["escalation_rate"] == 0.5


def test_stats_has_deflection_and_notes(client):
    client.post("/v1/chat", json={"session_id": "s1", "message": "how is billing charged?"})
    client.post(
        "/v1/chat", json={"session_id": "s2", "message": "what is the ceo's favorite food?"}
    )
    client.post("/v1/feedback", json={"session_id": "s1", "helpful": False, "note": "wrong answer"})
    stats = client.get("/v1/stats").json()
    assert stats["queries"] == 2
    assert stats["escalations"] == 1
    assert stats["escalation_rate"] == 0.5
    assert stats["deflection_rate"] == 0.5
    assert stats["feedback_with_notes"] == 1


def test_logs_records_queries_newest_first(client):
    client.post("/v1/chat", json={"session_id": "s1", "message": "how is billing charged?"})
    client.post("/v1/chat", json={"session_id": "s1", "message": "how do I deploy?"})
    body = client.get("/v1/logs?limit=10").json()
    assert body["total"] == 2
    assert body["logs"][0]["query"] == "how do I deploy?"
    assert body["logs"][1]["query"] == "how is billing charged?"
    assert all("escalated" in entry and "confidence" in entry for entry in body["logs"])


def test_logs_truncates_long_queries(client):
    long_q = "deploy " * 100
    client.post("/v1/chat", json={"session_id": "s1", "message": long_q})
    entry = client.get("/v1/logs?limit=1").json()["logs"][0]
    assert len(entry["query"]) <= 121  # 120 chars + ellipsis
    assert entry["query"].endswith("…")
