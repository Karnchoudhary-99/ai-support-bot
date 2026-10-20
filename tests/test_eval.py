"""Tests for the eval harness: gold-set loading, scoring, and aggregation."""

from pathlib import Path

import pytest

from supportbot.bot import SupportBot
from supportbot.eval import EvalCase, EvalReport, _score_case, load_gold_set, main, run_eval
from supportbot.memory import ConversationStore
from supportbot.models import ChatResponse, Citation
from supportbot.store import CorpusStore

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"
GOLD = Path(__file__).resolve().parent.parent / "examples" / "eval" / "gold.json"


@pytest.fixture()
def bot() -> SupportBot:
    return SupportBot(CorpusStore.from_directory(CORPUS), ConversationStore())


def _resp(citations: list[Citation], escalated: bool = False) -> ChatResponse:
    return ChatResponse(
        session_id="s",
        answer="answer",
        citations=citations,
        confidence=0.8,
        escalated=escalated,
        retrieved=len(citations),
    )


def test_load_gold_set_parses_cases():
    cases = load_gold_set(GOLD)
    assert len(cases) == 12
    assert any(c.should_escalate for c in cases)
    assert all(isinstance(c.expected_docs, set) for c in cases)


def test_load_gold_set_rejects_empty(tmp_path: Path):
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    with pytest.raises(ValueError):
        load_gold_set(empty)


def test_score_case_perfect_citations():
    citations = [
        Citation("billing", "Billing", "", "excerpt", 0.9),
        Citation("billing", "Billing", "", "excerpt", 0.8),
    ]
    result = _score_case(EvalCase("q", {"billing"}), _resp(citations))
    assert result.citation_precision == 1.0
    assert result.retrieval_recall == 1.0
    assert result.escalation_correct is True


def test_score_case_partial_citations():
    citations = [
        Citation("billing", "Billing", "", "excerpt", 0.9),
        Citation("api-reference", "API", "", "excerpt", 0.8),
    ]
    result = _score_case(EvalCase("q", {"billing"}), _resp(citations))
    assert result.citation_precision == 0.5
    assert result.retrieval_recall == 1.0


def test_score_case_correct_escalation():
    result = _score_case(EvalCase("q", set(), should_escalate=True), _resp([], True))
    assert result.escalation_correct is True


def test_score_case_wrong_escalation():
    result = _score_case(EvalCase("q", {"billing"}), _resp([], True))
    assert result.escalation_correct is False


def test_run_eval_on_bundled_gold_set(bot: SupportBot):
    cases = load_gold_set(GOLD)
    report = run_eval(bot, cases)
    assert isinstance(report, EvalReport)
    assert report.n == 12
    assert report.escalation_accuracy == 1.0
    assert 0.0 < report.answer_rate < 1.0
    assert report.groundedness >= 0.5
    summary = report.summary()
    assert summary["n"] == 12
    assert set(summary) == {
        "n",
        "groundedness",
        "citation_precision",
        "retrieval_recall",
        "escalation_accuracy",
        "answer_rate",
    }


def test_main_returns_zero_on_bundled_gold(capsys):
    rc = main([str(GOLD), "--corpus", str(CORPUS)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "groundedness" in out
