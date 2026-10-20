"""Tests for the answer composer: grounding, citations, escalation."""

from supportbot.bot import SupportBot
from supportbot.ingest import ingest_markdown
from supportbot.memory import ConversationStore
from supportbot.store import CorpusStore


def _bot() -> SupportBot:
    docs = [
        ingest_markdown("billing", "# Billing\nInvoices are charged monthly to your card."),
        ingest_markdown("deploy", "# Deploy\nPush to main to deploy your application."),
    ]
    return SupportBot(CorpusStore.from_documents(docs), ConversationStore())


def test_answer_cites_relevant_doc():
    response = _bot().ask("s1", "when are invoices charged?")
    assert not response.escalated
    assert response.retrieved > 0
    assert any(c.doc_id == "billing" for c in response.citations)
    assert 0.0 < response.confidence <= 1.0


def test_answer_mentions_citation_markers():
    response = _bot().ask("s1", "how do I deploy?")
    assert "[1]" in response.answer


def test_out_of_corpus_escalates():
    response = _bot().ask("s1", "how do I train a dragon for battle?")
    assert response.escalated
    assert response.confidence == 0.0
    assert response.citations == []


def test_empty_corpus_escalates():
    bot = SupportBot(CorpusStore(), ConversationStore())
    response = bot.ask("s1", "anything at all?")
    assert response.escalated


def test_turns_recorded_in_memory():
    bot = _bot()
    bot.ask("s1", "when are invoices charged?")
    history = bot.memory.get_history("s1")
    assert [m.role for m in history] == ["user", "assistant"]


def test_high_threshold_escalates_everything():
    bot = _bot()
    bot.escalation_threshold = 0.99
    assert bot.ask("s1", "when are invoices charged?").escalated
