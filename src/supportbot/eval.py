"""Eval harness: groundedness + escalation scoring on a gold question set.

The bot is extractive (no LLM), so "groundedness" here means concrete,
checkable properties instead of vibes:

- **citation precision**: cited docs all belong to the case's expected set
- **retrieval recall**: expected docs actually surfaced in the answer's citations
- **escalation accuracy**: the bot escalated exactly on the questions it
  cannot answer from the corpus (and answered the ones it can)

Run it with::

    python -m supportbot.eval                     # bundled sample gold set
    python -m supportbot.eval path/to/gold.json   # your own gold set
    python -m supportbot.eval --corpus path/to/docs

A gold set is a JSON list of cases::

    {"question": "...", "expected_docs": ["billing"], "should_escalate": false}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .bot import SupportBot
from .memory import ConversationStore
from .models import ChatResponse
from .store import CorpusStore

GOLD_DIR = Path(__file__).resolve().parent.parent.parent / "examples" / "eval"


@dataclass
class EvalCase:
    """One scored question: what docs must back a good answer."""

    question: str
    expected_docs: set[str] = field(default_factory=set)
    should_escalate: bool = False


@dataclass
class CaseResult:
    """Per-case outcome with the underlying bot response."""

    case: EvalCase
    response: ChatResponse
    citation_precision: float
    retrieval_recall: float
    escalation_correct: bool


@dataclass
class EvalReport:
    """Aggregate scores across the gold set."""

    results: list[CaseResult]

    @property
    def n(self) -> int:
        return len(self.results)

    def _mean(self, key: str) -> float:
        if not self.results:
            return 0.0
        return round(sum(getattr(r, key) for r in self.results) / len(self.results), 4)

    @property
    def citation_precision(self) -> float:
        """Mean over answerable cases: cited docs all expected?"""
        answered = [r for r in self.results if not r.response.escalated]
        if not answered:
            return 0.0
        return round(sum(r.citation_precision for r in answered) / len(answered), 4)

    @property
    def retrieval_recall(self) -> float:
        """Mean over answerable cases: expected docs surfaced?"""
        answered = [r for r in self.results if not r.response.escalated]
        if not answered:
            return 0.0
        return round(sum(r.retrieval_recall for r in answered) / len(answered), 4)

    @property
    def escalation_accuracy(self) -> float:
        return self._mean("escalation_correct")

    @property
    def answer_rate(self) -> float:
        answered = sum(1 for r in self.results if not r.response.escalated)
        return round(answered / max(1, len(self.results)), 4)

    @property
    def groundedness(self) -> float:
        """Headline score: citation precision weighted by escalation accuracy.

        A bot that answers everything but cites junk scores ~0 on precision;
        a bot that escalates everything scores 0 on answered coverage only
        implicitly — escalation_accuracy carries that signal.
        """
        return round(0.7 * self.citation_precision + 0.3 * self.escalation_accuracy, 4)

    def summary(self) -> dict:
        return {
            "n": self.n,
            "groundedness": self.groundedness,
            "citation_precision": self.citation_precision,
            "retrieval_recall": self.retrieval_recall,
            "escalation_accuracy": self.escalation_accuracy,
            "answer_rate": self.answer_rate,
        }


def load_gold_set(path: str | Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text())
    cases = []
    for item in raw:
        cases.append(
            EvalCase(
                question=item["question"],
                expected_docs=set(item.get("expected_docs", [])),
                should_escalate=bool(item.get("should_escalate", False)),
            )
        )
    if not cases:
        raise ValueError(f"gold set {path} is empty")
    return cases


def _score_case(case: EvalCase, response: ChatResponse) -> CaseResult:
    escalation_correct = response.escalated == case.should_escalate
    if response.escalated or not case.expected_docs:
        return CaseResult(case, response, 0.0, 0.0, escalation_correct)

    cited_docs = {c.doc_id for c in response.citations}
    citation_precision = (
        sum(1 for d in cited_docs if d in case.expected_docs) / len(cited_docs)
        if cited_docs
        else 0.0
    )
    retrieval_recall = sum(1 for d in case.expected_docs if d in cited_docs) / len(
        case.expected_docs
    )
    return CaseResult(
        case,
        response,
        round(citation_precision, 4),
        round(retrieval_recall, 4),
        escalation_correct,
    )


def run_eval(bot: SupportBot, cases: list[EvalCase]) -> EvalReport:
    """Ask every gold question on a fresh session and score the answers."""
    results = []
    for i, case in enumerate(cases):
        response = bot.ask(f"eval-{i}", case.question)
        results.append(_score_case(case, response))
    return EvalReport(results)


def _default_gold() -> Path:
    path = GOLD_DIR / "gold.json"
    if not path.exists():
        raise FileNotFoundError(f"bundled gold set missing: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the bot on a gold question set")
    parser.add_argument("gold", nargs="?", default=None, help="gold set JSON (default: bundled)")
    parser.add_argument("--corpus", default=None, help="corpus dir (default: bundled sample)")
    args = parser.parse_args(argv)

    gold_path = Path(args.gold) if args.gold else _default_gold()
    cases = load_gold_set(gold_path)
    corpus = Path(args.corpus) if args.corpus else None
    store = CorpusStore.from_directory(corpus) if corpus else CorpusStore.from_directory(
        Path(__file__).resolve().parent.parent.parent / "examples" / "corpus"
    )
    bot = SupportBot(store, ConversationStore())
    report = run_eval(bot, cases)

    print(f"eval: {report.n} cases -> {gold_path}")
    for key, value in report.summary().items():
        if key != "n":
            print(f"  {key:20s} {value:.4f}")
    failures = [
        r
        for r in report.results
        if not r.escalation_correct
        or (not r.response.escalated and r.citation_precision < 1.0)
    ]
    if failures:
        print(f"\n{len(failures)} imperfect case(s):")
        for r in failures:
            print(f"  - {r.case.question[:70]!r} precision={r.citation_precision} "
                  f"escalated={r.response.escalated}")
    return 0 if report.groundedness >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
