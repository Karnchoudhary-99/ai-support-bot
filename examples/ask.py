"""Offline demo: index the sample corpus and answer a few questions.

Usage: python examples/ask.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from supportbot.bot import SupportBot  # noqa: E402
from supportbot.memory import ConversationStore  # noqa: E402
from supportbot.store import CorpusStore  # noqa: E402

QUESTIONS = [
    "How do I deploy my first project?",
    "What happens if my payment fails?",
    "Why am I getting a 502 error after deploy?",
    "What is the API rate limit?",
    "How do I train a dragon?",  # out-of-corpus -> should escalate
]


def main() -> None:
    corpus = Path(__file__).resolve().parent / "corpus"
    store = CorpusStore.from_directory(corpus)
    bot = SupportBot(store, ConversationStore())
    print(f"Indexed {len(store)} chunks from {corpus}\n")
    for i, question in enumerate(QUESTIONS, start=1):
        response = bot.ask(f"demo-{i}", question)
        print(f"Q: {question}")
        print(f"A: {response.answer.splitlines()[0]}")
        print(
            f"   confidence={response.confidence:.2f} "
            f"escalated={response.escalated} "
            f"citations={[c.title for c in response.citations]}\n"
        )


if __name__ == "__main__":
    main()
