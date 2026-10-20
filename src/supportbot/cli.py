"""Interactive demo REPL: `python -m supportbot.cli`.

Builds the store from the bundled sample corpus (or SUPPORTBOT_CORPUS)
and chats in the terminal. Type /quit to exit.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from .api import DEFAULT_CORPUS
from .bot import SupportBot
from .memory import ConversationStore
from .store import CorpusStore

BANNER = """SupportBot demo — answers come from the bundled sample docs.
Type your question, /quit to exit."""


def main() -> None:
    corpus = Path(os.environ.get("SUPPORTBOT_CORPUS", DEFAULT_CORPUS))
    store = CorpusStore.from_directory(corpus)
    bot = SupportBot(store, ConversationStore())
    session_id = f"demo-{uuid.uuid4().hex[:8]}"
    print(BANNER)
    print(f"(session {session_id}, {len(store)} chunks indexed)\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if question in {"/quit", "/exit"}:
            print("bye!")
            break
        if not question:
            continue
        response = bot.ask(session_id, question)
        print(f"\nbot> {response.answer}")
        if response.escalated:
            print("(escalated to a human agent)")
        else:
            print(f"(confidence {response.confidence:.2f})\n")


if __name__ == "__main__":
    main()
