#!/usr/bin/env python3
"""CLI chat for v3 — ask plain-English questions about the DAD data-center IoT network.

    python3.12 apps/chat.py "how many attacks on wednesday?"
    python3.12 apps/chat.py --llm            # interactive, with local-LLM phrasing if Ollama is up

Every number comes from the deterministic query engine; the optional LLM only rewords it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3))

from src import interface, preprocess  # noqa: E402


def format_answer(ans) -> str:
    """Render an Answer with its trust labels for the terminal."""
    lines = [f"🔢 {ans.text}"]
    if getattr(ans, "used_llm", False):
        lines.append("   (reworded by local AI — numbers verified against the data)")
    if ans.context:
        lines.append(f"📚 {ans.context}")
    if ans.inferred:
        lines.append(f"🧠 {ans.inferred}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ask questions about the DAD IoT-network dataset.")
    parser.add_argument("question", nargs="*", help="a one-off question; omit for interactive mode")
    parser.add_argument("--llm", action="store_true", help="use a local Ollama model to reword answers")
    args = parser.parse_args(argv)

    print("Loading DAD dataset…", file=sys.stderr)
    iface = interface.Interface(preprocess.load_week(), use_llm=args.llm)

    if args.question:
        print(format_answer(iface.ask(" ".join(args.question))))
        return 0

    print("DAD chat — ask about attacks, attack types, the attacker, temperatures, or traffic. "
          "Ctrl-D to exit.")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            print()
            return 0
        if q:
            print(format_answer(iface.ask(q)))


if __name__ == "__main__":
    raise SystemExit(main())
