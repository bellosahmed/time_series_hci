"""Run the natural-language interface over the Jena data — with optional local LLM phrasing.

Examples:
    # template phrasing (no LLM), interactive
    python3.12 chat.py

    # local Ollama phrasing, interactive
    python3.12 chat.py --llm

    # one-shot question with the LLM
    python3.12 chat.py --llm "what was the hottest temperature?"

If Ollama isn't running, it falls back to the deterministic template automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../v1/apps
V1 = HERE.parent                                 # .../v1
sys.path.insert(0, str(V1))                      # make `src` importable from anywhere

from src import preprocess, interface, llm

DEFAULT_DATA = V1.parent / "data" / "jena_climate_2009_2016.csv"
PREFERRED = "llama3.2:3b"


def pick_model(requested: str | None) -> str:
    if requested:
        return requested
    models = llm.list_models()
    if PREFERRED in models:
        return PREFERRED
    return models[0] if models else PREFERRED


def main() -> None:
    p = argparse.ArgumentParser(description="Natural-language Q&A over the time-series data")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--sensor", default="T (degC)")
    p.add_argument("--llm", action="store_true", help="use a local Ollama model to phrase answers")
    p.add_argument("--model", default=None, help="Ollama model name (default: auto)")
    p.add_argument("question", nargs="*", help="ask once and exit; omit for interactive mode")
    args = p.parse_args()

    print("Loading + preprocessing data...")
    df = preprocess.align(preprocess.clean(preprocess.load_data(args.data)), freq="1h")

    use_llm = False
    model = None
    if args.llm:
        if llm.is_available():
            use_llm = True
            model = pick_model(args.model)
            print(f"Using local Ollama model: {model}")
        else:
            print("[warn] Ollama not reachable at localhost:11434 — using template phrasing.")
            print("       Start it with `ollama serve` and `ollama pull llama3.2:3b`.")

    iface = interface.Interface(df, use_llm=use_llm, model=model or "llama3.2")

    if args.question:
        print("\n" + iface.ask(" ".join(args.question)).text)
        return

    mode = f"LLM:{model}" if use_llm else "template"
    print(f"\nReady ({mode}). Ask e.g. 'max temperature?', 'correlation of temperature and density',")
    print("'any unusual readings?'. Ctrl-D or Ctrl-C to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q:
            print(iface.ask(q).text)


if __name__ == "__main__":
    main()
