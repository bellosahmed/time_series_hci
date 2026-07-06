"""Web UI for the v3 DAD interface — a small Flask server over the same `interface` module.

Built by `create_app(df)`: pass a DataFrame (tests pass a tiny one) or omit it to parse + load the
full DAD week. Serves a single chat page and a `/api/ask` endpoint; every number is computed by the
query engine, the server only relays it.

Run:  python3.12 apps/web/app.py   →  open the printed http://localhost:<port>
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
V3 = HERE.parent.parent
sys.path.insert(0, str(V3))

from src import interface, llm, preprocess  # noqa: E402

EXAMPLES = [
    "Show me the anomalies",
    "Show all the sensors in the dataset",
    "How many attacks on Wednesday?",
    "Which IP is the attacker?",
    "What types of attack are in the data?",
    "When did the attacks happen?",
    "What was the highest temperature on Friday?",
    "What was the busiest hour?",
    "Give me a summary of Thursday",
]


def create_app(df: pd.DataFrame | None = None) -> Flask:
    if df is None:
        print("Loading DAD dataset…")
        df = preprocess.load_week()
        print(f"Ready: {len(df):,} packets, {int(df['label'].sum())} labelled attacks.")

    app = Flask(__name__, static_folder=str(HERE), static_url_path="")

    @app.route("/")
    def index():
        return send_from_directory(str(HERE), "index.html")

    @app.route("/api/examples")
    def examples():
        return jsonify({"examples": EXAMPLES})

    @app.route("/api/models")
    def models():
        available = llm.is_available()
        return jsonify({"available": available, "models": llm.list_models() if available else []})

    @app.route("/api/ask", methods=["POST"])
    def ask():
        body = request.get_json(force=True) or {}
        question = (body.get("question") or "").strip()
        if not question:
            return jsonify({"error": "empty question"}), 400
        iface = interface.Interface(df, use_llm=bool(body.get("use_llm")))
        ans = iface.ask(question)
        return jsonify({
            "question": ans.question, "intent": ans.intent, "text": ans.text,
            "computed": ans.computed, "context": ans.context, "inferred": ans.inferred,
            "used_llm": ans.used_llm and llm.is_available(), "chart": ans.chart,
        })

    return app


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main() -> None:
    app = create_app()
    want = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 3040))
    port = want if _free(want) else next((p for p in range(want + 1, want + 20) if _free(p)), want)
    print(f"\n  Open  http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
