"""Web UI for the time-series HCI interface.

A small Flask server that reuses the existing `interface` / `llm` modules and serves
an interactive chat page on http://localhost:3030 (or the next free port).

The Flask app is built by `create_app(df)`: pass a DataFrame to serve (tests pass a tiny
in-memory one), or omit it to load + preprocess the full Jena dataset. Loading only happens
inside the factory, so importing this module is cheap and the routes are unit-testable.

Run:
    /usr/bin/python3.12 v1/apps/web/app.py
Then open the printed http://localhost:<port> in a browser.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent              # .../v1/apps/web
V1 = HERE.parent.parent                             # .../v1
sys.path.insert(0, str(V1))
DATA = V1.parent / "data" / "jena_climate_2009_2016.csv"

from src import interface, llm, preprocess  # noqa: E402

SUGGESTIONS = [
    "What was the hottest temperature?",
    "What was the coldest it got in 2010?",
    "Tell me the weather on 26 January 2015",
    "Hottest in summer 2012",
    "Is temperature rising over the years?",
    "Was 2013 hotter than 2014?",
    "How often was it above 35 degrees?",
    "Are there any unusual readings in 2014?",
    "Average humidity in 2015",
    "How does temperature relate to air density?",
]


def load_default_df() -> pd.DataFrame:
    """Load + preprocess the full Jena dataset (used when no df is injected)."""
    print("Loading dataset...")
    df = preprocess.align(preprocess.clean(preprocess.load_data(DATA)), freq="1h")
    print(f"Ready: {df.shape[0]:,} rows, {df.shape[1]} sensors.")
    return df


def create_app(df: pd.DataFrame | None = None) -> Flask:
    """Build the Flask app serving `df` (or the full dataset if None).

    Data loading lives here, not at import time, so the routes can be tested against a
    small in-memory frame without reading the 70k-row CSV.
    """
    if df is None:
        df = load_default_df()
    sensors = list(df.columns)

    app = Flask(__name__, static_folder=str(HERE), static_url_path="")

    @app.route("/")
    def index():
        return send_from_directory(str(HERE), "index.html")

    @app.route("/api/models")
    def models():
        available = llm.is_available()
        return jsonify({"available": available, "models": llm.list_models() if available else []})

    @app.route("/api/suggestions")
    def suggestions():
        return jsonify({"suggestions": SUGGESTIONS, "sensors": sensors})

    @app.route("/api/ask", methods=["POST"])
    def ask():
        body = request.get_json(force=True) or {}
        question = (body.get("question") or "").strip()
        use_llm = bool(body.get("use_llm"))
        model = body.get("model") or "llama3.2:3b"
        history = body.get("history") or []   # recent turns from the client (server stays stateless)
        if not question:
            return jsonify({"error": "empty question"}), 400
        try:
            iface = interface.Interface(df, use_llm=use_llm, model=model)
            # Seed conversation memory so follow-ups ("what about 2014?") resolve against prior turns.
            iface.history = [h for h in history if isinstance(h, dict)][-6:]
            ans = iface.ask(question)
            return jsonify({
                "question": ans.question,
                "intent": ans.intent,
                "sensors": ans.sensors,
                "text": ans.text,
                "computed": ans.computed,
                "context": ans.context,
                "inferred": ans.inferred,
                "chart": ans.chart,
                "resolved": ans.resolved,
                "used_llm": use_llm and llm.is_available(),
            })
        except Exception as exc:  # keep the UI alive on any error
            return jsonify({"error": str(exc)}), 500

    return app


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main() -> None:
    app = create_app()
    # Port: CLI arg > PORT env > 3030 (avoids Open WebUI on 3000). Falls back to next free port.
    want = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 3030))
    port = want
    if not _free(port):
        for p in range(want + 1, want + 20):
            if _free(p):
                print(f"[note] port {want} is in use; using {p} instead.")
                port = p
                break
    print(f"\n  Open  http://localhost:{port}\n")
    # This is Flask's built-in server — fine for local single-user use (the project's scope).
    # For a real deployment, serve `create_app()` behind a production WSGI server, e.g.:
    #   waitress-serve --port 3030 'apps.web.app:create_app'   (or gunicorn)
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
