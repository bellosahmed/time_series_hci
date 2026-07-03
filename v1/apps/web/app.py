"""Web UI for the time-series HCI interface.

A small Flask server that reuses the existing `interface` / `llm` modules and serves
an interactive chat page on http://localhost:3030 (or the next free port).

Run:
    /usr/bin/python3.12 v1/apps/web/app.py
Then open the printed http://localhost:<port> in a browser.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent              # .../v1/apps/web
V1 = HERE.parent.parent                             # .../v1
sys.path.insert(0, str(V1))
DATA = V1.parent / "data" / "jena_climate_2009_2016.csv"

from src import preprocess, interface, llm          # noqa: E402

app = Flask(__name__, static_folder=str(HERE), static_url_path="")

# Load + preprocess once at startup.
print("Loading dataset...")
DF = preprocess.align(preprocess.clean(preprocess.load_data(DATA)), freq="1h")
SENSORS = list(DF.columns)
print(f"Ready: {DF.shape[0]:,} rows, {DF.shape[1]} sensors.")

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


@app.route("/")
def index():
    return send_from_directory(str(HERE), "index.html")


@app.route("/api/models")
def models():
    available = llm.is_available()
    return jsonify({"available": available, "models": llm.list_models() if available else []})


@app.route("/api/suggestions")
def suggestions():
    return jsonify({"suggestions": SUGGESTIONS, "sensors": SENSORS})


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True) or {}
    question = (body.get("question") or "").strip()
    use_llm = bool(body.get("use_llm"))
    model = body.get("model") or "llama3.2:3b"
    if not question:
        return jsonify({"error": "empty question"}), 400
    try:
        iface = interface.Interface(DF, use_llm=use_llm, model=model)
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
            "used_llm": use_llm and llm.is_available(),
        })
    except Exception as exc:  # keep the UI alive on any error
        return jsonify({"error": str(exc)}), 500


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


if __name__ == "__main__":
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
    app.run(host="127.0.0.1", port=port, debug=False)
