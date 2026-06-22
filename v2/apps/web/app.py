"""
v2 web app — an interactive browser chat over the EU trade data.

Ask questions in plain English; the query engine computes the exact number, hybrid RAG adds context,
and (optionally) the local LLM phrases it. Numbers always come from the engine, never the model.

Run:  python3.12 apps/web/app.py [port]      (defaults to 3000, falls back to a free port)
"""
from __future__ import annotations
import os
import socket
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
V2 = HERE.parent.parent
sys.path.insert(0, str(V2))

from src import preprocess as pp
from src import interface as ui
from src import rag_system as rag
from src import llm

DATA = V2.parent / "data" / "eu_trade_monthly.csv"
DF = pp.load_trade(DATA)
RAG = rag.HybridRetriever().fit()

app = Flask(__name__, static_folder=str(HERE))


@app.route("/")
def index():
    return send_from_directory(str(HERE), "index.html")


@app.route("/api/suggestions")
def suggestions():
    return jsonify(ui.SUGGESTIONS)


@app.route("/api/models")
def models():
    return jsonify({"available": llm.is_available(), "models": llm.list_models()})


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    use_llm = bool(body.get("use_llm"))
    model = body.get("model") or llm.DEFAULT_MODEL
    if not question:
        return jsonify({"error": "Please type a question."})

    ans = ui.ask(DF, question, use_llm=use_llm, model=model)
    # add hybrid-RAG context (text only — never numbers)
    context = []
    if not ans.error and ans.computed:
        context = [t for _, t in RAG.retrieve_texts(question + " " + ans.computed, k=2)]
    return jsonify({
        "intent": ans.intent,
        "computed": ans.computed,
        "inferred": ans.inferred,
        "phrased": ans.phrased,
        "context": context,
        "error": ans.error,
    })


def _free_port(start: int) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


if __name__ == "__main__":
    want = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 3000))
    port = _free_port(want)
    print(f"\n  v2 trade chat → http://localhost:{port}")
    print(f"  Ollama: {'on' if llm.is_available() else 'off (templates used)'}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
