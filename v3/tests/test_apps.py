"""Tests for the app layer (Flask web routes + CLI helper), exercised against a tiny in-memory
MESSAGES frame so they don't need to parse the pcaps."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3 / "apps" / "web"))
sys.path.insert(0, str(V3 / "apps"))


def _small_df():
    idx = pd.to_datetime(["2019-10-23 02:00:00", "2019-10-23 02:00:01", "2019-10-21 00:00:00"])
    return pd.DataFrame({
        "day": ["wed", "wed", "mon"], "label": [1, 1, 0],
        "attack_type": ["modification", "modification", "none"],
        "src_ip": ["10.6.56.41", "10.6.56.41", "10.6.56.36"],
        "clientid": [131.0, 131.0, 152.0], "temperature": [np.nan, np.nan, 15.5],
    }, index=idx)


def _client():
    import app as webapp
    application = webapp.create_app(_small_df())
    application.testing = True
    return application.test_client()


def test_ask_route_returns_computed_answer():
    r = _client().post("/api/ask", json={"question": "how many attacks were there?"})
    assert r.status_code == 200
    assert "2" in r.get_json()["computed"]           # 2 attacks in the tiny frame


def test_ask_route_rejects_empty():
    assert _client().post("/api/ask", json={"question": "  "}).status_code == 400


def test_examples_route_lists_questions():
    d = _client().get("/api/examples").get_json()
    assert d["examples"] and isinstance(d["examples"], list)


def test_index_page_served():
    r = _client().get("/")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "DAD" in html
    assert "Explain with AI" in html and "drawChart" in html    # AI toggle + chart renderer present


def test_ask_route_includes_a_chart():
    d = _client().post("/api/ask", json={"question": "how many attacks were there?"}).get_json()
    assert d["chart"] and d["chart"]["kind"] == "bar"


def test_models_route_reports_availability():
    d = _client().get("/api/models").get_json()
    assert set(d) >= {"available", "models"} and isinstance(d["models"], list)


def test_cli_format_answer_includes_labels():
    import chat

    from src.interface import Answer
    a = Answer("q", "count", "There were 2 attacks.", 2, "There were 2 attacks.",
               "background note", None, False)
    out = chat.format_answer(a)
    assert "2 attacks" in out and "background note" in out
