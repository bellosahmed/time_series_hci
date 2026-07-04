"""Tests for the app layer (Flask web routes + CLI helpers).

The web app is exercised through a `create_app(df)` factory so routes can be tested
against a tiny in-memory DataFrame instead of loading the full 70k-row dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

V1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V1 / "apps" / "web"))
sys.path.insert(0, str(V1 / "apps"))


def _small_df(n=48):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"T (degC)": rng.normal(10, 5, n), "rh (%)": rng.normal(70, 10, n)}, index=idx
    )


def _client():
    import app as webapp

    application = webapp.create_app(_small_df())
    application.testing = True
    return application.test_client()


def test_ask_route_returns_exact_answer():
    r = _client().post("/api/ask", json={"question": "max temperature"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["intent"] == "max"
    assert data["computed"]  # exact computed fact is present


def test_ask_route_rejects_empty_question():
    r = _client().post("/api/ask", json={"question": "  "})
    assert r.status_code == 400


def test_suggestions_route_lists_sensors():
    r = _client().get("/api/suggestions")
    assert r.status_code == 200
    d = r.get_json()
    assert d["suggestions"]
    assert "T (degC)" in d["sensors"]


def test_models_route_shape():
    r = _client().get("/api/models")
    assert r.status_code == 200
    d = r.get_json()
    assert set(d) == {"available", "models"}
    assert isinstance(d["models"], list)


def test_ask_route_accepts_history_and_returns_resolved_key():
    r = _client().post("/api/ask", json={
        "question": "max temperature",
        "history": [{"question": "min temperature", "computed": "min ..."}],
    })
    assert r.status_code == 200
    assert "resolved" in r.get_json()   # follow-up transparency field is exposed


def test_index_page_serves_accessible_markup():
    r = _client().get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    # Screen-reader + keyboard affordances must be present (this is an HCI project).
    assert 'role="log"' in html and 'aria-live="polite"' in html   # live conversation region
    assert 'aria-label="Ask a question in plain English"' in html  # labelled input
    assert 'focus-visible' in html                                 # visible keyboard focus ring


def test_cli_pick_model_prefers_llama(monkeypatch):
    import chat
    from src import llm

    monkeypatch.setattr(llm, "list_models", lambda: ["llama3.2:3b", "other"])
    assert chat.pick_model(None) == "llama3.2:3b"
    monkeypatch.setattr(llm, "list_models", lambda: ["other"])
    assert chat.pick_model(None) == "other"
    monkeypatch.setattr(llm, "list_models", lambda: [])
    assert chat.pick_model(None) == "llama3.2:3b"  # PREFERRED fallback when none pulled
    assert chat.pick_model("custom") == "custom"  # explicit request always wins
