"""Tests for the app layer (Flask web routes + CLI helpers).

The web app is exercised through a `create_app(df)` factory so routes can be tested
against a tiny in-memory DataFrame instead of loading the full 70k-row dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def test_detectors_route_lists_choices():
    r = _client().get("/api/detectors")
    assert r.status_code == 200
    d = r.get_json()
    assert d["default"] == "portfolio"
    assert "portfolio" in d["detectors"] and "isolation_forest" in d["detectors"]


def test_ask_route_honours_detector_choice():
    r = _client().post("/api/ask",
                       json={"question": "any unusual readings?", "detector": "isolation_forest"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["intent"] == "anomaly"
    assert d["detector"] == "isolation_forest"       # the choice is applied and echoed back


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


def test_chart_js_is_served():
    r = _client().get("/chart.js")           # index.html depends on it
    assert r.status_code == 200
    assert "function legendHtml" in r.get_data(as_text=True)


def test_index_page_loads_react_app_and_render_module():
    r = _client().get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'src="chart.js"' in html          # the page loads the extracted render module
    assert 'src="app.js"' in html            # ...and the React app that drives the UI
    assert 'vendor/react.production.min.js' in html   # React is vendored locally (offline, no build)
    assert 'focus-visible' in html                    # visible keyboard focus ring (CSS)


def test_react_app_has_accessible_markup():
    # The UI is rendered by React (app.js), so the screen-reader + keyboard affordances live there.
    r = _client().get("/app.js")
    assert r.status_code == 200
    js = r.get_data(as_text=True)
    assert 'role="log"' in js and 'aria-live="polite"' in js        # live conversation region
    assert 'aria-label="Ask a question in plain English"' in js     # labelled input


def test_react_and_htm_vendored_and_served():
    client = _client()
    for asset in ("/app.js", "/vendor/react.production.min.js",
                  "/vendor/react-dom.production.min.js", "/vendor/htm.umd.js"):
        assert client.get(asset).status_code == 200, asset   # all served locally, no CDN


def test_cli_pick_model_prefers_larger_pulled_model(monkeypatch):
    import chat

    from src import llm

    # the CLI now delegates to llm.pick_model: prefer the largest pulled model, else the default
    monkeypatch.setattr(llm, "list_models", lambda: ["llama3.2:3b", "other"])
    assert chat.pick_model(None) == "llama3.2:3b"
    monkeypatch.setattr(llm, "list_models", lambda: ["llama3.1:8b", "llama3.2:3b"])
    assert chat.pick_model(None) == "llama3.1:8b"    # larger model auto-preferred
    monkeypatch.setattr(llm, "list_models", lambda: [])
    assert chat.pick_model(None) == llm.DEFAULT_MODEL  # fallback when none pulled
    assert chat.pick_model("custom") == "custom"      # explicit request always wins
