"""Tests for the LLM phrasing path through `Interface.ask`: open-range date handling when the
LLM plan supplies only one bound, and the faithfulness gate (reject invented numbers, fall back
to the template, log the failure; keep the LLM's wording when it's faithful)."""

import pytest

from src import interface


def test_llm_plan_one_sided_date_stays_an_open_range(year_df, monkeypatch):
    # If the LLM plan supplies only ONE of start/end (e.g. "from June onward"), the range must
    # stay open-ended -- not collapse into a single-day window (both start and end pinned to the
    # one supplied date), which is what happened before this fix.
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: {
        "intent": None, "sensors": [], "start": "2016-06-01", "end": None,
        "period": None, "op": None, "value": None,
    })
    iface = interface.Interface(year_df, use_llm=True)
    ans = iface.ask("what was the mean temperature these days?")
    expected = year_df.loc["2016-06-01":].mean()["T (degC)"]
    assert ans.value["mean"] == pytest.approx(expected)
    # a single collapsed day would have a much smaller (and different) mean than the whole
    # June-to-year-end window -- assert it's NOT equal to just the first day's mean either
    single_day_mean = year_df.loc["2016-06-01":"2016-06-01"].mean()["T (degC)"]
    assert ans.value["mean"] != pytest.approx(single_day_mean)


def test_llm_error_falls_back_to_template_and_logs(df, monkeypatch, caplog):
    # A dead/erroring LLM must not fail silently (the G2 "silently dead LLM" class of bug):
    # the answer falls back to the template AND the failure is logged, not swallowed.
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: None)  # isolate the phrasing path

    def boom(*a, **k):
        raise RuntimeError("ollama exploded")

    monkeypatch.setattr(llm, "explain", boom)
    iface = interface.Interface(df, use_llm=True)
    with caplog.at_level("WARNING"):
        ans = iface.ask("max temperature")
    assert "Computed" in ans.text  # template fallback still produced a correct answer
    assert any("ollama exploded" in r.getMessage() for r in caplog.records)


def test_llm_inventing_a_number_is_rejected_for_template(df, monkeypatch, caplog):
    # If the LLM fabricates a figure absent from the computed facts, the interface must
    # reject its phrasing and fall back to the trustworthy template — numbers stay exact.
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: None)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "The max was 999.99 degrees.")
    iface = interface.Interface(df, use_llm=True)
    with caplog.at_level("WARNING"):
        ans = iface.ask("max temperature")
    assert "999.99" not in ans.text        # fabricated number never reaches the user
    assert "Computed" in ans.text          # template fallback used instead
    assert any("invented" in r.getMessage().lower() for r in caplog.records)


def test_llm_faithful_phrasing_is_kept(df, monkeypatch):
    # When the LLM only rewords the exact facts (no invented numbers), keep its phrasing.
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: None)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "Nicely worded, no figures here.")
    ans = interface.Interface(df, use_llm=True).ask("max temperature in 2024")
    # the LLM's faithful phrasing is retained (an appended assumption note may follow it, but the
    # explicit period here means none is added)
    assert ans.text == "Nicely worded, no figures here."
