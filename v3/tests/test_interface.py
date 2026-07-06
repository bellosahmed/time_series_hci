"""Tests for the NL interface — routes a plain-English question to the exact query engine and
returns a trust-labelled answer. Runs against the real week (shared fixture)."""
from src import interface
from src import query_engine as qe


def test_count_scoped_to_a_day(week):
    a = interface.Interface(week).ask("how many attacks on wednesday?")
    assert a.intent == "count" and "104" in a.computed


def test_total_attacks(week):
    a = interface.Interface(week).ask("how many attacks were there in total?")
    assert "544" in a.computed


def test_attacker_ip(week):
    a = interface.Interface(week).ask("which IP is the attacker?")
    assert a.intent == "attacker" and "10.6.56.41" in a.computed


def test_attack_type_breakdown(week):
    a = interface.Interface(week).ask("what types of attack are in the data?")
    assert a.intent == "breakdown" and "modification" in a.computed


def test_temperature_stat_matches_query_engine(week):
    a = interface.Interface(week).ask("what was the highest temperature on friday?")
    assert a.intent == "temperature"
    assert a.value == qe.temperature_stat(week, "max", day="fri")   # exact number from the truth layer


def test_unknown_question_is_not_faked(week):
    a = interface.Interface(week).ask("please write me a poem")
    assert a.intent is None and a.value is None                     # no invented answer


def test_answers_are_trust_labelled(week):
    a = interface.Interface(week).ask("how many attacks on thursday?")
    assert a.computed and a.used_llm is False                       # exact computed fact, no LLM


def test_llm_rewords_a_clean_answer(week, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "In total, 544 attack packets were flagged.")
    a = interface.Interface(week, use_llm=True).ask("how many attacks were there in total?")
    assert a.used_llm is True and "544" in a.text and "544" in a.computed


def test_llm_fabrication_is_rejected(week, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "There were about 999 attacks.")
    a = interface.Interface(week, use_llm=True).ask("how many attacks were there in total?")
    assert a.used_llm is False and "544" in a.text                  # gate rejected it → exact template


def test_llm_off_falls_back_to_template(week, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    a = interface.Interface(week, use_llm=True).ask("how many attacks were there in total?")
    assert a.used_llm is False and "544" in a.text


def test_count_question_attaches_a_bar_chart(week):
    a = interface.Interface(week).ask("how many attacks were there?")
    assert a.chart and a.chart["kind"] == "bar" and len(a.chart["x"]) == 7   # attacks per day


def test_temperature_question_attaches_a_line_chart(week):
    a = interface.Interface(week).ask("what was the highest temperature on friday?")
    assert a.chart and a.chart["kind"] == "line" and len(a.chart["y"]) > 1   # temperature over time


def test_unknown_question_has_no_chart(week):
    a = interface.Interface(week).ask("write me a poem")
    assert a.chart is None


def test_show_all_sensors_lists_them(week):
    a = interface.Interface(week).ask("show all the sensors in the dataset")
    assert a.intent == "sensors" and "16" in a.computed and a.chart["kind"] == "bar"


def test_show_anomalies_summarises_attacks(week):
    a = interface.Interface(week).ask("show me the anomalies")
    assert a.intent == "anomalies" and "544" in a.computed and a.chart["kind"] == "bar"
