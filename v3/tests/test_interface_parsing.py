"""Tests for the deterministic NL parsing (intent, day, attack type, stat) — no data needed."""
from src import interface_parsing as ip


def test_detect_intent_routes_common_questions():
    assert ip.detect_intent("how many attacks on wednesday?") == "count"
    assert ip.detect_intent("which IP is the attacker?") == "attacker"
    assert ip.detect_intent("what types of attack are there?") == "breakdown"
    assert ip.detect_intent("what was the highest temperature on friday?") == "temperature"
    assert ip.detect_intent("what was the busiest hour?") == "traffic"
    assert ip.detect_intent("when did the attacks happen?") == "when"
    assert ip.detect_intent("show all the sensors in the dataset") == "sensors"
    assert ip.detect_intent("show me the anomalies") == "anomalies"
    assert ip.detect_intent("what are the unusual readings?") == "anomalies"
    assert ip.detect_intent("tell me a joke") is None


def test_resolve_day():
    assert ip.resolve_day("attacks on wednesday") == "wed"
    assert ip.resolve_day("the friday report") == "fri"
    assert ip.resolve_day("no day named here") is None


def test_resolve_attack_type():
    assert ip.resolve_attack_type("how many modification attacks") == "modification"
    assert ip.resolve_attack_type("duplicate/replay messages") == "duplication"
    assert ip.resolve_attack_type("packet interception") == "interception"
    assert ip.resolve_attack_type("nothing relevant") is None


def test_resolve_stat():
    assert ip.resolve_stat("highest temperature") == "max"
    assert ip.resolve_stat("coldest reading") == "min"
    assert ip.resolve_stat("average temperature") == "mean"      # default
