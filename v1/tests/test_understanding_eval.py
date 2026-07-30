"""The malformed-query scorer counts how many messy queries an interface answers correctly. Uses a
stub interface so it runs without Ollama."""

from src import understanding_eval as ue
from src.query_coverage import Case


class _StubAnswer:
    def __init__(self, intent, sensors, value):
        self.intent, self.sensors, self.value = intent, sensors, value


class _StubIface:
    def __init__(self, mapping):
        self.mapping = mapping
        self.history = []

    def ask(self, q):
        intent, sensors, value = self.mapping.get(q, (None, [], None))
        return _StubAnswer(intent, sensors, value)


def test_score_counts_correct_and_declined():
    cases = [Case("q1", "max", frozenset({"T (degC)"}), True),
             Case("q2", "mean", frozenset({"rh (%)"}), False)]
    iface = _StubIface({"q1": ("max", ["T (degC)"], {"value": 1.0}),
                        "q2": (None, [], None)})
    out = ue.score(iface, cases)
    assert out["n"] == 2 and out["correct"] == 1 and out["declined"] == 1
