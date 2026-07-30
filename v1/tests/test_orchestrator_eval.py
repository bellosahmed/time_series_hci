"""The multi-part scorer counts questions whose combined answer contains every expected fact. Uses a
stub so it runs without Ollama."""
from src import orchestrator_eval as oe


class _Stub:
    def run(self, q):
        class A:  # minimal duck-typed OrchestratedAnswer
            text = "max temperature 30.00; min humidity 20.00"
            computed = "max temperature 30.00; min humidity 20.00"
        return A()


def test_score_counts_complete_answers():
    cases = [("max temp and min humidity", ("30.00", "20.00")),
             ("something missing", ("999.99",))]
    out = oe.score(_Stub(), cases)
    assert out["n"] == 2 and out["complete"] == 1
