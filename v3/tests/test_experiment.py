"""Test the M3 experiment runner assembles the four configs into one comparison table."""
from src import experiment


def test_run_experiment_has_all_configs(week):
    res = experiment.run_experiment(week)
    assert set(res["config"]) == {"flow_only", "value_only", "early_fusion", "late_fusion", "supervised"}
    assert {"AP", "f1", "precision", "recall", "f1_pa", "recall_pa"} <= set(res.columns)
    assert (res["AP"].between(0, 1)).all()
