"""Tests for v3 query engine — the exact, deterministic 'truth layer'. Unit tests use a tiny
hand-built frame; the real-week checks assert both the known ground truth (544 attacks, the paper's
per-day schedule) and equality with an independent pandas computation."""
import numpy as np
import pandas as pd

from src import query_engine as qe


def _df():
    idx = pd.to_datetime(["2019-10-23 02:00:00", "2019-10-23 02:00:01", "2019-10-21 00:00:00"])
    return pd.DataFrame({
        "day": ["wed", "wed", "mon"],
        "label": [1, 1, 0],
        "attack_type": ["modification", "modification", "none"],
        "src_ip": ["10.6.56.41", "10.6.56.41", "10.6.56.36"],
        "clientid": [131.0, 131.0, 152.0],
        "temperature": [np.nan, np.nan, 15.5],
    }, index=idx)


def test_count_attacks_total_and_scoped():
    df = _df()
    assert qe.count_attacks(df) == 2
    assert qe.count_attacks(df, day="wed") == 2
    assert qe.count_attacks(df, day="mon") == 0
    assert qe.count_attacks(df, attack_type="modification") == 2


def test_attack_breakdown_and_attacker():
    df = _df()
    assert qe.attack_breakdown(df) == {"modification": 2}
    assert qe.attacker_ips(df) == ["10.6.56.41"]
    assert qe.total_messages(df) == 3


def test_temperature_stat():
    df = _df()
    assert qe.temperature_stat(df, "mean") == 15.5
    assert qe.temperature_stat(df, "max", sensor=152.0) == 15.5
    assert np.isnan(qe.temperature_stat(df, "mean", day="wed"))   # no readings in that scope


# ---- real week: known ground truth + equality with independent pandas ----

def test_real_attack_counts_match_paper(week):
    assert qe.count_attacks(week) == 544
    assert qe.attacks_by_day(week) == {"mon": 0, "tue": 0, "wed": 104, "thu": 56,
                                       "fri": 192, "sat": 192, "sun": 0}
    assert sum(qe.attack_breakdown(week).values()) == 544
    assert qe.attacker_ips(week) == ["10.6.56.41"]


def test_real_stats_equal_pandas(week):
    assert qe.total_messages(week) == 101583
    for stat in ("max", "min", "mean"):
        expected = float(getattr(week["temperature"].dropna(), stat)())
        assert qe.temperature_stat(week, stat) == expected            # == pandas, exactly
