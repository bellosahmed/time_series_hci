"""Tests for v3 preprocessing — parse the real DAD pcaps + join per-packet attack labels.

These run against the real dataset in code/data/dad/ (skipped if it isn't present), and lock in
the ground-truth totals from the paper so a parsing regression can't slip through.
"""
import pandas as pd
import pytest

from src import preprocess

pytestmark = pytest.mark.skipif(
    not preprocess.DATA_DIR.exists(), reason="DAD dataset not downloaded (run tools/get_dad_dataset.sh)"
)


def test_load_day_monday_shape_and_publishes():
    df = preprocess.load_day("mon")
    assert len(df) == 14431                      # == the Monday labels.csv row count
    assert (df["label"] == 0).all()              # Monday has no attacks (paper schedule)
    # 16 sensors × 288 samples = 4608 MQTT PUBLISH messages carry a temperature
    assert df["temperature"].notna().sum() == 4608
    # temperatures parse to plausible data-center readings
    temps = df["temperature"].dropna()
    assert temps.between(-10, 60).all()
    assert {"frame", "src_ip", "clientid", "msgid", "topic", "temperature", "label"} <= set(df.columns)


def test_load_week_real_attack_totals(week):
    assert isinstance(week.index, pd.DatetimeIndex)
    assert int(week["label"].sum()) == 544                     # the real labelled-attack total
    per_day = week.groupby(week["day"])["label"].sum()          # key on the filename day, not UTC
    order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    assert per_day.reindex(order).fillna(0).astype(int).tolist() == [0, 0, 104, 56, 192, 192, 0]


def test_attacker_ip_present_only_where_attacks_are(week):
    attacker_rows = week[week["label"] == 1]
    assert (attacker_rows["src_ip"] == preprocess.ATTACKER_IP).mean() > 0.9  # attacks come from .41
