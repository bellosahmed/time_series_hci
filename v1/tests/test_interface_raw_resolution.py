"""Regression test: extrema are computed at native resolution, not on the hourly-mean grid.

The analysis pipeline resamples the 10-minute Jena readings to an hourly mean
(``preprocess.align``). Averaging the six readings inside each clock hour smooths sub-hourly
peaks, so the maximum of the hourly means sits *below* the true reading: the "highest
temperature in July 2015" query returned 37.04 (hourly mean at 16:00) instead of the 37.28 a
spreadsheet reads straight off the raw column. The Interface now computes max/min/mean/summary on
a native-resolution ``raw`` frame when one is supplied, so those answers reconcile with the
source data. When no raw frame is given it falls back to the hourly frame (previous behaviour).
"""

import pandas as pd
import pytest

from src import interface, preprocess

SPIKE_TS = pd.Timestamp("2015-07-04 16:00")
SPIKE = 37.28
TROUGH_TS = pd.Timestamp("2015-07-19 04:50")
TROUGH = -4.35


def _raw_and_hourly():
    """A month of flat 10-min temperature with one sharp intra-hour peak and one trough."""
    idx = pd.date_range("2015-07-01 00:00", "2015-07-31 23:50", freq="10min")
    raw = pd.DataFrame({"T (degC)": 20.0}, index=idx)
    raw.loc[SPIKE_TS, "T (degC)"] = SPIKE      # a single 10-min spike inside the 16:00 hour
    raw.loc[TROUGH_TS, "T (degC)"] = TROUGH     # a single 10-min trough inside the 04:00 hour
    hourly = preprocess.align(raw, freq="1h")
    return raw, hourly


def test_hourly_only_max_is_smoothed_below_the_true_peak():
    """Without a raw frame the answer is the (lower) hourly mean — the original behaviour."""
    _, hourly = _raw_and_hourly()
    ans = interface.Interface(hourly).ask("highest temperature in July 2015")
    assert ans.value["value"] < SPIKE            # averaging pulled the peak down


def test_max_recovers_the_true_peak_from_the_raw_frame():
    raw, hourly = _raw_and_hourly()
    ans = interface.Interface(hourly, raw=raw).ask("highest temperature in July 2015")
    assert ans.value["value"] == pytest.approx(SPIKE)
    assert pd.Timestamp(ans.value["when"]) == SPIKE_TS


def test_min_recovers_the_true_trough_from_the_raw_frame():
    raw, hourly = _raw_and_hourly()
    ans = interface.Interface(hourly, raw=raw).ask("lowest temperature in July 2015")
    assert ans.value["value"] == pytest.approx(TROUGH)
    assert pd.Timestamp(ans.value["when"]) == TROUGH_TS


def test_summary_max_matches_the_raw_extreme():
    """A sensor summary reports the same exact peak as the dedicated max query."""
    raw, hourly = _raw_and_hourly()
    ans = interface.Interface(hourly, raw=raw).ask("summary of temperature in July 2015")
    assert ans.value["max"] == pytest.approx(SPIKE)
    assert ans.value["min"] == pytest.approx(TROUGH)


def test_no_raw_frame_is_unchanged_behaviour():
    """Passing no raw frame is identical to computing on the hourly frame directly."""
    _, hourly = _raw_and_hourly()
    with_default = interface.Interface(hourly).ask("highest temperature in July 2015")
    explicit = interface.Interface(hourly, raw=hourly).ask("highest temperature in July 2015")
    assert with_default.value["value"] == explicit.value["value"]
