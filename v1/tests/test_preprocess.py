"""Tests for the preprocessing layer (`src/preprocess.py`) — load, clean, align, impute."""

import numpy as np
import pandas as pd
import pytest

from src import preprocess


def _write_jena_csv(path, days: int = 6):
    idx = pd.date_range("2020-01-01", periods=days * 24 * 6, freq="10min")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Date Time": idx.strftime("%d.%m.%Y %H:%M:%S"),
        "T (degC)": 10 + rng.normal(0, 1, len(idx)),
        "wv (m/s)": rng.uniform(0, 5, len(idx)),
    })
    df.loc[7, "wv (m/s)"] = -9999.0                       # sentinel error code
    return df.to_csv(path, index=False) or path


def test_load_data_sorts_and_indexes(tmp_path):
    csv = _write_jena_csv(tmp_path / "j.csv")
    raw = preprocess.load_data(str(csv))
    assert isinstance(raw.index, pd.DatetimeIndex)
    assert raw.index.is_monotonic_increasing


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        preprocess.load_data("/no/such/dataset.csv")


def test_load_data_missing_time_col_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(KeyError):
        preprocess.load_data(str(p))


def test_clean_removes_sentinels_and_fills(tmp_path):
    raw = preprocess.load_data(str(_write_jena_csv(tmp_path / "j.csv")))
    cleaned = preprocess.clean(raw)
    assert (cleaned["wv (m/s)"] > -9000).all()            # -9999 replaced, not left in
    assert cleaned.isna().sum().sum() == 0                 # gaps filled


def test_align_resamples_to_hourly(tmp_path):
    cleaned = preprocess.clean(preprocess.load_data(str(_write_jena_csv(tmp_path / "j.csv"))))
    hourly = preprocess.align(cleaned, freq="1h")
    assert preprocess.infer_interval(hourly) == pd.Timedelta("1h")
    assert len(hourly) < len(cleaned)                      # downsampled from 10-min


def test_missing_report_only_lists_columns_with_gaps():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]})
    mr = preprocess.missing_report(df)
    assert mr["a"] == 1 and "b" not in mr.index


def test_compare_imputation_prefers_interpolation_on_a_trend():
    idx = pd.date_range("2020-01-01", periods=500, freq="h")
    s = pd.Series(np.arange(500, dtype=float), index=idx)  # strong linear trend
    res = preprocess.compare_imputation(s, frac=0.1, seed=0)
    assert res.loc["interpolation", "mae"] < res.loc["mean_fill", "mae"]


def test_compare_imputation_requires_datetime_index_for_time_method():
    with pytest.raises(TypeError):
        preprocess.compare_imputation(pd.Series([1.0, 2.0, 3.0]), method="time")
