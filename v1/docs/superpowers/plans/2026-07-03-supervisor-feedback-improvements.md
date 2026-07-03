# v1 Supervisor-Feedback Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the v1 supervisor feedback by adding the missing analysis (skewness, data availability, class imbalance, missing-data justification), verifying the pipeline across all 14 sensors, and describing it — with pass results recorded in-repo and in Notion.

**Architecture:** Additive, pure functions in the existing `src/` modules (no refactors), each with unit tests. A multi-sensor summary function proves every sensor was exercised. A new `report/DATA_QUALITY.md` describes real outputs. `verify_results.py` gains ground-truth checks. Pass evidence is captured from a real run and logged to `report/DATA_QUALITY.md` + a Notion daily-log page.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, statsmodels, pytest. Notion MCP for logging.

## Global Constraints

- **Core principle:** code computes ALL numbers; prose only explains them. `DATA_QUALITY.md` must cite only numbers produced by the code in this plan.
- **No new dependencies**, no new models, no refactors beyond the additions here.
- **Sensor names are opaque strings** (e.g. `rho (g/m**3)` contains `**`) — never parse them.
- **All 14 sensors:** `p (mbar)`, `T (degC)`, `Tpot (K)`, `Tdew (degC)`, `rh (%)`, `VPmax (mbar)`, `VPact (mbar)`, `VPdef (mbar)`, `sh (g/kg)`, `H2OC (mmol/mol)`, `rho (g/m**3)`, `wv (m/s)`, `max. wv (m/s)`, `wd (deg)`.
- **Every new function's docstring must state what the returned object MEANS**, not just its type (feedback item: "function description — when I return the data it means what").
- **Commits:** no AI co-author trailer (project convention). The working tree already contains unrelated changes, so each commit stages ONLY that task's files by explicit path. Run `python -m pytest` with the Python 3.12 interpreter (`python3.12` or the `py312-tshci` env).
- **Test command:** `python3.12 -m pytest tests/ -q`. **Verify command:** `python3.12 verify_results.py`.

---

## File Structure

- `src/analysis.py` — add `distribution_stats`, `data_availability`, `analyze_all_sensors` (shape/availability/multi-sensor).
- `src/evaluation.py` — add `anomaly_imbalance` (class imbalance of detector labels).
- `src/preprocess.py` — add `compare_imputation` (justify interpolation vs mean-fill).
- `tests/test_analysis.py` — NEW, covers all of the above.
- `verify_results.py` — add ground-truth checks (17 → 20).
- `report/DATA_QUALITY.md` — NEW, the written description (feedback item C).
- `notebooks/01_data_exploration_and_visualization.ipynb`, `notebooks/02_methods_and_evaluation.ipynb` — minimal new cells.
- Notion: new `Daily Log — 3 July 2026` page under `Msc_project`.

---

## Task 1: `distribution_stats` — shape & skewness

**Files:**
- Modify: `src/analysis.py` (append function)
- Test: `tests/test_analysis.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `analysis.distribution_stats(df: pd.DataFrame) -> pd.DataFrame` — index = sensor name; columns = `mean, median, std, skew, kurtosis, min, max`. `skew` is Fisher-Pearson (0 symmetric, >0 right tail); `kurtosis` is excess kurtosis.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analysis.py`:

```python
"""Tests for analysis/evaluation/preprocess additions (supervisor-feedback work)."""
import numpy as np
import pandas as pd

from src import analysis, evaluation, preprocess


def _df(n=500, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "T (degC)": rng.normal(10, 5, n),
            "rh (%)": rng.normal(70, 10, n),
            "p (mbar)": rng.normal(1000, 3, n),
        },
        index=idx,
    )


def test_distribution_stats_columns_and_symmetry():
    stats = analysis.distribution_stats(_df())
    assert set(stats.columns) == {"mean", "median", "std", "skew", "kurtosis", "min", "max"}
    assert list(stats.index) == ["T (degC)", "rh (%)", "p (mbar)"]
    assert abs(stats.loc["T (degC)", "skew"]) < 0.5  # normal data ~symmetric


def test_distribution_stats_detects_right_skew():
    idx = pd.date_range("2024-01-01", periods=1000, freq="h")
    rng = np.random.default_rng(1)
    skewed = pd.DataFrame({"wv (m/s)": rng.exponential(2.0, 1000)}, index=idx)
    stats = analysis.distribution_stats(skewed)
    assert stats.loc["wv (m/s)", "skew"] > 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/test_analysis.py -q`
Expected: FAIL with `AttributeError: module 'src.analysis' has no attribute 'distribution_stats'`

- [ ] **Step 3: Implement `distribution_stats`**

Append to `src/analysis.py`:

```python
def distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor distribution shape: mean, median, std, skew, kurtosis, min, max.

    Returns a DataFrame indexed by sensor name (one row per numeric column). ``skew``
    is Fisher-Pearson skewness — 0 means symmetric, >0 a right/long-upper tail, <0 a
    left tail. ``kurtosis`` is *excess* kurtosis — 0 is normal-tailed, >0 heavy-tailed
    (more outliers). Together they describe the *shape* of each sensor's distribution,
    the basis for judging skewness and heavy tails.
    """
    num = df.select_dtypes("number")
    return pd.DataFrame(
        {
            "mean": num.mean(),
            "median": num.median(),
            "std": num.std(),
            "skew": num.skew(),
            "kurtosis": num.kurt(),
            "min": num.min(),
            "max": num.max(),
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/test_analysis.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "Add distribution_stats (per-sensor shape + skewness)"
```

---

## Task 2: `data_availability` — coverage report

**Files:**
- Modify: `src/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `analysis.data_availability(raw_df, sentinels=(-9999.0,)) -> pd.DataFrame` — index = sensor; columns `n_total, n_missing, n_sentinel, n_real, available_pct`. Computed on the RAW (pre-clean) frame.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
def test_data_availability_exact_counts():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan, -9999.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=idx
    )
    rep = analysis.data_availability(df)
    assert rep.loc["x", "n_total"] == 10
    assert rep.loc["x", "n_missing"] == 1
    assert rep.loc["x", "n_sentinel"] == 1
    assert rep.loc["x", "n_real"] == 8
    assert abs(rep.loc["x", "available_pct"] - 80.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.12 -m pytest tests/test_analysis.py::test_data_availability_exact_counts -q`
Expected: FAIL with `AttributeError: ... has no attribute 'data_availability'`

- [ ] **Step 3: Implement `data_availability`**

Append to `src/analysis.py`:

```python
def data_availability(
    raw_df: pd.DataFrame, sentinels: tuple[float, ...] = (-9999.0,)
) -> pd.DataFrame:
    """Per-sensor data-availability report, computed on the RAW (pre-clean) frame.

    Returns a DataFrame indexed by sensor with columns:
      ``n_total``      — number of rows
      ``n_missing``    — NaN count (genuinely absent readings)
      ``n_sentinel``   — count of sentinel error codes (e.g. -9999 on wind sensors)
      ``n_real``       — ``n_total - n_missing - n_sentinel`` (usable readings)
      ``available_pct``— ``100 * n_real / n_total``
    This quantifies how much of each sensor is real versus imputed *before* cleaning —
    the honest denominator for any downstream statistic.
    """
    num = raw_df.select_dtypes("number")
    n_total = len(num)
    n_missing = num.isna().sum().astype(int)
    n_sentinel = num.isin(list(sentinels)).sum().astype(int)
    n_real = (n_total - n_missing - n_sentinel).astype(int)
    return pd.DataFrame(
        {
            "n_total": n_total,
            "n_missing": n_missing,
            "n_sentinel": n_sentinel,
            "n_real": n_real,
            "available_pct": 100.0 * n_real / n_total,
        }
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.12 -m pytest tests/test_analysis.py::test_data_availability_exact_counts -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "Add data_availability (per-sensor coverage report)"
```

---

## Task 3: `analyze_all_sensors` — verify all 14 sensors (B)

**Files:**
- Modify: `src/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `distribution_stats` (Task 1), `top_correlations` (existing), `anomaly.zscore_anomalies` (existing).
- Produces: `analysis.analyze_all_sensors(df, threshold=3.0) -> pd.DataFrame` — index = sensor; columns `skew, kurtosis, zscore_anomalies, top_corr_sensor, top_corr_r`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
def test_analyze_all_sensors_one_row_per_sensor():
    df = _df()
    out = analysis.analyze_all_sensors(df)
    assert list(out.index) == list(df.columns)
    assert {"skew", "kurtosis", "zscore_anomalies", "top_corr_sensor", "top_corr_r"} <= set(
        out.columns
    )
    assert (out["zscore_anomalies"] >= 0).all()
    # top-correlation partner must be a different, real sensor
    for col in df.columns:
        assert out.loc[col, "top_corr_sensor"] in df.columns
        assert out.loc[col, "top_corr_sensor"] != col
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.12 -m pytest tests/test_analysis.py::test_analyze_all_sensors_one_row_per_sensor -q`
Expected: FAIL with `AttributeError: ... has no attribute 'analyze_all_sensors'`

- [ ] **Step 3: Implement `analyze_all_sensors`**

Append to `src/analysis.py`:

```python
def analyze_all_sensors(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Run the core analysis across EVERY sensor (not just temperature).

    Returns a DataFrame indexed by sensor with one row each:
      ``skew``, ``kurtosis``   — distribution shape (from ``distribution_stats``)
      ``zscore_anomalies``     — count of global z-score outliers at ``threshold``
      ``top_corr_sensor``      — the other sensor it correlates most strongly with (by |r|)
      ``top_corr_r``           — that correlation coefficient (signed)
    This is the artifact that proves humidity, pressure, wind and the rest were all
    verified the same way as temperature — the answer to "verify the other sensors".
    """
    from . import anomaly

    stats = distribution_stats(df)
    num = df.select_dtypes("number")
    rows = {}
    for col in num.columns:
        top = top_correlations(num, col, n=1)
        rows[col] = {
            "skew": float(stats.loc[col, "skew"]),
            "kurtosis": float(stats.loc[col, "kurtosis"]),
            "zscore_anomalies": int(
                anomaly.zscore_anomalies(num[col], threshold=threshold).sum()
            ),
            "top_corr_sensor": top.index[0] if len(top) else None,
            "top_corr_r": float(top.iloc[0]) if len(top) else float("nan"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.12 -m pytest tests/test_analysis.py::test_analyze_all_sensors_one_row_per_sensor -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "Add analyze_all_sensors (multi-sensor verification table)"
```

---

## Task 4: `anomaly_imbalance` — class imbalance

**Files:**
- Modify: `src/evaluation.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `evaluation.anomaly_imbalance(flags: pd.Series) -> dict` — keys `n_anomaly, n_normal, n_total, anomaly_pct`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
def test_anomaly_imbalance_counts():
    flags = pd.Series([True, False, False, False, True])
    imb = evaluation.anomaly_imbalance(flags)
    assert imb["n_anomaly"] == 2
    assert imb["n_normal"] == 3
    assert imb["n_total"] == 5
    assert abs(imb["anomaly_pct"] - 40.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.12 -m pytest tests/test_analysis.py::test_anomaly_imbalance_counts -q`
Expected: FAIL with `AttributeError: ... has no attribute 'anomaly_imbalance'`

- [ ] **Step 3: Implement `anomaly_imbalance`**

Append to `src/evaluation.py`:

```python
def anomaly_imbalance(flags: pd.Series) -> dict:
    """Class-imbalance summary for a boolean anomaly flag Series.

    Returns ``{n_anomaly, n_normal, n_total, anomaly_pct}``. Anomalies are a tiny
    minority of all points, so this quantifies the imbalance — essential context when
    reading precision/recall, since a detector can score high 'accuracy' simply by
    predicting everything normal.
    """
    f = flags.astype(bool)
    n_total = int(len(f))
    n_anomaly = int(f.sum())
    n_normal = n_total - n_anomaly
    anomaly_pct = 100.0 * n_anomaly / n_total if n_total else 0.0
    return {
        "n_anomaly": n_anomaly,
        "n_normal": n_normal,
        "n_total": n_total,
        "anomaly_pct": anomaly_pct,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.12 -m pytest tests/test_analysis.py::test_anomaly_imbalance_counts -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation.py tests/test_analysis.py
git commit -m "Add anomaly_imbalance (detector class-imbalance summary)"
```

---

## Task 5: `compare_imputation` — justify the missing-data method

**Files:**
- Modify: `src/preprocess.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `preprocess.compare_imputation(series, method="time", frac=0.1, seed=42) -> pd.DataFrame` — index `["interpolation", "mean_fill"]`, column `mae`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
def test_compare_imputation_interpolation_beats_mean_on_ramp():
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    ramp = pd.Series(np.linspace(0, 100, 200), index=idx)  # smooth trend
    res = preprocess.compare_imputation(ramp)
    assert np.isfinite(res.loc["interpolation", "mae"])
    assert res.loc["interpolation", "mae"] <= res.loc["mean_fill", "mae"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.12 -m pytest tests/test_analysis.py::test_compare_imputation_interpolation_beats_mean_on_ramp -q`
Expected: FAIL with `AttributeError: ... has no attribute 'compare_imputation'`

- [ ] **Step 3: Implement `compare_imputation`**

Append to `src/preprocess.py`:

```python
def compare_imputation(
    series: pd.Series, method: str = "time", frac: float = 0.1, seed: int = 42
) -> pd.DataFrame:
    """Justify the interpolation choice by holding out known points and refilling.

    Randomly masks ``frac`` of the finite points, then refills them two ways:
      - interpolation (``method``, time-aware by default) — follows the local trend
      - mean-fill — replaces every gap with the series mean (ignores the trend)
    Returns a DataFrame indexed ``["interpolation", "mean_fill"]`` with a ``mae`` column:
    the mean absolute error against the held-out truth. Lower is better; on a trending
    signal interpolation wins, which is *why* the pipeline interpolates (following the
    graph's trend) rather than flattening gaps to an average.
    """
    s = series.astype(float)
    finite_idx = s.dropna().index
    rng = np.random.default_rng(seed)
    n_mask = max(1, int(len(finite_idx) * frac))
    mask_idx = rng.choice(finite_idx.to_numpy(), size=n_mask, replace=False)
    held = s.copy()
    held.loc[mask_idx] = np.nan
    interp = held.interpolate(method=method, limit_direction="both")
    meanfill = held.fillna(s.mean())
    true_vals = s.loc[mask_idx]
    mae_interp = float((interp.loc[mask_idx] - true_vals).abs().mean())
    mae_mean = float((meanfill.loc[mask_idx] - true_vals).abs().mean())
    return pd.DataFrame({"mae": {"interpolation": mae_interp, "mean_fill": mae_mean}})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.12 -m pytest tests/test_analysis.py::test_compare_imputation_interpolation_beats_mean_on_ramp -q`
Expected: PASS

- [ ] **Step 5: Full test suite + commit**

Run: `python3.12 -m pytest tests/ -q`
Expected: all previous tests still pass, plus the 6 new ones (was 48 passed / 1 skipped → now 54 passed / 1 skipped).

```bash
git add src/preprocess.py tests/test_analysis.py
git commit -m "Add compare_imputation (interpolation vs mean-fill justification)"
```

---

## Task 6: `verify_results.py` — ground-truth checks (17 → 20)

**Files:**
- Modify: `verify_results.py`

**Interfaces:**
- Consumes: `analysis.data_availability`, `analysis.distribution_stats`, `analysis.analyze_all_sensors` (Tasks 1–3).
- Produces: 3 additional checks; total passes 20/20.

- [ ] **Step 1: Add the checks**

In `verify_results.py`, after the existing analysis/correlation checks (around the correlation check near line 44–49) and before the anomaly section, insert:

```python
# NEW: data-quality additions recomputed a different way (ground-truth control)
raw = preprocess.load_data(DATA)
avail = analysis.data_availability(raw)
check("availability n_total == len(raw) for all sensors",
      bool((avail["n_total"] == len(raw)).all()), f"{len(raw)} rows")
check("distribution skew == pandas skew",
      np.isclose(analysis.distribution_stats(df).loc[S, "skew"], df[S].skew()))
multi = analysis.analyze_all_sensors(df)
check("multi-sensor table covers all 14 sensors", len(multi) == 14, f"{len(multi)}")
```

- [ ] **Step 2: Run verify_results**

Run: `python3.12 verify_results.py`
Expected: final line `20/20 checks passed.` and exit code 0.

Verify exit code: `echo $?` → `0`.

- [ ] **Step 3: Commit**

```bash
git add verify_results.py
git commit -m "verify_results: add data-quality ground-truth checks (20/20)"
```

---

## Task 7: `report/DATA_QUALITY.md` — the written description (C)

**Files:**
- Create: `report/DATA_QUALITY.md`
- Create (temporary): a scratch script to capture real numbers, then delete it.

**Interfaces:**
- Consumes: all functions from Tasks 1–5, run on the real Jena data.

- [ ] **Step 1: Capture real numbers**

Create `scratch_dataquality.py` at the repo root:

```python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from src import preprocess, analysis, anomaly, evaluation
pd.set_option("display.width", 200, "display.max_columns", 20)

DATA = ROOT.parent / "data" / "jena_climate_2009_2016.csv"
raw = preprocess.load_data(DATA)
df = preprocess.align(preprocess.clean(raw), freq="1h")

print("### AVAILABILITY ###")
print(analysis.data_availability(raw).round(3).to_markdown())
print("\n### DISTRIBUTION ###")
print(analysis.distribution_stats(df).round(3).to_markdown())
print("\n### MULTI-SENSOR ###")
print(analysis.analyze_all_sensors(df).round(3).to_markdown())
print("\n### IMPUTATION (T degC) ###")
print(preprocess.compare_imputation(df["T (degC)"]).round(4).to_markdown())
print("\n### IMBALANCE (z-score on T) ###")
print(evaluation.anomaly_imbalance(anomaly.zscore_anomalies(df["T (degC)"])))
```

Run: `python3.12 scratch_dataquality.py > /tmp/dq.txt 2>&1; cat /tmp/dq.txt`
Expected: five markdown tables/blocks with real numbers.

- [ ] **Step 2: Write `report/DATA_QUALITY.md`**

Create `report/DATA_QUALITY.md` with these sections, pasting the **real** tables captured in Step 1 (do not invent numbers). Structure:

```markdown
# Data Quality & Multi-Sensor Verification (v1)

_Generated 3 July 2026 from `src/` functions on the Jena Climate hourly series. All numbers
are computed by code (`analysis.py`, `preprocess.py`, `evaluation.py`); this document only
describes them._

## 1. Data availability
<paste AVAILABILITY table>
Every sensor is >99% real; the wind columns carry the `-9999` sentinels (see `n_sentinel`),
which preprocessing converts to NaN before interpolation.

## 2. Distribution & skewness (shape of the graph)
<paste DISTRIBUTION table>
Temperature is near-symmetric (skew ~0); wind speed is right-skewed (long upper tail);
`kurtosis` flags heavy-tailed sensors that produce more outliers.

## 3. How missing data is handled (average vs distribution vs trend)
The pipeline uses **time-aware interpolation** (`preprocess.clean`), which follows the local
trend of the graph, then forward/back-fills any leading/trailing gaps. It does NOT replace
gaps with the column mean. Justification (held-out points, MAE):
<paste IMPUTATION table>
Interpolation's error is far below mean-fill's, because it tracks the trend rather than
flattening it to an average.

## 4. Standard deviation → outlier threshold
`anomaly.zscore_anomalies` standardises each point as z = (x − mean) / std (population std,
`ddof=0`) and flags |z| > threshold. Std is the unit of "how far is far"; the threshold sweep
(`anomaly.threshold_sensitivity`) shows counts collapsing as the threshold rises, and the knee
is where only real events survive.

## 5. Class imbalance
<paste IMBALANCE dict>
Anomalies are a tiny minority (~a few %), so accuracy is misleading — precision/recall and the
point-adjusted metrics (`evaluation.py`) are the honest scores.

## 6. Multi-sensor verification (humidity and all other sensors)
<paste MULTI-SENSOR table>
The same analysis runs across all 14 sensors, not just temperature: each has its shape,
z-score anomaly count, and strongest correlation partner.

## 7. Ground-truth control (the algorithm gives the right answer)
`verify_results.py` re-derives results a different way and asserts agreement: 20/20 checks pass,
including query-engine == pandas and the data-quality functions == pandas. The query engine is
the exact truth layer; the LLM only explains its numbers.
```

- [ ] **Step 3: Delete the scratch script**

```bash
rm scratch_dataquality.py
```

- [ ] **Step 4: Commit**

```bash
git add report/DATA_QUALITY.md
git commit -m "Add DATA_QUALITY report (availability, skewness, imbalance, multi-sensor)"
```

---

## Task 8: Notebook cells (minimal)

**Files:**
- Modify: `notebooks/01_data_exploration_and_visualization.ipynb`
- Modify: `notebooks/02_methods_and_evaluation.ipynb`

Use the NotebookEdit tool. Keep additions tiny so **Restart & Run All** stays safe.

- [ ] **Step 1: nb 01 — availability + skewness**

Add a markdown cell "## Data availability & distribution shape" and a code cell:

```python
from src import analysis
display(analysis.data_availability(raw_df))          # raw_df = pre-clean frame already in the nb
display(analysis.distribution_stats(df))             # df = cleaned hourly frame already in the nb
```

(If the notebook names the raw/clean frames differently, use those existing names — do not reload.)

- [ ] **Step 2: nb 02 — multi-sensor table + std/threshold note**

Add a markdown cell "## Multi-sensor verification (all 14 sensors)" and a code cell:

```python
from src import analysis
display(analysis.analyze_all_sensors(df))
```

Add a short markdown cell describing the std→threshold link (mirror `DATA_QUALITY.md` §4).

- [ ] **Step 3: Restart & Run All each notebook (py312-tshci kernel); confirm no errors, then commit**

```bash
git add notebooks/01_data_exploration_and_visualization.ipynb notebooks/02_methods_and_evaluation.ipynb
git commit -m "Notebooks: add availability, skewness, multi-sensor cells"
```

---

## Task 9: Record pass results (repo + Notion) & verify

**Files:**
- Modify: `report/DATA_QUALITY.md` (append verification section)
- Notion: create `Daily Log — 3 July 2026` under `Msc_project` (id `38639cc9-db44-81b3-a6bb-f1b0ea5c77cb`).

**Interfaces:**
- Consumes: real output of `pytest` and `verify_results.py`.

- [ ] **Step 1: Capture the real pass output**

Run and keep the exact output:
```bash
python3.12 -m pytest tests/ -q 2>&1 | tail -3
python3.12 verify_results.py 2>&1 | tail -3
```
Expected: pytest `54 passed, 1 skipped` (or the true count); verify `20/20 checks passed.`

- [ ] **Step 2: Append a "Verification results" section to `report/DATA_QUALITY.md`**

Paste the REAL captured lines (no paraphrase):

```markdown
## 8. Verification results (3 July 2026)

- `python3.12 -m pytest tests/ -q` → `<paste real summary line>`
- `python3.12 verify_results.py` → `<paste real final line>`

Feedback status: **A** (skewness/availability/imbalance) ✅ · **B** (all 14 sensors verified) ✅ ·
**C** (documented in this file + notebooks) ✅.
```

- [ ] **Step 3: Create the Notion daily-log page**

Using the Notion MCP `notion-create-pages` tool, create a page with parent =
`38639cc9-db44-81b3-a6bb-f1b0ea5c77cb`, title `Daily Log — 3 July 2026`, body containing the same
pass summary and the A/B/C status. Cross-check the numbers match `DATA_QUALITY.md` exactly.

- [ ] **Step 4: Commit the repo change**

```bash
git add report/DATA_QUALITY.md
git commit -m "DATA_QUALITY: record pytest + verify_results pass evidence"
```

- [ ] **Step 5: Final verification (evidence before claiming done)**

Re-run both commands one last time and confirm the numbers in the repo doc and the Notion page
match the live output. Only then report completion, quoting the real numbers.

---

## Self-Review (completed)

- **Spec coverage:** availability (T2/T7), correlation (T3), shape/skewness (T1), missing-data method (T5/T7), std→outlier (T6 check + T7 §4), imbalance (T4/T7), multi-sensor/humidity (T3/T8), function-return docs (docstrings in T1–T5 + global constraint), ground-truth control (T6/T7 §7), results logged repo+Notion+verified (T9). Cloud/speed intentionally excluded. ✅
- **Placeholder scan:** all code blocks complete; the only `<paste …>` markers are real-number captures the executor fills from live runs (by design — numbers must not be invented). ✅
- **Type consistency:** `distribution_stats`/`data_availability`/`analyze_all_sensors` return DataFrames; `anomaly_imbalance` returns dict; `compare_imputation` returns DataFrame indexed `interpolation`/`mean_fill`. Names consistent across tasks, tests, verify, and report. ✅
