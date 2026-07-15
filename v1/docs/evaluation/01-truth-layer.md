# 1. The truth layer — proven, not benchmarked

## What's being evaluated

`src/query_engine.py` — the functions that compute every exact number the interface ever shows
a user (`get_max`, `get_min`, `get_mean_std`, `get_correlation`, `count_missing`,
`threshold_events`, ...). This is the one component in v1 where "evaluation" isn't really the
right word: a mean is either equal to pandas's mean or it's a bug. There's no meaningful
percentage score for "how close to correct" arithmetic is.

## Method

Independently recompute the same number a *different* way — plain pandas, called directly, not
through any of v1's own code — and assert exact equality (`rtol/atol` effectively 0 for these
operations; floating-point equality holds because both paths use the same underlying numpy
reduction).

```python
# from verify_results.py
check("max == pandas", iface.ask("max temperature").value.value == df[S].max())
check("mean == pandas", abs(qe.get_mean_std(df, S)["mean"] - df[S].mean()) < 1e-9)
check("correlation == pandas", abs(qe.get_correlation(df, S, "rho (g/m**3)") - df[S].corr(df["rho (g/m**3)"])) < 1e-9)
```

This is deliberately the strongest and simplest evaluation method in the project: no sampling,
no confidence interval, no benchmark corpus — just "compute it twice, independently, and check
they agree." That's appropriate because this is the one place in the pipeline where the *entire
point* is that the number must be provably correct, since every other trust mechanism in the
system (the LLM's faithfulness gate, the explainability trace) is built on the assumption that
the underlying computed fact is right.

## Real evidence (reproduced 15 July 2026)

```bash
python3.12 verify_results.py
```

```
✓  data shape sane (rows>60k, 14 sensors)
✓  max == pandas
✓  min == pandas
✓  mean == pandas
✓  correlation == pandas
✓  availability n_real & n_sentinel == pandas recomputation (all sensors)
✓  distribution skew == pandas skew
✓  multi-sensor table covers every numeric sensor
✓  2013-scoped max == pandas slice
✓  threshold count == pandas
```

Plus the unit test suite exercises the same functions directly:

```bash
python3.12 -m pytest tests/test_query_engine.py -v
```

Every one of these passed on the real 70,129-row Jena Climate dataset (2009–2016, after
cleaning/alignment), not a synthetic sample — so this isn't just "the formula is right," it's
"the formula is right on the actual data this project reports numbers from."

## Honest caveats

- This method only proves *arithmetic* correctness — it says nothing about whether the
  *question* was routed to the right computation in the first place. That's a separate concern,
  covered in [`06-query-routing-coverage.md`](06-query-routing-coverage.md).
- Floating-point equality checks use a small tolerance (`1e-9`) rather than bit-for-bit equality,
  since the two code paths (v1's own aggregation vs. raw pandas) can differ in summation order at
  the level of the last floating-point bit. This tolerance is far tighter than anything a human
  would call "rounding" — it exists purely to absorb floating-point non-associativity, not to
  hide a real discrepancy.
