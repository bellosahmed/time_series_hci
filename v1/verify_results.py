"""Independent verification that v1's results are TRUE, not just plausible.

Run:  python3.12 verify_results.py

Each check recomputes a result a *different* way (pandas ground truth, known real
events, cross-method agreement) and asserts the system matches. The point of the
project is trust: numbers come from code, and here we prove that code is correct.
Exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import (  # noqa: E402
    analysis,
    anomaly,
    clustering,
    config,
    evaluation,
    forecasting,
    interface,
    llm,
    preprocess,
    query_coverage,
    rag_system,
    reasoning,
    retrieval_eval,
)
from src import query_engine as qe  # noqa: E402

DATA = config.DATA_PATH
S = "T (degC)"
checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


advisories: list[tuple[str, bool, str]] = []


def advise(name: str, ok: bool, detail: str = "") -> None:
    """Record an informational / non-deterministic check that is REPORTED but does not
    count toward the pass/fail total — keeping the headline number stable."""
    advisories.append((name, bool(ok), detail))


print("Loading + preprocessing…")
df = preprocess.align(preprocess.clean(preprocess.load_data(DATA)), freq="1h")
iface = interface.Interface(df)

# 1. shape
check("data shape sane (rows>60k, 14 sensors)", len(df) > 60_000 and df.shape[1] == 14,
      f"{df.shape}")

# 2-5. query engine == pandas ground truth (the truth layer is exact)
check("max == pandas", qe.get_max(df, S)["value"] == df[S].max())
check("min == pandas", qe.get_min(df, S)["value"] == df[S].min())
check("mean == pandas", np.isclose(qe.get_mean_std(df, S)["mean"], df[S].mean()))
check("correlation == pandas",
      np.isclose(qe.get_correlation(df, S, "rho (g/m**3)"), df[S].corr(df["rho (g/m**3)"])))

# NEW: data-quality additions recomputed a different way (ground-truth control)
raw = preprocess.load_data(DATA)
avail = analysis.data_availability(raw)
num_raw = raw.select_dtypes("number")
exp_sentinel = (num_raw == -9999.0).sum()
exp_missing = num_raw.isna().sum()
exp_real = len(raw) - exp_sentinel - exp_missing
check("availability n_real & n_sentinel == pandas recomputation (all sensors)",
      bool((avail["n_real"] == exp_real).all())
      and bool((avail["n_sentinel"] == exp_sentinel).all()),
      f"wv real={int(exp_real.get('wv (m/s)', -1))}")
check("distribution skew == pandas skew",
      np.isclose(analysis.distribution_stats(df).loc[S, "skew"], df[S].skew()))
multi = analysis.analyze_all_sensors(df)
check("multi-sensor table covers every numeric sensor",
      len(multi) == df.select_dtypes("number").shape[1], f"{len(multi)}")

# 6. scoped query via the interface == pandas slice
a13 = iface.ask("hottest in 2013")
check("2013-scoped max == pandas slice", a13.value["value"] == df.loc["2013", S].max())

# 7. threshold count == pandas
n_above = int((df[S] > 35).sum())
ath = iface.ask("how often above 35 degrees?")
check("threshold count == pandas", ath.value["count"] == n_above, f"{n_above}")

# 8. known-event corroboration: the all-time heat record is the July 2015 heatwave
mx = qe.get_max(df, S)
check("all-time max is a summer event (Jul 2015 heatwave)",
      mx["when"].year == 2015 and mx["when"].month in (6, 7, 8), f"{mx['when']}")
# cold extreme falls in winter (physical sanity)
mn = qe.get_min(df, S)
check("all-time min is a winter event", mn["when"].month in (12, 1, 2), f"{mn['when']}")

# 8b. reasoning is PHYSICALLY correct on the real extremes (validates the heuristic, not just
# that it produces a plausible-sounding story)
zmax = reasoning.sensor_zscores(df, mx["when"])
zmin = reasoning.sensor_zscores(df, mn["when"])
check("reasoning: temperature reads HIGH at the all-time heat extreme",
      zmax["T (degC)"] > 1.5, f"{zmax['T (degC)']:+.1f} sigma")
check("reasoning: temperature reads LOW at the all-time cold extreme",
      zmin["T (degC)"] < -1.5, f"{zmin['T (degC)']:+.1f} sigma")
check("reasoning verdict at the heat extreme is a real event (not a sensor glitch)",
      "real weather event" in reasoning.describe_moment(df, mx["when"]))

# 9. anomaly answer carries multi-sensor reasoning (the novelty) with a verdict
aa = iface.ask("any unusual readings in 2014?")
check("anomaly answer has reasoning", aa.inferred is not None and "σ" in (aa.inferred or ""))
check("reasoning gives a real-event / fault verdict",
      any(w in (aa.inferred or "") for w in ("real weather event", "sensor glitch", "moved")))

# 10. forecast is well-formed (lower<=mean<=upper, future dates)
fc = forecasting.forecast(df[S], steps=14)
check("forecast bounds ordered & future",
      (fc["lower"] <= fc["mean"]).all() and (fc["mean"] <= fc["upper"]).all()
      and fc.index[0] > df.index[-1])

# 10b. real-data recall: the z-score detector flags most DOCUMENTED events (not synthetic)
resid_full = analysis.decompose(df[S], period=24)["resid"].dropna()
ev = evaluation.events_recall(anomaly.zscore_anomalies(resid_full, threshold=4.0))
check("z-score catches >= 3 of 5 documented real weather events", ev["caught"] >= 3,
      f"caught {ev['caught']}/{ev['total']}, missed {ev['missed']}")

# 11. point-adjusted recall >= strict recall (it can only help, never hurt)
noisy, labels = evaluation.inject_spikes(df[S].iloc[:5000], n=40, magnitude=6.0)
dec = analysis.decompose(noisy, period=24)
pred = anomaly.zscore_anomalies(dec["resid"].dropna(), threshold=4.0)
strict = evaluation.score(pred, labels)
padj = evaluation.score_point_adjusted(pred, labels)
check("point-adjusted recall >= strict recall", padj.recall >= strict.recall,
      f"strict={strict.recall:.2f} adj={padj.recall:.2f}")

# 12. confusion matrix sums to N (every point is exactly one of TP/FP/FN/TN)
c = evaluation.confusion(pred, labels)
check("confusion sums to N", sum(c.values()) == len(labels), str(c))

# 13. forecast backtest is useful (MASE < 1.5; < 1 means it beats persistence). 20 folds (not 3) so
# a confidence interval across folds is meaningful, not just a pooled point estimate.
bt = forecasting.backtest(df[S], horizon=14, folds=20)
check("forecast backtest MASE finite & < 1.5", np.isfinite(bt["mase"]) and bt["mase"] < 1.5,
      f"MASE={bt['mase']:.2f} (per-fold mean={bt['mase_mean']:.2f}, "
      f"95%CI=[{bt['mase_ci_low']:.2f},{bt['mase_ci_high']:.2f}], n_folds={bt['folds']})")

# 14. multi-seed anomaly benchmark recall is solid
def det(noisy):
    return anomaly.zscore_anomalies(
        analysis.decompose(noisy, period=24)["resid"].dropna(), threshold=4.0)
bench = evaluation.benchmark(df[S].iloc[:4000], det, seeds=range(4), n=30, magnitude=6.0)
check("benchmark recall mean > 0.5", bench["recall"]["mean"] > 0.5, f"R={bench['recall']['mean']:.2f}")

# 15. statistical rigor (deterministic, seeded): bootstrap recall CI + paired McNemar.
# Reuses the labelled noisy series from check 11 (`noisy`, `labels`, `dec`, `pred`).
resid = dec[["resid"]].dropna()
ifpred = anomaly.isolation_forest_anomalies(resid, contamination=0.02)
ci = evaluation.bootstrap_ci(pred, labels, metric="recall", n_boot=300, seed=0)
check("bootstrap recall CI brackets the point estimate and is within [0,1]",
      ci["lower"] <= ci["point"] <= ci["upper"] and 0.0 <= ci["lower"] <= ci["upper"] <= 1.0,
      f"recall={ci['point']:.2f} 95%CI[{ci['lower']:.2f},{ci['upper']:.2f}]")
mc = evaluation.mcnemar_test(ifpred, pred, labels)
check("McNemar (IForest vs z-score) returns a valid p-value in [0,1]",
      0.0 <= mc["p_value"] <= 1.0, f"p={mc['p_value']:.3g} n_discordant={mc['n_discordant']}")

# 16. clustering day-types are meaningful (temperature-ordered) AND reachable via the interface.
feats_c = clustering.daily_features(df)
labels_c = clustering.kmeans_cluster(feats_c, k=4)
prof_c = clustering.cluster_profiles(feats_c, labels_c)
names_c = clustering.day_type_labels(prof_c)
k_summer, k_winter = pd.Timestamp(2015, 7, 4), pd.Timestamp(2015, 1, 15)
warmer = (k_summer in labels_c.index and k_winter in labels_c.index
          and prof_c.loc[int(labels_c.loc[k_summer]), "mean_T (degC)"]
          > prof_c.loc[int(labels_c.loc[k_winter]), "mean_T (degC)"])
check("clustering: a summer day is a warmer day-type than a winter day", warmer,
      f"{names_c[int(labels_c.loc[k_summer])]} vs {names_c[int(labels_c.loc[k_winter])]}")
adt = iface.ask("what kind of day was 4 July 2015?")
check("interface exposes day-type classification for a real day",
      adt.intent == "daytype" and isinstance(adt.value, dict) and "label" in adt.value,
      adt.value.get("label") if isinstance(adt.value, dict) else "n/a")

# 17. retrieval-quality evaluation (IR metrics on a hand-labelled query set). Evaluates the RAG
# *retriever* (not the truth layer) with Precision@k / Recall@k / MRR / MAP / NDCG, and checks
# the LSA retriever produces valid metrics on the same ground truth. See docs/archive/RETRIEVAL_EVALUATION.md.
gt = retrieval_eval.GROUND_TRUTH
tfidf_metrics = retrieval_eval.evaluate_retriever(rag_system.Retriever(), gt, k=3)
check("retriever IR metrics valid & TF-IDF MRR >= 0.8 on labelled queries",
      all(0.0 <= tfidf_metrics[m] <= 1.0 for m in ("p_at_k", "r_at_k", "mrr", "map", "ndcg"))
      and tfidf_metrics["mrr"] >= 0.8,
      f"MRR={tfidf_metrics['mrr']:.2f} MAP={tfidf_metrics['map']:.2f} R@3={tfidf_metrics['r_at_k']:.2f}")
lsa_metrics = retrieval_eval.evaluate_retriever(rag_system.LSARetriever(), gt, k=3)
check("LSA retriever runs & returns valid IR metrics in [0,1]",
      all(0.0 <= lsa_metrics[m] <= 1.0 for m in ("mrr", "map", "r_at_k"))
      and lsa_metrics["mrr"] >= 0.8,
      f"MRR={lsa_metrics['mrr']:.2f} MAP={lsa_metrics['map']:.2f}")

# 18. query-routing coverage (does the interface correctly route realistic questions to the right
# intent/sensor/scope?) — run against the REAL dataset, not a synthetic fixture, so this is the
# same benchmark the report cites, exercised on real data. See src/query_coverage.py.
cov = query_coverage.score_coverage(interface.Interface(df))
check("query-routing coverage: intent/sensor/scope all correct on >= 95% of labelled questions",
      cov["overall_pct"] >= 95.0,
      f"overall={cov['overall_pct']:.1f}% intent={cov['intent_pct']:.1f}% "
      f"sensors={cov['sensors_pct']:.1f}% scope={cov['scope_pct']:.1f}% (n={cov['n']})")

# 18b. HELD-OUT generalisation check: a separate, freshly-authored batch of questions the fixes
# above were never tuned on -- proves the fixes generalise, not just that the specific 4 gaps
# found in GROUND_TRUTH were closed. See query_coverage.HELD_OUT_GROUND_TRUTH.
cov_ho = query_coverage.score_coverage(interface.Interface(df), query_coverage.HELD_OUT_GROUND_TRUTH)
check("query-routing coverage generalises to a held-out question set (>= 90%)",
      cov_ho["overall_pct"] >= 90.0,
      f"overall={cov_ho['overall_pct']:.1f}% (n={cov_ho['n']})")

# 19. recommend intent, analog restriction: candidates for an explicitly-scoped analog search
# stay INSIDE the named month, recomputed a different way -- directly filtering the raw column
# by calendar month, not trusting the interface's own restriction logic. See docs/evaluation/
# 07-recommend-and-explainability.md.
a_apr = iface.ask("recommend days similar to 6 April 2013 in the month of April 2013")
apr_dates = [c["when"] for c in a_apr.value["candidates"]]
check("recommend analog restriction: every candidate independently confirmed inside April 2013",
      bool(apr_dates) and all(d.year == 2013 and d.month == 4 for d in apr_dates),
      f"{[d.date().isoformat() for d in apr_dates]}")

# 20. recommend intent, directional honesty threshold: the reported "not genuinely hot" threshold
# is independently recomputed from the raw column (75th percentile of daily means), not read back
# from the system's own stored value.
a_dec = iface.ask("recommend hot days in December")
raw_daily = df[S].resample("1D").mean().dropna()
expected_threshold = float(raw_daily.quantile(0.75))
check("recommend honesty threshold == independently-recomputed 75th percentile of daily means",
      not a_dec.value["meets_threshold"]
      and np.isclose(a_dec.value["threshold"], expected_threshold, atol=0.05),
      f"reported={a_dec.value['threshold']:.2f} recomputed={expected_threshold:.2f}")

# 21. correlation breakdown: R² and the seasonal split are recomputed independently from raw
# pandas groupby, not by calling the same query_engine function again (which would just prove
# the function agrees with itself).
bd = qe.correlation_breakdown(df, S, "rh (%)")
r_expected = df[S].corr(df["rh (%)"])
summer_mask = df.index.month.isin((6, 7, 8))
summer_expected = df[S][summer_mask].corr(df["rh (%)"][summer_mask])
check("correlation breakdown R² and summer split == independent pandas recomputation",
      np.isclose(bd["r_squared"], r_expected ** 2)
      and np.isclose(bd["by_season"]["summer"], summer_expected),
      f"R²={bd['r_squared']:.3f} summer_r={bd['by_season']['summer']:.3f}")

# 22. is_causal_query word-boundary fix: a bare "because" (no causal claim) must NOT hijack a
# real question into the correlation branch -- run against the real interface, not just the
# parsing function in isolation, so this proves the fix actually reaches user-facing behaviour.
a_because = iface.ask("I went out because the weather was nice, what was the average temperature in 2013?")
check("bare 'because' no longer hijacks an unrelated question into the correlation intent",
      a_because.intent == "mean", f"intent={a_because.intent!r}")

# 17. LLM faithfulness — ADVISORY + majority-of-3. The 3B model is stochastic, so we vote
# over 3 runs per question and REPORT the result without letting it flip the headline count
# (previously this check made the total flap between 19/20 and 20/20 run to run).
if llm.is_available() and llm.list_models():
    model = "llama3.2:3b" if "llama3.2:3b" in llm.list_models() else llm.list_models()[0]
    on = interface.Interface(df, use_llm=True, model=model)
    clean = 0
    for q in ["hottest in 2013?", "how often above 35?", "average humidity in 2015"]:
        votes = sum(
            not llm.faithfulness(
                " ".join(filter(None, [(a := on.ask(q)).computed, a.inferred])), a.text
            )["invented"]
            for _ in range(3)
        )
        clean += votes >= 2  # majority of 3 runs invented no numbers
    advise("LLM invented 0 numbers (majority-of-3)", clean == 3, f"{clean}/3 questions clean")

    # 17b. Faithfulness ABLATION — the gate's value is asserted above ("0 invented"); this
    # measures what the RAW (ungated) rate would have been, so that claim is actually comparable
    # to something, not just a bare number. See docs/evaluation/05-llm-faithfulness.md.
    abl = llm.faithfulness_ablation(
        interface.Interface(df, use_llm=True, model=model),
        ["hottest in 2013?", "how often above 35?", "average humidity in 2015",
         "was 2013 warmer than 2014?", "any anomalies in 2013?"],
        model=model, n_repeats=3)
    advise("faithfulness ablation: raw (ungated) vs gated invention rate", True,
           f"raw={abl['raw_invention_rate']:.0%} gated={abl['gated_invention_rate']:.0%} "
           f"(n={abl['n_total_runs']})")

    # 18c. Independent-LLM-judge cross-check for query-routing coverage — a partial, automatable
    # substitute for a second human rater (see docs/evaluation/06-query-routing-coverage.md, §3).
    judge = query_coverage.score_against_llm_judge(
        query_coverage.GROUND_TRUTH + query_coverage.HELD_OUT_GROUND_TRUTH, model=model)
    if judge["available"]:
        advise("query-routing ground truth vs. an independent LLM judge (informational, not "
               "a pass/fail claim)", True,
               f"intent={judge['intent_agreement_pct']:.1f}% sensors={judge['sensors_agreement_pct']:.1f}% "
               f"scope={judge['scope_agreement_pct']:.1f}% (n={judge['n_judged']})")

    # 17d. Adversarial faithfulness probe -- deliberately mismatched (question, facts) pairs, to
    # test whether the gate actually gets exercised, not just whether ordinary questions happen to
    # avoid tempting it. See docs/evaluation/05-llm-faithfulness.md.
    probe = llm.adversarial_faithfulness_probe(model=model, n_repeats=3)
    advise("adversarial faithfulness probe: invention rate under deliberately mismatched facts",
           True, f"invention_rate={probe['invention_rate']:.0%} (n={probe['n_total_runs']} runs "
           f"across {len(probe['per_case'])} adversarial cases)")
else:
    print("  (skipped LLM faithfulness advisory — Ollama not available)")

# ---- report ----
print("\n" + "=" * 64)
passed = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"  {'✓' if ok else '✗ FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
if advisories:
    print("-" * 64)
    print("  advisory (reported, not counted toward pass/fail):")
    for name, ok, detail in advisories:
        print(f"  {'✓' if ok else '•'}  {name}" + (f"   [{detail}]" if detail else ""))
print("=" * 64)
print(f"{passed}/{len(checks)} checks passed.")
sys.exit(0 if passed == len(checks) else 1)
