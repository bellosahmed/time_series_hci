"""
Independent verification of every headline v2 result.

Re-derives each claim from the raw data and checks it, printing PASS/FAIL. Run:
    cd code/v2 && python3.12 verify_results.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))

import numpy as np
import pandas as pd
from src import preprocess as pp, analysis as an, anomaly as ad
from src import forecasting as fc, query_engine as qe, reasoning as rz
from src import clustering as cl, deep_anomaly as da, rag_system as rag

DATA = V2.parent / "data" / "eu_trade_monthly.csv"
checks: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


df = pp.load_trade(DATA)
logged = pp.log_transform(df)
truth = ad.label_windows(df.index, [("2008-09-01", "2009-04-01"), ("2020-03-01", "2020-08-01")])

# 1. data integrity
r = pp.validate(df)
check("Data: 292x54, clean", df.shape == (292, 54) and r.is_clean, str(r))

# 2. query engine == pandas (truth layer)
e = qe.get_max(df, "FR", "EXP")
qe_ok = (e.value == df["FR_EXP"].max()
         and qe.get_mean_std(df, "IT", "EXP")[0] == df["IT_EXP"].mean()
         and abs(qe.get_correlation(df, "DE", "FR", "EXP")
                 - df["DE_EXP"].pct_change().corr(df["FR_EXP"].pct_change())) < 1e-12
         and qe.trade_balance(df, "FR", "2024-03-01")
             == df.loc["2024-03-01", "FR_EXP"] - df.loc["2024-03-01", "FR_IMP"])
check("Query engine == pandas (max/mean/corr/balance)", qe_ok)

# 3. ADF: level non-stationary, seasonal diff stationary
adf_lvl = an.adf_test(logged["FR_EXP"]).stationary
adf_diff = an.adf_test(logged["FR_EXP"].diff(12)).stationary
check("ADF: level non-stationary -> diff stationary", (not adf_lvl) and adf_diff)


# 4. detectors catch BOTH crises (event-level)
def detected(flags, w, tol="31D"):
    s = pd.Timestamp(w[0]) - pd.Timedelta(tol); ed = pd.Timestamp(w[1]) + pd.Timedelta(tol)
    f = flags.fillna(False); return bool(f.loc[(f.index >= s) & (f.index <= ed)].any())

z = ad.zscore_anomalies(logged["DE_EXP"], 3.0)
sn = ad.seasonal_naive_anomalies(logged["DE_EXP"], 3.0)
iso = ad.isoforest_anomalies(ad.systemic_features(logged), 0.05)
both08 = all(detected(f, ("2008-09-01", "2009-04-01")) for f in [z, sn, iso])
bothcov = all(detected(f, ("2020-03-01", "2020-08-01")) for f in [z, sn, iso])
check("All 3 detectors catch 2008 & COVID (event-level)", both08 and bothcov)

# 5. confusion matrix sums to N
s = ad.score(sn, truth)
check("Confusion sums to 292", s.tp + s.fp + s.fn + s.tn == 292,
      f"TP={s.tp} FP={s.fp} FN={s.fn} TN={s.tn}")

# 6. forecasting generalises (backtest)
fr = df["FR_EXP"] + df["FR_IMP"]
bt = fc.backtest(fr, horizon=24, log=True)
check("Forecast backtest MAPE < 8% (predicts, not memorises)", bt.mape < 8.0, f"MAPE={bt.mape:.1f}%")
btc = fc.backtest_from(fr, cutoff="2019-12-01", horizon=18, log=True)
check("COVID backtest breaches band (shock detected)", (btc.actual < btc.lower).sum() >= 1)

# 7. deep model loses to baseline (honest negative result)
ae = da.ae_anomalies(ad.systemic_features(logged), 0.05,
                     trained=da.train_autoencoder(ad.systemic_features(logged), epochs=60, seed=0))
f1_ae = ad.score(ae, truth).f1
f1_iso = ad.score(iso, truth).f1
check("Deep AE F1 <= Isolation Forest F1 (negative result)", f1_ae <= f1_iso,
      f"AE={f1_ae:.2f} IsoForest={f1_iso:.2f}")

# 8. clustering partitions all 27 countries
feats = cl.country_features(df)
lab = cl.cluster_countries(feats, k=4)
check("Clustering: 27 countries -> 4 groups", len(lab) == 27 and lab.nunique() == 4)

# 9. reasoning: COVID = shared shock
rep = rz.explain_month(rz.cross_section_z(logged, "_EXP"), "2020-04-01")
check("Reasoning: 2020-04 = shared shock", rep.kind == "shared shock")

# 10. RAG KB has no numbers (trust boundary)
kb = rag.knowledge_base()
check("RAG KB contains no euro amounts", "EUR" not in " ".join(t for _, t in kb))

# 11. BERT beats lexical/LSA on a no-overlap paraphrase
q = "factories shut and shipping stopped worldwide"
bert_top = [i for i, _ in rag.BertRetriever().fit(kb).search(q, 3)]
bm_top = [i for i, _ in rag.BM25().fit(kb).search(q, 3)]
check("BERT finds COVID on paraphrase (BM25 doesn't)",
      "event_covid" in bert_top and "event_covid" not in bm_top,
      f"BERT={bert_top[:2]} BM25={bm_top[:2]}")

# 12. LLM (if up) preserves the exact number
from src import interface as ui, llm
if llm.is_available():
    a = ui.ask(df, "What was Spain's worst monthly export drop?", use_llm=True)
    mag = a.computed.split("was ")[1].split("%")[0].lstrip("-").rstrip("0").rstrip(".")  # "35"
    ok = a.phrased is not None and mag in a.phrased and "2020" in a.phrased
    check("LLM phrasing preserves the exact number+date", ok, f"magnitude={mag}, year=2020")
else:
    check("LLM phrasing (skipped — Ollama off)", True)

# ---- report ----
print("\n" + "=" * 64)
print(" v2 RESULTS VERIFICATION")
print("=" * 64)
passed = 0
for name, ok, detail in checks:
    print(f" [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
    passed += ok
print("-" * 64)
print(f" {passed}/{len(checks)} checks passed")
print("=" * 64)
sys.exit(0 if passed == len(checks) else 1)
