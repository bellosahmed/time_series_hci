# Final Report — Verifiable NL Analytics across Three Domains (v1 · v2 · v3)

A cross-version synthesis of the project: **one architecture, three domains, three label regimes.** This
folder is the consolidated write-up of *what was built, what was found, which techniques were chosen (and
why over the alternatives), the limitations, and the future work.*

> **The single idea, held constant across all three versions:** **computation = deterministic code (the
> "truth layer"); the LLM only *explains*; an automated *faithfulness check* verifies it invented no
> number.** We call the pattern **Verifiable NL Analytics**. Everything below is an argument that this
> pattern is *domain-agnostic* and that *honest evaluation* — not model complexity — is what carries it.

## Contents
1. **[01_systems_and_data.md](01_systems_and_data.md)** — the three systems, their domains/data, and the shared architecture.
2. **[02_techniques_and_rationale.md](02_techniques_and_rationale.md)** — every major technique and *why we chose it over the alternative*.
3. **[03_results_and_findings.md](03_results_and_findings.md)** — consolidated results and the cross-cutting findings.
4. **[04_limitations_and_future_work.md](04_limitations_and_future_work.md)** — honest limitations and future improvements.

---

## Executive summary

Three independent systems were built on the **same five-layer architecture** (preprocess → analyse →
detect → reason → **query engine (truth)** → interface → **LLM + faithfulness gate**), on three
deliberately different domains:

- **v1 — Weather** (Jena climate, physical sensors): the *reference implementation*, richest method set,
  but only **synthetic** anomaly labels.
- **v2 — Trade** (EU international trade, economics): **real, datable** macro anomalies (2008, COVID,
  2022) — external validity for the detectors.
- **v3 — IoT network security** (DAD, MQTT captures): **real, per-packet** attack labels — the strongest
  evaluation, plus a two-channel + fusion study.

The consistent thread across all three is an **honesty-first evaluation** (point-adjusted metrics,
bootstrap CIs, `verify_results == pandas`, LLM faithfulness), and a repeated, defensible finding:
**simple, robust methods match or beat complex ones on these datasets**, and *good evaluation design*
(real labels, per-type breakdowns, blocked CV) matters more than model sophistication.

---

## Master comparison table

| Aspect | **v1 — Weather** | **v2 — Trade** | **v3 — IoT network security** |
| --- | --- | --- | --- |
| **Domain** | Physical temperature sensors | Macro-economics | Data-center IoT / MQTT network |
| **Dataset** | Jena Climate 2009–2016 | EU monthly trade 2002–2026 | DAD (Vigoya 2020), 7-day MQTT captures |
| **Shape** | 70,129 hourly rows × 14 sensors | 292 months × 54 country-flows | ~101,583 packets, 16 sensors, 5-min cadence |
| **Anomaly labels** | **Synthetic** (injected spikes) | **Real, coarse** (crisis *months*) | **Real, fine** (544 labelled attack *packets*) |
| **Anomaly nature** | statistical outliers in a physical signal | macro shocks (shared across countries) | network attacks (dup / intercept / modify) |
| **Detectors** | z-score(STL), ARIMA, IsolationForest, LOF, OCSVM, Matrix Profile, CNN-AE, LSTM-AE, LSTM-forecaster | STL z-score, seasonal-naive, IsolationForest, multivariate AE | value-channel z, flow IsolationForest, early/late fusion, supervised combiner |
| **Best detector result** | precision **0.85–1.00** on a *controlled* signal; point-adjusted F1 **0.71** on synthetic | IsolationForest **F1 0.36** (strict); ensemble point-adjusted **F1 0.90**; catches 2008 & COVID | value-only **AP 0.183**; fusion **point-adjusted recall 1.00**; CIs overlap → no config sig. best |
| **Forecasting** | ARIMA — MAE 3.55 °C, **MASE 0.83** (beats persistence) | SARIMA backtest **MAPE 4.0 %** (14.4 % across COVID → shocks *detected*, not predicted) | (n/a — detection is the focus) |
| **Reasoning** | multi-sensor co-movement (real event vs faulty sensor) | multi-country **shared-shock vs isolated** | which flow feature moved (dup / intercept) |
| **Retrieval (RAG)** | TF-IDF · BM25 · LSA · **dense MiniLM**; IR metrics (P@k/MRR/MAP/NDCG) | BM25 + LSA + **BERT** + chunking | KB over attack types + sensors (TF-IDF) |
| **Interface extras** | agentic multi-step tool-use; **offline conversational memory** | Flask chat | web charts + AI toggle; sensors/anomalies intents |
| **Deep learning outcome** | *no single best* detector | **AE (0.21) loses to IsolationForest (0.36)** | **supervised combiner is worst** (day-blocked CV) |
| **Quality gates** | **182 tests**, verify **29/29**, ruff+mypy, cov 92 % | **74 tests**, verify **13/13** | **56 tests**, verify **12/12**, ruff |
| **Headline finding** | the pattern works; deep ≠ better; synthetic labels cap the claim | real labels confirm detection; honest AE loss; forecast is honest about shocks | real labels prove channel complementarity; fusion trades precision for recall; learned fusion overfits |

*(Full numbers and their sources in [03_results_and_findings.md](03_results_and_findings.md).)*

---

## The three cross-cutting findings (the "so what")

1. **The architecture is domain-agnostic.** The *identical* five-layer design — truth layer + LLM
   explanation + faithfulness gate — transferred from weather to economics to network security with only
   the data layer changing. That is evidence the contribution is a *reusable method*, not a one-off.

2. **Label quality is the real ladder, not model complexity.** v1 (synthetic) → v2 (real, coarse) → v3
   (real, fine) is a progression in *how honestly detection can be measured*. The most valuable upgrades
   were **better labels and better metrics**, not fancier models.

3. **Simple beats complex, repeatedly and honestly.** Deep/learned methods **lost or tied** simpler ones
   in every version (v1: no single best; v2: AE < IsolationForest; v3: supervised combiner worst under
   honest CV). Reported as negative results, not hidden — which is itself the contribution.

See the detail files for the full argument, the technique-by-technique rationale, and the roadmap.
