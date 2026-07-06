# 01 — The Three Systems, Their Data, and the Shared Architecture

## The shared architecture (constant across v1, v2, v3)

Every version is the same five layers; only the **data layer** changes per domain:

```
  1. preprocess / analysis        clean + describe the data
            │
            ▼
  2. detect (statistical · ML · deep) + evaluation      find anomalies, measure honestly
            │
            ▼
  3. query_engine  ── THE TRUTH LAYER ──   exact numbers, == pandas, independently verified
            │
            ▼
  4. reasoning · rag_system · llm          explain WHY, in words
            │
            ▼
  5. interface (+ agent) → apps            question in → verified answer out
```

**Invariants in all three:**
- **Truth layer:** every reported number is one exact deterministic computation, cross-checked by a
  `verify_results.py` harness (`== pandas` + known-event checks).
- **LLM = explanation only:** a local **Ollama** model rewords the computed facts; a **faithfulness gate**
  rejects any answer that introduces a number absent from the facts.
- **Trust labels:** answers are marked 🔢 *computed* / 📚 *context* / 🧠 *inferred*.
- **Fully local / offline:** no cloud, no external APIs; runs with just Python + a local model.

This constancy is the point: the same trust machinery works regardless of whether the "anomaly" is a
heat spike, a financial crisis, or a spoofed sensor packet.

---

## v1 — Weather (the reference implementation)

- **Domain / data:** Jena Climate 2009–2016 — **70,129 hourly rows × 14 physical sensors** (temperature,
  humidity, pressure, wind, …); target `T (degC)`. Wind columns carry `-9999` sentinel errors (cleaned).
- **Why this domain:** multivariate, strongly seasonal, real data-quality issues — an ideal testbed to
  build the whole pipeline. But **Jena has no anomaly labels.**
- **What is richest here:** the widest method set (6 anomaly families + 3 deep detectors), a full
  **retrieval-evaluation** suite (four retrievers + IR metrics), **agentic** multi-step tool-use, and
  (added late) **deterministic offline conversational memory** in the chat.
- **The catch:** detector accuracy can only be measured on **synthetic** injected anomalies (or a
  controlled clean signal) — honest, but it caps how strong the detection claim can be.

## v2 — Trade (real, datable macro anomalies)

- **Domain / data:** EU monthly international trade — **292 months × 54 columns** (27 EU countries ×
  {exports, imports}), 2002–2026, € millions, not seasonally adjusted.
- **Why this domain:** it has **real, datable anomalies** — the **2008 financial crisis**, **COVID-2020**,
  the **2022 energy crisis** — so detectors are tested against events that *actually happened*, and
  anomalies are **shared across countries** (a multivariate, correlated-shock structure).
- **What is distinctive:** **multi-country shared-shock vs isolated** reasoning; **SARIMA** forecasting
  with an out-of-sample **backtest** (does it predict, or memorise?); a hybrid **BM25 + LSA + BERT +
  chunking** retriever; a multivariate **autoencoder** benchmarked against the baselines.
- **The catch:** labels are **real but coarse** — crisis *windows* (~14 anomaly months), and only **292**
  monthly points, which is little data for deep models.

## v3 — IoT network security (real, fine-grained labels)

- **Domain / data:** DAD (Vigoya et al., 2020) — **7 days of MQTT sensor traffic** captured as network
  packets (~**101,583 packets**), 16 cooling-sensors publishing every 5 min, with **544 human-labelled
  attack packets** in three types (**duplication, interception, modification**) from one attacker IP.
- **Why this domain:** it provides **real, per-packet ground-truth labels** — directly fixing v1's biggest
  weakness (synthetic labels) — and the attacks split cleanly across **two signals** (sensor value vs
  network flow), which motivates a **fusion** study.
- **What is distinctive:** a **pcap → labelled table** pipeline (scapy), a **two-channel + fusion**
  detector (value + flow, early/late/supervised), and the most rigorous evaluation — **per-attack-type
  recall, point-adjusted metrics, and bootstrap confidence intervals** on real labels.
- **The catch:** the traffic is **"semi-synthetic"** (authors' term — normal data generated from an STL
  model), it is one data center / 7 days / one attacker, and labels are **binary** (71 % only typeable as
  "mixed"), with strong class imbalance (186:1 packet, 21:1 window).

---

## Why three domains at all?

The domains were chosen to be **maximally different** — physical, economic, cyber — precisely so that a
*shared* architecture transferring across them is meaningful evidence. And the label regimes were chosen
to **climb a ladder of evaluation honesty**: synthetic (v1) → real-but-coarse (v2) → real-and-fine (v3).
Each version is therefore both a standalone system *and* a stress test of the same underlying claim.
