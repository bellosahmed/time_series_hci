# v3 — Plan: Verifiable NL Analytics for a Data-Center IoT Sensor Network (DAD)

A third domain for the project, built on the **same architecture as v1** (preprocess → analyse →
detect → reason → query engine → interface → verified LLM explanation) but on a **data-center IoT
network-security** dataset with **real, labelled anomalies**.

> **The one idea (unchanged from v1/v2):** **computation = code (the source of truth); the LLM only
> explains.** Every number is produced by a deterministic query engine and checked; a local LLM only
> rewords it, and a faithfulness gate rejects any explanation that invents a figure.

> **Framing decision (chosen):** v3 is a **sensor + network** time-series anomaly system in the v1
> mould — it extracts the IoT temperature readings *and* network-flow features from the capture, and
> detects anomalies **evaluated against the dataset's real attack labels**. Alternatives considered:
> pure network-IDS (flow only) or sensor-only (temperature only). Sensor-only is rejected because the
> attacks are network-level (message duplication/interception/modification) and mostly do **not** move
> the temperature value — so flow features are essential to catch them. This is noted so it can be
> revisited.

---

## 1. Why this dataset — and what it fixes from v1

**Dataset: DAD** — Vigoya, Fernández, Carneiro & Cacheda (2020), *"Annotated Dataset for Anomaly
Detection in a Data Center with IoT Sensors"*, *Sensors* 20(13), 3745 (doi:10.3390/s20133745).
Source: <https://github.com/dad-repository/dad>. Already cited in v1's references.

**The headline advantage over v1:** v1's single biggest limitation was that anomaly detection was
evaluated on **synthetic, injected** anomalies (no real labels in the Jena data). **DAD ships real,
human-annotated ground-truth attack labels**, so v3 can report **genuine precision/recall/F1** on
*real* anomalies — directly closing v1's most-criticised gap.

**Verified real (already fetched to `code/data/dad/`, gitignored).** The per-day attack counts match
the paper's stated schedule exactly:

| Day (2019) | Rows | Attack pkts | Paper's schedule |
| --- | ---: | ---: | --- |
| Mon 21 Oct | 14,431 | 0 | no attacks |
| Tue 22 Oct | 14,398 | 0 | packets *removed* (not labelled as attack) |
| Wed 23 Oct | 14,524 | 104 | modification, 04:00–06:00 |
| Thu 24 Oct | 14,538 | 56 | insertion, ~03:00 (<5 min) |
| Fri 25 Oct | 14,544 | 192 | mix (intercept/dup/modify), 06:00 & 14:00–16:00 |
| Sat 26 Oct | 14,630 | 192 | mix (intercept/dup/modify), 06:00 & 14:00–16:00 |
| Sun 27 Oct | 14,518 | 0 | no attacks |

**544 labelled attack packets / ~101,600 (0.54%)** — binary labels, attacker IP `10.6.56.41`, three
attack types (duplication, interception, modification of MQTT messages).

**Honesty notes to carry into the report (state, don't hide):**
- The authors describe the traffic as **"semi-synthetic"** — a controlled reproduction of real
  behaviours, not a fully organic capture.
- **One** data center, **7** days, **one** attacker IP → external-validity limits (as with v1's single
  Jena domain; mitigated by this being a *third* independent domain for the shared architecture).
- **Strong class imbalance** (0.54%) → evaluate with PR-appropriate, point-adjusted metrics, not
  accuracy.

---

## 2. What is genuinely different from v1 (and why)

| Aspect | v1 (Jena) | v3 (DAD) |
| --- | --- | --- |
| Raw input | ready **CSV** of sensor readings | **`.pcap`** captures of MQTT traffic + per-packet label CSVs |
| Preprocessing | load/clean/resample | **parse pcaps** → extract MQTT messages + temperature payloads → **join labels** → window |
| Anomaly labels | **synthetic** (injected) | **real** (human-annotated attacks) — the big win |
| Anomaly nature | statistical outliers in a physical signal | **network attacks** on the message flow |
| Features | 14 physical sensors | **sensor** (temperature/device) **+ network-flow** (rates, msgid dup, per-IP) |
| Evaluation ceiling | capped by synthetic labels | **real precision/recall/F1**, per attack-type |

**New preprocessing dependency:** pcap parsing. `tshark` is not installed and needs sudo; **`scapy`**
(pip, pure-Python, no sudo, has an MQTT dissector in `scapy.contrib.mqtt`) is the chosen tool. The
per-packet `*_labels.csv` (`frame.number; ip.src; tcp.srcport; mqtt.clientid; mqtt.msgid; label`)
joins to the parsed packets by **`frame.number`**, giving each message its attack label for free.

---

## 3. Architecture → modules (`v3/src/`, mirroring v1)

```
  preprocess (pcap→messages→windows)     (1. get & describe the data)
        │
        ▼
  analysis · anomaly · forecasting · deep (2. detect attacks — REAL labels)
  evaluation · clustering
        │
        ▼
  query_engine                           (3. EXACT counts/stats — truth layer)
        │
        ▼
  reasoning · rag_system · llm           (4. explain WHY, in words)
        │
        ▼
  interface · agent → apps/              (5. question in, answer out)
```

| Layer | Module | Role in v3 |
| --- | --- | --- |
| **1. Data** | `preprocess.py` | scapy-parse the 7 pcaps → per-message table `(ts, src_ip, dst_ip, mqtt_clientid, mqtt_msgid, topic, temperature, label)` by joining `*_labels.csv` on `frame.number`; clean; resample to fixed **windows** (e.g. 1-min) producing a **sensor** series (temperature per device) and a **flow-feature** matrix |
| | `features.py` (new) | window/flow features: message rate, inter-arrival stats, **duplicate-msgid ratio**, per-source-IP counts, attacker-IP share, publish/subscribe mix, payload-change rate |
| **2. Detect** | `anomaly.py` | z-score (rate/temperature residual), **Isolation Forest, LOF, One-Class SVM** on flow windows; Matrix Profile optional |
| | `forecasting.py` | ARIMA forecast-error on message-rate / temperature (deseasonalised) — optional |
| | `deep.py` | windowed **autoencoder** (reconstruction error), GPU-aware (reuse v1 `get_device`) — optional |
| | `channels.py` (new) | the **two-channel** detector: a **value** score (temperature residual) + a **flow** score (rate/inter-arrival) per window |
| | `fusion.py` (new) | **early** (concat→one detector) and **late** (max/OR/weighted) fusion of the two channels + an optional supervised combiner — **the headline experiment** |
| | `evaluation.py` | **REAL-label** precision/recall/F1 + **point-adjusted** metrics + confusion + **bootstrap CIs** + **per-attack-type** breakdown (duplication/interception/modification) |
| | `clustering.py` | K-means over windows/days → "normal vs attack" day/period types |
| **3. Truth** | `query_engine.py` | **exact** counts & stats: attacks per day/hour/type, per-IP packet counts, temperature max/min/mean per sensor & scope, message rate — all **== pandas** |
| **4. Explain** | `reasoning.py` | *why* a window is flagged: which flow feature moved (duplicate msgids ↑, insertion burst, attacker-IP share ↑) |
| | `rag_system.py` | KB over MQTT, the three attack types, the sensors and the data center; TF-IDF/BM25/LSA/dense retrievers (reuse v1) |
| | `llm.py` | optional local **Ollama** layer — rewords facts, never computes; faithfulness gate (reuse v1) |
| **5. Interface** | `interface.py` | NL question → query engine → trust-aware answer (intents below) |
| | `agent.py` | optional multi-step tool-use (reuse v1 pattern) |
| | `main.py` | end-to-end pipeline entry point |

> **Status update (checked against the real `src/` directory, not against this table):** `anomaly.py`
> was superseded by `channels.py`+`fusion.py` (the two-channel design that became the headline
> experiment, §6) — a deliberate, documented change. `analysis.py`, `rag_system.py` and `agent.py`
> remain undone, honestly, for a stated reason: v3's question set is narrow enough that a full
> retrieval/agent layer would mostly retrieve the same handful of facts already inlined in
> `interface.py`'s `ATTACK_KB`. **Correction to an earlier version of this note:** it previously
> claimed `notebooks/standalone_full_pipeline.ipynb` §13.1 closed three of the six gaps as
> notebook-only demonstrations — that section was never actually written into the notebook on disk,
> and the claim was wrong. `reasoning.py`, `clustering.py` and `forecasting.py` have since been built
> as real `src/` modules (not notebook cells), each with unit tests and wired into `interface.py` as
> the `explain`, `day-type` and `forecast` intents respectively. Both `clustering.py` (day-type
> grouping) and `forecasting.py` (SARIMA vs naive persistence) were tested against the real DAD week
> and are **honest negative results**, kept in the code as tested findings rather than tuned to look
> better: whole-day temperature/traffic averages do not separate attack days from clean days (DAD's
> attacks last hours, not the whole day), and the SARIMA backtest does not beat naive persistence
> (MASE ≈ 1.4) on this short 7-day history. `main.py` (a unified entry point, for parity with v1/v2)
> is the one item from the original six still genuinely missing.

Much of v1's code (`llm.py`, `rag_system.py`, `evaluation.py` metrics, the trust-labelling, the
query-engine/interface scaffolding, the deep `get_device`) **ports over**; the domain-specific work is
`preprocess.py`, `features.py`, and the domain intents/KB.

---

## 4. Preprocessing pipeline (the main new build)

1. **Parse** each `capture_19102X0000_*.pcap` with scapy → rows of
   `(frame_number, epoch_ts, src_ip, dst_ip, tcp_srcport, mqtt_type, mqtt_clientid, mqtt_msgid,
   mqtt_topic, payload)`.
2. **Extract temperature** from MQTT **PUBLISH** payloads (verify the payload format during
   implementation — likely a numeric temperature per topic/sensor).
3. **Join labels** from `*_labels.csv` on `frame_number` → attach `label ∈ {0,1}` (+ derive attack
   *type* from the time window per the README schedule, for the per-type evaluation).
4. **Concatenate** the 7 days into one time-indexed frame; sanity-check counts against the known totals
   (**544 attacks**, per-day distribution above) — this is a built-in correctness gate.
5. **Resample** to fixed windows → (a) **sensor** series (per-device temperature) and (b) **flow
   features** (`features.py`), each window carrying a window-level label (attack if any labelled packet
   falls in it — the standard weak-labelling for windowed detection).

---

## 5. NL interface — domain-adapted intents

The v1 interface pattern (deterministic intent routing → query engine → trust-labelled answer),
re-pointed at DAD questions:

- **count / when** — "how many attacks on Wednesday?", "when did the modification attack happen?",
  "were there any attacks on Monday?" → exact counts + time windows.
- **attack-type** — "what kinds of attack are in the data?", "how many duplication attacks?"
- **attacker** — "which IP is the attacker?" → `10.6.56.41` (computed from the labelled packets).
- **sensor stats** — "what was the highest temperature reported on Friday?", "average temperature per
  sensor" → max/min/mean over the extracted readings.
- **traffic / rate** — "how many MQTT messages per hour?", "busiest hour".
- **removed packets** — "was anything unusual about Tuesday?" → the *removed-packet* day (a different,
  unlabelled anomaly the README documents — a nice honesty/edge case).
- **anomaly (+reasoning)** — "flag the unusual periods and say why" → detector output + `reasoning`.
- **summary / compare** — day report; compare two days.

Every answer stays **trust-labelled** (🔢 computed / 📚 context / 🧠 inferred), same as v1.

---

## 6. Headline experiment — two complementary channels + fusion

The attacks are **disjoint in signature**, so the key experiment compares four configurations against
the **real** labels, **per attack type**:

| Config | Value channel (temperature residual) | Flow channel (rate/inter-arrival) |
| --- | --- | --- |
| **Flow-only** | — | Isolation Forest / z-score |
| **Value-only** | STL/SARIMA/AE residual → threshold | — |
| **Early fusion** | concatenate both feature sets → **one** detector | |
| **Late fusion** | **max / OR / weighted** of the two channel scores | |

**Hypothesis + mechanism (to be confirmed with real numbers, not asserted):**
- **Single channels have a coverage ceiling** — flow-only misses **modification** (timing stays normal);
  value-only misses **duplication/interception** (values stay on-pattern). Their low overall recall is a
  *coverage* failure, not a tuning one.
- **Late fusion (OR/max) should win overall recall/F1** — each attack is caught by whichever specialist
  owns it (complementary ensemble ≈ additive coverage) — **but precision dips** because an OR-rule
  **sums both channels' false positives** (the trade-off to state explicitly).
- **Early fusion may *under*-perform the best specialist** — a sharp single-dimension anomaly
  (modification) gets **diluted** in the mixed high-dimensional space (Isolation Forest splits on random
  features; normal flow dims mask the one anomalous value dim).

**Why the per-attack-type table is the explanation:** it will show flow-only ≈ 0 recall on modification
and value-only ≈ 0 on duplication/interception — the aggregate F1 alone hides *why* fusion helps.
An optional **supervised combiner** (logistic regression over the two channel scores, day-blocked CV)
is reported separately (only ~544 positives → overfitting risk). Full design + code sketch in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 7. Evaluation & quality gates (mirror v1)

- **Detection (the headline):** the fusion comparison above — precision/recall/F1 + **point-adjusted**
  metrics against the **real** labels, per attack-type, with **bootstrap CIs**; compare detectors
  (z-score / IsolationForest / LOF / OCSVM / AE) on one harness — honestly, as v1 did (report losses).
- **Truth layer:** every `query_engine` output **== pandas**, checked by a `verify_results.py` harness
  (e.g. "544 attacks total", per-day counts, attacker IP, temperature extrema) — an independent gate.
- **LLM faithfulness:** the local model invents **0** numbers (majority-of-N vote), as in v1.
- **Engineering gates:** `pytest`, `ruff`, `mypy`, coverage; `reproduce.sh`; seeded; pinned deps.

---

## 8. Apps, notebooks, docs

- **`apps/chat.py`** (CLI) + **`apps/web/`** (React UI) — reuse v1's app layer and the vendored,
  build-free React front-end + unit-tested `chart.js`.
- **Notebooks** `00`–`04` mirroring v1 (walkthrough, EDA/traffic viz, methods+evaluation, interface,
  detector comparison).
- **`report/`** REPORT · EVALUATION · MODELS · DATA_QUALITY (packet parsing, label join, imbalance).

---

## 9. Milestones (phased, TDD like v1)

- **M0 — Acquire & verify dataset. ✅ DONE.** Real DAD fetched to `code/data/dad/`; labels verified
  against the paper's schedule (544 attacks, per-day distribution); `tools/get_dad_dataset.sh`
  reproduces it.
- **M1 — Preprocess. ✅ DONE.** `preprocess.py` (scapy pcap→labelled MESSAGES frame; join on
  `frame.number`) **and** `features.py` (VALUE = per-sensor temperature; FLOW = `msg_rate`,
  `publish_rate`, `iat_mean/std`, `dup_msgid_ratio`; weak `window_labels`). **9 tests green**, ruff
  clean, locking the real totals (101,583 packets · 544 attacks · per-day `[0,0,104,56,192,192,0]` ·
  32,288 readings). *Confirmed on real data:* payload = plain float string; single topic `IoT`;
  `clientid` encodes InRow-unit+probe (131 = InRow 13 / probe 1); times are UTC. **Flow features
  deliberately omit attacker-identity** (`attacker_share`/`unique_src`) to avoid label leakage — see
  `report/EVALUATION.md`.
- **M2 — Query engine. ✅ DONE.** `query_engine.py` (exact counts/stats == pandas) + `verify_results.py`
  (**12/12**, incl. an independent raw-CSV cross-check of the 544 total). 14 tests green, ruff clean.
  Noted: `attack_breakdown` = {modification 104, duplication 56, **mixed 384**} — 71% only typeable at
  day level (binary labels), an honest limit for the per-type evaluation.
- **M3 — Detection + REAL-label evaluation (the headline). ✅ DONE.** `channels.py`/`fusion.py`/
  `evaluation.py`/`experiment.py`; 22 tests green, ruff clean. **Result (real labels):** the per-type
  table *proves* complementarity — flow-only 0.00 recall on modification but 1.00 on duplication; value
  the opposite. Fusion lifts recall 0.44→0.53 but cuts precision 0.33→0.21, and — the honest finding —
  **does not beat `value-only` on AP/F1** (0.181/0.376) because the flow channel is weak here, so fusion
  helps recall at a precision cost that isn't worth it. Numbers deliberately un-inflated (no attacker-IP
  leakage). Full table + why in `report/EVALUATION.md`. **Follow-ups:** tune detectors, point-adjusted
  metrics, seasonal value model, bootstrap CIs.
- **M4 — NL interface. ✅ DONE.** `interface_parsing.py` + `interface.py` (intents: count · attacker ·
  breakdown · temperature · traffic · when · summary; 🔢/📚/🧠 trust labels). 33 tests green, ruff clean.
  Every answer's number comes from the query engine; unknown questions are refused, not faked.
- **M5 — LLM explanation + faithfulness gate. ✅ DONE.** `llm.py` (Ollama `explain` + `faithfulness`)
  wired into the interface via `_phrase`: rewords computed facts only if no number is invented, else the
  exact template. 39 tests green, verify 12/12, ruff clean. Guards IPs + temperatures + counts; Ollama
  optional (prose deferred to write-up, as v1).
- **M6 — Apps, notebooks, report, docs; quality gates green. ✅ DONE.** CLI (`apps/chat.py`), Flask web
  app (`apps/web/`, smoke-tested live), `reproduce.sh`, `requirements.txt`, `report/REPORT.md` +
  `report/MODELS.md`, an **executed** walkthrough notebook (`notebooks/00_walkthrough.ipynb`), and the
  **M3 rigor pass** — point-adjusted metrics + bootstrap CIs (which showed the value-vs-fusion AP gap is
  *not* significant and fusion reaches 1.00 event-level recall). **47 tests green, verify 12/12, ruff
  clean.** Plus the **future work**: a supervised score-combiner (`fusion.supervised_scores`, day-blocked
  CV — an *honest negative result*: worst config, AP 0.101, because attack type is confounded with day)
  and a **comprehensive EDA notebook** (`notebooks/01_exploratory_analysis.ipynb` — STL trend/seasonality,
  ACF, correlations, attacks, detection). **48 tests green, verify 12/12, ruff clean.** v3 complete.

---

## 10. Environment & reproducibility

- **Python 3.12** (as v1/v2). **New dep: `scapy`** (`python3.12 -m pip install scapy` — pure Python,
  no sudo, MQTT dissector in `scapy.contrib.mqtt`). Torch cu128 already present for the optional AE.
- Data is **gitignored**; `tools/get_dad_dataset.sh` re-downloads it. Seeds fixed; deps pinned in a
  `requirements.txt`; `reproduce.sh` runs tests + verification.

---

## 11. Open questions to confirm before/while building

1. **Framing** — proceed with **sensor + network** (chosen), or pivot to pure network-IDS / sensor-only?
2. **Temperature payload format** — confirm how the reading is encoded in the MQTT PUBLISH payload
   (topic-per-sensor? JSON? raw float?) once the first pcap is parsed; the sensor-series feature family
   depends on it, but the network-flow family and the real-label evaluation do **not**.
3. **Window size** for flow features — **resolved to 5 min** by the data: sensors publish every 5 min,
   so 1-min windows are only 39.3% occupied vs 100% at 5 min (measured in M1). A sensitivity sweep
   around 5 min can still refine it in M3.
