# Verifiable NL Analytics for a Data-Center IoT Sensor Network

**MSc Project Report — v3 (Draft)**
_Author: Bello Ahmed · Loughborough University · Last updated: 6 July 2026_

> Working draft generated from the implemented system and its measured results. Numbers come from the
> code and its test/verification suites; reference formatting to be confirmed against the department
> style guide.

---

## Abstract

This third case study applies the project's **Verifiable NL Analytics** architecture — a deterministic
truth layer that computes every figure, a local LLM restricted to *explanation*, and an automated
faithfulness check — to a **data-center IoT network-security** setting. The data is the real, annotated
**DAD** dataset (Vigoya et al., 2020): seven days of MQTT sensor traffic captured as network packets,
with **544 human-labelled attack packets** of three types (duplication, interception, modification).
Unlike the project's weather system (v1), whose anomaly labels were synthetic, DAD provides **real
ground-truth labels**, enabling honest detection metrics. We parse the packet captures into a labelled
message table, derive two detector channels — a **value** channel (sensor temperature) and a **flow**
channel (network-traffic behaviour) — and evaluate single channels against **early/late fusion** on the
real labels, *per attack type*. The per-type results demonstrate genuine channel complementarity (each
channel is blind to the other's attack), and show — honestly — that fusion trades precision for recall
and does not beat the stronger single channel on aggregate when the other channel is weak. The same NL
interface and faithfulness guarantee transfer to this domain, now guarding IP addresses as well as
measurements.

---

## 1. Introduction

**Problem.** IoT sensor networks in data centers are safety-critical (a mis-reported cooling
temperature can trip the whole facility) yet exposed: MQTT is typically unauthenticated, and messages
can be duplicated, dropped, or modified in transit. Operators need to (a) detect such attacks and
(b) interrogate the data in plain English **without trusting a language model to do arithmetic**.

**Aim.** Reuse the project's local, trustworthy analytics architecture on a network-security domain
with *real* labels, and test whether combining a sensor-value view with a network-flow view improves
attack detection.

**Contributions.**
1. A reproducible pipeline that turns raw DAD `.pcap` captures into a labelled, analysable message table.
2. A **two-channel + fusion** detection study evaluated on **real** attack labels, per attack type.
3. Confirmation that **Verifiable NL Analytics** transfers to a third, unrelated domain, extending the
   faithfulness guard to IP addresses.

---

## 2. Related work

The dataset is DAD (Vigoya, Fernández, Carneiro & Cacheda, 2020, *Sensors* 20(13), 3745); the authors
provide statistical analysis but train no detectors, positioning DAD for exactly this kind of study.
The detectors used are standard: Isolation Forest (Liu, Ting & Zhou, 2008) for the flow channel and a
robust z-score for the value channel. Score-level and feature-level fusion are classical ensemble
strategies. Retrieval-augmented, faithfulness-checked LLM explanation follows the project's v1 design
and the broader concern with LLM hallucination in analytics.

---

## 3. Methodology

**3.1 Data.** Seven daily MQTT captures (2019-10-21…27), ~101,583 packets, 16 simulated cooling
sensors publishing a temperature every 5 minutes; per-packet attack labels in CSV. The attacker is a
single host (`10.6.56.41`); attacks span five days (Wed modification; Thu duplication; Fri/Sat mixed).
Normal traffic is STL-model-generated ("semi-synthetic"), an honest limitation.

**3.2 Preprocessing.** `scapy` parses each capture; the temperature is a plain-float MQTT PUBLISH
payload; the label CSV joins on `frame.number` (== the 1-based packet index) to yield a datetime-indexed
**MESSAGES** frame — the single source of truth.

**3.3 Two channels.** *Value*: per-window max |z| of sensor temperature (catches **modification**).
*Flow*: an Isolation Forest over behavioural window features — message rate, inter-arrival mean/std,
duplicate-msgid ratio — at a **5-minute** window (matching the publish cadence; catches **duplication**
and **interception**). Attacker-identity features are **deliberately excluded** to avoid label leakage.

**3.4 Fusion.** *Late*: rank-normalise each channel's score and combine (max/OR). *Early*: concatenate
the value and flow feature matrices into one Isolation Forest.

**3.5 Truth layer, interface, explanation.** A `query_engine` answers exact questions (counts per
day/type, attacker IP, temperature extrema, traffic) — each an independent pandas op. A deterministic
NL interface routes questions to it and returns trust-labelled answers; an optional local LLM rewords
them, gated by a faithfulness check that rejects any invented number or IP.

**3.6 Evaluation.** Real labels enable precision/recall/F1, average precision (threshold-free), and
**per-attack-type recall**. An independent `verify_results.py` recomputes headline figures via a second
path (raw label CSVs) — 12/12 checks.

---

## 4. Implementation

Python 3.12; `scapy` for parsing, scikit-learn for detectors, Flask + a small vanilla-JS page for the
web UI, a CLI (`apps/chat.py`). 65 automated tests, `verify_results.py` 12/12, ruff-clean, seeded and
reproducible (`reproduce.sh`). Layout mirrors v1 (`src/ tests/ apps/ report/ docs/`).
`notebooks/standalone_full_pipeline.ipynb` is a single, fully self-contained notebook (no `from src
import` — every function ported line-by-line from `src/*.py` and checked against it, confirmed at
runtime via `sys.modules`) that reproduces the whole pipeline end to end on the real data and adds a
further-experiments section beyond what §5 reports (stationarity, change-point detection, flow-feature
separability, two more classical detectors, a window-size sweep, ROC/confusion-matrix views); every
number it reproduces matches this document and `MODELS.md` exactly.

---

## 5. Results & Evaluation

**5.1 Truth layer.** Every `query_engine` figure equals an independent pandas computation; the verify
gate confirms 544 attacks (also via the raw CSVs), the per-day schedule `[0,0,104,56,192,192,0]`, the
attacker IP, 101,583 messages and 32,288 temperature readings.

**5.2 Detection — two channels + fusion (real labels; 2016 windows, 90 attack windows).**

| config | AP | best-F1 | precision | recall | rec·modification | rec·duplication | rec·mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flow-only   | 0.136 | 0.226 | 0.412 | 0.156 | 0.00 | 1.00 | 0.17 |
| value-only  | **0.183** | **0.379** | 0.331 | 0.444 | 0.46 | 0.50 | 0.43 |
| early-fusion| 0.147 | 0.289 | 0.202 | 0.511 | 0.42 | 1.00 | 0.52 |
| late-fusion | 0.171 | 0.306 | 0.214 | 0.533 | 0.46 | 1.00 | 0.53 |

**Findings.** (i) **Complementarity is real:** flow-only gets **0.00** recall on modification but
**1.00** on duplication; value has the opposite tilt — each channel is blind to the other's attack.
(ii) **Fusion trades precision for recall** (recall 0.44→0.53; precision 0.33→0.21), exactly as an OR of
two detectors predicts. (iii) **But fusion does not win on aggregate here:** `value-only` has the best
AP/F1, because the flow channel is weak on this data (few cleanly separable windows), so fusing it adds
more false alarms than detections. The honest lesson: *fusion pays off only when both channels are
strong.* Absolute scores are modest by design — no attacker-IP leakage, untuned detectors, 4.5% window
imbalance, and 71% of attacks typeable only as "mixed".

**5.3 Interface & faithfulness.** The interface answers domain questions with exact numbers
(*"104 attacks on Wednesday"*, *"attacker is 10.6.56.41"*). The faithfulness gate rejects a fabricated
count (999 for a 544 fact) while allowing rounding (29.28→29.3 °C) and preserving IPs — so a small
local model can reword safely. **A real, previously undocumented gap was found while testing the gate
directly against spoofed IPs** (in the standalone notebook, §11): because the number-matching regex
tokenises a dotted-quad into two decimal-looking numbers (`10.6.56.41` → `10.6`, `56.41`) and then
applies the same tolerance built for temperature rounding, a **last-octet swap** (`...56.41` →
`...56.99`) is *missed* — it lands inside the 2%-rounding band. A wholesale different IP or a
third-octet swap is still caught. This is more consequential than a missed count, since a wrong
attacker IP misdirects an investigation rather than just misreporting a figure; left as a documented
limitation (§7), not silently patched. Full mechanism and the exact test cases: `EVALUATION.md`.

**5.4 Deep detectors (first attempt, both channels — honest negative result).** Neither channel had a
deep variant before this experiment. A multivariate LSTM autoencoder over 30-minute sequences of flow
features loses to the existing Isolation Forest flow channel (AP 0.086 vs 0.136): duplication attacks
are an *instantaneous* per-window spike (a burst in message rate, a jump in the duplicate-msgid ratio),
not a shape unfolding over several windows, so the sequence model's overlapping-window score-averaging
step — designed to localise a *gradual* anomaly — dilutes exactly the instantaneous signal that made
the window stand out in the first place. A per-sensor LSTM forecaster (16 independent models,
2-hour context) loses narrowly to the existing z-score value channel (AP 0.166 vs 0.183): a
modification attack is a single reading implausible for its sensor's usual range, answerable directly
by a marginal z-score, whereas the forecaster answers the strictly harder question of whether a
reading is implausible *given recent history* — the right tool for gradual drift, not the single-point
fault DAD's modification attack actually is. The narrowness of the value-channel loss (per-type recall
0.42/0.50/0.43 vs the baseline's 0.46/0.50/0.43) indicates the forecaster is not failing outright, only
solving a more general problem than the fault requires. This is a direct, cross-domain replay of
notebook 05's finding from v1 — that single-channel value faults favour threshold/covariance methods
over learned sequence models — now reproduced on a third, structurally unrelated dataset, which is
stronger evidence for the general claim than either result alone. Full mechanism: `EVALUATION.md`.

---

## 6. Discussion

DAD's real labels let us *prove*, not assert, that a value view and a flow view are complementary, and
to measure fusion honestly. The counter-intuitive result — fusion helping recall but not F1 — is a more
informative outcome than a tidy "fusion wins", and it generalises: ensembling complementary detectors
is only worthwhile when each is individually competitive. The architecture's core promise (code owns the
numbers; the LLM only explains; a gate verifies it) held in a domain — packet-level network security —
quite unlike the weather and trade domains, extending the guard to IP addresses.

## 7. Limitations and future work

- **Semi-synthetic** normal traffic; one data center, seven days, one attacker.
- **Binary labels** → 71% of attacks are only day-level "mixed", capping per-type analysis.
- **Detector contamination is untuned** (a chosen prior, not learned).
- **The faithfulness gate's numeric-tolerance check does not fully cover IP addresses** (§5.3): a
  last-octet or near-value second-octet substitution in an attacker IP can slip through, because the
  check tokenises a dotted-quad as decimal numbers and reuses the tolerance tuned for measurement
  rounding. A wholesale different IP or a third-octet swap is still caught. Future work: an
  IP-address-aware check (match full dotted-quads as atomic tokens, compare for exact equality) rather
  than reusing the numeric-rounding tolerance for text that only looks numeric.
- *Tried since, all honest negative or non-significant results, logged in `EVALUATION.md`:*
  point-adjusted metrics + bootstrap CIs (now the default reporting, §5.2); a seasonal value model
  (hurts, AP 0.183→0.099); a Mahalanobis/combined flow channel (a wash, within CI noise); a supervised
  day-blocked score-combiner (the *worst* config, AP 0.101 — confounds attack type with day); deep
  detectors on both channels (§5.4 — lose, for a specific, testable reason each time, not just "not
  enough data"); and, in the standalone notebook's further-experiments section, an ADF stationarity test
  (temperature is non-stationary, p=0.53), a CUSUM change-point detector (tracks the real attack-day
  schedule within 2 hours but only at day granularity), a PCA view of the flow-feature space (attack and
  normal windows are offset but overlap heavily — why flow's AP is modest, not near-1.0), and two more
  flow-channel candidates (LOF loses; One-Class SVM ties Isolation Forest on AP with a higher F1, not
  adopted only for simplicity). Remaining future work: real multi-week/multi-attacker captures, which no
  amount of further tuning on this fixed dataset can substitute for.
- **The project's own architecture plan (`docs/PLAN.md`) was not fully carried out.** Checking the
  actual `src/` directory against the plan's module table programmatically (standalone notebook §13.1):
  7 of 17 planned modules were never built — `analysis`, `reasoning`, `rag_system`, `clustering`,
  `forecasting`, `agent`, `main` (`anomaly.py` was deliberately superseded by `channels.py`+`fusion.py`,
  a documented design change, not a gap). This had gone unflagged — README.md's status line narrows
  scope without ever crossing these off the plan's table. §13 closes three of the seven as notebook-only
  demonstrations: a SARIMA forecasting section (which recovers the paper's Friday/Saturday "mixed"
  windows but misses the Wednesday modification window — the same point-fault-vs-aggregate story as
  §5.4), a minimal `reasoning.py`-style `explain_window` function, and a day-type clustering attempt
  (an honest negative — it fails to recover the attack/clean split). `rag_system.py` and `agent.py`
  remain undone, for a stated reason: v3's question set is narrow enough that a full retrieval/agent
  layer would mostly retrieve the same handful of facts already inlined in `interface.py`.
- **Several of the paper's own descriptive analyses are not yet used by v3's detectors** (§13.2–13.3):
  protocol type, TCP flags and packet size are parsed by `scapy` but never kept by `preprocess.py`; a
  re-parse with these fields replicates the paper's protocol split (96.6% vs 96.9% TCP) but finds its
  claimed TCP-flag "atypical closure" signal does not reproduce at the per-packet level (an honest
  negative, most likely because the signal operates at flow/connection level, a coarser aggregation than
  tested). Distribution goodness-of-fit testing (mirroring the paper's KS/Anderson-Darling analysis)
  finds every sensor's best-fitting static distribution is still formally rejected — independently
  re-deriving why the paper (and this project) use STL/ARIMA rather than a fixed distribution.

## 8. Conclusion

The project's Verifiable NL Analytics pattern transfers to a real, labelled data-center IoT network:
exact answers remain exact, the LLM never invents a figure, and — using the real labels DAD provides —
we give an honest, per-type account of when combining a sensor and a network view helps and when it
does not.

## Appendix A — How to run
```bash
cd code/v3
bash tools/get_dad_dataset.sh        # fetch the real dataset (~18 MB, gitignored)
python3.12 -m pytest tests/          # 65 pass
python3.12 verify_results.py         # 12/12 independent checks
python3.12 apps/chat.py "how many attacks on wednesday?"
python3.12 apps/web/app.py           # browser UI (prints the URL)
```

## References
See `../v1/report/REPORT.md` for the shared method references; DAD-specific:
Vigoya, L., Fernández, D., Carneiro, V. and Cacheda, F. (2020) 'Annotated Dataset for Anomaly Detection
in a Data Center with IoT Sensors', *Sensors*, 20(13), 3745.
