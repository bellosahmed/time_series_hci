# v3 — Running Evaluation & Build Record (DAD)

A **live** log of the v3 build: for each step, *what was done → the result → the trade-off → what was
discovered*. Updated as the work happens (not at the end), so the report and viva have the reasoning,
not just final numbers. Newest milestone at the bottom. Companion to `docs/PLAN.md` /
`docs/ARCHITECTURE.md`; headline numbers here must always match the code + `verify_results.py`.

---

## M0 — Acquire & verify the dataset ✅

**Did:** fetched the real DAD dataset (Vigoya et al. 2020) to `code/data/dad/` via
`tools/get_dad_dataset.sh`; counted the shipped per-packet labels per day.

**Result:** 22 files, ~18 MB. Labelled-attack counts per day = **Mon 0 · Tue 0 · Wed 104 · Thu 56 ·
Fri 192 · Sat 192 · Sun 0 = 544** / ~101.6k packets (0.54%). Attacker IP `10.6.56.41`.

**Trade-off:** the authors call the traffic **"semi-synthetic"** (normal data generated from an STL
model fitted to 14 days of real readings) — realistic and *labelled*, but not a fully organic capture;
single data center, 7 days, one attacker.

**Discovery:** the per-day counts reproduce the paper's stated attack schedule **exactly**, which both
proves the download is genuine and gives us free, real ground-truth labels — the thing v1 lacked
(v1 could only inject *synthetic* anomalies). This is v3's core advantage.

---

## M1 — Preprocess: pcap → labelled MESSAGES frame 🔨 (in progress)

### Step 1 — spike to learn the payload format (throwaway)
**Did:** installed `scapy` 2.7.0; parsed the Monday pcap to inspect the MQTT structure.

**Result:** 14,431 packets == 14,431 label rows; 4,608 PUBLISH/day; single topic `IoT`; payload is a
**plain ASCII float string** (`b"15.54"`).

**Trade-off:** scapy noisily tries to auto-load optional layers (smb/kerberos/tls) that need a newer
`cryptography` than is installed — cosmetic errors only. Handled by **targeted imports** (not
`scapy.all`) + silencing its logger, rather than upgrading a system dependency.

**Discovery:** `frame.number` **equals the 1-based packet index**, so joining the label CSV to the
parsed packets is exact and trivial — no fuzzy timestamp matching needed. And **4,608 = 16 sensors ×
288 samples/day**, matching the paper, so the capture is complete.

### Step 2 — `preprocess.py` (TDD)
**Did:** wrote 3 tests first (locking the real totals), watched them fail, then implemented
`parse_pcap` / `load_day` / `load_week`. `clientid`/`msgid`/`label` are taken from the authoritative
labels CSV (tshark-derived) rather than re-parsed from scapy.

**Result:** MESSAGES frame = **101,583 packets × 10 cols** (`ts,frame,src_ip,src_port,topic,
temperature,clientid,msgid,label,attack_type`). Tests green: label total **544**, per-day
`[0,0,104,56,192,192,0]`, **32,288** temperature readings (8.67–29.40 °C, mean 18.4), attacks from
`10.6.56.41`. ruff clean.

**Trade-off:** the shipped labels are **binary** (attack/normal) with no per-packet *type*, so
`attack_type` is assigned at **day granularity** from the paper's schedule (Wed→modification,
Thu→duplication, Fri/Sat→"mixed"). Finer per-packet typing isn't recoverable from the data — an honest
limit for the per-type evaluation in M3.

**Discovery:** `clientid` decodes to the paper's sensor scheme — `131–134` = InRow **13** probes 1–4,
`151–154` = InRow **15** — so we get per-sensor identity for the value channel for free. Also, capture
timestamps are **UTC** (day boundaries shift ~2 h vs the local schedule), so attack-day logic is keyed
on the filename `day`, not the UTC index, to stay correct — a subtlety that would silently corrupt the
per-day counts if missed.

### Step 3 — `features.py` (TDD)
**Did:** wrote the VALUE channel (`sensor_series` — temperature per sensor client id) and the FLOW
channel (`flow_features` — `msg_rate`, `publish_rate`, `iat_mean`, `iat_std`, `dup_msgid_ratio` per
window) plus weak `window_labels`. 7 unit tests (hand-built frames) + 2 real-week checks; all green,
ruff clean.

**Result:** the two channels build from the MESSAGES frame; exact invariants hold on the real week
(`msg_rate` sums to 101,583; `publish_rate` sums to 32,288). One test I wrote first **failed** — it
assumed >50% of 1-min windows carry traffic; the real figure is **39.3%**.

**Trade-off:** we **deliberately exclude attacker-identity features** (`attacker_share`, `unique_src`).
In DAD the attacker IP `10.6.56.41` appears *only* during attacks, so such a feature is the label in
disguise — it would make detection look perfect while proving nothing. The flow features are all
**behavioural aggregates** (rate/timing/replay) a real detector could compute without knowing the
attacker. This *lowers* the headline detection numbers we'll report but makes them **honest and
defensible** — the right trade for a viva.

**Discovery:** the failing test wasn't a bug — it revealed the **traffic is bursty on a 5-minute
cadence** (each sensor publishes every 5 min). Measured: **1-min windows are 39.3% occupied vs 5-min
windows 100%**. So the natural flow-feature window is **5 minutes**, matching the publish period — this
resolves Open-Q #3 (window size) directly from the data rather than a blind sweep, and the fix was to
assert *exact packet-count invariants* instead of a fragile occupancy fraction.

**Next:** M2 — `query_engine.py` (exact counts/stats == pandas: attacks per day/type, attacker IP,
temperature extrema, message rate), with a `verify_results.py` gate.

---

## M2 — Query engine: the truth layer ✅

**Did:** built `query_engine.py` (`count_attacks`, `attacks_by_day`, `attack_breakdown`,
`attacker_ips`, `temperature_stat`, `message_rate`, `busiest_hour`, `total_messages`) — every figure a
thin, exact pandas op the NL layer will only *present*. TDD: 5 tests first (hand-built + real week),
then implementation. Added `verify_results.py`, an independent gate.

**Result:** 14 tests green, ruff clean, **`verify_results.py` 12/12**. Confirmed on the real week:
544 attacks, per-day `[0,0,104,56,192,192,0]`, attacker `10.6.56.41`, 101,583 messages, 32,288
readings, temperature max/min/mean **== pandas** exactly.

**Trade-off:** the query engine is deliberately *dumb and exact* — no smoothing, no inference, one
pandas expression per answer. That caps what it can answer (only what the data literally says) but is
exactly what makes every number verifiable and hallucination-proof — the project's core promise.

**Discovery:** the `attack_breakdown` is `{modification: 104, duplication: 56, mixed: 384}` — because
the shipped labels are binary and Fri/Sat interleave all three attack types, 384/544 (**71%**) of
labelled attacks can only be typed as "mixed". So a clean per-type detection score is possible for
Wed (modification) and Thu (duplication) but not for the mixed days — an honest constraint the M3
per-type evaluation must state. The verify gate also cross-checks the attack total via a **second,
independent path** (summing the raw label CSVs, no pcap involved) so a parser *or* query-engine bug
can't pass silently.

**Next:** M3 — the two-channel + fusion detector, graded per attack type on the real labels (§6).

---

## M3 — Two channels + fusion, graded on the real labels ✅ (the headline)

**Did:** built `channels.py` (value = max |z| of sensor temperature; flow = Isolation Forest over the
behavioural flow features), `fusion.py` (late = rank-normalised OR/max; early = concat → one Isolation
Forest), `evaluation.py` (average precision, best-F1 operating point, per-type recall) and
`experiment.py` (the runner). TDD: 9 tests first, then implementation. Window = 5 min (M1 cadence).

**Result (real labels, 2016 windows, 90 attack windows = mixed 60 / modification 26 / duplication 4):**

| config | AP | best-F1 | precision | recall | rec·modification | rec·duplication | rec·mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flow-only   | 0.136 | 0.226 | 0.412 | 0.156 | **0.00** | **1.00** | 0.17 |
| value-only  | **0.183** | **0.379** | 0.331 | 0.444 | 0.46 | 0.50 | 0.43 |
| early-fusion| 0.147 | 0.289 | 0.202 | 0.511 | 0.42 | 1.00 | 0.52 |
| late-fusion | 0.171 | 0.306 | 0.214 | **0.533** | 0.46 | 1.00 | **0.53** |

**Trade-off (confirmed, exactly as hypothesised):** fusion **lifts recall** (0.44 → 0.53 — the union of
two specialists) but **cuts precision** (0.33 → 0.21 — an OR-rule sums both channels' false positives).

**Discovery — the honest, non-obvious finding: fusion is *not* a free win here.** On the aggregate,
threshold-free metric, **`value-only` is best (AP 0.183, F1 0.379)** — *neither* fusion beats it. The
reason is visible in the per-type columns, which is exactly why we report them:
- The **complementarity is real and provable:** flow-only gets **0.00 recall on modification** (a bad
  temperature keeps timing normal, so the flow features never move) yet **1.00 on duplication**; the
  value channel has the opposite tilt. Each channel is genuinely blind to the other's attack — the core
  scientific claim, demonstrated on real labels, not asserted.
- **But the flow channel is weak on this data** (only 4 duplication windows are cleanly separable; it is
  noisy elsewhere), so fusing it into the already-broad value channel **adds more false alarms than true
  positives** → recall up, F1/AP down. *So fusion pays off only when both channels are strong; when one
  is weak, it helps recall at a precision cost that isn't worth it.* This refines the plan's hypothesis
  (which hedged that "whether F1 improves depends on how noisy each channel is" — now measured: it
  doesn't).
- **early ≈ late** (late marginally better AP), a mild version of the predicted early-fusion dilution.

**Why the absolute numbers are low (AP 0.14–0.18) — and honestly so:** (1) we **deliberately excluded
the attacker IP** (no label leakage) — this is a genuinely hard unsupervised problem; (2) untuned
detectors (default Isolation Forest, plain z-score); (3) 4.5% window imbalance; (4) weak window labels
and the 71%-"mixed" blur. All are improvable refinements (tune contamination, point-adjusted metrics, a
seasonal value model) — recorded as M3 follow-ups — but the first-pass numbers already **prove the
mechanism** cleanly, which is the point. Reproduce: `experiment.run_experiment(load_week())`
(deterministic, `random_state=0`).

**Next:** M4 — the NL interface (domain intents over the query engine, trust-labelled), then M5 (LLM
explanation + faithfulness gate).

---

## M4 — Natural-language interface ✅

**Did:** built `interface_parsing.py` (deterministic intent/day/attack-type/stat routing) and
`interface.py` (`Interface.ask` → `Answer` with 🔢 computed / 📚 context / 🧠 inferred trust labels).
Intents: count · attacker · breakdown · temperature · traffic · when · summary. TDD: 11 tests first,
then implementation.

**Result:** 33 tests green, ruff clean, verify 12/12. Real Q&A works, every number from the query
engine — e.g. *"How many attacks on Wednesday?"* → "104", *"Which IP is the attacker?"* →
"10.6.56.41", *"Highest temperature on Friday?"* → 29.28 °C, *"Summary of Thursday"* → "14,538
messages, 56 attack packets, 9.0–29.3 °C".

**Trade-off:** routing is deterministic keyword matching (as in v1) — transparent and un-foolable, but
limited to phrasings it recognises; free-form wording is deferred to the optional LLM planner (M5).
One concrete bug found + fixed via TDD: `\battack\b` missed the **plural** "attacks", so intent
matching now tolerates a trailing `s`.

**Discovery:** the *busiest hour* answer (Fri 04:00 UTC / 06:00 local, 824 messages) **coincides with a
mixed-attack window** — duplication inflates traffic, so an innocuous traffic question already surfaces
an attack. A neat cross-check that the flow signal M3 relies on is real, and a good example of the
interface making an attack visible without being asked about attacks directly.

**Next:** M5 — wire the optional local LLM to reword these computed facts, gated by a faithfulness
check that rejects any invented number (port v1's `llm.py`).

---

## M5 — LLM explanation + faithfulness gate ✅

**Did:** ported `llm.py` (Ollama `explain` + the `faithfulness` number-guard) and wired `_phrase` into
the interface: with the LLM on and reachable, it rewords the computed facts **only if** the
faithfulness check finds no invented number; otherwise it returns the exact template. TDD: faithfulness
+ gating tests first.

**Result:** 39 tests green, ruff clean, verify 12/12. The gate behaves correctly on the DAD facts:
fabricating "999" for a 544 fact → **rejected** → exact template; rounding "29.28 → 29.3 °C" → allowed;
the IP "10.6.56.41" preserved → ok. **Verified live (M6+):** a local Ollama `llama3.2:3b` reworded
"There were 104 labelled attack packets on Wednesday" → "On Wednesday, there were 104 labelled attack
packets" (`used_llm=True`, the **104 preserved** by the gate) — so the LLM path is exercised, not just
monkeypatched. (Earlier notes assumed Ollama was absent; it is in fact running here.)

**Trade-off:** the friendlier prose comes from a small local model, but the gate is deliberately
simple number-matching — it enforces decimals, counts, years and the larger IP octets (56, 41), yet
tolerates small integers ≤ 12 (clock hours, the "10"/"6" octets). So a fabricated *small* count could
slip through — the same honest residual documented for v1, now noted for IPs too.

**Discovery:** the gate is exactly what lets a **cheap 3B model be used safely** — because the numbers
are already exact upstream (query engine) and re-checked downstream (faithfulness), the model only has
to be a competent *re-phraser*, never a calculator. The **Verifiable NL Analytics** pattern therefore
transfers cleanly to a third, unrelated domain (network security) — with the new wrinkle of guarding
**IP addresses**, not just physical measurements.

**Core pipeline (M0–M5) complete.** Remaining: **M6** — apps (CLI/web), notebooks, the report
write-up, and the detection refinements logged in M3.

---

## M6 — Apps, reproducibility, write-up 🔨 (apps + report done)

**Did:** built the **CLI** (`apps/chat.py` — one-off or interactive, `--llm`), the **web app**
(`apps/web/app.py` Flask `create_app(df)` factory + a small vanilla-JS `index.html`, no build step),
`reproduce.sh`, and the **report draft** (`report/REPORT.md`). TDD for the app layer (5 tests over a
tiny in-memory frame). Then smoke-tested the live server.

**Result:** 44 tests green, ruff clean, verify 12/12. The web app serves (HTTP 200) and `/api/ask`
returns real answers end-to-end ("104 attacks on Wednesday", "attacker 10.6.56.41"). `README.md` now
has run instructions; `REPORT.md` carries the honest M3 table and findings.

**Trade-off:** the web UI is plain HTML+JS rather than v1's vendored-React app — less polished but zero
build, easy to run and test, proportionate for a research prototype. A per-request `Interface` is built
on each `/api/ask` (cheap — it only holds a df reference), keeping the server stateless.

**Discovery:** driving the real server (not just `test_client`) confirmed the whole chain — pcap →
MESSAGES frame → query engine → interface → JSON — works on the full dataset behind Flask, and the
faithfulness/`used_llm` flag is correctly `false` with no Ollama present.

**Remaining M6 (deferred):** walkthrough notebooks, a `MODELS.md` detector table (currently folded into
`REPORT.md` §5.2), and the M3 detector refinements (tuning, point-adjusted metrics, bootstrap CIs).

---

## M3-refined — point-adjusted metrics + bootstrap CIs ✅ (the rigor pass)

**Did:** added `evaluation.point_adjust`, `best_f1_point_adjusted` and `bootstrap_ci` (TDD, 3 tests),
wired them into the experiment (columns `AP_lo/AP_hi`, `f1_pa`, `recall_pa`), re-ran, and wrote
`report/MODELS.md`. 47 tests green, ruff clean.

**Result — the interpretation genuinely changed:**

| config | AP [95% CI] | F1 | F1·point-adj | recall·point-adj |
| --- | :---: | ---: | ---: | ---: |
| flow-only   | 0.136 [0.077, 0.222] | 0.226 | 0.294 | 0.178 |
| value-only  | 0.183 [0.136, 0.245] | 0.379 | 0.804 | 0.889 |
| early-fusion| 0.147 [0.108, 0.206] | 0.289 | 0.732 | **1.000** |
| late-fusion | 0.171 [0.123, 0.246] | 0.306 | 0.789 | **1.000** |

**Trade-off:** point-adjusted scoring is the standard event-level protocol but is **optimistic** (one
hit credits a whole segment) — so we report **both** window-level and point-adjusted, never just the
flattering one.

**Discovery — this pass overturned the earlier headline in two honest ways:**
1. **The value-only "win" is not significant.** The AP bootstrap CIs **overlap heavily** (value
   [0.136, 0.245] vs late-fusion [0.123, 0.246]), so we **cannot** claim value-only beats fusion on AP —
   the M3 first-pass conclusion was within noise. (Only bootstrap CIs revealed this; the point estimate
   alone was misleading.)
2. **At the event level, fusion is best where it matters:** both fusions reach **1.00 point-adjusted
   recall** (every attack *episode* caught) with F1 within 0.014 of value-only. So the honest verdict is
   *goal-dependent*: fusion for catching every episode; the simpler value channel for window-level
   precision — and no config is significantly best on AP. This is a stronger, more defensible result
   than the first pass, and a textbook case of why CIs + event-level metrics matter.

**Formal significance test (ported from v1: `evaluation.mcnemar_test`) ✅.** v1 pairs its bootstrap CIs
with a McNemar test; v3 only had the CI. Ported it (TDD, +2 tests → 60 tests), applied to `value-only`
vs `late-fusion` **each thresholded at its own best-F1 operating point** (not the AP ranking): only
value-only-correct = 95, only late-fusion-correct = 8, **p ≈ 5×10⁻²⁰** — highly significant.

**This does not contradict point 1 above — it answers a different question.** AP is threshold-free: it
asks whether the two configs *rank* windows similarly well across every possible cutoff, and there the
CIs overlap, so no — you cannot say one out-ranks the other overall. McNemar asks a narrower, sharper
question: *at each config's own best single cutoff*, do they make significantly different point-level
calls? There, yes, decisively — value-only's classification decisions are right far more often than
late-fusion's at their respective best-F1 thresholds. Together: the two configs are indistinguishable as
*rankers* (AP), but value-only is the significantly better *classifier at its optimal threshold*
(McNemar) — a real nuance a single metric would have hidden either way, and the reason this project
reports point estimate + CI + paired test rather than any one of them alone.

**Next:** walkthrough notebook (`notebooks/00_walkthrough`).

---

## Future work — supervised score-combiner ✅ (a negative result worth reporting)

**Did:** added `fusion.supervised_scores` — a logistic regression over the two rank-normalised channel
scores, evaluated with **day-blocked** leave-one-day-out CV (an attack episode never spans train and
test). TDD (1 test), added as a fifth config. 48 tests green, ruff clean.

**Result:** the supervised combiner is the **worst** config — AP **0.101** [0.075, 0.141], below every
unsupervised one (flow 0.136, late-fusion 0.171, value 0.183).

**Trade-off:** a learned combiner *can* weight the channels optimally, but it needs enough i.i.d.
labelled data to generalise. Here it has neither.

**Discovery (the honest lesson):** attack **type is confounded with day** (Wed=modification,
Thu=duplication, Fri/Sat=mixed) and there are only four attack days, so under day-blocked CV the model
trains on days whose attacks need a *different* channel than the held-out day → the learned weighting
doesn't transfer and it overfits. A random-split CV would have **leaked** (same episode in train+test)
and looked great — the honest day-blocked split exposes the failure. **With scarce, episode-clustered
labels a simple unsupervised OR generalises better than a learned fusion.** This is a strong,
defensible negative result, and a caution about CV design that examiners reward.

**Window/contamination tuning:** the flow window is already fixed at 5 min by the sensor cadence (M1);
finer window or Isolation-Forest contamination sweeps are low-value given the flow channel's weak
signal and are left as optional tuning rather than pursued.

**Next:** a comprehensive EDA walkthrough notebook (trends, seasonality, correlations, attacks).

---

## M6+ — richer chat (charts + AI toggle) and a build-along tutorial ✅

**Did:** (1) the web chat now shows **charts** like v1 — the interface attaches a small presentation-only
`chart` payload (bar of attacks-per-day / by-type for count/when/breakdown/summary; hourly line for
temperature/traffic), rendered as inline SVG in `index.html`; added an **"Explain with AI" toggle** +
status lamp driven by a new `/api/models` route. (2) A fresh **build-along tutorial notebook**
(`notebooks/02_building_the_system.ipynb`, executed, 22 cells) that teaches *how* the app is built and
*why*, step by step, for anyone. TDD for the backend (chart + route tests). 53 tests green, ruff clean.

**Result:** verified live — `/api/ask` returns bar/line chart payloads (attacks-per-day
`[0,0,104,56,192,192,0]`; 24-point hourly temperature line), and `/api/models` reports the local Ollama
as available.

**Discovery / correction:** a local Ollama **is** running here (`llama3.2:3b`) — my earlier M5 note that
it wasn't was wrong. With the toggle on, the model reworded "There were 104 labelled attack packets on
Wednesday" → "On Wednesday, there were 104 labelled attack packets" (`used_llm=True`), the **104
preserved** by the faithfulness gate — so the explain-with-AI experience now matches v1, and the
"local model not enabled" impression was just the toggle defaulting off. **Trade-off:** charts are
deliberately presentation-only (the query engine still owns every number; the SVG just draws them), so
a chart can never introduce a figure the truth layer didn't compute.

---

## M6++ — more intents + a full course-style tutorial ✅

**Did:** (1) fixed gaps the user hit in the chat — added **`sensors`** ("show all sensors" → lists the 16
client ids + a per-sensor readings chart) and **`anomalies`** ("show me the anomalies" / "unusual
readings" → the 544 labelled attacks, by day/type, with a chart) intents (TDD); added both as example
chips. (2) **Rewrote the build-along tutorial** `notebooks/02_building_the_system.ipynb` into a
comprehensive, course-style notebook (**44 cells**, executed, 6 plots): learning objectives,
prerequisites, TOC, background on the LLM-hallucination problem + the pattern + MQTT/pcap/attacks, a
data dictionary, parsing deep-dive, sensor/temperature exploration, the truth layer built from scratch,
both detectors explained *with plots*, a **metrics primer** (precision/recall/F1/AP/point-adjusted/CI),
the honest results interpretation, the interface + faithfulness gate with a live LLM demo, the apps, and
**three exercises** (including one that extends the truth layer with a `== pandas` assertion). 55 tests
green, verify 12/12, ruff clean.

**Discovery:** the "couldn't map that" fallback the user saw was simply missing intents — the router is
deterministic, so unmapped phrasings are *refused rather than guessed* (by design), which is why adding
the two intents fixed it cleanly and the answers stay exact.

## M3 follow-ups — seasonal value model + robust flow channel ✅ (tried, honest negative result)

The M3 note (§ above) flagged a *seasonal value model* and a *stronger flow channel* as tuning follow-ups
to lift the modest absolute AP (value-only 0.183, flow-only 0.136). Both were implemented **TDD** and
measured on the full real week — and both **fail to help**, which is the honest, reportable outcome.

- **`channels.seasonal_value_score`** — z-scores each sensor against a **time-of-day** baseline (mean/std
  within each hour-of-day) instead of a global one, to catch a reading that is normal for the day but wrong
  for its hour. **Result: it *hurts* — AP 0.183 → 0.099.** The data-centre temperature has no exploitable
  diurnal cycle for the attacks to violate, and the per-hour baseline is starved of samples (noisier std),
  so it *dilutes* the modification signal the global z-score catches cleanly. A clean case of *adding
  structure the data doesn't have makes things worse.*
- **`channels.flow_score(method="mahalanobis" | "combined")`** — a Ledoit-Wolf covariance (Mahalanobis)
  distance over the standardised flow features (the notebook-05 lesson: covariance beats axis-aligned
  Isolation-Forest splits for *cross-sectional* anomalies). **Result: a wash on AP — 0.136 → 0.125, CIs
  overlapping heavily** — though best-F1 nudges up (0.226 → 0.252). Not a significant improvement.

**Decision:** the new methods are kept as **tested, optional** functions (they document the attempt and are
reusable), but the **defaults stay global-z + Isolation Forest** — swapping in a detector that is worse or
within noise would be wrong. This is the same pattern the whole project keeps finding: on these small,
label-scarce datasets, *simple robust methods hold up and the honest negative result is the contribution.*
(+2 channel tests → **58 tests** green, ruff clean.)

## Deep models — first attempt, both channels ✅ (tried, honest negative result)

v3 had never tried a deep detector (unlike v1 and, now, v2). Two candidates ported from v1's `deep.py`
pattern, both TDD, both new (`src/deep.py`, +5 tests → 65 tests):

- **`deep.flow_lstm_ae_score`** — a multivariate LSTM autoencoder over 6-window (30-minute) sequences
  of flow features, replacing the flow channel's Isolation Forest with a sequence-aware reconstruction
  model — testing whether a *temporal* duplication pattern is visible to a sequence model where
  hand-aggregated window features wash it out. **Result: it loses — AP 0.086 vs 0.136 for Isolation
  Forest, best-F1 0.158 vs 0.226.** Per-type recall: 0.0 on modification (expected — this is the flow
  channel), 0.5 on duplication (same as the baseline), 0.2 on mixed (down from the baseline's typical
  level). 2,016 windows is enough data that this isn't purely a scale story the way v2's was; the more
  likely explanation is that duplication's signal (a replayed msgid, a rate spike) is already exactly
  what the hand-crafted `dup_msgid_ratio`/`msg_rate` features capture directly — there's no extra
  *temporal shape* for a sequence model to find beyond what the aggregates already see.
- **`deep.value_lstm_forecast_score`** — v1's `LSTMForecaster`, applied independently per sensor (16
  small models) and combined via max-|forecast-error z| across sensors, the same reduction the z-score
  value channel already uses. **Result: it loses, but narrowly — AP 0.166 vs 0.183 for the z-score,
  best-F1 0.318 vs 0.379.** This is the more interesting negative result of the two: it is a direct,
  cross-domain test of notebook 05's finding (from v1) that a *single-channel value fault* is a
  threshold/covariance problem, not a sequence problem, so a learned temporal model has no structural
  advantage over a marginal z-score for this specific fault geometry — and that is exactly what happened
  here, on a third, unrelated dataset. The gap is narrow (not a rout) because the forecaster still learns
  *some* useful structure (per-type recall 0.42/0.50/0.43 — close to the z-score's 0.46/0.50/0.43), just
  not enough to beat the simpler, purpose-fit baseline.

**Decision:** kept both as tested, permanent modules — not deleted — for the same reason as the M3
follow-ups above: the comparison itself, and the fact that the value-channel result specifically
corroborates a finding from a different dataset (v1), is the reportable contribution, independent of
either architecture "winning." Neither is wired into `experiment.py`'s default configs; both are
reproducible via `deep.flow_lstm_ae_score` / `deep.value_lstm_forecast_score` directly.

### How each model works, and why each result happened (methodology + mechanism)

*Written for direct use in the Methods/Discussion sections of the write-up.*

**`flow_lstm_ae_score` — architecture and training.** Six consecutive 5-minute flow-feature rows
(30 minutes of `msg_rate`/`iat_mean`/`iat_std`/`dup_msgid_ratio` context) are windowed into a sequence;
an LSTM encoder compresses it to an 8-dimensional latent vector, an LSTM decoder reconstructs the full
30-minute sequence, and the mean squared reconstruction error is the anomaly score. Training uses
non-overlapping 6-window blocks; scoring uses overlapping windows with per-timestep error averaging
(the same localisation approach used throughout the project), seeded and run for 100 epochs.

**Why it lost — the mechanism, not just the number.** A duplication attack manifests as an
**instantaneous** spike within a single 5-minute window (a burst in `msg_rate`, a jump in
`dup_msgid_ratio`), not as a shape that unfolds gradually across several windows. Isolation Forest
scores each window independently and fully credits an outlier window the moment it sees one. The
LSTM-AE, by construction, reconstructs a *sequence*: the overlapping-window averaging step that
localises the score also means a single spiky window's error gets blended with the errors of five
neighbouring, normal windows before it reaches its final per-window score — diluting precisely the
signal that made that window anomalous. The model is well-suited to finding a *slow drift* across
30 minutes; the attack it needed to find is not that kind of anomaly, so its structural advantage
(temporal context) is not just unused here, it is actively counter-productive to a scoring step
(averaging) that was designed for a different, gradual-anomaly use case.

**`value_lstm_forecast_score` — architecture and training.** Sixteen independent, small LSTM
forecasters, one per sensor, each conditioned on the previous 24 five-minute readings (2 hours of
history) to predict the next reading. Forecast error is standardised per sensor (so a naturally noisier
sensor cannot dominate the comparison), then the window-level score is the **maximum** standardised
error across all 16 sensors — the same "largest single-sensor deviation wins" reduction the existing
z-score `value_score` channel already uses, so the two are directly comparable.

**Why it lost, narrowly, and why the narrowness itself is the finding.** A modification attack is a
single reading that is simply implausible for that sensor's usual distribution — detecting it only
requires the question "is this value far from normal?", which a marginal z-score answers in one
comparison against the sensor's mean/std. The forecaster instead answers a strictly harder question —
"is this value far from what recent history would predict?" — which is the correct tool for *gradual
drift* or *context-dependent* deviations, neither of which describes this attack. The forecaster is not
a worse model in general; it is solving a more general problem than the specific fault requires, and
pays a complexity cost (more parameters, more can go wrong in training 16 separate small models on
short per-sensor series) without a matching benefit. Because the forecaster still learns *some*
genuinely useful per-sensor structure (its per-type recall — 0.42/0.50/0.43 — sits close to, not far
below, the z-score's 0.46/0.50/0.43), the result is a narrow loss rather than a rout, which is itself
evidence the mechanism above is the right explanation: the model isn't failing outright, it's solving
the wrong-shaped problem competently. **This is the same conclusion v1's notebook 05 reached on a
completely different dataset (Jena weather sensors) for the same reason (single-channel value faults
favour threshold/covariance methods over learned sequence models) — the fact that it reproduces on DAD,
a third and structurally unrelated domain, is stronger evidence for the general claim than either
result alone would be.**

---

## Standalone notebook — a real, previously undocumented faithfulness-gate gap ✅ (found, honest negative result)

Built `notebooks/standalone_full_pipeline.ipynb` — a single, self-contained notebook (no `from src
import` anywhere; every function ported line-by-line from `src/*.py` and checked against it) that
reproduces the whole v3 pipeline end to end on the real DAD data, then adds a further-experiments
section (§12: ADF stationarity, CUSUM change-point detection, PCA separability of the flow-feature
space, LOF/One-Class SVM as two more flow-channel candidates, a window-size sensitivity sweep,
ROC/confusion-matrix views, and a wider interface question battery — see below). Every number
reproduced there matches this document's and `MODELS.md`'s figures exactly, confirming the port is
faithful.

**While building the faithfulness-gate demo (§11), a real bug was found, not assumed.** The M5 note
above already flags that small integers ≤ 12 are deliberately exempted (clock hours, small octets).
Testing the gate directly against a *spoofed attacker IP* surfaced a second, more consequential gap
that was not previously documented:

- `faithfulness()` has no concept of "IP address" — it just finds every token matching
  `-?\d+(?:\.\d+)?` in the text. Applied to a dotted-quad, that regex greedily consumes `10.6.56.41`
  as **two** tokens, `10.6` and `56.41` (each match stops at the digit-run boundary, so the four octets
  collapse into two decimal-looking numbers), and then applies the **measurement-rounding tolerance**
  (2% relative + 0.1 absolute) that was tuned for temperatures like `29.28 → "29.3 °C"`.
- **Result, tested directly:** a last-octet swap (`10.6.56.41 → 10.6.56.99`) is **missed** — `56.99` is
  only 3.5% away from `56.41`, inside the tolerance band, so the gate reports `ok: True` for a
  genuinely different attacker IP. A near-value second-octet swap (`10.6 → 10.9`) is **also missed** for
  the same reason (a 0.3 gap against a ~0.31 tolerance). A third-octet swap or a wholesale unrelated IP
  (`192.168.1.1`) **is** caught, because the resulting decimal token lands far enough outside the
  tolerance band.
- **Why this matters more here than a wrong count would:** in a security domain, a wrong attacker IP
  does not just misreport a number — it misdirects an investigation toward the wrong host. This is the
  more operationally serious failure mode the M5 gate was never explicitly tested against.

**Decision:** left as a **documented limitation**, not silently patched, consistent with how this
project treats every other discovered gap (the small-integer exemption noted in M5, the "biggest
exporter" routing limitation found in v2's standalone notebook). A correct fix is straightforward to
scope for future work — match full dotted-quads as atomic tokens and compare them for exact string
equality, rather than reusing the numeric-rounding tolerance for text that only *looks* numeric — but
implementing it was out of scope for this pass; the honest finding is the contribution here. Reproduced
directly in `notebooks/standalone_full_pipeline.ipynb` §11, including a passing regression-style
assertion that documents the current (gap-having) behaviour rather than hiding it.

### §12 further experiments — headline results (all freshly computed, not copied from §M3/§deep-models)

- **ADF stationarity:** the hourly temperature series is **non-stationary** (p=0.53, cannot reject the
  unit-root null) despite STL showing only a mild (~0.5 °C) seasonal amplitude — relevant to any future
  forecasting attempt on this channel (v3 does not currently forecast temperature, unlike v2's SARIMA).
- **CUSUM change-point detection** on the hourly message rate (threshold=3, drift=0.5): every one of
  the 4 real attack-day boundaries has a detected change point within 2 hours of it — a cheap, coarse,
  **day-level** triage signal that broadly tracks the real schedule, but cannot localise to the
  5-minute windows the value/flow channels operate on, and several detected points are not
  attack-related (ordinary end-of-day traffic shifts also trigger it).
- **PCA of the flow-feature space** (2 components, 84% of variance): attack and normal window centroids
  are clearly offset but the point clouds overlap heavily — a visual explanation for why the flow
  channel's AP is modest (0.136) rather than near-1.0.
- **LOF and One-Class SVM** as two more flow-channel candidates (same features, same evaluation as
  Isolation Forest): LOF loses clearly (lower AP and F1 — a local-density method doesn't suit a
  global distributional shift). One-Class SVM is a near-tie on AP (within 0.001) with a **higher**
  best-F1 than Isolation Forest — a legitimate alternative, not adopted as the new default only because
  Isolation Forest is simpler (no kernel/nu tuning) and the F1 gap was not re-verified against a
  bootstrap CI in this pass.
- **Window-size sweep** (1/5/15 min): AP rises sharply with window size (value 0.10→0.18→0.42; flow
  0.03→0.14→0.17), but this is **not** simply "bigger windows detect better" — a longer window is more
  likely to contain at least one attack packet by construction (`window_labels` uses any-packet-in-window),
  so part of the AP gain is a labelling-coarseness artefact, not a detection improvement. The 5-minute
  default trades some of that apparent AP for finer localisation, matching the real publish cadence — a
  deliberate choice, not the AP-maximising one.
- **ROC-AUC vs AP:** ROC-AUC ranks the configs differently from AP (flow-only 0.594, barely above
  chance, vs its non-trivial AP of 0.136) because ROC-AUC integrates over the full false-positive-rate
  range, most of which is free true-negative credit under 21:1 imbalance — the textbook reason AP
  stayed the primary metric throughout this project rather than ROC-AUC or accuracy.

---

## Standalone notebook §13 — checked v3 against its own plan and against the source paper

Prompted by "v3 is inspired from the paper — is there anything that can be added, and what did it
miss": rather than guess, `notebooks/standalone_full_pipeline.ipynb` §13 checked two concrete things —
(a) whether `docs/PLAN.md`'s own architecture table was actually built, and (b) whether the source
paper's own descriptive analyses (protocol/TCP-flag/packet-size stats, distribution goodness-of-fit)
are used anywhere in v3 — and added two multivariate techniques requested explicitly as "beyond STL"
(Granger causality, DTW). Every result below is freshly computed against the real data, not asserted.

**§13.1 — the module gap (a real, verifiable finding).** Listing `src/*.py` programmatically against
`docs/PLAN.md`'s own architecture table: **10 of 17 planned modules were built** (`preprocess`,
`features`, `channels`, `fusion`, `evaluation`, `query_engine`, `interface`, `interface_parsing`,
`llm`, `deep` — `anomaly.py` was deliberately superseded by `channels.py`+`fusion.py`, a documented
design change, not a gap). **7 were never built:** `analysis`, `reasoning`, `rag_system`, `clustering`,
`forecasting`, `agent`, `main`. README.md's status line narrows scope to "M0–M5 core + M6 apps" without
ever crossing these off PLAN.md's table, so the plan document silently reads as if they are still
coming. `docs/PLAN.md` now carries a status note recording this. Three of the seven are partially
closed as notebook-only demonstrations (§13.6–13.8 below); `rag_system.py` and `agent.py` are left as
honestly-stated, un-closed gaps — v3's question set is narrow enough that a full retrieval/agent layer
would mostly retrieve the same handful of facts already inlined in `interface.py`'s `ATTACK_KB`, so
building one would be scope for its own sake.

**§13.2 — what the paper analysed that v3's channels don't use.** The paper reports a protocol split
(96.9% TCP / 3.4% UDP / 63.3% MQTT), TCP-flag-based detection of "atypical closures" during
interception, and flow-level byte counts — none of which `preprocess.py`/`channels.py` currently keep
(only `src_ip`, `src_port`, `topic`, `temperature` are extracted). Re-parsing with protocol, TCP flags
and packet size added:
- **Protocol split replicates the paper almost exactly:** 96.6% TCP / 3.4% UDP here vs the paper's
  96.9% / 3.4% — an independent validation of the pcap re-parse itself.
- **The TCP-flag "atypical closure" signal does NOT reproduce at packet level (honest negative):**
  every single labelled attack packet carries a plain `PA` (push-ack) flag — 100% of them — because a
  modification/duplication attack *is* a data packet, just a manipulated one, so its flags look
  identical to a normal PUBLISH. The reset-flag rate by day does not single out Tuesday (interception)
  or any attack day either — Sunday (clean) has a similar rate to Wed/Thu/Fri/Sat. Read as: the paper's
  claim most likely operates at flow/connection-level state transitions across a whole session, a
  coarser aggregation than a per-packet flag-frequency check reproduces — not a claim the paper is
  wrong.

**§13.3 — distribution goodness-of-fit (mirrors the paper's KS/Anderson-Darling analysis).** Fitting
normal/lognormal/gamma/Weibull to each of the 16 sensors' temperatures: **every sensor's best-fitting
distribution is still formally rejected by the KS test** (p < 0.05, all 16). Not a data-quality issue —
with n≈2000 per sensor the test has power to flag even small deviations, and temperature has a real
~24h diurnal cycle, so no time-invariant distribution can fit a series whose mean genuinely shifts by
time of day. This is precisely the reasoning that leads to STL/ARIMA (not a static distribution) as the
right model for the series — the paper's own conclusion, now independently re-derived rather than just
cited.

**§13.4 — Granger causality (requested, beyond-STL).** Tested a same-InRow sensor pair (131→132) and a
cross-InRow pair (131→231): **both show significant Granger causality at every lag (p ≪ 0.05), even
across different cooling units.** Read with the standard caveat applied at full force: this almost
certainly reflects a **shared external driver** (the data centre's ambient diurnal cycle moving every
sensor's baseline together — consistent with §2's 0.43 mean off-diagonal sensor correlation), not true
causal influence between units.

**§13.5 — Dynamic Time Warping (requested, beyond-STL), two experiments.** (a) *Sensor-pair shape
distance:* DTW agrees with Euclidean/Pearson on ranking (same-InRow far more alike than cross-InRow)
but narrows the gap between the two pairs (DTW ratio ~3× vs Euclidean's ~14×) — some of what Euclidean
reports as "difference" between cross-unit sensors is a small phase offset DTW's flexible alignment
absorbs, not a genuinely different shape. (b) *Daily-profile shape distance (baseline = Monday):*
**does NOT separate attack days from clean ones** — Sunday (0 attacks) is about as shape-different from
Monday as Friday/Saturday (192 attacks each). Extends the project's established "point fault, not shape
fault" finding (v1 notebook 05, replayed in v3 §9) down to a coarse, smoothed 24-point daily profile —
even relaxing "shape" this far still doesn't surface the attacks.

**§13.6 — forecasting (closes the `forecasting.py` gap, mirrors the paper's own ARIMA use).** A SARIMA
backtest (order (1,1,1)×(1,0,1,24), rolled forward 24h at a time) on the hourly-mean temperature: MAPE
0.61%. The **top forecast-error anomalies (z=4.11, 3.15, 2.76, 2.66) fall almost exactly on the paper's
documented Friday/Saturday "mixed" windows (6h and 14–16h)** — a genuine recovery of those windows from
forecasting alone. But it does **not** clearly flag the Wednesday modification window (all z-scores
under 2.2 in 03:00–07:00) — the same point-fault-vs-aggregate story as §13.5: averaging 16 sensors into
one hourly mean dilutes a single-sensor, single-reading fault below the forecast's noise floor. The
per-sensor z-score value channel (§5), which never aggregates across sensors, remains the right tool
for that attack type.

**§13.7 — a minimal reasoning layer (closes the `reasoning.py` gap).** `explain_window(ts)` reports
which single sensor (value channel) or flow feature (flow channel) moved furthest from its own mean at
a flagged window. Sharp and correct for a modification window (sensor 134, z=−4.28 — an obvious single
culprit); appropriately weak for "mixed" windows (top z-scores under 0.5, no single dominant feature) —
matching §5.2's per-type recall table exactly: the explanation is only as sharp as the underlying
attack is single-cause, which is the honest behaviour a reasoning layer should have, not a bug.

**§13.8 — day-type clustering (closes the `clustering.py` gap — an honest negative result).** K-means
(k=2) on day-level summary features (temperature mean/std, message-rate std — `n_attacks` deliberately
excluded to avoid leaking the label into its own evaluation) **does not recover the attack/clean
split**: it groups {Mon, Tue, Wed, Thu} together and {Fri, Sat, Sun} together, putting Sunday (0
attacks) with the two heaviest attack days and separating the real Wed/Thu attack days into the
"clean-looking" cluster. Cause: `msg_rate_std` trends upward across the week (22.8→65.8, roughly
Mon–Sat) for reasons unrelated to the attacks and dominates the standardised feature space. A clean
demonstration that generic day-level summary statistics do not substitute for the window-level,
attack-sensitive features (§4) the real detectors use.

**Overall:** one genuine replication (protocol split), five honest negative-or-partial results
(TCP-flag signal, distribution fitting, Granger's causation caveat, DTW's day-level result, SARIMA's
Wednesday miss, clustering's failure) — none silently dropped for not confirming a hypothesis, which is
the same standard applied throughout this document.
