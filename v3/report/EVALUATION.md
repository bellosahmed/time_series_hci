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
| value-only  | **0.181** | **0.376** | 0.325 | 0.444 | 0.46 | 0.50 | 0.43 |
| early-fusion| 0.151 | 0.304 | 0.212 | 0.533 | 0.50 | 1.00 | 0.52 |
| late-fusion | 0.168 | 0.306 | 0.214 | **0.533** | 0.46 | 1.00 | **0.53** |

**Trade-off (confirmed, exactly as hypothesised):** fusion **lifts recall** (0.44 → 0.53 — the union of
two specialists) but **cuts precision** (0.33 → 0.21 — an OR-rule sums both channels' false positives).

**Discovery — the honest, non-obvious finding: fusion is *not* a free win here.** On the aggregate,
threshold-free metric, **`value-only` is best (AP 0.181, F1 0.376)** — *neither* fusion beats it. The
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
| value-only  | 0.181 [0.135, 0.243] | 0.376 | 0.800 | 0.889 |
| early-fusion| 0.151 [0.111, 0.213] | 0.304 | 0.732 | **1.000** |
| late-fusion | 0.168 [0.121, 0.242] | 0.306 | 0.786 | **1.000** |

**Trade-off:** point-adjusted scoring is the standard event-level protocol but is **optimistic** (one
hit credits a whole segment) — so we report **both** window-level and point-adjusted, never just the
flattering one.

**Discovery — this pass overturned the earlier headline in two honest ways:**
1. **The value-only "win" is not significant.** The AP bootstrap CIs **overlap heavily** (value
   [0.135, 0.243] vs late-fusion [0.121, 0.242]), so we **cannot** claim value-only beats fusion on AP —
   the M3 first-pass conclusion was within noise. (Only bootstrap CIs revealed this; the point estimate
   alone was misleading.)
2. **At the event level, fusion is best where it matters:** both fusions reach **1.00 point-adjusted
   recall** (every attack *episode* caught) with F1 within 0.014 of value-only. So the honest verdict is
   *goal-dependent*: fusion for catching every episode; the simpler value channel for window-level
   precision — and no config is significantly best on AP. This is a stronger, more defensible result
   than the first pass, and a textbook case of why CIs + event-level metrics matter.

**Next:** walkthrough notebook (`notebooks/00_walkthrough`).

---

## Future work — supervised score-combiner ✅ (a negative result worth reporting)

**Did:** added `fusion.supervised_scores` — a logistic regression over the two rank-normalised channel
scores, evaluated with **day-blocked** leave-one-day-out CV (an attack episode never spans train and
test). TDD (1 test), added as a fifth config. 48 tests green, ruff clean.

**Result:** the supervised combiner is the **worst** config — AP **0.101** [0.076, 0.141], below every
unsupervised one (flow 0.136, late-fusion 0.168, value 0.181).

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
