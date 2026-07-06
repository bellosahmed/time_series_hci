# Work Log — time_series_hci

A running log of what we built, tried, and decided. Newest entries at the top.
(Complements `v1/EVALUATION.md`, `v1/MODELS.md`, and `CLAUDE.md`.)

---

## 2026-07-06

### v3 started: real DAD IoT-network dataset fetched + verified, plan written
Created **`code/v3/`** as a third domain (data-center IoT/MQTT network security) and grounded it in a
**real** dataset before planning.

- **Got the real dataset.** DAD (Vigoya et al. 2020, `github.com/dad-repository/dad`) → `code/data/dad/`
  (gitignored, 22 files, ~18 MB) via a reproducible `v3/tools/get_dad_dataset.sh`. **Verified genuine,
  not fabricated:** the per-day labelled-attack counts reproduce the paper's schedule exactly (Mon 0,
  Tue 0, Wed 104, Thu 56, Fri 192, Sat 192, Sun 0 = **544** attack packets / ~101.6k; attacker IP
  `10.6.56.41`; three attack types).
- **Key finding that shaped the plan:** unlike v1/v2's ready sensor CSVs, DAD is **`.pcap` network
  captures** — temperature readings sit inside MQTT payloads (need `scapy` to parse) and the anomalies
  are **network attacks** on the message flow, shipped with **real, human-annotated labels**. That last
  point is the draw: it **fixes v1's biggest limitation** (synthetic-only anomaly labels), enabling
  genuine precision/recall/F1 on real anomalies.
- **Framing decision (documented, revisitable):** v3 = **sensor + network** time-series anomaly system
  in the v1 mould, not sensor-only — because network attacks mostly don't move the temperature value,
  so flow features (msgid duplication, insertion bursts, attacker-IP share) are essential to catch them.
- **Plan written** — `v3/docs/PLAN.md` (v1-style: module map, pcap→labelled-window preprocessing,
  detectors + real-label evaluation, domain-adapted NL intents, milestones, honest limitations: the
  authors call the traffic "semi-synthetic", single data center, 7 days, 0.54% class imbalance).
  `v3/` mirrors the v1 layout (src/tests/notebooks/apps/report/docs/tools); nothing under `src/` built
  yet. **M0 (acquire + verify) done; M1 (preprocess) is next.**
- **Headline experiment designed + blueprint written.** Because the three attacks are disjoint in
  signature (modification→value, duplication/interception→flow), M3 is framed as a **two-channel +
  fusion** comparison (flow-only / value-only / early / late fusion) graded **per attack type** on the
  real labels — the honest expected finding is late-fusion wins overall recall (complementary
  specialists) but trades precision (OR sums false positives), and early-fusion can dilute the sharp
  single-channel signal. Added to `v3/docs/PLAN.md` (§6) and a full `v3/docs/ARCHITECTURE.md` (file
  tree, data-flow diagram, message-frame schema, per-module signatures + key algorithm sketches,
  test plan, build order).
- **M1 started — `preprocess.py` built + green (TDD).** Installed `scapy` (2.7.0). Spike on the real
  Monday pcap resolved the two unknowns: MQTT PUBLISH payload is a **plain ASCII float** (`b"15.54"`)
  and **`frame.number` == the 1-based packet index** (14,431 packets == 14,431 label rows), so the
  label join is exact. Wrote `v3/src/preprocess.py` (`parse_pcap`/`load_day`/`load_week` → the
  datetime-indexed MESSAGES frame; `clientid`/`msgid`/`label` taken from the authoritative labels CSV;
  targeted scapy imports to silence its optional-layer auto-load noise). 3 tests lock the real totals
  (101,583 packets · 544 attacks · per-day `[0,0,104,56,192,192,0]` · 32,288 temperature readings ·
  attacker IP `10.6.56.41`); ruff clean. **Discovery:** `clientid` encodes InRow-unit+probe (131 =
  InRow 13 / probe 1), and capture timestamps are **UTC** (day boundaries tz-shifted ~2 h) — attack-day
  grouping keyed on the filename `day`, not the UTC index, to stay correct.
- **M1 finished — `features.py` built + green (TDD).** VALUE channel (`sensor_series`, temperature per
  sensor) + FLOW channel (`flow_features`: `msg_rate`, `publish_rate`, `iat_mean/std`,
  `dup_msgid_ratio`) + weak `window_labels`. 9 tests green (added a session `week` fixture in
  `conftest.py` so the 7 pcaps parse once), ruff clean. **Trade-off (honesty):** flow features
  **deliberately exclude the attacker IP** (`attacker_share`/`unique_src`) — since `10.6.56.41` appears
  only during attacks, using it would be label leakage and make detection trivially perfect; behavioural
  aggregates keep the eventual numbers honest. **Discovery:** a first-written test failed (assumed >50%
  of 1-min windows have traffic; real = 39.3%) — not a bug but a finding: traffic is **bursty on a
  5-minute publish cadence** (1-min windows 39.3% occupied vs 5-min 100%), so the flow window is set to
  5 min (resolves the window-size open question from the data). Fixed the test to assert exact
  packet-count invariants instead. Documented in `v3/report/EVALUATION.md`. **M1 done; M2 (query
  engine) next.**
- **M2 done — `query_engine.py` (truth layer) + `verify_results.py` gate (TDD).** Exact pandas ops
  (`count_attacks`, `attacks_by_day`, `attack_breakdown`, `attacker_ips`, `temperature_stat`,
  `message_rate`, `busiest_hour`). 14 tests green; **verify 12/12** (incl. an independent raw-label-CSV
  path confirming 544, plus temperature stats == pandas); ruff clean. **Trade-off:** the engine is
  deliberately dumb-and-exact (one pandas expression per answer) — caps expressiveness but guarantees
  verifiability. **Discovery:** 384/544 (**71%**) of labelled attacks are only typeable as "mixed"
  (binary labels + Fri/Sat interleave all three types), so clean per-type scoring is possible for Wed
  (modification) and Thu (duplication) only — a constraint M3's per-type evaluation must state.
  Documented in `v3/report/EVALUATION.md`. **M3 (detection + fusion) next.**
- **M3 done — two-channel + fusion detector, graded on REAL labels (TDD, the headline).**
  `channels.py` (value = temperature max|z|; flow = Isolation Forest), `fusion.py` (early/late),
  `evaluation.py` (AP / best-F1 / per-type recall), `experiment.py`. 22 tests green, ruff clean.
  **Result (2016 windows, 90 attack windows):** the per-type table *proves* complementarity — flow-only
  **0.00** recall on modification vs **1.00** on duplication; value the opposite tilt. Fusion lifts
  recall **0.44→0.53** but cuts precision **0.33→0.21** (OR sums false positives). **Honest headline
  finding:** fusion does **not** beat `value-only` on AP (0.181) / F1 (0.376) — because the flow channel
  is weak here (only 4 clean duplication windows), fusing it adds more false alarms than true positives.
  *Discovery: fusion pays off only when both channels are strong; here it trades precision for recall
  and loses on aggregate — refining the plan's hedged hypothesis into a measured result.* Numbers kept
  honest (attacker IP deliberately excluded → no leakage; untuned detectors). Full table + reasoning in
  `v3/report/EVALUATION.md`. **M4 (NL interface) next.**
- **M4 done — NL interface (TDD).** `interface_parsing.py` (deterministic intent/day/attack-type/stat)
  + `interface.py` (`Interface.ask` → trust-labelled `Answer`: 🔢 computed / 📚 context / 🧠 inferred).
  Intents: count · attacker · breakdown · temperature · traffic · when · summary. 33 tests green, ruff
  clean, verify 12/12. Every number comes from the query engine; unknown questions refused not faked.
  **Bug found+fixed (TDD):** `\battack\b` missed the plural "attacks" → matcher now tolerates trailing
  `s`. **Discovery:** the "busiest hour" answer (Fri 04:00 UTC, 824 msgs) coincides with a mixed-attack
  window — duplication inflates traffic, so a plain traffic question already surfaces an attack (a live
  cross-check that M3's flow signal is real). Documented in `v3/report/EVALUATION.md`. **M5 (LLM
  explanation + faithfulness gate) next.**
- **M5 done — LLM explanation + faithfulness gate (TDD).** Ported `llm.py` (Ollama `explain` +
  `faithfulness` number-guard) and wired `interface._phrase`: rewords computed facts only if the gate
  finds no invented number, else the exact template. 39 tests green, verify 12/12, ruff clean. Gate
  verified on DAD facts (fabricated 999→rejected; 29.28→29.3 rounding allowed; IP 10.6.56.41 preserved).
  **Trade-off:** simple number-matching guards decimals/counts/years/larger IP octets but tolerates
  small ints ≤12 — same honest residual as v1, now noted for IPs. **Discovery:** the gate is what lets a
  cheap 3B model be used *safely* (numbers exact upstream + checked downstream → the model only
  re-phrases), so **Verifiable NL Analytics transfers to a third domain (network security)**, now
  guarding IP addresses too. Ollama not installed here → live prose deferred (gate+fallback fully tested
  via monkeypatch), as in v1. **Core pipeline M0–M5 complete; M6 (apps/notebooks/report) remains.**
- **M6 (apps + report) done — v3 is runnable and written up.** CLI `apps/chat.py`, Flask web app
  `apps/web/` (create_app factory + plain HTML+JS page, no build), `reproduce.sh`, `report/REPORT.md`
  (draft with the real M3 table), README run section. TDD app layer (5 tests). **44 tests green, verify
  12/12, ruff clean.** Smoke-tested the live server (not just test_client): `GET /` 200 and `/api/ask`
  returns real answers ("104 attacks on Wednesday", "attacker 10.6.56.41") end-to-end on the full
  dataset. **Trade-off:** plain HTML+JS web UI vs v1's vendored React — zero build, easy to run/test,
  proportionate for a prototype. **Discovery:** driving the real server confirmed the whole pcap→
  frame→query→interface→JSON chain works behind Flask. **Remaining M6:** notebooks, `MODELS.md`, and
  the M3 detector refinements. Documented in `v3/report/EVALUATION.md`.
- **M6 finished — rigor pass + notebook + models doc.** Added `evaluation.point_adjust`,
  `best_f1_point_adjusted`, `bootstrap_ci` (TDD, 3 tests) and wired them into the experiment; wrote
  `report/MODELS.md`, an **executed** `notebooks/00_walkthrough.ipynb` (nbconvert, exit 0), and
  `requirements.txt`. **47 tests green, verify 12/12, ruff clean.** **Discovery (the rigor pass changed
  the story, honestly):** the AP bootstrap CIs **overlap** (value [0.135,0.243] vs late-fusion
  [0.121,0.242]) → the earlier "value-only beats fusion on AP" is **not statistically significant**;
  and at the event level both fusions reach **1.00 point-adjusted recall**. So the verdict is now
  goal-dependent (fusion for catching every episode; the simpler value channel for window precision),
  with no config significantly best on AP — a stronger, more defensible result than the first pass, and
  a textbook case for why CIs + event-level metrics matter. **Trade-off:** point-adjusted scoring is
  optimistic, so both window-level and event-level are always reported. Documented in
  `v3/report/EVALUATION.md` + `MODELS.md`. **v3 M0–M6 complete.**
- **Future work + comprehensive notebook + web-UI docs.** (1) **Supervised score-combiner**
  (`fusion.supervised_scores` — logistic regression over the two channel scores, **day-blocked** LOGO CV;
  TDD, added as a 5th config). **Honest negative result:** it is the *worst* config (AP 0.101) — attack
  *type* is confounded with *day*, so under day-blocked CV the learned channel-weighting doesn't transfer
  and it overfits; a random split would have leaked and looked great. Lesson: with scarce,
  episode-clustered labels a simple unsupervised OR generalises better than a learned fusion. Window
  (5 min, from cadence) and contamination tuning judged low-value given the weak flow signal. 48 tests
  green, ruff clean. (2) **Comprehensive EDA notebook** `notebooks/01_exploratory_analysis.ipynb` (34
  cells, executed, 10 plots): data dictionary, quality, temperature distributions, time series, **STL
  trend/seasonality**, **autocorrelation**, sensor-correlation heatmap, traffic, attack timeline, the
  two channels, detection results with CIs, NL demo. (3) **Web-UI run+what-it-does** documented in
  `v3/README.md`. Documented in `v3/report/EVALUATION.md` + `MODELS.md`.
- **Richer web chat (charts + AI toggle) + build-along tutorial notebook (TDD).** (1) Interface now
  attaches a presentation-only `chart` payload (bar of attacks/day or /type; hourly line for
  temperature/traffic); `index.html` renders it as inline SVG and gained an **"Explain with AI" toggle**
  + status lamp fed by a new `/api/models` route — matching v1. **Correction (honesty):** a local
  Ollama **is** running here (`llama3.2:3b`); my earlier "not installed" note was wrong — verified live
  the model rewords ("...104 attack packets on Wednesday", `used_llm=True`, 104 preserved by the gate),
  so the AI-explain experience now works, it was just the toggle defaulting off. (2) New **tutorial
  notebook** `notebooks/02_building_the_system.ipynb` (executed, 22 cells) that builds the system from
  scratch and explains each step (packets→table→truth layer→detectors→English interface+gate) for
  others to learn. **53 tests green, verify 12/12, ruff clean.** **Trade-off:** charts are
  presentation-only — the query engine still owns every number, so a chart can't introduce an
  uncomputed figure. Documented in `v3/report/EVALUATION.md`.
- **More chat intents + expanded course-style tutorial.** The user hit "couldn't map that" for
  *"show all sensors"* and *"show me the anomalies"* — added **`sensors`** (lists the 16 client ids +
  per-sensor readings chart) and **`anomalies`** (544 labelled attacks by day/type + chart) intents
  (TDD), and as example chips. **Discovery:** the fallback was just missing intents — the router refuses
  rather than guesses (by design), so adding intents fixes it while keeping answers exact. Rewrote
  `notebooks/02_building_the_system.ipynb` into a full course-style tutorial (44 cells, executed, 6
  plots): objectives, TOC, background (LLM hallucination, MQTT/pcap/attacks), data dictionary, parsing
  deep-dive, exploration, truth layer from scratch, both detectors with plots, a metrics primer, honest
  results, interface + gate + live LLM, apps, and 3 exercises. **55 tests green, verify 12/12, ruff
  clean.**
- **Data quality (missing data + imbalance) + on-prem Open WebUI spec notebook.** (A) Audited DAD:
  **0** missing temperatures on PUBLISH packets; only **0.6%** value-grid gaps → added
  `sensor_series(fill=True)` to **time-interpolate the value channel** (TDD), while **deliberately
  keeping flow-channel gaps** (missing messages = the interception signal — filling there would erase an
  attack). Effect on detection is within the bootstrap CIs (value AP 0.181→0.183) — refreshed
  `MODELS.md` + re-executed the 3 notebooks. Class **imbalance is real** (packet 186:1, window 21:1;
  duplication only 4 windows) — documented in new `report/DATA_QUALITY.md`; handled by imbalance-aware
  metrics (AP, point-adjusted, per-type, bootstrap CIs), **not** resampling (which would falsify a
  security dataset). (B) Wrote `tools/build_webui_spec_notebook.py` → generates
  `v3/local_webui_dataset_spec.ipynb` (valid v4, **35 cells**): a fully **offline/on-prem** spec for
  local Open WebUI + Ollama + Docker + **RTX 5070 GPU pass-through** — host health checks (py3.12/SSL,
  nvidia-smi, docker nvidia runtime, Ollama), ASCII architecture + data-flow, annotated `docker run`
  (`--gpus all`, `3000:8080`, read-only dataset mount), generated `docker-compose.yaml` + `.env`,
  offline/telemetry-off env, local-dataset RAG wiring, in-container GPU verification, privacy boundaries,
  troubleshooting. Grounded in this host's real values; **no external image assets** (offline). **56
  tests green, verify 12/12, ruff clean.**
- **Notebooks restructured to the v1/v2 in-depth style (5 notebooks, all executed, 0 errors).** Replaced
  the uneven set with: `00_code_walkthrough` (module-by-module tour, 21 cells), `01_data_exploration_
  and_visualization` (full EDA, 34 cells, 10 plots), `02_methods_and_evaluation` (parsing correctness +
  missingness/imbalance + the two detectors + fusion + metrics-explained + honest findings + trust gate,
  21 cells), `03_interface_and_explanation` (intents, trust labels, charts, LLM + faithfulness gate, 17
  cells), `04_build_along_tutorial` (from-scratch teaching walkthrough, 44 cells). 137 cells total,
  matching v1's depth; each re-executed after the value-channel gap-fill so embedded numbers are current.
- **Cross-version final report** in **`code/final_report/`** (5 docs, ~4,200 words): `README.md`
  (executive summary + **master v1/v2/v3 comparison table** + 3 cross-cutting findings),
  `01_systems_and_data.md` (the three domains + shared 5-layer architecture),
  `02_techniques_and_rationale.md` (**why-this-not-that** tables per layer — truth layer vs LLM-computes,
  IsolationForest vs deep, point-adjusted/bootstrap/day-blocked-CV vs naïve, real vs synthetic labels,
  don't-resample-imbalance, scapy vs tshark, etc.), `03_results_and_findings.md` (consolidated real
  numbers per version + the cross-cutting findings), `04_limitations_and_future_work.md` (per-version +
  shared limitations, prioritised roadmap). **Headline thesis:** one domain-agnostic architecture across
  weather/trade/network-security; **honest evaluation (real labels, event-level metrics, CIs, blocked CV)
  matters more than model complexity** — deep/learned methods lost or tied simpler ones in all three
  (v1 no single best; v2 AE 0.21 < IF 0.36; v3 supervised combiner worst). Grounded in each version's own
  `report/` + `verify_results` numbers.

## 2026-07-06

### Instructor-review fixes: report references + counts, and offline conversational memory
Acted on an examiner-style review of v1. **Tests 175 → 182 passing (1 skipped); verify 29/29; ruff +
mypy clean.** Scope note: the user de-scoped the **HCI user study** and all **GitHub** work (CI/tags/
credibility) — both dropped from the priorities/memory; credibility goes in the report, not the repo.

- **Report references were a placeholder — the single most damaging fixable defect.** `report/REPORT.md`
  had `## References` as a literal `[ADD]` stub (and a truncated word) with **zero in-text citations**.
  An MSc report with no reference list risks an automatic cap regardless of the engineering. Fixed:
  rewrote the Literature Review with proper in-text author–date citations grounded in the methods the
  system actually implements (LOF, One-Class SVM, Isolation Forest, Matrix Profile, STL, ARIMA, LSTM/TCN
  detectors, RAG, TF-IDF/BM25/LSA/SBERT, plus a hallucination-survey citation that motivates the whole
  faithfulness design), and added a complete alphabetised Harvard reference list. **Why it happened:**
  the draft was auto-generated from the code and left citations for the author to fill; they never were.
  **Trade-off/honesty:** the canonical method papers are cited confidently; the note to confirm exact
  formatting against the department style guide is kept (legitimate, not a placeholder), and I did *not*
  fabricate precise page numbers I was unsure of.
- **Stale numbers in the report contradicted the artefact.** The report cited **"87 tests / 21/21"** in
  five places (Results, Testing, Appendix A) while the suite is now **179 / 29/29** — a submission whose
  own numbers disagree with the code invites "which is true?". Synced every count. **Why:** the report
  was written at an earlier commit and never re-synced when the suite grew; the verification harness is
  exactly the tripwire for this, so the report must track it. Also refreshed the same stale counts
  across `docs/` (the historical "state after this work" snapshots kept, each annotated with current
  gates) and `CLAUDE.md`.
- **Fix #4 — conversational memory now works with the LLM OFF (the real functional bug).** Previously an
  elliptical follow-up (*"what about 2014?"*) was only resolved via `llm.rewrite_followup`, gated on the
  LLM being on; in the default offline mode the chat silently forgot context and returned "couldn't map
  that". Implemented **`interface_parsing.resolve_followup()`** (TDD: 4 tests written first, watched fail
  — the end-to-end one reproduced the exact `intent=None` bug — then implemented): it carries the prior
  turn's intent + sensor and swaps in the new year/period/day/sensor, composing a canonical standalone
  question the deterministic parser routes ("hottest in 2013" + "what about 2014?" → "max T (degC) in
  2014"). **Design choice/trade-off:** the LLM rewrite stays the *first* choice when available (richer,
  and it preserves the existing tested contract), with the deterministic resolver as the offline path
  **and** as a backup when an LLM rewrite is unroutable — so this is a strict improvement with no
  regression, rather than the plan's "deterministic-first" ordering which would have broken the
  LLM-rewrite test. Fully deterministic: it only rewrites wording; every number is still computed by the
  query engine. See `docs/CHAT_MEMORY_PLAN.md` (Phase 1 marked done; Phases 2–4 still open).
- **Discovery:** the existing `resolve_sensors` substring behaviour means "hottest" already yields the
  temperature sensor (via the "hot" alias), so a bare-year follow-up correctly carries `T (degC)` even
  when the previous question named no sensor explicitly — the deterministic carry-forward needed no
  special-casing for the most common phrasing.
- **Follow-on: scope carried across an intent swap.** Running the 5-turn demo surfaced an edge case —
  *"and the coldest?"* answered the coldest reading **ever** (−22.65 °C, 2009) instead of within the
  year under discussion. Cause: "cold**est**" supplies both an intent (`min`) and, via the "cold"
  alias, a sensor, so the question looked self-contained and the "needs context" gate never attempted
  carry-forward. **Why the fix is a *cue*, not a looser gate:** a fresh question like *"what was the
  coldest?"* also has intent+sensor+no-period and must **not** inherit a stale year, so the
  discriminator is an explicit continuation cue. Added `is_followup_cue` ("and…", "what about…") and
  let the resolver run for cue questions too; now *"and the coldest?"* → *"min T (degC) in 2015"*
  (−6.18 °C) while *"what was the coldest?"* stays whole-record. TDD: 3 tests (cue detector, cue
  carries scope end-to-end, non-cue stays fresh). **Trade-off:** genuinely ambiguous phrasing is
  resolved by leaning on the cue words rather than guessing — a fresh non-cue question is never
  silently scoped to a prior year.

## 2026-07-05

### Web UI rebuilt in React (vendored, no build) + chart fixes: compare chart + day/month toggle
Rewrote the browser chat (`apps/web/`) from vanilla JS to a **React app**, and fixed two chart gaps the
user hit. **Tests 173 → 175 passing (1 skipped); verify 29/29; ruff + mypy clean; JS unit suite green.**

- **React, but vendored + build-free.** React + ReactDOM (production UMD) and `htm` (JSX-like tagged
  templates, ~1 KB, no Babel) live in `apps/web/vendor/` and are served by Flask alongside `app.js`.
  A thin `h()` shim lets templates use plain HTML `class`/`for` (mapped to `className`/`htmlFor`).
  **Trade-off:** a full Vite/webpack build would give HMR + tree-shaking, but it drags a `node_modules`
  toolchain and a `npm run build` step into a single local page and breaks the "run `app.py`, done"
  story. Vendoring keeps the app **fully offline, zero-build, one-command** — matching the project's
  local, self-contained ethos — at the cost of no bundler niceties (irrelevant at this size).
  **Discovery:** the unit-tested render core (`chart.js` `drawChart`/`legendHtml`) ported over untouched
  — React just calls it via `dangerouslySetInnerHTML`, so the migration cost was UI plumbing only, and
  the Node behavioural tests still guard the SVG output.
- **"Graph doesn't show" — root cause & fix.** The backend only attached a chart for
  max/min/mean/trend/threshold/anomaly and returned **none for `compare`** — and "Was 2013 hotter than
  2014?" (an example chip) is a `compare`, so it showed no graph. Added `_compare_chart`: it overlays
  the two years' **month-of-year profiles** (both mapped to a synthetic shared year, Jan–Dec axis,
  `monthaxis` month-only labels) as two lines (blue vs amber) with a naming legend. `chart.js` extended
  additively with an optional `series2` + compare legend (existing tests untouched, new ones added).
  **Discovery:** a year-on-year overlay is a *better* comparison visual than the single line the other
  intents get — the previously-missing chart is now the most informative one.
- **Day-by-day / month-by-month toggle.** `interface._chart_for` now aggregates the scoped series into
  aligned **daily** and **monthly** means (`views`), and the React chart shows a Day/Month segmented
  switch. Verified in-browser: "Average temperature in 2013" → 365-pt daily line ↔ 12-pt monthly line.
  **Trade-off:** the old fixed ~220-pt raw down-sample was jagged and unaligned to calendar units;
  resampling to whole days/months is cleaner and calendar-honest but hides sub-daily detail — so a
  sub-2-day scope (e.g. a single day report) keeps the **raw hourly** series and hides the toggle.
- **Verified end-to-end** in a real browser (headless Chromium via Playwright): React mounts (5 sections,
  no `class`/`for` warnings), no console/page errors bar the harmless `favicon.ico` 404, the toggle
  swaps resolutions, and the compare chart draws both lines. Light + dark screenshots both clean.
- A11y unchanged in substance but moved into `app.js` (React renders the markup); `test_apps.py` updated
  to assert `role="log"`/`aria-live`/input `aria-label` in `app.js`, that `index.html` loads
  `app.js`+`chart.js`+vendored React, and that every vendored asset is served locally (no CDN).

### Held-out retrieval test + k-justification (pre-empting examiner feedback)
Closed the two cheap evaluation gaps a supervisor would raise. **Tests 169 → 173; verify 29/29; ruff +
mypy clean.**
- **Held-out paraphrase set** (`retrieval_eval.HELDOUT_GROUND_TRUTH`, n=6) — answers "did you teach to
  the test?". Every descriptive word is **verified absent** from the enriched KB (sweltering, nippy,
  oppressive, sultry, blustery, gale, buffeting, weighty, compressed…). Result: keyword methods collapse
  to **MRR 0.250** (only generic-word luck), the **dense encoder genuinely generalises to 0.556** — real
  but honestly imperfect. This is the fair generalisation number. Tests guard it (keyword weak; dense >
  keyword). Documented in `docs/RETRIEVAL_EVALUATION.md`.
- **k-justification** — new `clustering.kmeans_selection(features, k_range)` returns inertia (elbow) +
  silhouette per k; plotted in notebook 02. **Honest finding:** silhouette actually peaks at **k=2**
  (0.287); **k=4** (0.173) is a *deliberate interpretability choice* (four seasonal day-types) at a
  modest silhouette cost, with the inertia elbow bending ~k=3–4. Justified, not assumed. Tested.

### Notebooks refreshed to match current v1 (00 + 03), re-executed clean
Updated the existing notebooks to reflect the modules/features added this cycle, and fixed a stale
reference. All executed via nbconvert (Restart & Run All), 0 errors, outputs saved.
- **`00_code_walkthrough`** — added the two new modules (**`agent`**, **`retrieval_eval`**), extended the
  anomaly section (**LOF · One-Class SVM · Matrix Profile**), showed all four retrievers
  (TF-IDF/BM25/LSA/**dense**), added the **day-type** interface example and an **agentic** multi-step
  demo, renumbered sections (now 14), and **fixed the stale `08_model_comparison` recap link** →
  `04_deep_models_comparison`.
- **`03_interface_reasoning`** — refreshed the "no LLM yet" intro, added **threshold** + **day-type**
  questions, added an **agentic** subsection, and removed a stray incomplete comment.
- **`02_methods_and_evaluation`** — broadened the anomaly section to the full detector set (z-score,
  Isolation Forest, **LOF**, **One-Class SVM**, **Matrix Profile**) and added a **retrieval + IR-metrics**
  section (TF-IDF vs LSA: P@3/R@3/MRR/MAP/NDCG).
- **`01`** (data viz) and **`04`** (deep models, GPU) re-executed against current code — 0 errors.
- **All 5 notebooks now run top-to-bottom clean (0 errors) on current v1.**
- (An HCI-study *notebook* was drafted then removed — the study belongs in `report/HCI_USER_STUDY.md`,
  not a notebook.)

### v1 review fixes (clustering, coverage) + KB enrichment
Acted on a supervisor-style review of the v1 folder. **Tests 156 → 169; coverage 88% → 92%; verify
27/27 → 29/29; ruff + mypy clean.**

- **Clustering wired into the HCI interface** — it was implemented but only reachable from `main.py`.
  Added a **`daytype` intent** ("what kind of day was 4 July 2015?" → *"a 'warm' day…"*): lazy K-means
  day-types (`clustering.day_type_labels` maps clusters to cold/cool/mild/warm by mean temp),
  `interface` branch, tests (`test_clustering.py` + interface test), and 2 verify checks (summer day is
  a warmer type than a winter day; interface exposes it). **verify → 29/29.**
- **Coverage gaps closed** — `clustering.py` 0 → 100%, `preprocess.py` 62 → 100% (new
  `test_preprocess.py`: load/clean/sentinels/align/impute + error paths), `main.py` 0 → 63% (new
  `test_main.py` end-to-end smoke on a tiny generated Jena-format CSV). Overall **92%**.
- **KB enrichment (acting on the retrieval evaluation)** — rewrote the 14 one-line `SENSOR_DOCS` with
  everyday synonyms ("muggy/sticky/damp", "windy/breezy/gusty", "heavy/thin air"). **Result flips the
  earlier conclusion:** paraphrase MRR jumped for all retrievers (TF-IDF 0.267 → **1.000**, BM25 0.067
  → 1.000, LSA 0.267 → 1.000, Dense 0.500 → 0.867) — cheap literal **TF-IDF is now perfect and ahead of
  the dense encoder**. Takeaway: for a small curated KB, **document quality dominates retriever choice**.
  Honest caveat noted (enrichment vocab overlaps the paraphrase probes). Updated the two tests that
  encoded the pre-enrichment finding. Detail: `docs/RETRIEVAL_EVALUATION.md`.

### Dense + BM25 retrievers, and Matrix Profile detector (follow-through items 6–8)
Pulled three more COP509 techniques in. **Tests 149 → 156; verify 27/27; ruff + mypy clean.** Retriever
numbers in `v1/docs/RETRIEVAL_EVALUATION.md`; detector note in `report/MODELS.md`.

- **(6) Dense sentence-transformer retriever** — `rag_system.sentence_transformer_embedder` +
  `dense_retriever()` (module's recommended `all-MiniLM-L6-v2`, via the existing `EmbeddingRetriever`,
  with automatic TF-IDF fallback). **This closes the loop on the earlier LSA finding with numbers:**
  on standard queries dense lifts MRR 0.900 → **0.933** (R@3 → 1.0, catching the "how hot" miss); on
  out-of-vocabulary **paraphrases** it ~doubles TF-IDF/LSA (MRR 0.267 → **0.500**) and triples BM25 —
  measured proof that dense embeddings, not LSA, fix vocabulary mismatch. Honest: still only 0.500 on
  paraphrases (not a silver bullet).
- **(7) BM25 retriever** — `rag_system.BM25Retriever` (Okapi, `rank_bm25`): the probabilistic sparse
  baseline. Ties TF-IDF on standard queries; trails on paraphrases (MRR 0.067). **Fairness fix:** BM25
  now removes English stop words like TF-IDF does — diagnosing the low paraphrase score showed BM25 was
  matching the stop-word "it" and returning artefacts; with that removed the comparison is
  apples-to-apples and BM25 correctly returns nothing when only stop words match.
- **(8) Matrix Profile detector** — `anomaly.matrix_profile_anomalies` (discords, via `stumpy`).
  **Honest caveat documented:** on injected *point* spikes it scores P/R/F1 = 0.04/0.12/0.06 — expected,
  because a spike isn't a discord; it detects anomalous *patterns*, not points. (LOF + One-Class SVM
  already existed, so Matrix Profile was the genuine new detector.)
- Optional deps grouped under `[project.optional-dependencies] advanced` (rank-bm25, sentence-transformers,
  stumpy). Kept OUT of `verify_results.py` (stays dependency-light for CI); dense/BM25/MP tests skip
  gracefully when the package/model is absent.

### Retrieval evaluation + LSA retriever (from the COP509 IR material)
Reviewed the COP509 (NLP/IR) lecture slides and pulled the two most useful techniques into the
project. Full write-up + numbers in `v1/docs/RETRIEVAL_EVALUATION.md`. **Tests 140 → 149; verify
25/25 → 27/27; ruff + mypy clean.**

- **Gap closed:** anomaly detection and the truth layer were rigorously evaluated, but the **RAG
  retriever** was never *quantitatively* evaluated. Added the module's **IR Evaluation Measures**.
- **`src/retrieval_eval.py`** — `Precision@k / Recall@k / MRR / MAP / NDCG@k` + `evaluate_retriever`,
  over a 10-query hand-labelled ground-truth set (query → relevant sensor doc; relevance judged by us,
  not the system — the standard IR principle).
- **`rag_system.LSARetriever`** — the taught **LSI/LSA** method (TF-IDF → truncated SVD, 13 latent
  dims → cosine in latent space). Fully local (sklearn only), deterministic. Drop-in with `Retriever`.
- **Result (mean over 10 queries):** TF-IDF and LSA **tie** at MRR/MAP **0.900**, R@3 0.900. On this
  small, low-redundancy KB LSA gives no measurable gain (SVD needs term co-occurrence it doesn't have).
  The one shared miss — *"how hot does it get"* → nothing, because "hot" isn't in the KB — is the
  classic **vocabulary-mismatch** limitation; paraphrases ("muggy", "breezy") confirm it. **Honest
  finding:** semantic gain needs true embeddings (the `EmbeddingRetriever`, item 7b) — now evidenced by
  measurement, not assertion.
- **Verify:** 2 new checks (TF-IDF baseline MRR ≥ 0.8; LSA valid metrics) → **27/27**.
- **Report framing:** retriever = module's VSM baseline; this = IR Evaluation Measures; the project's
  `llm.faithfulness` gate = the RAG lecture's **Faithfulness** measure — together they evaluate both
  halves of a RAG system.

### Agentic multi-step tool-use + polish (items 6 & 7 from the assessment)
Implemented the one ambitious extension and the polish tier from `docs/ASSESSMENT_AND_RECOMMENDATIONS.md`.
Full detail in `v1/docs/AGENTIC_AND_POLISH.md`. **Tests 120 → 140 (1 skipped); verify 25/25; ruff +
mypy clean.**

- **(6) Agentic, still-numberless tool-use** — new `src/agent.py`. For *multi-step* questions
  ("compare the hottest of 2013, 2014 and 2015") the LLM **plans** which deterministic query-engine
  tools to call (extremum / average / correlation / threshold / anomaly_count) with which args; the
  executor runs each against pandas (results == pandas), and a final synthesis step is gated by the
  **same `llm.faithfulness` check** as the single-step path. The LLM never emits a data value — the
  truth layer still owns every number. Falls back to the single-step interface when the LLM is off or
  gives no usable plan. New `llm.plan_steps` (planner) + `llm.explain_synthesis` (synthesis). Exposed
  via `chat.py --agent`. Fully testable offline (`tests/test_agent.py`, 12 tests): executor is pure
  Python; planning/synthesis monkeypatched.
- **(7a) Larger-model option** — `DEFAULT_MODEL` now reads `TSHCI_MODEL`; new `llm.pick_model` auto-
  selects the largest *pulled* preferred model (8B/7B before 3B), else the default. `chat.py` delegates
  to it (auto-upgrades when a bigger model is present). Rationale: the faithfulness gate already
  guarantees the *numbers*; a bigger model mainly fixes wrong *qualifiers* — the residual it can't catch.
- **(7b) Embedding-RAG option** — new `rag_system.EmbeddingRetriever` (Ollama `nomic-embed-text`, cosine
  over unit-normalised vectors). Semantic match instead of keyword overlap; **drop-in** (same interface)
  and always safe — transparently falls back to the TF-IDF `Retriever` when no backend is reachable.
  TF-IDF stays the default. Tested offline via an injected `embed_fn` (`tests/test_rag.py`).
- **(7c) `resolve_threshold` year edge-case fixed** — threshold is now the number **following** the
  comparison word, so "above 30 in 2013" reads 30 and "above 2000" correctly reads 2000 as a genuine
  threshold (a year-range number is only taken when it sits directly after the comparison word), while
  "above average in 2015" yields no threshold. Replaces the old blanket "drop 1900–2100" heuristic.
- **(7d) mypy** — added to `dev` deps + a lightweight non-strict `[tool.mypy]` config (source layer,
  `ignore_missing_imports`, a few dynamic codes disabled for the deliberate `value: object` reuse).
  `python3.12 -m mypy` is clean on 16 source files.

### Full v1 audit + code-quality / architecture pass (fourth pass)
Read every v1 source module (13), both apps, tests and docs; verified the whole system and applied
best-practice organisation. Kept `src/` **flat** (user chose this over a subpackage restructure) so
notebook/app/report imports are untouched — nothing to re-run. **120 tests pass (1 skipped),
verify_results 25/25, ruff clean.**

- **Bug fixed (test-first): LSTM detectors weren't reproducibly seeded** (`deep.py`). `torch.manual_seed`
  ran *inside* `_train`, **after** `_build_lstm_ae()` / `_build_lstm_forecaster()` had already built the
  model — so LSTM **weight init was unseeded** and scores drifted run-to-run (the CNN path seeds before
  building, so it was inconsistent, and the "seeded/reproducible deep models" claim didn't hold for the
  LSTMs). Added a failing reproducibility test (`test_lstm_ae_scores_reproducible` diffed 100%), then
  added `_seed_torch(seed)` before building both LSTM models. 5 → 7 deep tests.
- **Packaging (`pyproject.toml`, new).** Declares the package + abstract deps (loose bounds; the pinned
  reproducible set stays in `requirements.txt`), and configures **pytest** (`pythonpath=["."]`,
  `testpaths`) and **ruff** (E/F/W/I/B/UP). Verified the pytest config resolves `from src import …` from
  both `v1/` and the repo root, so the empty `conftest.py` bootstrap was **removed**.
- **Lint clean-up (ruff, 14 → 0).** Real ones: removed an unused `pytest` import; `zip(..., strict=True)`
  in `inject_spikes`; `benchmark(seeds=None)` instead of a `range()` default; moved `interface.py`'s
  `logger` below its imports (E402); dropped a stale `"Answer"` string annotation; `def` instead of a
  bound `lambda` in a test; import ordering. No behaviour change (full suite still green).
- **Docs organised.** Moved working notes (`impro.md`, `IMPROVEMENTS.md`, `WRITEUP.md`, `RESULTS_DATA.md`)
  into **`report/notes/`**, leaving deliverables (REPORT, EVALUATION, MODELS, DATA_QUALITY, HCI_USER_STUDY,
  FINETUNING) at top level. Fixed the live cross-references + stale counts (`17/17`→`25/25`, conftest→
  pyproject) in `v1/README.md` and `code/CLAUDE.md`. Cleaned `__pycache__` from the tree.
- **Minor correctness/doc fixes.** Corrected `main.py`'s stale "LLM/interface out of scope" docstring;
  documented the `resolve_threshold` year-vs-threshold edge case (a bare integer 1900–2100 is read as a
  year — harmless on this dataset's magnitudes).
- **Audit verdict:** the codebase is genuinely high quality (multiply reviewed already) — the LSTM seeding
  was the only real bug found across ~2800 LOC; everything else was hygiene/organisation.

### Supervisor-feedback code pass (v1) — correctness, rigor, robustness, a11y
Worked through the code-only items in `v1/report/impro.md` (write-up items deferred). All changes
test-first; **84 tests pass (1 skipped)**, `verify_results.py` **21/21** (+1 advisory). Highlights:

- **Circular wind-direction fix (correctness).** The interface routed `wd (deg)` through *linear*
  max/min/mean/correlation — "maximum wind direction" returned a meaningless ~359.8° regardless of the
  actual wind. Added `analysis.circular_stats_deg` + `compass_point`; the interface now answers with
  the circular mean / std (prevailing direction, e.g. **196.6° S** on the real data) and uses the
  circular–linear coefficient for correlation. Closes the hole in the "exact numbers" claim.
- **No more silent failures.** The four `except Exception` swallows in `interface.py` (LLM phrasing,
  chart, forecast, reasoning) now log a warning first — the G2 "silently-dead LLM for weeks" bug lived
  in one of these.
- **LLM trust gate.** Every LLM explanation is now checked with `llm.faithfulness`; if it invents a
  number absent from the computed facts, the interface **discards it and uses the exact template**, so
  a hallucinated figure can never reach the user. Prompt also hardened against invented qualifiers
  (the 3B model calling a July date "winter").
- **Statistical rigor.** Added `evaluation.bootstrap_ci` (percentile CI on precision/recall/f1) and
  `evaluation.mcnemar_test` (paired significance test, e.g. Isolation Forest vs z-score). Both wired
  into `verify_results.py` as deterministic checks.
- **Stable verification.** The stochastic LLM-faithfulness check is now **advisory + majority-of-3**,
  so the headline count no longer flaps 19/20 ↔ 20/20 (was flaky). Now 3/3 clean, reported separately.
- **App tests.** `apps/web/app.py` refactored to a `create_app(df)` factory (data loads in the factory,
  not at import) so the Flask routes are unit-tested against a tiny frame; added a CLI `pick_model`
  test. Closes the "no tests for apps/" gap.
- **Reproducibility.** Pinned `requirements.txt` to exact tested versions (Python 3.12.13; added scipy,
  dropped unused seaborn) and added `reproduce.sh` (one command → seeded tests + verification banner).
- **Accessibility (HCI).** Web UI: real `<button>` chips (keyboard/Enter/Space), `role=log` +
  `aria-live` conversation region, labelled input/select, visible `:focus-visible` rings (was
  `outline:none`), and `role=img`/`<title>` on the chart SVG.

### Report refresh + borderline-anomaly marker (2026-07-04, second pass)
Brainstormed scope with the user (the HCI study itself was **deferred** — it needs real participants,
which cannot be fabricated). Then did the agreed write-up items in `v1/report/REPORT.md` plus one
code item. **87 tests pass (1 skipped).**

- **Borderline-anomaly marker (code, test-first).** New `anomaly.zscore_borderline` flags near-threshold
  points (0.75×–1× threshold); the interface exposes them (`Answer.borderline`) and the web chart draws
  them as dashed amber rings ("uncertain, not flagged") with a caption hint. Real 2014 query: 24 anomaly
  windows **+ 81 borderline near-misses**. Shows uncertainty instead of a hard binary.
- **Report Results refreshed** to current numbers: 87 tests / 21-of-21 harness (was "21 tests" / "7/7");
  LLM now integrated + faithfulness-checked (was "future"). New subsections — **§5.7** statistical rigor
  (z-score recall 0.78, 95% CI 0.64–0.90; precision 0.89, 0.76–0.98; McNemar IForest-vs-z p≈1e-19, read
  honestly against class imbalance), **§5.8** circular wind fix (359.8° → prevailing 196.6° S), **§5.9**
  measured LLM faithfulness (0 invented numbers, majority-of-3).
- **Limitations → Limitations & Threats to Validity** (§7): construct / statistical-conclusion /
  external / reproducibility validity, and an honest "no human study yet" note.
- **Generalisation appendix** (Appendix B): the identical architecture on v2 (EU trade) — catches 2008 +
  COVID, forecast MAPE 4.0%, AE 0.21 ≤ IForest 0.36, 0 invented numbers — evidence the method is general.
- **"Verifiable NL Analytics" framing** (Abstract, §1.6, §6): reframed the contribution from a weather
  demo to a reusable pattern (truth layer + LLM explanation + automated faithfulness verification).

### Chart legend + legend Q&A + conversational follow-ups (2026-07-04, third pass)
User feedback while using the web page: the chart markers were unlabelled, and they wanted to *ask*
about them and get real answers, plus dataset follow-ups "like a normal AI". **96 tests pass.**

- **Adaptive chart legend** (web): labels each marker with a matching swatch (observed line, max, min,
  anomaly, borderline ring, forecast band); shows only markers present; omitted on plain line charts.
- **Legend / marker Q&A.** New deterministic handler: *"what is the red dot?"*, *"what does the dotted
  circle mean?"*, *"explain the legend"* now return real explanations (from a fixed knowledge base) instead
  of the default fallback. The meanings are facts in code; the LLM only rephrases them — no hallucination
  risk. (`interface_parsing.resolve_legend_query` + `LEGEND_EXPLANATIONS`.)
- **Conversational follow-ups (memory).** The interface keeps a short rolling history; with the LLM on, an
  elliptical follow-up is rewritten into a standalone question (`llm.rewrite_followup`) which the query
  engine then answers exactly — the model rewrites *language*, code still computes *numbers*. Verified live:
  "show anomalies in 2013" → "what about 2014?" correctly answered 2014; "what is the red dot?" → legend.
  Web sends recent turns per request (server stays stateless) and shows an "understood as…" line for
  transparency; the CLI keeps history in-session. LLM-off stays single-shot (documented limitation).
- **Report:** §5.5 gains "Reading the chart (legend)" and "Conversational follow-ups" notes.

---

## 2026-07-02

### v1 tidy-up: dedupe deep modules + robust test bootstrap
- **Scope:** best-practices pass on **v1 only** (arrange / combine / document) — no behaviour change.
- **Combine (dedupe deep detectors):** the CNN- and LSTM-autoencoder scorers each carried their own
  copy of (a) the z-standardisation step and (b) the overlapping-window error-averaging loop. Both
  are now single shared helpers in `deep_anomaly.py` — `standardise()` and
  `average_overlapping_windows()` — imported by `deep_lstm.py`. Removed `deep_lstm._standardise` and
  two duplicated averaging loops.
- **Arrange:** added `v1/conftest.py` so the suite resolves `from src import ...` from any working
  directory (previously only worked as `python -m pytest` run from inside `v1/`; a bare `pytest` or a
  run from the repo root failed to import `src`).
- **Verify:** **48 passed, 1 skipped**; `verify_results.py` **17/17**. Confirmed the same results
  when running the suite from the repository root (the conftest fix).

### v1 made newcomer-readable: merge deep modules + rewrite README
- **Goal:** make v1 easy for a newcomer (and me) to understand — clear where to start, grouped,
  documented — without changing behaviour.
- **Combine:** merged `deep_anomaly.py` + `deep_lstm.py` into a single **`src/deep.py`** ("all deep
  detectors in one place": CNN-AE, LSTM-AE, LSTM-forecaster). Public API unchanged. Merged the two
  test files into `tests/test_deep.py`. Updated the notebook imports via aliasing
  (`deep_anomaly as da` → `deep as da`, `deep_lstm as dl` → `deep as dl`) so every existing `da.*` /
  `dl.*` call still resolves — no in-cell code changes and no re-run needed. src modules: 14 → 13.
- **Document:** rewrote `v1/README.md` as a guided onboarding doc — plain "what it does", a numbered
  **reading order** (README → walkthrough notebook → query_engine → interface → anomaly/reasoning), a
  **layers diagram** grouping the 13 modules into Data → Detect → Truth → Explain → Interface, and
  copy-paste run/test commands. Updated `CLAUDE.md` module table + test list to match `deep.py`.
- **Verify:** **48 passed, 1 skipped**; `verify_results.py` **17/17** (unchanged).

### v2 made newcomer-readable: declutter root + rewrite README
- **Goal:** same treatment as v1 — arrange / clean / document — no behaviour change. (v2 has a single
  deep module, so there was no deep-module merge to do here.)
- **Arrange (declutter root):** moved the 16 developer scripts off the top level into **`v2/tools/`** —
  14 `build_*_notebook.py` (notebook generators) + `get_trade_data.py` + `get_bilateral_demo.py`
  (Eurostat fetchers). Root now holds only `verify_results.py` beside the real structure
  (`src/ tests/ notebooks/ apps/ report/`). Fixed each script's output path to survive the move:
  builders now write to `Path(__file__).parents[1]/"notebooks"`, fetchers to
  `HERE.parent.parent/"data"`. Verified the builders resolve to `v2/notebooks/*.ipynb` via a
  write-stubbed dry run (executed notebooks left untouched) and the fetchers to `code/data/`.
- **Arrange:** added `v2/conftest.py` (same `src`-on-path fix as v1).
- **Document:** rewrote `v2/README.md` in the same onboarding style — "what it does", a numbered
  **reading order**, the **layers diagram** (Data → Detect → Truth → Explain → Interface) over the 12
  `src/` modules, corrected run commands (`tools/get_trade_data.py`, web app) and the test count
  (**79**). Updated `CLAUDE.md` + `GLOSSARY.md` references to `tools/get_trade_data.py`.
- **Verify:** **79 passed**; `verify_results.py` **13/13**; identical when run from the repo root
  (the conftest fix); executed notebooks unchanged.

## 2026-06-28

### v1 review, fixes, and notebook consolidation (9 → 6)
- **Scope:** this case study uses **v1 only** (temperature / Jena climate). Reviewed the whole v1
  codebase, fixed issues, and consolidated the notebooks.
- **Bugs / improvements (src):**
  - `reasoning.co_movement` hardcoded the word "temperature" in its output — wrong label when the
    queried sensor was anything else. Now uses the actual `target` sensor name.
  - `interface.detect_intent` matched intent keywords as **substrings** ("most" inside "almost",
    "top" inside "stop"). Switched to **word-boundary** regex matching (consistent with v2).
  - Added `tests/test_reasoning.py` (2 tests) + 2 word-boundary asserts to `test_interface`.
    **Tests: 23 → 25 passing** (1 skipped when Ollama is up).
- **Notebooks consolidated 9 → 5** (re-executed clean on the `py312-tshci` GPU kernel):
  - old `01_data_exploration` + `04_visualizations` → **`01_data_exploration_and_visualization`** (Part 1 EDA, Part 2 visual gallery)
  - old `02_evaluation` + `03_methods_demo` → **`02_methods_and_evaluation`** (Part 1 methods, Part 2 correctness)
  - old `06_deep_model` + `07_deep_comparison` + `08_model_comparison` → **`04_deep_models_comparison`** (Parts A/B/C)
  - old `05_interface_demo` → **`03_interface_reasoning`**
  - final order: 00 walkthrough · 01 EDA+viz · 02 methods+eval · 03 interface · 04 deep models
- **Kernel:** registered `py312-tshci` (→ `/usr/bin/python3.12`, has torch+CUDA) because the old
  `python3` kernel pointed at a missing `python` and `tshci` is the torch-less 3.14 venv.
- **Docs synced:** `notebooks/README.md`, `v1/README.md`, `CLAUDE.md` (notebook table + test counts).

### Interface: time-scoped + threshold questions (fixes "always 2015")
- **Problem reported:** every "hottest/coldest" question returned the all-time record (37.04 °C,
  2015-07-04) because the interface always called `get_max`/`get_min` over the **whole** dataset —
  it never read a year or a value from the question.
- **Fix (`src/interface.py` + `query_engine.py`):**
  - `resolve_period()` — parses a year (`2013`), month+year (`july 2015`), or range
    (`between 2012 and 2014`); `ask()` slices the dataframe to it before computing.
  - `resolve_threshold()` + `query_engine.threshold_events()` — answers "how often above 30?",
    "colder than -15 in 2012?" with exact counts + first/last occurrence.
  - Every answer now states the period it used (`max … in 2013 was …`).
- **Verified on real data:** hottest overall 37.04 (2015); in 2013 → 34.93; coldest 2010 → -17.71;
  Jan 2012 → 11.97; above 35 °C → 18 times; below -15 in 2012 → 92 times.
- **Tests:** +4 (`resolve_period`, `resolve_threshold`, year-scoped ask, threshold count) → **29 pass,
  1 skip.** Web app suggestion chips updated; notebook 03 re-executed.

### New docs + verification recheck
- `v1/report/HCI_USER_STUDY.md` — full **user-study protocol** (within-subjects: spreadsheet vs
  template vs AI; task success graded against ground truth; time, SUS, NASA-TLX, trust).
- `v1/report/WRITEUP.md` — dissertation **draft skeleton** (structure + measured evidence embedded;
  lit review/prose marked TODO — for later, not for marking yet).
- **Verification rechecked twice:** test suite 48 pass / 1 skip; `verify_results.py` **17/17** on both
  runs (exit 0); independent fresh-pandas spot-check matches the system exactly (max 37.04 @ 2015-07-04
  16:00, min −22.65, mean 9.4447, >35 °C = 18).

### Evaluation rigor (examiner-driven) + de-emphasised "novelty"
- **Forecast backtest** (`forecasting.backtest`, rolling-origin, MAE/RMSE/**MASE**): on T, MAE 3.55 °C,
  RMSE 4.28 °C, **MASE 0.83** (beats persistence). MAPE avoided (temp crosses 0 °C).
- **Parameter sensitivity** (`anomaly.threshold_sensitivity` / `contamination_sensitivity`):
  z-threshold 4.0 is the knee (526 flags, ~0.75%); contamination 0.01 conservative — choices justified.
- **LLM faithfulness** (`llm.faithfulness`, rounding-tolerant): measures dropped vs **invented**
  numbers. Live result: **0 invented** (no hallucinated figures — trust property holds); occasional
  benign omission when summarising multi-value facts.
- **Repeatable anomaly benchmark** (`evaluation.benchmark`, 10 seeds): strict P 0.77±0.13, R 0.66±0.07,
  F1 0.71; point-adjusted similar — mean±std, not a single run.
- **`verify_results.py` now 17/17** (added forecast-MASE, benchmark recall, LLM-no-invented checks).
- **EVALUATION.md** gained a "Quantitative rigor" section with the measured numbers.
- **De-emphasised novelty** (per user): softened the three "novelty" mentions to "cross-sensor
  explanation feature"; saved a memory that the core objective is the evaluated explainable trust
  system + HCI, not novelty.
- **Tests:** +5 (backtest, threshold/contamination sensitivity, faithfulness, benchmark) → **48 pass, 1 skip.**

### More: forecast, reasoning confidence, web charts, v1 verification
- **Forecast intent** — `forecasting.forecast()` fits ARIMA on the last year of daily means and
  returns a 14-day forecast with an 80% band; interface answers "what will it be next week?".
- **Reasoning confidence** — `describe_moment` now adds a verdict: many sensors moving together →
  "real weather event"; one alone → "possible sensor glitch".
- **Web charts** — `Answer.chart` carries a downsampled series (+ markers / forecast band); the web
  UI renders an inline SVG (anomaly markers, the extreme point, or history+forecast).
- **`verify_results.py` for v1** — **14 independent checks** (query engine == pandas, scoped queries,
  thresholds, known-event placement, reasoning verdict, forecast bounds, point-adjusted ≥ strict
  recall, confusion sums to N). All pass.
- **EVALUATION.md** — added an honest "Ground truth & honesty" section: exact stats are hard truth
  (== pandas); anomaly accuracy is on synthetic labels + corroborated against known real events
  (Jul 2015 heatwave, winter cold extremes).
- **Tests:** +3 (forecast shape, forecast intent, reasoning verdict) → **43 pass, 1 skip.**

### Capability + UX upgrades (reasoning novelty, trend, compare, cleaner web UI)
- **Wired the multi-sensor reasoning (the project's novelty) into the interface.** Anomaly answers
  now *explain why* via `reasoning.describe_moment` — e.g. "Tdew low (-2.9σ); T low (-2.7σ); rho high
  (+2.5σ)". It was built but previously only used in a notebook, not the NL interface.
- **New `trend` intent** — linear warming/cooling rate (e.g. "T (degC) rising at +0.40 per year").
- **New `compare` intent** — two-period questions ("was 2013 hotter than 2014?"): exact stat per year
  + which was warmer/colder. `resolve_compare()` fires on 2 years + a comparison word.
- **Richer weather report** — added dew point.
- **Robustness:** unknown-sensor questions now say "I don't track X" instead of defaulting to
  temperature; the day-climatology label clamps to years actually present (no more spurious "2017");
  `anomaly.anomaly_windows` now delegates to `query_engine.get_anomaly_windows` (one implementation).
- **Evaluation honesty:** added `evaluation.point_adjust` + `score_point_adjusted` (the fairer
  time-series anomaly metric, mirroring v2).
- **Cleaner web UI** (`apps/web/index.html`): light, high-contrast, larger type; friendly trust labels
  (📊 exact figures · ℹ️ background · 🧠 what it suggests) + a one-line trust banner; no duplicated
  facts in template vs AI mode; refreshed suggestion chips.
- **Tests:** +8 (compare, trend, missing-sensor, anomaly-reasoning, point-adjust, mocked `llm.plan`
  ×2) → **40 pass, 1 skip.** Notebook 03 re-executed.
- **Friendly dates:** added `query_engine.fmt_when` ("19 Aug 2012 13:00") used across answers and the
  reasoning text instead of raw ISO timestamps.

### Interface: specific-day "weather report" answers
- **Ask:** questions like "tell me the weather in 26 january" should give a daily report.
- **Added:** `resolve_day()` parses a specific date — "26 january", "january 26", "26th of jan 2015",
  "26/01/2015", "2015-01-26"; year optional. New `weather` intent + `query_engine.weather_report()`
  give a multi-sensor snapshot (temp avg/low/high, humidity, pressure, wind).
- **No year → climatology:** "26 January" with no year reports across all years (e.g. 192 readings);
  with a year it's that exact 24-hour day. Stats also scope to a day ("hottest on 26 January 2015").
- **Verified:** 26 Jan 2015 → avg 1.1 °C (low -0.5, high 2.7), humidity ~93%; 5 Jul 2014 → avg 22.6 °C.
  **+2 tests → 32 pass, 1 skip.** Web chips + notebook 03 updated.

### Interface: LLM-as-parser so it isn't limited to a keyword list
- **Ask:** make it understand *any* phrasing, not just fixed keywords.
- **Design (trust-preserving):** the local LLM (`llm.plan`) translates free text into a **structured
  plan** {intent, sensors, start/end dates, op, value} — it interprets language (seasons, ranges,
  loose phrasing) but **never computes a number**; the query engine still does all the maths.
- **Merge rule:** deterministic parser wins when confident (reliable keyword/explicit-year matches);
  the LLM only *fills gaps* it left open. Prevents the small 3B model from flipping a good parse
  (e.g. "typical"→mean becoming max).
- **Seasons** added to the offline resolver too: spring/summer/autumn/fall, and winter rolls back to
  the previous December (`winter 2013` = Dec 2012–Feb 2013).
- **Verified (llama3.2:3b):** "how warm did it get during the summer of 2012?" → max 35.52 (summer
  2012); "typical temperature toward the end of 2014" → mean 10.70 (2014); "brutally cold in early
  2010" → anomaly scan in 2010. **30 tests pass, 1 skip.**

### CUDA out-of-memory fix in notebook 04 (deep models)
- **Symptom:** notebook 04 OOM'd on the 8 GB GPU when run interactively.
- **Cause:** the scoring step pushed the *entire* ~70k-window tensor through the model in **one
  forward pass**; the merged notebook also trains 3 models per session without freeing GPU memory,
  so stale kernels left ~4.5 GB held and the big single allocation no longer fit.
- **Fix (src):** batched inference — `deep_anomaly._squared_errors` and a new
  `deep_lstm._predict_batched` run the forward pass in mini-batches (8192 / 4096), so peak GPU
  memory is bounded regardless of series length. Added `deep_anomaly._free_cuda()` and call it after
  each model's scoring to return memory to the driver.
- **Notebook:** setup cell now clears the cache and prints GPU headroom on start.
- **Result:** runs clean in ~35 s with only **3.2 GB free**; CNN-AE scores 70,129 points in 1.4 s.
  Deep tests (5) + full suite (25 pass, 1 skip) green.

---

## 2026-06-22

### v2 kick-off — new domain dataset: international trade (supply chain)
- **Objective today:** start v2 by getting a richer, supply-chain-flavoured time series — the user
  wanted **trade between countries** (not the Jena climate data). v2 is a fresh iteration and does
  **not** have to mirror v1's structure.
- **Source survey (offline-friendly):** no Kaggle (no CLI/creds). Probed UN Comtrade (free API now
  deprecated → web page), IMF DOTS (host unreachable), World Bank (OK, annual only), **Eurostat (OK,
  monthly)**. Picked Eurostat for monthly resolution + genuine country-to-country trade.
- **Dataset chosen:** Eurostat `ei_eteu27_2020_m` — *EU27 international trade, monthly, 2002–2026*.
  Pulled raw trade **value (million EUR, not seasonally adjusted)**, total goods, partner = WORLD,
  so seasonality and real shocks survive.
- **Built:** `code/v2/get_trade_data.py` — reproducible fetch via Eurostat statistics/1.0 JSON API
  (the SDMX endpoint ignored query filters; statistics/1.0 filters correctly). Each country's monthly
  EXPORTS and IMPORTS = one column ("channel"), analogous to a sensor in v1.
- **Result:** `code/data/eu_trade_monthly.csv` — **292 months × 54 columns** (27 EU countries ×
  exports/imports), 2002-01 → 2026-04, **0 missing cells**, 114 KB (gitignored like Jena).
- **Why it's good for the project:** real, *explainable* anomalies are baked in — **2008 crisis**
  (DE exports €88.6bn Oct'08 → €63.5bn Jan'09) and **COVID** (€109bn Mar'20 → €75.9bn Apr'20, ~30%
  drop). Great for Objective 2 (drift / event detection) and the explanation layer.
### v2 EDA (notebook 01) — done
- Structure decided: **hybrid** v2 — analysis steps as notebooks, engine/interface/llm as importable
  `src` (so the website + pytest still work). Scaffolded `v2/{src,notebooks,tests}`.
- Built `v2/notebooks/01_eda.ipynb` (authored via `build_eda_notebook.py`, executed clean, 0 errors).
  **Documented for readers:** 13 markdown cells (What/Why/How-to-read per step) + line-commented code,
  and **8 plots** — value distribution (raw vs log), total trade with balance shading, big-traders bar,
  full STL 4-panel, month boxplot, COVID small-multiples, correlation heatmap + EXP-vs-IMP scatter, and a
  residual z-score anomaly preview (flags 2008-04, COVID 2020-04/05/06, 2022-09, 2024-03).
  Added `v2/GLOSSARY.md` + a v2 section on the Notion Glossary page.
- **Findings:** 292 months × 54 cols, 0 missing, no index gaps, all values > 0 → minimal cleaning.
  Total EU exports grew **3.1×** (2002→2026), roughly exponential → **log-transform** before anomaly
  detection. Clear yearly seasonality (summer/Dec troughs), DE export seasonal swing ~€32.6 bn →
  **STL period = 12**. Real shocks confirmed: 2008 **−28.3%**, COVID **−30.7%** (built-in ground truth).
  Countries co-move strongly (mean export-growth corr **0.57**; top pairs ES–IT, AT–DE, CZ–HU/PL ≈ 0.91).
- Tooling note: fixed a broken system `rpds`/`jsonschema` by installing newer ones into the py3.12
  user site so `nbformat`/`nbconvert` import. Notebook kernel = `python3`.
### v2 bilateral DEMO — EU trade by partner country
- To explain *country-to-country* (bilateral) trade, pulled Eurostat `ext_lt_maineu` (Extra-EU trade
  by partner, annual 2002-2025) → `code/data/eu_trade_by_partner.csv` (24 years × 22 cols = 11 major
  partners × EXP/IMP). Script: `v2/get_bilateral_demo.py`.
- Caveat: reporter = EU as a whole, breakdown by partner country (EU→USA, EU→China, EU→Russia…).
  True individual pairs (FR→BE) need Comext `DS-018995`, which the public API doesn't serve (404);
  intra-EU members can't be "partners" in extra-EU datasets.
- Real, explainable bilateral events: **EU imports from Russia −78%** (€163.6bn 2021 → €36.3bn 2024,
  sanctions); EU exports to USA grew €246bn→€536bn; China/UK trends visible. (This is a teaching
  example; the main v2 dataset remains `eu_trade_monthly.csv`.)
### v2 Preprocess (notebook 02 + src/preprocess.py + tests) — done
- Hybrid in action: logic in **`v2/src/preprocess.py`** (load_trade, validate→QualityReport, countries,
  flow_columns, trade_balance, total_trade, add_features, log_transform, one-call `preprocess()`),
  demonstrated in **`v2/notebooks/02_preprocess.ipynb`** (12 cells, 2 plots, 0 errors), covered by
  **`v2/tests/test_preprocess.py`** (8 tests, all pass).
- Implements the EDA decisions: validate says **CLEAN ✓** (0/0/0); derived **balance** (FR deficit /
  BE surplus) + **total**; **log transform** makes crises comparable across years (FR_EXP change-std:
  raw 4,608 EUR m vs log 0.131 = scale-free).
- Saved `data/eu_trade_features.csv` (108 cols = 54 flows + 27 BAL + 27 TOT) for downstream notebooks.
### v2 Behaviour analysis (notebook 03 + src/analysis.py + tests) — done
- `v2/src/analysis.py`: trend_slope(s)/trend_slopes, STL decompose + trend/seasonal strength
  (Hyndman), rolling_stats, adf_test→ADFResult, acf_pacf, correlations(growth), top_pairs,
  lead_lag, coverage. Notebook `03_analysis.ipynb` (18 cells, 6 plots, 0 errors). Tests:
  `test_analysis.py` (8) → **16 tests total pass**.
- Findings: fastest export growth = CY 9.9, LV 8.9, PL 8.7 %/yr (catch-up economies); slowest =
  LU 0.4, FI 1.7, MT 1.9. STL strength FR trend 0.93 / seasonal 0.67, BE 0.96 / 0.44. **ADF:** log
  level non-stationary (p=0.81) → seasonal-diff stationary (p=0.0013) ⇒ ARIMA needs differencing
  (d≥1, D≥1). STL residuals have **fat tails** (skew −3.18, kurt 28) — driven by the crises, so the
  |z|>3 rule is usable and the tails ARE the anomalies. Co-movement high (ES-IT, AT-DE, CZ-HU/PL
  ≈0.91); DE vs PL lead-lag = 0 (move together).
- Folded in the two preprocessing caveats: full coverage (27/27 countries, n=292) and a note that
  the last 1-2 months are **provisional** (Eurostat revises them).
### v2 Anomaly detection (notebook 04 + src/anomaly.py + tests) — done
- `v2/src/anomaly.py`: three detector families — (1) **statistical** z-score on STL residual,
  (2) **forecasting** seasonal-naive error z-score, (3) **ML multivariate** Isolation Forest over
  all 54 channels' year-on-year growth (systemic-shock detector). Plus eval helpers: label_windows,
  Scores, score() (P/R/F1/confusion). Notebook `04_anomaly.ipynb` (16 cells, 4 plots, 0 errors).
  Tests `test_anomaly.py` (6) → **22 tests total pass**.
- Evaluated vs **real labelled crises** (2008-09→2009-04, 2020-03→2020-08; 14 months). Strict
  month-by-month: seasonal-naive best (P=0.62 R=0.36 F1=0.45), STL-z (0.50/0.21/0.30), IsoForest
  (0.36/0.36/0.36). Low recall is a labelling artifact (wide window vs detectors firing on the
  sharpest months) — so added **event-level detection**: all three catch **both** 2008 & COVID (±1mo).
- Honest notes captured: **precision artifact** (real-but-unlabelled 2022 energy shock & 2021 rebounds
  count as FPs → measured precision is a lower bound); three families fail differently (z-score assumes
  near-normal, seasonal-naive blind to drift, IsoForest needs multivariate view) → agreement = confidence.
### v2 Multi-country reasoning (notebook 05 + src/reasoning.py + tests) — done
- `v2/src/reasoning.py`: cross_section_z (months×27 standardised growth), systemic_timeline,
  ShockReport, shared_shock/explain_month (classifies shared-shock vs isolated vs normal +
  evidence + sentence), find_isolated_months, co_movement. Notebook `05_reasoning.ipynb`
  (14 cells, 4 plots, 0 errors). Tests `test_reasoning.py` (6) → **28 tests total pass**.
- Demonstrated: **Apr-2020 COVID** = 18/27 countries down (67%) → "shared shock" (top FR/RO/IT/SK/ES
  ≈ −5σ); **Jan-2009 GFC** = 23/27 down (85%); **isolated** example Aug-2014 (only CY +2.2σ).
  57 isolated months found overall.
- **Trust-aware output** (the LLM bridge): emits 🔢 COMPUTED facts (counts, z-scores, fraction, kind)
  separately from 🧠 INFERRED cause — the model may rephrase computed facts but never invents them.
- Deep-learning model shortlist recorded (for the later deep step): USE multivariate Autoencoder
  (systemic) + 1D CNN-AE baseline; LSTM-AE/forecaster/TCN/VAE optional; Transformer/GNN/TFT
  literature-only (292 pts too few). Classical baselines must be beaten to justify deep.
  Saved as a **side note**: `v2/DEEP_MODELS.md` (full table + recommendation; not yet implemented).
### v2 Forecasting (notebook 06 + src/forecasting.py + tests) — done
- `v2/src/forecasting.py`: SARIMA (1,1,1)(1,1,1)12 — fit_sarima, forecast()→Forecast(mean/band),
  **backtest()** (train on past, predict UNSEEN future, MAPE/RMSE/coverage), backtest_from(cutoff).
  Notebook `06_forecasting.ipynb` (12 cells, 4 plots, 0 errors). Tests `test_forecasting.py` (5)
  → **33 tests total pass**.
- **Predict-vs-memorise answered with a backtest:** France trade, last 24 unseen months →
  **MAPE 4.0%, 100% band coverage** = genuine prediction (SARIMA's ~5 params can't memorise 292 pts).
- **Shock backtest across COVID** (train ≤2019-12 → forecast 2020-21): MAPE 14.4%, **6 months fall
  BELOW the 95% band** → proves shocks are *detected, not predicted* (the forward early-warning idea).
- Forecast to 2030: France €116.4 bn → central €127.6 bn (+10%), band €78-208 bn (widens with horizon).
  Projected fastest growers: CY 9.9, LV 8.9, PL 8.7, BG 8.3, LT 8.0 %/yr.
- Limitations stated: shocks unpredictable, band widens, structural breaks, provisional tail.
- ANSWER to "when do we train models": SARIMA trained+validated here; deep models train later (deep
  notebook) with the same backtest discipline (small/regularised to avoid memorising). See DEEP_MODELS.md.
### v2 Query engine (notebook 07 + src/query_engine.py + tests) — done
- `v2/src/query_engine.py` — THE TRUTH LAYER: get_max/get_min (Extremum), get_mean_std,
  get_correlation, trade_balance, total_trade, biggest_surplus/deficit, growth, worst_drop,
  rank_countries, country_summary. Notebook `07_query_engine.ipynb` (16 cells, 0 errors).
  Tests `test_query_engine.py` (10) → **43 tests total pass**.
- **Verification: 8/8 checks confirm engine == pandas** (exact). Sample answers: FR deficit
  −7.5 bn (Mar-2024), DE biggest surplus +21.8 bn; PL exports +244% (2010→2024); worst monthly
  export drops ES −35% / DE −30.7% (both Apr-2020 COVID); top exporters DE>NL>IT>FR>BE; NL in
  surplus 100% of months.
- This is the foundation for the trust principle: code computes exact numbers; the interface/LLM
  (next) only phrase them.
### v2 Interface + local LLM (notebook 08 + src/interface.py + src/llm.py + tests) — done
- `v2/src/interface.py`: NL question → detect_intent (word-boundary match) + resolve country/flow/
  dates → `query_engine` → trust-aware `Answer` (🔢 computed / 🧠 inferred / 💬 LLM). Intents: max,
  min, mean, correlation, surplus/deficit/balance, growth, drop, rank, summary, shock(→reasoning).
  `src/llm.py`: stdlib-HTTP Ollama, is_available/list_models/pick_model/generate/explain; strict
  system prompt forbids changing numbers; graceful fallback to templates if Ollama down.
- Notebook `08_interface.ipynb` (14 cells, 0 errors). Tests `test_interface.py` (10) → **53 tests pass**.
- **Live LLM verified** (Ollama up; picked `llama3.2:3b`): it phrases the computed facts and keeps
  numbers EXACT (e.g. "Poland's exports … growing by 173% … Dec-2010 to Dec-2024"; Spain "-35% …
  peak of COVID-19") — adds context but never alters a figure. Trust principle proven end-to-end.
- **Bug fixed:** growth keyword "rise" matched inside "summa**rise**" (same class as v1 usual/unusual)
  → switched intent matching to word boundaries; added regression test.
- Sample answers: FR peak export 57,238 (2025-10); DE-FR export corr 0.79; ES worst drop -35% (2020-04);
  top exporters DE>NL>FR>IT>BE; NL in surplus 100% of months.
### v2 Hybrid RAG (notebook 09 + src/rag_system.py + tests) — done
- User chose **hybrid (lexical + semantic)**. Findings before building: `rank_bm25` not installed and
  the installed Ollama models **can't embed** (501 Not Implemented). So built fully offline, no new
  deps: **BM25 from scratch** (lexical) + **LSA** = TF-IDF→TruncatedSVD (semantic, sklearn). Same
  `search()` interface so neural embeddings (pull `nomic-embed-text`) can swap in later.
- `v2/src/rag_system.py`: knowledge_base() (20 context docs — countries, trade terms, 2008/COVID/2022
  events, methods; **no euro amounts**), BM25, SemanticLSA, HybridRetriever (min-max normalise +
  weighted combine + re-rank), explain() (🔢/📚/🧠). Notebook `09_rag.ipynb` (12 cells, 0 errors).
  Tests `test_rag.py` (7) → **60 tests total pass**.
- Verified semantic match: "pandemic lockdown halted commerce" → event_covid (0.97); "war disrupted
  energy and trade" → event_2022 (0.98) — meaning matched without literal keywords. Full trust-aware
  answer assembled: 🔢 ES -35% (2020-04, from query engine) + 📚 COVID/2008 context + 🧠 inference.
- Trust boundary kept: RAG supplies CONTEXT only; numbers always come from the query engine (KB has
  no euro amounts — unit-tested).
- Note on retrieval method (answer to user): v2 now uses **BM25 + LSA hybrid**; earlier interface used
  keyword routing only. Confusion-matrix confirmed (seasonal-naive DE: TP=5 FP=3 FN=9 TN=275, sums to
  292; FP=real-but-unlabelled 2009-tail/2021-rebound; FN=wide-window recovery tail).
### v2 Evaluation + report (notebook 10 + report/ + README) — done
- `v2/notebooks/10_evaluation.ipynb` (13 cells, 0 errors): detector comparison averaged over 27
  countries (STL-z F1 0.27, seasonal-naive 0.22, **IsoForest 0.36** — multivariate wins on average);
  event-level all three catch both crises; confusion-matrix heatmap; forecasting recap (normal MAPE
  **4.0%**/100% coverage, COVID MAPE 14.4%/6 breaches); v1↔v2 comparison table.
- Wrote **`v2/report/REPORT.md`** (full MSc draft: abstract, aim/objectives, dataset, methodology
  per step, results, evaluation discussion, limitations, v1↔v2, conclusion/future work) +
  **`v2/report/EVALUATION.md`** (all numbers) + **`v2/README.md`** (folder/module map + how to run).
- Note: user declined pulling `nomic-embed-text` (real neural embedded search) — LSA stays the
  semantic half; documented as future work. Ollama registry reachable + ollama CLI 0.30.7 if revisited.
- v2 status: **10 notebooks, 60 tests passing**, full pipeline (EDA→preprocess→analysis→anomaly→
  reasoning→forecasting→query engine→interface+LLM→hybrid RAG→evaluation) + report. Remaining
  optional: web app, deep models, clustering.

### v2 final batch — clustering, deep model, web UI, report finalised (user: "do 4 2 1 5")
- **#4 Clustering** (`src/clustering.py`, notebook 11, 6 tests): country_features (size, balance_ratio,
  growth, seasonality, volatility) → K-means. Silhouette best=2 (just isolates CY/MT outliers) so used
  **k=4** for interpretable groups: Eastern catch-up (BG/CZ/EE/LT/LV/PL/RO/SI/SK), Western established
  (AT/BE/DE/DK/ES/FI/FR/HU/IE/IT/NL/PT/SE), CY alone, EL/HR/LU/MT. Satisfies Objective 4.
- **#2 Deep model** (`src/deep_anomaly.py`, notebook 12, 4 tests, GPU cuda): multivariate autoencoder.
  **Honest negative result** — AE F1 **0.21** (catches COVID, misses 2008) **loses to** Isolation
  Forest 0.36. Overfits fast (more epochs → worse); early-stop at 60. Insight: systemic shocks are
  low-rank so an AE reconstructs them well → tree-based IsoForest better. Deep needs more data; IsoForest
  stays default. (DEEP_MODELS.md prediction confirmed.)
- **#1 Web UI** (`apps/web/app.py` + `index.html`): Flask chat — model selector, use-LLM toggle,
  suggestion chips, trust-labelled answers (🔢/📚/🧠/💬) incl. hybrid-RAG context. Auto free-port
  (3000→3001 when busy). Verified endpoints (/, /api/suggestions, /api/models, /api/ask).
- **#5 Report finalised**: REPORT.md (added clustering §3.9, deep §3.10/§4.5, web §3.11; v1↔v2 table;
  future work), EVALUATION.md (deep-vs-baseline table), README.md (12 notebooks, 70 tests, web app).
- **Status: 12 notebooks, 70 tests passing**, full pipeline + clustering + deep + web + report.
  User declined neural embedded search (nomic-embed-text) — kept as future work.

### v2 Chunking + BERT retrieval (notebook 13 + rag_system extensions + tests) — done
- User chose "both: chunker + BERT". Installed **sentence-transformers 5.6.0**; downloaded
  **all-MiniLM-L6-v2** (384-dim, ~90MB). HF Hub reachable.
- `src/rag_system.py` additions: **chunk_text()** (sentence-aware, overlapping), **BertRetriever**
  (neural embeddings; lazy-loads model; drop-in for SemanticLSA), and `HybridRetriever(semantic=
  "lsa"|"bert")`. Notebook `13_chunking_bert.ipynb` (12 cells, 0 errors). Tests +4 → **74 total pass**
  (BERT tests skip gracefully if lib absent via importorskip).
- **Key demo (BERT earns its keep):** paraphrase query "factories shut and shipping stopped worldwide"
  (no shared words) → BM25 & LSA return wrong country docs; **BERT returns event_covid + event_2008**.
  Chunking a long doc → 4 overlapping chunks; chunk-level BERT retrieval answers "how did governments
  react?" with the stimulus chunk.
- Honest note kept: for 20 atomic KB docs BM25/LSA already suffice; BERT + chunking pay off as the
  corpus grows (long docs, paraphrase). Retrieval still returns CONTEXT only.
- Docs updated: REPORT §3.8 + v1↔v2 table + future work, README (13 notebooks, BERT/chunking).
  New dep: sentence-transformers. **v2 now: 13 notebooks, 74 tests passing.**

### v2 results verification (verify_results.py + report/VERIFICATION.md) — done
- Built `v2/verify_results.py`: independently re-derives every headline claim from raw data and checks
  it (separate from unit tests — tests check the code, this checks the claims). **13/13 PASS.**
- Verified: data 292×54 clean; query engine == pandas (truth layer); ADF level→diff; all 3 detectors
  catch 2008 & COVID (event-level); confusion sums to 292 (TP5/FP3/FN9/TN275); forecast backtest MAPE
  **4.0%** (predicts not memorises); COVID breaches band; deep AE F1 0.21 ≤ IsoForest 0.36 (negative
  result holds); clustering 27→4; reasoning 2020-04 = shared shock; **RAG KB has no numbers**; **BERT
  finds COVID on paraphrase where BM25 fails**; **LLM keeps magnitude 35 + year 2020 exact**.
- Fixed an over-strict LLM check (was matching literal "-35.0"; LLM correctly wrote "35%"/"April 2020"
  — magnitude+date preserved, sign/decimal dropped = not a trust violation).
- Documented in `report/VERIFICATION.md` (one-command reproducible proof of the trust chain).

### v2 Evaluation upgrade (notebook 14 + src/evaluation.py + tests) — done
- Addresses the review's #1 weakness (low strict F1). `src/evaluation.py`: **point_adjust** (TS-anomaly
  standard — credit a whole anomaly segment if any point flagged), score_point_adjusted, **ensemble**
  (vote across detectors), **pr_curve** + pr_auc. Notebook `14_evaluation_upgrade.ipynb` (10 cells,
  0 errors). Tests `test_evaluation.py` (5) → **79 tests total pass**.
- **Big improvement:** point-adjusted F1 — STL 0.30→0.52, **seasonal-naive 0.45→0.90**, IsoForest
  0.36→0.76. **Ensemble (≥2 votes) point-adjusted: P=0.82, R=1.00, F1=0.90** (perfect recall, 0 FN).
  PR curves trace each detector's full threshold trade-off (PR-AUC).
- Takeaway: the earlier strict numbers were honest but pessimistic; under the standard metric the
  system detects the real 2008 & COVID crises strongly, and the ensemble is the best single choice.
- Fixed NumPy 2.x `trapz`→`trapezoid` in pr_auc. Docs updated: REPORT §4.1, EVALUATION.md (point-adj +
  ensemble table, 79 tests), README (14 notebooks, evaluation.py).

---

## 2026-06-21

### v1 restructured + documented
- Tidied the scattered v1 layout: `apps/` (chat.py + web/), `report/` (REPORT moved in from
  `code/report/`, plus EVALUATION/MODELS/FINETUNING), `src` / `tests` / `notebooks` unchanged.
- New docs: `v1/README.md` (folder map + how to run), `notebooks/README.md` (reading order),
  and **`notebooks/00_code_walkthrough.ipynb`** — tours every src module with runnable examples.
- Fixed path lines in moved `chat.py` / `web/app.py`; updated root README + CLAUDE.md.
- Verified: chat.py + web import OK, 23 tests pass, walkthrough executes.

### Web UI (Flask) — interactive chat in the browser
- Built `v1/web/app.py` (Flask) + `v1/web/index.html`: chat page with **model selector**,
  **Use-LLM toggle**, **clickable suggested questions**, and **trust-labelled answers**
  (computed/context/inferred shown).
- Reuses `interface`/`llm` directly; endpoints `/api/ask`, `/api/models`, `/api/suggestions`.
- **Port note:** the user wanted :3000, but a **FastAPI app already occupies :3000** — app.py
  auto-falls back to the next free port (currently **3030**). Run: `python3.12 v1/web/app.py [port]`.
- Verified all endpoints return correct answers; numbers stay exact.

### Interface: removed the keyword limitation
- **Problem:** the NL interface only understood fixed keywords (max/min/mean/correlation/missing/
  anomaly/summary).
- **Change:** expanded the keyword synonyms, and added an **LLM fallback** — when keywords don't
  match and `use_llm=True`, the local model classifies the question into an allowed intent + sensor
  (`llm.classify`), then the **query engine still computes the exact number**. Trust preserved.
- **Verified live:** "how warm did the air ever get?" → max → 37.04 °C; "is humidity connected to
  how warm it is?" → correlation(rh, T). 23 tests pass.
- **Bug found & fixed:** adding "usual" to the *mean* keywords matched "un**usual**" (anomaly) — a
  substring collision. Removed it.

### Ollama explanation layer (the final step) — implemented & verified
- `src/llm.py` talks to local Ollama (stdlib HTTP); strict prompt forbids changing numbers.
- `interface.Interface(use_llm=True)` phrases answers via the LLM, falls back to template if Ollama
  is down. Live demo kept numbers exact (37.04 °C, −0.963).
- Added `chat.py` runnable entry point (`python3.12 chat.py --llm`). `FINETUNING.md` = side note.
- Note: only the community Gemma models are installed; recommend `ollama pull llama3.2:3b` for a
  cleaner explainer.

### Evaluation: confusion matrix + precision clarified
- Added TP/TN/FP/FN (`evaluation.confusion`). Controlled signal: Isolation Forest perfect (40 TP,
  0 FP); z-score weakest (misses a few spikes STL absorbs).
- **Key finding:** real-data precision (~0.14) is an *artifact* — genuine background anomalies count
  as false positives against synthetic-only labels. Controlled-signal precision is 0.85–1.00.

### Models: full comparison + ARIMA
- Added ARIMA forecast-error detector (deseasonalised; ~3 s on 70k points).
- Six detectors compared (statistical / forecasting / classical / 3 deep) in `MODELS.md` and
  `notebook 08`. CNN-AE is the best simple default.

### GPU enabled
- Installed cu128 PyTorch for the RTX 5070 (Blackwell, sm_120). Deep models are device-aware.
- Verified: 21 tests pass on GPU; CPU/GPU equivalent; ~1.7× speedup (small models, so modest).

### Deep models
- CNN autoencoder, LSTM autoencoder, LSTM forecaster — unsupervised, per-point reconstruction/forecast
  error. All recall 1.00 on injected anomalies.

### Core pipeline (earlier in the day)
- preprocess (incl. −9999 sentinel cleaning) → analysis (STL) → anomaly (z-score, Isolation Forest)
  → clustering (K-means day-types) → **query engine** (exact, 6/6 vs pandas) → interface + RAG +
  reasoning. Dataset: Jena Climate 2009–2016 (70k hourly rows).
- Environment gotcha: Jupyter runs on **Python 3.12** (deps there), separate 3.14 venv.

---

## How to use this log
Add a dated entry whenever we build/try/decide something: what the problem was, what changed, what
the result was, and any bug or surprise. This feeds the report's methodology and discussion.
