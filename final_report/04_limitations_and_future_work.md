# 04 — Limitations and Future Work

Stated honestly — the same discipline as the evaluation. Limitations first (per version, then shared),
then a prioritised roadmap.

## Limitations

### v1 — Weather
- **Synthetic anomaly labels.** Jena has no ground truth, so detection precision is measured on injected
  spikes or a controlled signal — honest, but it caps the strength of the detection claim.
- **Single primary dataset** → external-validity risk (partly answered by v2/v3 being other domains).
- **Weak local model.** A 3B LLM: the faithfulness gate guarantees *numbers*, but *qualifiers* (adjectives,
  framing) rely on prompting and can still be imperfect.
- **No human evaluation.** For an "HCI interface", usability/trust are argued from design, not measured
  with users (a deliberately de-scoped item).
- **Agentic layer is bounded by the 3B model (numbers are not).** The added Understanding agent,
  multi-agent orchestrator, and their critics are all **model-bound in quality**: the 3B model produces
  the occasional plausible-but-wrong query rewrite, mis-assigns a sub-question's role, and runs a
  **noisy LLM-critic that over-flags**; the orchestrator's multi-part coverage is even **non-deterministic
  run-to-run** (2/3 vs 3/3). None of this touches correctness — every number is still computed by the
  truth layer, mis-routes surface a *correct number under a visible interpretation*, and the trustworthy
  gate is the **deterministic** critic (the LLM-critic is advisory). The honest reading: the agentic layer
  widens *understanding and coverage*, but its language-level judgments are the weakest link and scale
  with model strength. The **COSMA harness cluster run was not executed** — its core and generated
  `sbatch` are verified offline; DiRAC submission remains an untested (if reproducible) step.

### v2 — Trade
- **Coarse real labels.** Ground truth is crisis *windows* (~14 anomaly months), so per-point precision is
  necessarily low; point-adjusted metrics are the fairer read.
- **Little data for deep models.** 292 monthly points is small — a reason the autoencoder underperforms.
- **Monthly resolution** hides intra-month dynamics; some shocks are gradual.

### v3 — IoT network security
- **"Semi-synthetic" dataset.** The authors generated the normal traffic from an STL model; it is one data
  center, 7 days, one attacker IP.
- **Binary labels.** 71 % of attacks are only typeable as "mixed" (Fri/Sat interleave all three types), so
  clean per-type scores exist only for Wednesday (modification) and Thursday (duplication, just 4 windows).
- **Strong imbalance** (186:1 / 21:1) and a **weak flow channel** (few cleanly separable windows) cap the
  absolute detection scores.
- **Untuned detectors** and window-level weak labels; the analysis is retrospective, not streaming.

### Shared across all three
- **Small local models** — the faithfulness gate has a residual gap (it tolerates small integers ≤ 12, so a
  fabricated *small* count could slip through).
- **Retrospective, batch** analysis — not real-time / operational.
- **RAG is text-oriented** — raw sensor/pcap data must be summarised to markdown/CSV before indexing.

## Threats to validity

The limitations above, restated as the standard validity threats — *what could make the reported
conclusions wrong*, and why we judge the headline claims to survive.

- **v1 — Weather.** The chief threat is **construct validity**: detection is scored on *injected* faults,
  so "F1/AP on Jena" measures response to a synthetic proxy, not to real weather anomalies. We mitigate by
  (a) reporting on multiple fault *geometries* (spike, drift, stuck, phase, oscillation) rather than one, so
  the ranking reflects a method's inductive bias not a single label choice, and (b) separating the *truth
  layer* (query engine, verified `== pandas`) from the detection claim, so the exact-analytics contribution
  does not depend on the labels at all. **External validity** (one dataset) is partly answered by v2/v3
  being other domains. The **conclusion** that *no single detector wins and Isolation Forest is a poor
  multivariate default* is robust: it is a mechanistic result (axis-aligned splits dilute a single-channel
  fault), independently re-derived, and now encoded in the shipped `sensor_network_anomalies` portfolio.
- **v2 — Trade.** The chief threat is **conclusion validity** from small n and coarse labels: 292 monthly
  points with ~14 crisis *months* make strict per-point precision unstable and starve the deep models. We
  mitigate with point-adjusted metrics and an out-of-sample SARIMA backtest, and we do **not** over-claim —
  the openly-reported negative result (autoencoder 0.21 < Isolation Forest 0.36) is exactly the conclusion
  small data supports. **Construct validity** is comparatively strong here: the anomalies (2008, COVID,
  2022) are real, datable events, not injected.
- **v3 — Network security.** The chief threats are **construct and external validity**: the traffic is
  "semi-synthetic" (STL-modelled normal), from one data centre, 7 days, one attacker, and 71% of attacks are
  only typeable as "mixed." We mitigate by using the **real per-packet labels** (the best in the project),
  excluding the attacker-IP feature to avoid **label leakage** (an internal-validity safeguard), and
  reporting **bootstrap confidence intervals** and per-type recall so the modest absolute AP (0.10–0.18) is
  presented with its uncertainty rather than as a point estimate. The honest conclusion — *no fusion config
  is significantly best on AP* — is a direct consequence of taking those CIs seriously.
- **Shared.** A residual **construct-validity** threat to the trust claim is the faithfulness gate's
  tolerance of small integers (≤ 12), so a fabricated *small* count could in principle slip past the numeric
  check; qualifiers (adjectives, framing) are prompt-controlled, not gated. All three systems are
  **retrospective/batch**, so conclusions about operational, streaming use are extrapolations, not measured.

## Future work (prioritised)

### High-value, cross-cutting
1. **A unified multi-domain framework.** Factor the shared truth-layer + interface + faithfulness gate into
   one installable package that the three domains plug into — turning "three sibling systems" into "one
   framework, three adapters." This is the strongest research framing.
2. **Larger / better local models** (e.g. 8B, or a fine-tuned small model) to reduce wrong *qualifiers* —
   the one residual the numeric gate can't catch — while staying fully local.
3. **Strengthen the faithfulness gate** to close the small-integer gap (e.g. entity-aware checking of
   counts and IPs), and add *qualifier* faithfulness (not just numeric).
4. **Run the agentic layer on DiRAC/COSMA and re-measure with a stronger model.** The batch harness
   (`batch_harness.py`) already fans the evaluations out as a SLURM array job and is proven
   *result-preserving* across shard counts; the untested step is the cluster submission itself. Doing so
   would (a) quantify the 3B **non-determinism** cheaply by fanning the same jobs out many times, and
   (b) let a larger local model be benchmarked at scale on the decomposition/critic quality that is the
   agentic layer's weakest link — directly attacking limitation "agentic layer is bounded by the 3B
   model" above, while the trust gates stay model-independent.

### Detection / evaluation
4. **Real-time / streaming detection** — move from retrospective batch to an operational monitor (all three
   domains have a live analogue).
5. **v3: lift the absolute detection scores — but the real lever is data, not tuning.** The obvious tuning
   knobs were *tried and did not help* (recorded honestly): a seasonal, time-of-day value model **hurt**
   (AP 0.183 → 0.099 — the data-centre temperature has no exploitable diurnal cycle for the attack to
   violate, and the per-hour baseline is sample-starved), and a Mahalanobis/covariance flow channel was
   **within the confidence interval** (AP 0.136 → 0.125). Both are kept as tested optional methods, but the
   defaults stay global-z + Isolation Forest. The genuine remaining lever is therefore **more real captures**
   (more attack types, more days, more than one attacker) to enlarge the tiny per-type strata — which the
   **fixed DAD dataset (one attacker, 7 days) cannot provide**. This is a *dataset ceiling*, not a modelling
   gap: closing it means a new capture campaign or a second labelled network-security corpus, not more tuning.
6. **v2: a higher-resolution companion series so the deep models and point metrics can breathe.** With only
   292 *monthly* points and ~14 crisis months, per-point precision is unstable and the autoencoder is
   starved. Official trade statistics are, however, **published monthly at source** — so daily/weekly *trade*
   values do not exist to fetch. The honest route is a **higher-frequency economic *proxy*** for the same
   shocks — e.g. a daily/weekly shipping or freight-rate index, port-call / container-throughput data, or an
   equity/commodity series that co-moves with trade — run through the *same* pipeline as a companion. This is
   a deliberate **dataset decision, not a silent swap**: the proxy measures the shock at higher resolution but
   is a different variable, so it must be framed as corroboration of the monthly finding, not a replacement.
7. **Causal, not just correlational** reasoning — e.g. does a pressure drop *cause* the humidity spike
   (v1), or does one country's shock *cause* another's (v2)? Causal discovery done responsibly.
8. **Cross-domain transfer studies** — can a detector/retriever trained on one domain inform another?

### HCI / deployment
9. **A human evaluation** of the interface (task accuracy vs a spreadsheet, trust, SUS) to substantiate the
   "usable, trustworthy" claim beyond design argument.
10. **On-prem hardening** — the offline Open WebUI + Ollama + GPU stack (documented in
    `v3/local_webui_dataset_spec.ipynb`) taken to a real air-gapped deployment with a domain user.

## The one-line conclusion
Across weather, economics, and network security, a **deterministic truth layer + an explanation-only LLM +
a faithfulness gate** delivers **verifiable** natural-language analytics — and the project's most durable
lesson is that **honest evaluation (real labels, event-level metrics, confidence intervals, blocked CV)
matters more than model complexity**, with simple methods repeatedly matching or beating complex ones.
