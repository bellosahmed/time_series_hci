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

## Future work (prioritised)

### High-value, cross-cutting
1. **A unified multi-domain framework.** Factor the shared truth-layer + interface + faithfulness gate into
   one installable package that the three domains plug into — turning "three sibling systems" into "one
   framework, three adapters." This is the strongest research framing.
2. **Larger / better local models** (e.g. 8B, or a fine-tuned small model) to reduce wrong *qualifiers* —
   the one residual the numeric gate can't catch — while staying fully local.
3. **Strengthen the faithfulness gate** to close the small-integer gap (e.g. entity-aware checking of
   counts and IPs), and add *qualifier* faithfulness (not just numeric).

### Detection / evaluation
4. **Real-time / streaming detection** — move from retrospective batch to an operational monitor (all three
   domains have a live analogue).
5. **v3: better detectors** — a seasonal value model, tuned Isolation-Forest contamination, and a stronger
   flow channel; more attack types and more captures to lift the tiny per-type strata.
6. **Causal, not just correlational** reasoning — e.g. does a pressure drop *cause* the humidity spike
   (v1), or does one country's shock *cause* another's (v2)? Causal discovery done responsibly.
7. **Cross-domain transfer studies** — can a detector/retriever trained on one domain inform another?

### HCI / deployment
8. **A human evaluation** of the interface (task accuracy vs a spreadsheet, trust, SUS) to substantiate the
   "usable, trustworthy" claim beyond design argument.
9. **On-prem hardening** — the offline Open WebUI + Ollama + GPU stack (documented in
   `v3/local_webui_dataset_spec.ipynb`) taken to a real air-gapped deployment with a domain user.

## The one-line conclusion
Across weather, economics, and network security, a **deterministic truth layer + an explanation-only LLM +
a faithfulness gate** delivers **verifiable** natural-language analytics — and the project's most durable
lesson is that **honest evaluation (real labels, event-level metrics, confidence intervals, blocked CV)
matters more than model complexity**, with simple methods repeatedly matching or beating complex ones.
