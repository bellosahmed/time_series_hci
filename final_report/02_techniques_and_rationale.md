# 02 — Techniques and Rationale (*why this, not that*)

Every significant design choice, the alternative it was chosen over, and the reason. Grouped by layer.

## A. The core trust design

| Choice | Alternative rejected | Why |
| --- | --- | --- |
| **Deterministic truth layer** computes every number | Let the LLM compute/answer directly | LLMs **hallucinate numbers**; a confidently wrong figure is worse than none. Exact pandas ops are verifiable and cannot be fabricated. |
| **LLM = explanation only** + **faithfulness gate** | Trust the LLM's prose | The gate *structurally* guarantees safety: numbers are exact upstream and re-checked downstream, so even a **small 3B local model** is safe to use. |
| **Local Ollama** (on-prem) | Cloud LLM API (OpenAI etc.) | **Privacy** (data never leaves the host), **offline** operation, **zero cost**, and **reproducibility** (pinned local model). |
| **`verify_results.py` independent harness** | Trust the unit tests alone | A *second, independent* recomputation (e.g. summing raw label files, or `== pandas`) catches bugs the tests share; it is the trust *gate*, not just coverage. |
| **Trust labels** (🔢/📚/🧠) | Undifferentiated prose | The user can see *what is computed vs contextual vs inferred* — trust is legible, not asserted. |

## B. Anomaly detection

| Choice | Alternative rejected | Why |
| --- | --- | --- |
| **Isolation Forest / z-score as the workhorses** | Deep learning by default | On **every** dataset, deep/learned methods **lost or tied** simpler ones (see §D and `03_results`). Simple = faster, interpretable, and no over-fitting on small data. |
| **STL decomposition** for the value/residual signal | Raw values, or plain moving average | STL cleanly separates trend + **seasonality** + residual; anomalies live in the residual. (v3's temperatures are strongly daily-seasonal; v1/v2 too.) |
| **A *family* of detectors, compared** | Pick one detector and defend it | Different detectors catch different anomaly *shapes*; comparing them on one harness is more honest and informative than advocacy. |
| **v3: two-channel design (value + flow)** | One detector on all features | The attacks are **disjoint in signature** — modification shows in the *value*, dup/intercept in the *flow*. No single view sees all three. |
| **v3: late (OR) fusion** | Early fusion / a single big model | Late fusion preserves each specialist's sensitivity; early fusion **dilutes** a sharp single-channel signal in a high-dim space (measured). |
| **v3: exclude the attacker-IP feature** | Use it (it's highly predictive) | The attacker IP appears **only** during attacks — using it is **label leakage** that makes detection trivially perfect and proves nothing. Behavioural features keep the result honest. |

## C. Evaluation methodology (the real backbone)

| Choice | Alternative rejected | Why |
| --- | --- | --- |
| **Average precision + F1, never accuracy** | Accuracy | Under heavy imbalance (v3 21:1 windows) accuracy is meaningless (always-normal ≈ 95 %). AP is threshold-free and imbalance-honest. |
| **Point-adjusted (event-level) metrics** *alongside* strict | Strict point metrics only | Anomalies are **events/segments**; catching any point of an episode is a real detection. We report **both**, never just the flattering one. |
| **Bootstrap confidence intervals** | Single point estimates | v3's "value-only beats fusion" turned out **not significant** — only the CIs revealed it. Point estimates mislead. |
| **Per-attack-type recall** (v3) | One aggregate score | The aggregate hides *why* fusion helps; the per-type table **proves** channel complementarity (flow 0.00 on modification, 1.00 on duplication). |
| **Day-blocked CV** for the supervised combiner (v3) | Random-split CV | Attacks are **episode-clustered**; a random split leaks the same episode into train and test. Day-blocked is the honest split — and it exposed the supervised model's overfitting. |
| **Real labels (v2, v3)** | Synthetic only (v1) | External validity — detectors are tested on events that *actually occurred*. |
| **Do NOT resample the imbalance** (v3) | SMOTE / over/under-sampling | Rebalancing a **security** dataset falsifies its real rarity and leaks across clustered episodes; imbalance-aware *metrics* are the right fix, not fake data. |
| **Fill value-channel gaps but NOT flow gaps** (v3) | Fill everything, or fill nothing | A *missing message is the interception signal* — filling flow gaps erases attacks. The value channel is interpolated so its residual isn't distorted. |

## D. Deep learning — used, benchmarked, and reported honestly

Deep models were **implemented in every version** (v1: CNN-AE, LSTM-AE, LSTM-forecaster; v2: multivariate
AE; v3: supervised combiner) — but **not privileged**. In each case a simpler method matched or beat them:

- **v1:** no single detector is best; deep detectors tie Isolation Forest on the controlled signal.
- **v2:** the autoencoder (F1 **0.21**) **loses** to Isolation Forest (**0.36**) — reported as a loss.
- **v3:** the supervised (learned) fusion is the **worst** config under honest day-blocked CV.

**Why keep them then?** To *measure* the trade-off rather than assume it — and because "we tried the
sophisticated method and it didn't help *here*, and here's why" is a stronger, more defensible result
than either hype or omission. (Deep models earn their keep on large, i.i.d. data — not the small,
clustered, real-label regimes these datasets provide.)

## E. Retrieval / RAG

| Choice | Alternative rejected | Why |
| --- | --- | --- |
| **Compare TF-IDF / BM25 / LSA / dense** (v1) with IR metrics | Assume dense embeddings win | Measured, not asserted: keyword methods **tie** on in-vocabulary queries; dense wins on **paraphrases** (MRR 0.267 → 0.500). Document quality often dominates. |
| **Local embeddings** (MiniLM / Ollama) | Hosted embedding API | Offline + private, consistent with the whole-stack principle. |

## F. Engineering / infrastructure

| Choice | Alternative rejected | Why |
| --- | --- | --- |
| **scapy** for pcap parsing (v3) | `tshark` / pyshark | Pure-Python, **no sudo**, has an MQTT dissector — installs and runs anywhere. |
| **Vendored React (v1) / plain HTML+JS (v3)** web UI | A Vite/webpack build | **Zero build step**, fully offline, one-command run — matches the local/self-contained ethos. |
| **`frame.number` join** (v3) | Fuzzy timestamp matching | `frame.number` equals the packet index, so the label join is **exact** — no guesswork. |
| **TDD throughout** | Test-after | Every module was red→green; a failing test even *discovered* findings (v3's 5-min cadence, the plural-"attacks" parser bug). |
| **On-prem Docker + GPU pass-through** (Open WebUI spec) | Cloud deployment | Privacy-first, air-gappable; `--gpus all` uses the local RTX 5070; no external calls. |
