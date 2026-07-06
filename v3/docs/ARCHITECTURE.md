# v3 — Code Structure, Architecture & Implementation Blueprint

Companion to [`PLAN.md`](PLAN.md). This is the *how*: the file layout, the data-flow architecture, the
data contracts between modules, and the implementation sketch (signatures + key algorithms) for each
module. Nothing here is built yet — it is the blueprint the M1→M6 build follows (TDD, like v1).

---

## 1. Code structure (file tree)

```
code/
├── data/dad/                         # REAL dataset (gitignored) — 7×{.pcap,_labels.csv,_labels.xml}
└── v3/
    ├── README.md
    ├── requirements.txt              # pandas, numpy, scikit-learn, statsmodels, scapy, flask, (torch)
    ├── pyproject.toml                # pytest/ruff/mypy config + puts src/ on path (copy v1)
    ├── reproduce.sh                  # tests + verify_results, with an environment banner
    ├── verify_results.py             # independent “== pandas” + known-totals gate
    ├── tools/
    │   └── get_dad_dataset.sh        # reproducible download (done)
    ├── docs/
    │   ├── PLAN.md
    │   └── ARCHITECTURE.md           # this file
    ├── src/
    │   ├── __init__.py
    │   ├── preprocess.py             # (1) pcap+labels → per-message DataFrame → week frame
    │   ├── features.py               # (1) message frame → value series + flow-feature windows
    │   ├── analysis.py               # rolling stats, STL/decomposition, correlations (port v1)
    │   ├── channels.py               # (2) value-channel score + flow-channel score
    │   ├── fusion.py                 # (2) early/late fusion + optional supervised combiner  ← headline
    │   ├── anomaly.py                # (2) z-score / IsolationForest / LOF / OCSVM (port v1)
    │   ├── forecasting.py            # (2) ARIMA residual on rate/temperature (port v1, optional)
    │   ├── deep.py                   # (2) windowed autoencoder, GPU-aware (port v1, optional)
    │   ├── evaluation.py             # (2) real-label P/R/F1 + point-adjusted + per-type + bootstrap
    │   ├── clustering.py             # (2) K-means day/window types (port v1)
    │   ├── query_engine.py           # (3) exact counts/stats — the truth layer
    │   ├── reasoning.py              # (4) why a window is flagged (which feature moved)
    │   ├── rag_system.py             # (4) KB retrieval (port v1)
    │   ├── llm.py                    # (4) Ollama phrasing + faithfulness gate (port v1)
    │   ├── interface.py              # (5) NL question → query engine → trust-aware answer
    │   ├── interface_parsing.py      # (5) deterministic intent/scope parsing (port v1 pattern)
    │   ├── agent.py                  # (5) optional multi-step tool-use (port v1)
    │   └── main.py                   # (5) end-to-end pipeline entry point
    ├── tests/                        # pytest, one file per module (mirrors v1)
    ├── notebooks/                    # 00_walkthrough … 04_detector_comparison
    └── apps/
        ├── chat.py                   # CLI
        └── web/                      # Flask + vendored React (reuse v1)
```

**Ported from v1 with little change:** `analysis.py`, `anomaly.py`, `forecasting.py`, `deep.py`,
`evaluation.py` (metrics), `clustering.py`, `rag_system.py`, `llm.py`, `interface_parsing.py`,
`agent.py`, the app layer. **Net-new for this domain:** `preprocess.py`, `features.py`, `channels.py`,
`fusion.py`, and the domain-specific `query_engine.py` / `interface.py` intents / KB.

---

## 2. Architecture (data flow)

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  code/data/dad/  ── 7× (capture_*.pcap  +  capture_*_labels.csv)             │
  └───────────────┬─────────────────────────────────────────────────────────────┘
                  │  preprocess.load_week()          scapy parse + join on frame.number
                  ▼
        ┌──────────────────────────┐   MESSAGES df  (one row per MQTT message)
        │  messages: DataFrame      │   index=ts; cols: frame, src_ip, clientid,
        │  (the single source       │   msgid, topic, temperature, label, attack_type
        │   of truth for v3)        │
        └───────┬──────────────┬────┘
                │              │
   features.sensor_series()    features.flow_features()          query_engine.*  (3. TRUTH)
                │  VALUE               │  FLOW                    exact counts/stats == pandas
                ▼                      ▼                                   │
     channels.value_score()   channels.flow_score()                       ▼
                │                      │                          interface (5) → llm (4, explain)
                └──────┬───────────────┘                                   │        faithfulness gate
                       ▼                                                    ▼
                 fusion.{early,late}()  ← HEADLINE                     apps/ (CLI, web)
                       │
                       ▼
      evaluation.evaluate(scores, REAL labels, point_adjust)  →  P/R/F1 per attack type + bootstrap CI
```

Two independent consumers of the MESSAGES frame: the **detection path** (features → channels → fusion →
evaluation, graded on real labels) and the **answer path** (query_engine → interface → llm → apps). The
LLM never touches the numbers on either path.

---

## 3. Data contracts (the schema everything agrees on)

**MESSAGES frame** (output of `preprocess.load_week`) — the contract the whole system depends on:

| column | dtype | meaning |
| --- | --- | --- |
| *(index)* `ts` | datetime64 | packet time (from pcap) |
| `frame` | int | `frame.number` (join key to labels) |
| `src_ip` | str | source IP (`10.6.56.41` = attacker) |
| `clientid` | str | MQTT client, e.g. `131` = InRow 13 / TAS sensor |
| `msgid` | int | MQTT message/token id |
| `topic` | str | MQTT topic (sensor identity) |
| `temperature` | float | value parsed from the PUBLISH payload (NaN if not a publish) |
| `label` | int8 | **real** ground truth: 0 normal / 1 attack |
| `attack_type` | category | modification / duplication / interception / `none` (from schedule) |

**VALUE frame** (`features.sensor_series`): index=window ts, columns = one temperature column per sensor.
**FLOW frame** (`features.flow_features`): index=window ts, columns = the flow features (below).
**WINDOW labels** (`features.window_labels`): index=window ts → `label`, `attack_type` (a window is an
attack if any labelled packet falls in it — standard weak labelling for windowed detection).

---

## 4. Implementation blueprint (module by module)

### 4.1 `preprocess.py` — pcap + labels → MESSAGES frame  *(M1, the main new build)*
```python
from scapy.all import PcapReader
from scapy.contrib.mqtt import MQTT, MQTTPublish

def parse_pcap(pcap_path: str) -> pd.DataFrame:
    """One row per packet: frame, ts, src_ip, clientid, msgid, topic, temperature."""
    rows = []
    for i, pkt in enumerate(PcapReader(pcap_path), start=1):   # frame.number is 1-based
        if IP not in pkt:
            continue
        rec = {"frame": i, "ts": float(pkt.time), "src_ip": pkt[IP].src,
               "clientid": None, "msgid": None, "topic": None, "temperature": np.nan}
        if pkt.haslayer(MQTT):
            m = pkt[MQTT]
            rec["msgid"] = getattr(m, "msgid", None)
            if pkt.haslayer(MQTTPublish):                       # temperature lives here
                p = pkt[MQTTPublish]
                rec["topic"] = _as_str(p.topic)
                rec["temperature"] = _parse_temperature(p.value)   # ← confirm format on real data
        rows.append(rec)
    return pd.DataFrame(rows)

def load_day(pcap_path, labels_csv_path) -> pd.DataFrame:
    pkts = parse_pcap(pcap_path)
    labels = pd.read_csv(labels_csv_path, sep=";")             # frame.number;...;label
    df = pkts.merge(labels[["frame.number", "mqtt.clientid", "label"]],
                    left_on="frame", right_on="frame.number", how="left")
    df["label"] = df["label"].fillna(0).astype("int8")
    return df

def load_week(data_dir=DATA_DIR) -> pd.DataFrame:
    """Concat the 7 days, add attack_type from the README schedule, set the ts index."""
    df = pd.concat([load_day(p, c) for p, c in _day_files(data_dir)]).sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df["attack_type"] = _attack_type_from_schedule(df["ts"], df["label"])
    return df.set_index("ts")
```
> **Two things get confirmed against the *real* pcap on first run** (Open Q #2): the exact
> `_parse_temperature` payload format and the topic→sensor mapping. The flow channel and the real-label
> evaluation do **not** depend on either.

### 4.2 `features.py` — the two feature families  *(M1)*
```python
def sensor_series(df, freq="5min") -> pd.DataFrame:
    """VALUE channel: temperature per sensor (clientid), resampled to a common grid."""
    pub = df.dropna(subset=["temperature"])
    return (pub.pivot_table("temperature", index=pub.index, columns="clientid")
               .resample(freq).mean())

def flow_features(df, window="1min") -> pd.DataFrame:
    """FLOW channel: per-window network-traffic features (where dup/interception live)."""
    g = df.resample(window)
    feats = pd.DataFrame({
        "msg_rate":       g.size(),                                   # packets per window
        "iat_mean":       g.apply(lambda w: w.index.to_series().diff().dt.total_seconds().mean()),
        "iat_std":        g.apply(lambda w: w.index.to_series().diff().dt.total_seconds().std()),
        "dup_msgid_ratio":g["msgid"].apply(lambda s: 1 - s.nunique()/max(len(s), 1)),  # duplication
        "unique_src":     g["src_ip"].nunique(),
        "attacker_share": g["src_ip"].apply(lambda s: (s == ATTACKER_IP).mean()),
        "publish_ratio":  g["topic"].apply(lambda s: s.notna().mean()),
    })
    return feats.fillna(0.0)

def window_labels(df, window="1min") -> pd.DataFrame:
    lab = df["label"].resample(window).max().fillna(0).astype("int8")   # attack if ANY in window
    typ = df["attack_type"].resample(window).agg(_dominant_type)
    return pd.DataFrame({"label": lab, "attack_type": typ})
```

### 4.3 `channels.py` — one score per channel  *(M3)*
```python
def value_score(sensor_df) -> pd.Series:
    """Per-window anomaly score from the temperature residual (catches MODIFICATION).
    Deseasonalise each sensor (STL or a rolling model), z-score the residual, take the
    max |z| across sensors per window."""
    resid = analysis.deseasonalise(sensor_df)          # reuse v1
    return resid.abs().max(axis=1)

def flow_score(flow_df, method="iforest") -> pd.Series:
    """Per-window anomaly score from flow features (catches DUPLICATION/INTERCEPTION)."""
    model = anomaly.fit_detector(flow_df, method)      # reuse v1: iforest/lof/ocsvm/zscore
    return pd.Series(-model.score_samples(flow_df), index=flow_df.index)   # higher = more anomalous
```

### 4.4 `fusion.py` — the headline experiment  *(M3)*
```python
def _norm(s):                                   # comparable scales before combining
    return (s.rank(pct=True))                    # rank-normalise to [0,1]

def late_fusion(value_s, flow_s, rule="max", w=0.5) -> pd.Series:
    v, f = _norm(value_s), _norm(flow_s)
    if rule == "max": return pd.concat([v, f], axis=1).max(axis=1)   # OR-like union
    if rule == "mean":return w*v + (1-w)*f
    raise ValueError(rule)

def early_fusion(value_feats, flow_feats, method="iforest") -> pd.Series:
    X = pd.concat([value_feats, flow_feats], axis=1).fillna(0.0)     # concat → ONE detector
    model = anomaly.fit_detector(X, method)
    return pd.Series(-model.score_samples(X), index=X.index)

def supervised_combiner(value_s, flow_s, labels):                   # optional, reported separately
    """LogReg over the two channel scores, day-blocked CV (only ~544 positives → overfit risk)."""
    ...
```
The M3 harness runs **flow-only / value-only / early / late**, then
`evaluation.per_type_report(...)` grades each against the real labels **per attack type**.

### 4.5 `evaluation.py` — graded on REAL labels  *(M3, port v1 + add per-type)*
```python
def evaluate(scores, labels, threshold=None, point_adjust=True) -> dict:  # P/R/F1/(+PA), confusion
def per_type_report(scores, labels, attack_type, threshold) -> pd.DataFrame:
    """One row per config × attack type — the table that *explains* why fusion wins."""
def bootstrap_ci(scores, labels, metric="f1", n=1000) -> tuple[float, float]:
```

### 4.6 `query_engine.py` — the truth layer  *(M2, == pandas)*
```python
def count_attacks(df, scope=None, attack_type=None) -> int
def attack_breakdown(df) -> dict[str, int]          # {modification: .., duplication: .., ...}
def attacker_ips(df) -> list[str]                   # computed from labelled packets → ['10.6.56.41']
def message_rate(df, freq="1H") -> pd.Series
def temperature_stat(df, sensor, stat, scope=None) -> float   # max/min/mean over parsed payloads
def removed_packet_report(df) -> dict               # Tuesday’s interception (unlabelled) edge case
```
Every function has a one-line pandas equivalent in `verify_results.py` (e.g. total attacks == 544,
per-day counts, attacker IP) — the independent trust gate.

### 4.7 `interface.py` — NL → answer  *(M4, port v1 pattern)*
Deterministic intent routing (`interface_parsing.py`) → `query_engine` → trust-labelled `Answer`
(🔢 computed / 📚 context / 🧠 inferred). Domain intents: `count` · `when` · `attack_type` · `attacker`
· `sensor_stat` · `rate` · `anomaly(+reasoning)` · `summary` · `compare`. LLM (`llm.py`) only rephrases;
faithfulness gate rejects any invented number — identical guarantee to v1.

---

## 5. Test plan (TDD, one file per module)

| Test file | Locks in |
| --- | --- |
| `test_preprocess.py` | week frame row count ≈ 101.6k; **label total == 544**; per-day counts match the schedule; join keys unique |
| `test_features.py` | window labelling (attack iff a labelled packet inside); `dup_msgid_ratio` rises on Thu; `iat_mean` rises on Tue |
| `test_channels.py` | value_score flags a modified window; flow_score flags a duplicated window |
| `test_fusion.py` | late-fusion recall ≥ max(single-channel recall) on a crafted mixed set; early-fusion API |
| `test_evaluation.py` | P/R/F1 vs hand-computed; point-adjust behaviour; per-type table shape |
| `test_query_engine.py` | every stat **== pandas** on the real frame |
| `test_interface.py` | each intent routes to the right query; trust labels present |

`verify_results.py`: independent recomputation (544 attacks, per-day distribution, attacker IP,
temperature extrema, one detection metric) — the gate that proves the numbers.

---

## 6. Build order (maps milestones → files)

1. **M1** `preprocess.py` + `features.py` (+ tests) — parse one real pcap first to fix the payload
   format, then the week frame + windows.
2. **M2** `query_engine.py` (+ tests + verify) — exact counts/stats on the frame.
3. **M3** `channels.py` + `fusion.py` + `evaluation.py` — the two-channel + fusion experiment, graded
   per attack type on real labels (**the headline**).
4. **M4** `interface.py` (+ `interface_parsing.py`, `rag_system.py`) — the NL layer.
5. **M5** `llm.py` wiring + faithfulness gate.
6. **M6** `apps/`, notebooks, `report/`, quality gates (pytest/verify/ruff/mypy) green.

Each step is red→green→refactor with the real dataset, exactly as v1 was built.
```
