"""Independent verification gate for v3 — recompute the headline figures a different way and confirm
the query engine agrees with pandas and with the paper's ground truth. Run: `python3.12 verify_results.py`.

The strongest check reads the 7 raw label CSVs directly (a path independent of the pcap parsing) and
confirms the attack total, so a bug in either the parser or the query engine cannot pass silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

V3 = Path(__file__).resolve().parent
sys.path.insert(0, str(V3))

from src import features, preprocess  # noqa: E402
from src import query_engine as qe  # noqa: E402


def main() -> int:
    df = preprocess.load_week()
    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))
        print(f"  {'✓' if ok else '✗'}  {name}")

    # (1) independent path: sum the label column straight from the raw CSVs (no pcap involved)
    raw = pd.concat([pd.read_csv(preprocess.DATA_DIR / f"capture_{s}_labels.csv", sep=";")
                     for s in preprocess.DAY_STAMPS.values()])
    raw_total = int(raw["label"].sum())
    check("attack total == 544 from the raw label CSVs (independent of the pcap path)", raw_total == 544)
    check("query_engine.count_attacks == raw-CSV total == 544",
          qe.count_attacks(df) == raw_total == 544)

    # (2) per-day schedule and type breakdown
    check("per-day attacks == paper schedule [0,0,104,56,192,192,0]",
          list(qe.attacks_by_day(df).values()) == [0, 0, 104, 56, 192, 192, 0])
    check("attack_breakdown sums to 544", sum(qe.attack_breakdown(df).values()) == 544)
    check("attacker IP == ['10.6.56.41']", qe.attacker_ips(df) == ["10.6.56.41"])

    # (3) totals
    check("total messages == 101,583", qe.total_messages(df) == 101583 == len(df))
    check("temperature readings == 32,288", int(df["temperature"].notna().sum()) == 32288)

    # (4) query engine == pandas (exact stats)
    for stat in ("max", "min", "mean"):
        check(f"temperature {stat} == pandas",
              qe.temperature_stat(df, stat) == float(getattr(df["temperature"].dropna(), stat)()))

    # (5) feature invariants: every packet/reading counted exactly once
    f = features.flow_features(df, "5min")
    check("flow msg_rate sums to total packets", int(f["msg_rate"].sum()) == len(df))
    check("flow publish_rate sums to readings", int(f["publish_rate"].sum()) == 32288)

    n_pass = sum(ok for _, ok in checks)
    print("=" * 64)
    print(f"{n_pass}/{len(checks)} checks passed.")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
