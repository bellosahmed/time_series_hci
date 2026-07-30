"""Run the automated leg of the comparative benchmark: OUR APP vs LOCAL Ollama models,
scored against the oracle ground truth. Emits a results table (and a CSV of raw replies for audit).

Usage:
    python3.12 docs/evaluation/benchmark/run_local.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V1))

from src import benchmark_external as bx  # noqa: E402
from src import interface, llm, preprocess  # noqa: E402


def main() -> None:
    raw_csv = preprocess.clean(preprocess.load_data(str(bx.DATA)))
    df = preprocess.align(raw_csv, freq="1h")
    iface = interface.Interface(df, raw=raw_csv)      # LLM off: pure deterministic pipeline
    oracle_raw = bx.load_raw()

    local_models: list[str] = []
    if llm.is_available():
        local_models = llm.list_models()
    print(f"Local models available: {local_models or '(none — is `ollama serve` running?)'}\n")

    contestants = ["our app"] + local_models
    correct = {c: 0 for c in contestants}
    rows = []

    for case in bx.GOLD:
        gt = bx.ground_truth(oracle_raw, case)
        result = {"question": case.question, "sensor": case.sensor, "kind": case.kind, "truth": gt}

        app_val = bx.ask_app(iface, case)
        result["our app"] = app_val
        correct["our app"] += bx.is_correct(app_val, gt)

        for m in local_models:
            val, reply = bx.ask_local(m, oracle_raw, case)
            result[m] = val
            result[m + "__reply"] = reply.replace("\n", " ")[:200]
            correct[m] += bx.is_correct(val, gt)
        rows.append(result)

    # ---- results table -------------------------------------------------------------------------
    n = len(bx.GOLD)
    qw = max(len(r["question"]) for r in rows)
    header = f"{'question'.ljust(qw)}  {'truth':>10}  " + "  ".join(f"{c[:16]:>16}" for c in contestants)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['question'].ljust(qw)}  {r['truth']:>10.2f}  "
        cells = []
        for c in contestants:
            v = r.get(c)
            ok = bx.is_correct(v, r["truth"])
            cell = f"{v:.2f}" if isinstance(v, float) else "—"
            cells.append(f"{('✓ ' if ok else '✗ ')+cell:>16}")
        print(line + "  ".join(cells))
    print("-" * len(header))
    print("ACCURACY (exact within ±0.01):")
    for c in contestants:
        print(f"  {c:24} {correct[c]}/{n}  ({100*correct[c]/n:.0f}%)")

    # ---- audit CSV -----------------------------------------------------------------------------
    out = Path(__file__).resolve().parent / "results_local.csv"
    fields = ["question", "sensor", "kind", "truth", *contestants,
              *[m + "__reply" for m in local_models]]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nRaw replies + parsed values written to {out.relative_to(V1)}")


if __name__ == "__main__":
    main()
