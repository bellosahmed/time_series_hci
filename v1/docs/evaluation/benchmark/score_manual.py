"""Score a filled-in answers sheet from the commercial tools against the oracle answer key.

Usage:
    python3.12 docs/evaluation/benchmark/score_manual.py docs/evaluation/benchmark/answers_template.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V1))

from src import benchmark_external as bx  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    sheet = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "answers_template.csv"
    raw = bx.load_raw()
    truth = {i: bx.ground_truth(raw, c) for i, c in enumerate(bx.GOLD, 1)}

    with sheet.open() as f:
        rows = list(csv.DictReader(f))
    tools = [c for c in rows[0].keys() if c not in ("n", "question")]

    correct = {t: 0 for t in tools}
    answered = {t: 0 for t in tools}
    print(f"{'n':>2}  {'truth':>10}  " + "  ".join(f"{t:>12}" for t in tools))
    print("-" * (16 + 14 * len(tools)))
    for r in rows:
        i = int(r["n"])
        gt = truth[i]
        cells = []
        for t in tools:
            val = bx.parse_number(r[t])
            if val is not None:
                answered[t] += 1
                ok = bx.is_correct(val, gt)
                correct[t] += ok
                cells.append(f"{('✓ ' if ok else '✗ ')+format(val, '.2f'):>12}")
            else:
                cells.append(f"{'—':>12}")
        print(f"{i:>2}  {gt:>10.2f}  " + "  ".join(cells))
    print("-" * (16 + 14 * len(tools)))
    n = len(rows)
    for t in tools:
        print(f"  {t:12} {correct[t]}/{n} correct  ({answered[t]}/{n} answered)")


if __name__ == "__main__":
    main()
