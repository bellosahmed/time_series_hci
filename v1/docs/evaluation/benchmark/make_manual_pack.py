"""Generate the manual benchmark pack for commercial tools that can't be scripted from here
(ChatGPT, Claude, Kimi, GLM — no API keys in this environment).

Writes:
  * manual_pack.md        — copy/paste instructions + the exact questions to ask each tool
  * answers_template.csv  — a blank sheet; fill one column per tool with the number it gave
  * answer_key.csv        — the oracle ground truth (kept separate so filling the sheet stays blind)

Then score with:  python3.12 docs/evaluation/benchmark/score_manual.py answers_template.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V1))

from src import benchmark_external as bx  # noqa: E402

HERE = Path(__file__).resolve().parent
TOOLS = ["chatgpt", "claude", "kimi", "glm"]


def main() -> None:
    raw = bx.load_raw()
    keys = [(c, bx.ground_truth(raw, c)) for c in bx.GOLD]

    # --- instructions + questions ---------------------------------------------------------------
    lines = [
        "# Manual benchmark pack — commercial LLMs as contestants",
        "",
        "These tools can't be called from the project environment (no API keys), so run them by "
        "hand and record what they say. **Do this identically for each tool** so the comparison is fair.",
        "",
        "## Procedure (do it the way a real user would)",
        "",
        "1. Upload the dataset file `data/jena_climate_2009_2016.csv` to the tool "
        "(ChatGPT, Claude, Kimi and GLM all accept file uploads + can run code on them).",
        "2. Paste this framing once:",
        "",
        "   > This CSV has a `Date Time` column (format `dd.mm.yyyy HH:MM:SS`, 10-minute readings) "
        "and weather sensor columns. Some rows use `-9999` as a missing-value code — treat those as "
        "missing. Answer each question with just the number, rounded to 2 decimals.",
        "",
        "3. Ask the questions below **verbatim, one at a time**, and write the number each tool "
        "returns into `answers_template.csv` (one column per tool).",
        "",
        "## Questions",
        "",
    ]
    for i, case in enumerate(bx.GOLD, 1):
        lines.append(f"{i}. {case.question}")
    lines += [
        "",
        "## Notes to watch for (these are the interesting failure modes)",
        "",
        "- Does the tool silently average to hourly first (biasing max/min)? Ours computes on the "
        "native 10-minute grid.",
        "- Does it keep the dataset's **duplicate timestamps** (183 in 2014 alone)? Ours drops them.",
        "- Does it invent a number when the code errors, instead of saying it failed?",
        "",
        "When the sheet is filled: `python3.12 docs/evaluation/benchmark/score_manual.py "
        "docs/evaluation/benchmark/answers_template.csv`",
        "",
    ]
    (HERE / "manual_pack.md").write_text("\n".join(lines))

    # --- blank answer sheet ---------------------------------------------------------------------
    with (HERE / "answers_template.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "question", *TOOLS])
        for i, case in enumerate(bx.GOLD, 1):
            w.writerow([i, case.question, *["" for _ in TOOLS]])

    # --- separate answer key --------------------------------------------------------------------
    with (HERE / "answer_key.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "question", "sensor", "kind", "truth"])
        for i, (case, gt) in enumerate(keys, 1):
            w.writerow([i, case.question, case.sensor, case.kind, f"{gt:.4f}"])

    print("Wrote manual_pack.md, answers_template.csv, answer_key.csv to", HERE.relative_to(V1))


if __name__ == "__main__":
    main()
