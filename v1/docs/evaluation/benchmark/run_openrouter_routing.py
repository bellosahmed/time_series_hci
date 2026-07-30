"""Routing/understanding contest: can hosted LLMs map a plain-English question to the right
(intent, sensors, has_scope) as well as our deterministic parser?

Complements the numeric benchmark (`run_openrouter.py`, which tests *computation*). Here the task
is *understanding* — the same job scored in `docs/evaluation/06-query-routing-coverage.md`. Each
model gets the identical classification contract used by `llm.independent_intent_judgment`; scores
are against the hand-authored `query_coverage.GROUND_TRUTH`, side by side with our app.

KEY: same as run_openrouter.py (OPENROUTER_API_KEY env or .openrouter_key file).

USAGE:
    python3.12 docs/evaluation/benchmark/run_openrouter_routing.py
    python3.12 docs/evaluation/benchmark/run_openrouter_routing.py --models inclusionai/ling-3.0-flash:free
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V1))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_openrouter as ro  # noqa: E402

from src import benchmark_external as bx  # noqa: E402
from src import interface, preprocess  # noqa: E402
from src import query_coverage as qc  # noqa: E402
from src.interface_parsing import INTENTS  # noqa: E402

HERE = Path(__file__).resolve().parent
# fast + reliable free models by default (the slow/rate-limited ones make 49 questions impractical)
DEFAULT_MODELS = ["inclusionai/ling-3.0-flash:free", "nvidia/nemotron-3-super-120b-a12b:free"]

JUDGE_SYSTEM = ("You are a strict classifier of weather-data questions. Reply with JSON only, no "
                "prose. Choose the single best intent from the allowed list, list only sensor "
                "columns actually named or clearly implied, and decide if a specific time window "
                "is named.")


def judge_prompt(question: str, intents: list[str], sensors: list[str]) -> str:
    return (
        f"{JUDGE_SYSTEM}\n\n"
        f"Question: {question}\n"
        f"Allowed intents: {intents}\n"
        f"Allowed sensor columns: {sensors}\n"
        'Return JSON ONLY: {"intent": <one allowed intent or null>, '
        '"sensors": [<zero or more allowed columns actually named or clearly implied>], '
        '"has_scope": <true if a specific date/month/year/season/relative-period is named, else false>}'
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judgment(text: str, intents: list[str], sensors: list[str]) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    intent = data.get("intent")
    return {
        "intent": intent if intent in intents else None,
        "sensors": [s for s in data.get("sensors", []) if s in sensors],
        "has_scope": bool(data.get("has_scope")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated ids (default: fast free models)")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",")] if args.models else DEFAULT_MODELS

    key = ro.read_key()
    cases = qc.GROUND_TRUTH
    intents = list(INTENTS.keys())
    sensors = qc.ALL_SENSOR_COLUMNS
    n = len(cases)

    # --- our app's routing (deterministic) ------------------------------------------------------
    raw = preprocess.clean(preprocess.load_data(str(bx.DATA)))
    df = preprocess.align(raw, freq="1h")
    app = qc.score_coverage(interface.Interface(df, raw=raw), cases)
    print(f"our app (deterministic): intent {app['intent_pct']:.0f}%  sensors {app['sensors_pct']:.0f}%  "
          f"scope {app['scope_pct']:.0f}%  overall {app['overall_pct']:.0f}%\n")

    # --- hosted LLM judges ----------------------------------------------------------------------
    summary = {"our app": app}
    rows = []
    for m in models:
        i_ok = s_ok = sc_ok = all_ok = judged = 0
        for case in cases:
            reply, err = ro.ask_with_retry(key, m, judge_prompt(case.question, intents, sensors), tries=3)
            j = parse_judgment(reply, intents, sensors)
            rec = {"model": m, "question": case.question,
                   "exp_intent": case.intent, "got_intent": (j or {}).get("intent"),
                   "exp_scope": case.has_scope, "got_scope": (j or {}).get("has_scope"),
                   "error": err}
            rows.append(rec)
            if j is None:
                continue
            judged += 1
            ii = j["intent"] == case.intent
            ss = set(j["sensors"]) == set(case.sensors)
            cc = j["has_scope"] == case.has_scope
            i_ok += ii
            s_ok += ss
            sc_ok += cc
            all_ok += (ii and ss and cc)
            print(f"  [{m.split('/')[-1][:18]:18}] {('OK' if ii else 'xx')} "
                  f"{str(case.intent):11}->{str(j['intent']):11} {case.question[:40]}", flush=True)
            time.sleep(0.4)
        summary[m] = {"n": n, "judged": judged,
                      "intent_pct": 100*i_ok/n, "sensors_pct": 100*s_ok/n,
                      "scope_pct": 100*sc_ok/n, "overall_pct": 100*all_ok/n}

    # --- table ----------------------------------------------------------------------------------
    print(f"\n=== ROUTING ACCURACY vs hand-labelled ground truth (n={n}) ===")
    print(f"{'contestant':42} {'intent':>7} {'sensors':>8} {'scope':>7} {'overall':>8}")
    for name, s in summary.items():
        print(f"{name:42} {s['intent_pct']:6.0f}% {s['sensors_pct']:7.0f}% "
              f"{s['scope_pct']:6.0f}% {s['overall_pct']:7.0f}%")

    out = HERE / "results_routing.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "question", "exp_intent", "got_intent",
                                          "exp_scope", "got_scope", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-question detail -> {out.relative_to(V1)}")


if __name__ == "__main__":
    main()
