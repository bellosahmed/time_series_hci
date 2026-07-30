"""Run the OpenRouter leg of the comparative benchmark: real hosted LLMs (free tier) as
contestants, scored against the same oracle ground truth as the local leg.

The API is OpenAI-compatible; called here with the standard library only (no new deps), matching
`src/llm.py`'s style.

KEY (required — free models still need auth):
    Get one at https://openrouter.ai/keys, then either
      * export it:      OPENROUTER_API_KEY=sk-or-... python3.12 .../run_openrouter.py
      * or drop it in:  docs/evaluation/benchmark/.openrouter_key   (gitignored)

USAGE:
    python3.12 docs/evaluation/benchmark/run_openrouter.py                 # auto-pick free chat models
    python3.12 docs/evaluation/benchmark/run_openrouter.py --models moonshotai/kimi-k2,z-ai/glm-4.6
    python3.12 docs/evaluation/benchmark/run_openrouter.py --max-models 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

V1 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(V1))

from src import benchmark_external as bx  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = "https://openrouter.ai/api/v1"

# id substrings that aren't text-chat contestants (audio, vision-only, safety classifier, meta-router)
_NON_CHAT = ("lyria", "content-safety", "-vl", "openrouter/free", "-omni-")


def read_key() -> str:
    import os
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        kf = HERE / ".openrouter_key"
        if kf.exists():
            key = kf.read_text().strip()
    if not key:
        sys.exit("No OpenRouter key. Set OPENROUTER_API_KEY or create "
                 "docs/evaluation/benchmark/.openrouter_key (see this file's docstring).")
    return key


def free_chat_models(max_models: int) -> list[str]:
    with urllib.request.urlopen(f"{BASE}/models", timeout=15) as r:
        data = json.loads(r.read())["data"]
    free = [m for m in data if str(m.get("pricing", {}).get("prompt", "1")) in ("0", "0.0")]
    ids = [m["id"] for m in free if not any(x in m["id"].lower() for x in _NON_CHAT)]
    # prefer larger / well-known families first for a meaningful comparison
    pref = ("nemotron-3-ultra", "nemotron-3-super", "gpt-oss", "gemma-4-31", "gemma-4-26",
            "kimi", "glm", "deepseek", "qwen", "ling-3")
    ids.sort(key=lambda i: next((k for k, p in enumerate(pref) if p in i.lower()), len(pref)))
    return ids[:max_models]


def chat(key: str, model: str, prompt: str, timeout: float = 90.0) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.benchmark",     # OpenRouter attribution headers (optional)
        "X-Title": "ts-hci-benchmark",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return (data["choices"][0]["message"]["content"] or "").strip()


def ask_with_retry(key: str, model: str, prompt: str, tries: int = 4) -> tuple[str, str]:
    """Return (reply_text, error). Backs off on 429 (free-tier rate limit)."""
    for attempt in range(tries):
        try:
            return chat(key, model, prompt), ""
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:
                time.sleep(2 ** attempt * 3)          # 3s, 6s, 12s
                continue
            return "", f"HTTP {e.code}: {e.read()[:120].decode('utf-8', 'replace')}"
        except Exception as e:                        # noqa: BLE001 - report, don't crash the run
            return "", f"{type(e).__name__}: {e}"
    return "", "exhausted retries"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated model ids (overrides auto-pick)")
    ap.add_argument("--max-models", type=int, default=5)
    args = ap.parse_args()

    key = read_key()
    models = ([m.strip() for m in args.models.split(",")] if args.models
              else free_chat_models(args.max_models))
    print(f"Contestants: {models}\n")

    raw = bx.load_raw()
    n = len(bx.GOLD)
    correct = {m: 0 for m in models}
    answered = {m: 0 for m in models}
    rows = []

    for case in bx.GOLD:
        gt = bx.ground_truth(raw, case)
        prompt = bx.build_prompt(raw, case)
        rec = {"question": case.question, "sensor": case.sensor, "kind": case.kind, "truth": gt}
        for m in models:
            reply, err = ask_with_retry(key, m, prompt)
            val = bx.parse_number(reply)
            rec[m] = val
            rec[m + "__reply"] = (err or reply).replace("\n", " ")[:200]
            if val is not None:
                answered[m] += 1
                correct[m] += bx.is_correct(val, gt)
            time.sleep(1.0)                            # be gentle with the free tier
        rows.append(rec)

    qw = max(len(r["question"]) for r in rows)
    header = f"{'question'.ljust(qw)}  {'truth':>9}  " + "  ".join(f"{m.split('/')[-1][:18]:>18}" for m in models)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for m in models:
            v = r.get(m)
            ok = bx.is_correct(v, r["truth"])
            cell = f"{v:.2f}" if isinstance(v, float) else "—"
            cells.append(f"{('✓ ' if ok else '✗ ')+cell:>18}")
        print(f"{r['question'].ljust(qw)}  {r['truth']:>9.2f}  " + "  ".join(cells))
    print("-" * len(header))
    print("ACCURACY (exact within ±0.01):")
    print(f"  {'our app (reference)':40} {n}/{n}  (100%)")
    for m in models:
        print(f"  {m:40} {correct[m]}/{n}  ({100*correct[m]/n:.0f}%)   [{answered[m]}/{n} answered]")

    out = HERE / "results_openrouter.csv"
    fields = ["question", "sensor", "kind", "truth",
              *models, *[m + "__reply" for m in models]]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nRaw replies written to {out.relative_to(V1)}")


if __name__ == "__main__":
    main()
