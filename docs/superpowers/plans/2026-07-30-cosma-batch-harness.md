# COSMA Batch Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A workload-agnostic SLURM-array batch harness that shards the project's eval workloads, runs each shard, and aggregates results — fully offline-testable, with a generated `sbatch` script for COSMA/DiRAC.

**Architecture:** New `v1/src/batch_harness.py` with a `JobSpec` model, deterministic round-robin sharding, per-shard runner, JSONL aggregation, and `render_sbatch`. A `local` CLI mode runs all shards in-process (no SLURM) and is the path tests exercise. Default runners wrap the existing `benchmark_external`/`orchestrator_eval`/`understanding_eval` corpora; the harness computes nothing itself.

**Tech Stack:** Python 3.12, stdlib (`dataclasses`, `json`, `argparse`, `glob`, `pathlib`), pandas (only in default runners), pytest, ruff, mypy.

## Global Constraints

- Python 3.12 (`python3.12`); run tests with `python3.12 -m pytest`.
- Commits authored **solely by the user** — NO `Co-Authored-By: Claude` trailer.
- Do NOT push to GitHub.
- **Trust invariant:** the harness only schedules/collects; all numbers come from the existing deterministic evaluators. No new metrics.
- No cluster credentials, accounts, or Durham-specific paths hard-coded — they are CLI/env inputs.
- Must be fully offline-testable: `local` mode + a stub runner cover the whole pipeline; no SLURM/Ollama needed for tests.
- `ruff check` and `mypy` must stay clean.

---

### Task 1: Job model + deterministic sharding

**Files:**
- Create: `v1/src/batch_harness.py`
- Test: `v1/tests/test_batch_harness.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) JobSpec(id: str, kind: str, payload: dict)`
  - `assign(index: int, num_shards: int) -> int` (round-robin: `index % num_shards`)
  - `shard(jobs: list[JobSpec], num_shards: int) -> list[list[JobSpec]]`

- [ ] **Step 1: Write the failing tests**

```python
# v1/tests/test_batch_harness.py
"""The batch-harness core is workload-agnostic and offline: sharding is a deterministic, balanced
partition; run_shard/aggregate/render_sbatch need no SLURM or Ollama (stub runner)."""
from src import batch_harness as bh


def _jobs(n):
    return [bh.JobSpec(id=f"j{i:03d}", kind="stub", payload={"i": i}) for i in range(n)]


def test_assign_is_round_robin():
    assert [bh.assign(i, 3) for i in range(7)] == [0, 1, 2, 0, 1, 2, 0]


def test_shard_is_a_balanced_partition():
    jobs = _jobs(10)
    shards = bh.shard(jobs, 3)
    ids = [j.id for s in shards for j in s]
    assert sorted(ids) == sorted(j.id for j in jobs)          # union == all jobs
    assert len({*ids}) == len(ids)                            # disjoint (no dup)
    sizes = sorted(len(s) for s in shards)
    assert sizes[-1] - sizes[0] <= 1                          # balanced
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_batch_harness.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.batch_harness'`).

- [ ] **Step 3: Implement the model + sharding**

```python
# v1/src/batch_harness.py
"""COSMA/DiRAC batch harness (sub-project 3): shard the project's embarrassingly-parallel eval
workloads across SLURM array tasks, run each shard, aggregate the per-shard outputs. The harness only
schedules and collects -- every number still comes from the existing deterministic evaluators. The
whole pipeline is offline-testable via ``local`` mode and an injected runner; no cluster required.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobSpec:
    """One unit of work: a stable unique id, a dispatch ``kind``, and kind-specific ``payload``."""

    id: str
    kind: str
    payload: dict = field(default_factory=dict)


def assign(index: int, num_shards: int) -> int:
    """Round-robin shard for the job at position ``index`` (the single source of truth for sharding)."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    return index % num_shards


def shard(jobs: list[JobSpec], num_shards: int) -> list[list[JobSpec]]:
    """Partition ``jobs`` into ``num_shards`` balanced, disjoint, deterministic shards (round-robin)."""
    buckets: list[list[JobSpec]] = [[] for _ in range(num_shards)]
    for i, job in enumerate(jobs):
        buckets[assign(i, num_shards)].append(job)
    return buckets
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_batch_harness.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit** _(hold until end per session preference; listed for completeness)_

```bash
git add v1/src/batch_harness.py v1/tests/test_batch_harness.py
git commit -m "feat(harness): JobSpec + deterministic round-robin sharding"
```

---

### Task 2: run_shard (per-shard execution + JSONL)

**Files:**
- Modify: `v1/src/batch_harness.py`
- Test: `v1/tests/test_batch_harness.py` (append)

**Interfaces:**
- Consumes: `assign`, `JobSpec`.
- Produces: `run_shard(jobs, shard_id, num_shards, runner, out_path=None) -> list[dict]` where `runner: Callable[[JobSpec], dict]`. A runner exception → `{"id", "kind", "ok": False, "error": str}`.

- [ ] **Step 1: Write the failing tests**

```python
# append to v1/tests/test_batch_harness.py
import json


def test_run_shard_selects_only_its_jobs_in_order():
    jobs = _jobs(6)
    seen = []

    def runner(job):
        seen.append(job.id)
        return {"id": job.id, "ok": True, "i": job.payload["i"]}
    rows = bh.run_shard(jobs, shard_id=0, num_shards=3, runner=runner)
    assert [r["id"] for r in rows] == ["j000", "j003"]   # indices 0,3 -> shard 0, order kept
    assert seen == ["j000", "j003"]


def test_run_shard_captures_runner_error_and_writes_jsonl(tmp_path):
    jobs = _jobs(3)

    def runner(job):
        if job.id == "j000":
            raise RuntimeError("boom")
        return {"id": job.id, "ok": True}
    out = tmp_path / "shard0.jsonl"
    rows = bh.run_shard(jobs, 0, 1, runner, out_path=out)   # 1 shard -> all jobs
    bad = next(r for r in rows if r["id"] == "j000")
    assert bad["ok"] is False and "boom" in bad["error"]
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["id"] == "j000"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_batch_harness.py -k run_shard -v`
Expected: FAIL (`no attribute 'run_shard'`).

- [ ] **Step 3: Implement run_shard**

```python
# add imports at top
import json
from collections.abc import Callable
from pathlib import Path

Runner = Callable[["JobSpec"], dict]


def run_shard(jobs: list[JobSpec], shard_id: int, num_shards: int, runner: Runner,
              out_path: "str | Path | None" = None) -> list[dict]:
    """Run every job assigned to ``shard_id`` through ``runner``; one job's failure becomes an
    ``ok=False`` row, never an exception. If ``out_path`` is given, write the rows as JSONL."""
    mine = [j for i, j in enumerate(jobs) if assign(i, num_shards) == shard_id]
    rows: list[dict] = []
    for job in mine:
        try:
            row = dict(runner(job))
            row.setdefault("id", job.id)
        except Exception as exc:  # noqa: BLE001 - a bad job must not sink the shard
            row = {"id": job.id, "kind": job.kind, "ok": False, "error": str(exc)}
        rows.append(row)
    if out_path is not None:
        with open(out_path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    return rows
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_batch_harness.py -k run_shard -v`
Expected: PASS.

- [ ] **Step 5: Commit** _(held)_

```bash
git add v1/src/batch_harness.py v1/tests/test_batch_harness.py
git commit -m "feat(harness): run_shard with per-job error capture + JSONL output"
```

---

### Task 3: aggregate (merge + dedup + summary) and write_csv

**Files:**
- Modify: `v1/src/batch_harness.py`
- Test: `v1/tests/test_batch_harness.py` (append)

**Interfaces:**
- Consumes: JSONL shard files.
- Produces:
  - `aggregate(shard_paths: list[str | Path]) -> list[dict]` (sorted by `id`, dedup by `id`, last wins)
  - `summarise(rows: list[dict]) -> dict` → `{"n", "ok", "correct"}` (counts present keys)
  - `write_csv(rows: list[dict], path: str | Path) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# append to v1/tests/test_batch_harness.py
def test_aggregate_merges_sorts_and_dedups(tmp_path):
    (tmp_path / "s0.jsonl").write_text(
        '{"id": "j001", "ok": true, "correct": true}\n{"id": "j000", "ok": true, "correct": false}\n')
    (tmp_path / "s1.jsonl").write_text(
        '{"id": "j000", "ok": true, "correct": true}\n')   # duplicate id -> last wins
    rows = bh.aggregate([tmp_path / "s0.jsonl", tmp_path / "s1.jsonl"])
    assert [r["id"] for r in rows] == ["j000", "j001"]      # sorted, deduped
    assert next(r for r in rows if r["id"] == "j000")["correct"] is True
    summ = bh.summarise(rows)
    assert summ == {"n": 2, "ok": 2, "correct": 1}


def test_write_csv_roundtrips(tmp_path):
    rows = [{"id": "a", "ok": True}, {"id": "b", "ok": False}]
    out = tmp_path / "r.csv"
    bh.write_csv(rows, out)
    text = out.read_text()
    assert "id" in text and "a" in text and "b" in text
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_batch_harness.py -k "aggregate or write_csv" -v`
Expected: FAIL (`no attribute 'aggregate'`).

- [ ] **Step 3: Implement aggregate/summarise/write_csv**

```python
# add import at top
import csv
import glob as _glob


def aggregate(shard_paths: list["str | Path"]) -> list[dict]:
    """Merge per-shard JSONL into one list, sorted by id and de-duplicated by id (last write wins, so
    a re-run shard is safe). Blank lines are ignored."""
    by_id: dict[str, dict] = {}
    for path in shard_paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_id[row["id"]] = row
    return [by_id[k] for k in sorted(by_id)]


def summarise(rows: list[dict]) -> dict:
    """Small counts over result rows: total, how many ok, how many correct (keys counted only when
    present, so it works across job kinds)."""
    return {
        "n": len(rows),
        "ok": sum(1 for r in rows if r.get("ok")),
        "correct": sum(1 for r in rows if r.get("correct")),
    }


def collect_glob(pattern: str) -> list[str]:
    """Sorted shard files matching a glob (used by the aggregate CLI)."""
    return sorted(_glob.glob(pattern))


def write_csv(rows: list[dict], path: "str | Path") -> None:
    """Write result rows to CSV. Columns are the union of keys, id first, in first-seen order."""
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    if "id" in cols:
        cols.remove("id")
        cols.insert(0, "id")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_batch_harness.py -k "aggregate or write_csv" -v`
Expected: PASS.

- [ ] **Step 5: Commit** _(held)_

```bash
git add v1/src/batch_harness.py v1/tests/test_batch_harness.py
git commit -m "feat(harness): aggregate (merge/sort/dedup) + summarise + write_csv"
```

---

### Task 4: render_sbatch (SLURM array script generation)

**Files:**
- Modify: `v1/src/batch_harness.py`
- Test: `v1/tests/test_batch_harness.py` (append)

**Interfaces:**
- Produces: `render_sbatch(n_shards, kinds, models, out_dir, account, partition, time="01:00:00", python="python3.12") -> str`

- [ ] **Step 1: Write the failing test**

```python
# append to v1/tests/test_batch_harness.py
def test_render_sbatch_has_array_and_runshard_command():
    script = bh.render_sbatch(n_shards=8, kinds=["benchmark", "orchestrator"],
                              models=["llama3.2:3b"], out_dir="runs/x",
                              account="dp999", partition="cosma", time="02:00:00")
    assert "#SBATCH --array=0-7" in script
    assert "#SBATCH --account=dp999" in script
    assert "#SBATCH --partition=cosma" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "run-shard" in script and "$SLURM_ARRAY_TASK_ID" in script
    assert "--num-shards 8" in script
    assert "benchmark,orchestrator" in script
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3.12 -m pytest tests/test_batch_harness.py -k sbatch -v`
Expected: FAIL (`no attribute 'render_sbatch'`).

- [ ] **Step 3: Implement render_sbatch**

```python
def render_sbatch(n_shards: int, kinds: list[str], models: list[str], out_dir: str,
                  account: str, partition: str, time: str = "01:00:00",
                  python: str = "python3.12") -> str:
    """Emit a SLURM array-job script: one array task per shard, each running run-shard for its
    ``$SLURM_ARRAY_TASK_ID``. Account/partition/time/out-dir are inputs -- nothing site-specific is
    hard-coded. Aggregation is documented as a dependent job, not auto-submitted."""
    kinds_csv = ",".join(kinds)
    models_csv = ",".join(models)
    return f"""#!/bin/bash
#SBATCH --job-name=jena-eval
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --array=0-{n_shards - 1}
#SBATCH --time={time}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output={out_dir}/slurm_%A_%a.out

set -euo pipefail
mkdir -p {out_dir}

{python} -m src.batch_harness run-shard \\
  --shard $SLURM_ARRAY_TASK_ID --num-shards {n_shards} \\
  --kinds {kinds_csv} --models {models_csv} \\
  --out {out_dir}/shard_$SLURM_ARRAY_TASK_ID.jsonl

# After all array tasks finish, aggregate (submit as a dependent job):
#   sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID aggregate.slurm
# or run locally:
#   {python} -m src.batch_harness aggregate --glob '{out_dir}/shard_*.jsonl' --out {out_dir}/results.csv
"""
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3.12 -m pytest tests/test_batch_harness.py -k sbatch -v`
Expected: PASS.

- [ ] **Step 5: Commit** _(held)_

```bash
git add v1/src/batch_harness.py v1/tests/test_batch_harness.py
git commit -m "feat(harness): render_sbatch SLURM array script (no site-specific values baked in)"
```

---

### Task 5: build_jobs + default_runner + run_local (result-preserving)

**Files:**
- Modify: `v1/src/batch_harness.py`
- Test: `v1/tests/test_batch_harness.py` (append)

**Interfaces:**
- Consumes: `benchmark_external`, `orchestrator_eval`, `understanding_eval`, `preprocess`, `interface`, `orchestrator`.
- Produces:
  - `build_jobs(kinds: list[str], models: list[str]) -> list[JobSpec]` (deterministic, unique ids)
  - `default_runner(job, df=None, raw=None, iface=None, orch=None) -> dict`
  - `run_local(jobs, num_shards, runner) -> list[dict]` (runs all shards in-process, then aggregates in memory) — asserts sharding does not change results.

- [ ] **Step 1: Write the failing tests**

```python
# append to v1/tests/test_batch_harness.py
def test_build_jobs_deterministic_unique_ids():
    a = bh.build_jobs(kinds=["orchestrator", "understanding"], models=["m1"])
    b = bh.build_jobs(kinds=["orchestrator", "understanding"], models=["m1"])
    ids = [j.id for j in a]
    assert [j.id for j in b] == ids           # deterministic
    assert len(set(ids)) == len(ids)          # unique
    assert all(j.kind in ("orchestrator", "understanding") for j in a)


def test_run_local_is_result_preserving_across_shard_counts():
    jobs = _jobs(11)

    def runner(job):
        return {"id": job.id, "ok": True, "val": job.payload["i"] * 2}
    one = bh.run_local(jobs, num_shards=1, runner=runner)
    many = bh.run_local(jobs, num_shards=4, runner=runner)
    assert one == many                        # sharding never changes the aggregated result
    assert len(one) == 11
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_batch_harness.py -k "build_jobs or run_local" -v`
Expected: FAIL (`no attribute 'build_jobs'`).

- [ ] **Step 3: Implement build_jobs / default_runner / run_local**

```python
def build_jobs(kinds: list[str], models: list[str]) -> list[JobSpec]:
    """Enumerate eval jobs from the existing corpora, in a fixed order with stable unique ids. For
    ``benchmark`` each case is crossed with every model; the eval corpora are model-agnostic."""
    from . import benchmark_external as be
    from . import orchestrator_eval as oe
    from . import understanding_eval as ue
    jobs: list[JobSpec] = []
    if "benchmark" in kinds:
        for model in models:
            for i, case in enumerate(be.GOLD):
                jobs.append(JobSpec(id=f"bench-{model}-{i:03d}", kind="benchmark",
                                    payload={"case_index": i, "model": model}))
    if "orchestrator" in kinds:
        for i, (q, expected) in enumerate(oe.MULTIPART_CASES):
            jobs.append(JobSpec(id=f"orch-{i:03d}", kind="orchestrator",
                                payload={"question": q, "expected": list(expected)}))
    if "understanding" in kinds:
        for i, case in enumerate(ue.MALFORMED_CASES):
            jobs.append(JobSpec(id=f"und-{i:03d}", kind="understanding",
                                payload={"question": case.question, "intent": case.intent,
                                         "sensors": list(case.sensors)}))
    return jobs


def default_runner(job: JobSpec, df=None, raw=None, iface=None, orch=None) -> dict:
    """Dispatch a job to the real evaluator for its kind, returning a result row. Data/interfaces are
    passed in (loaded once by the caller). The harness adds no numbers -- it just records the
    evaluator's verdict."""
    kind = job.kind
    if kind == "orchestrator":
        ans = orch.run(job.payload["question"])
        blob = f"{ans.text} {ans.computed}"
        complete = all(s in blob for s in job.payload["expected"])
        return {"id": job.id, "kind": kind, "ok": True, "correct": complete}
    if kind == "understanding":
        iface.history = []
        ans = iface.ask(job.payload["question"])
        correct = (ans.value is not None and ans.intent == job.payload["intent"]
                   and set(job.payload["sensors"]).issubset(set(ans.sensors)))
        return {"id": job.id, "kind": kind, "ok": ans.value is not None, "correct": bool(correct)}
    if kind == "benchmark":
        from . import benchmark_external as be
        case = be.GOLD[job.payload["case_index"]]
        truth = be.ground_truth(raw, case)
        pred = be.ask_app(iface, case)
        return {"id": job.id, "kind": kind, "ok": pred is not None,
                "correct": be.is_correct(pred, truth)}
    return {"id": job.id, "kind": kind, "ok": False, "error": f"unknown kind {kind!r}"}


def run_local(jobs: list[JobSpec], num_shards: int, runner: Runner) -> list[dict]:
    """Run every shard in-process and aggregate in memory -- the SLURM-free path (dev/offline, and the
    correctness oracle: the aggregated result is independent of ``num_shards``)."""
    by_id: dict[str, dict] = {}
    for sid in range(num_shards):
        for row in run_shard(jobs, sid, num_shards, runner):
            by_id[row["id"]] = row
    return [by_id[k] for k in sorted(by_id)]
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_batch_harness.py -k "build_jobs or run_local" -v`
Expected: PASS.

- [ ] **Step 5: Commit** _(held)_

```bash
git add v1/src/batch_harness.py v1/tests/test_batch_harness.py
git commit -m "feat(harness): build_jobs + default_runner + result-preserving run_local"
```

---

### Task 6: CLI (`plan` / `run-shard` / `aggregate` / `local`)

**Files:**
- Modify: `v1/src/batch_harness.py` (add `main()` + `if __name__`)
- Test: `v1/tests/test_batch_harness.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv: list[str] | None = None) -> int`. `local` mode uses a stub-free default: it loads data lazily only for kinds that need it; a `--dry-run` on `plan` writes files without submitting.

- [ ] **Step 1: Write the failing test**

```python
# append to v1/tests/test_batch_harness.py
def test_cli_plan_writes_jobs_and_sbatch(tmp_path):
    rc = bh.main(["plan", "--kinds", "orchestrator,understanding", "--models", "m1",
                  "--num-shards", "4", "--out-dir", str(tmp_path),
                  "--account", "dp999", "--partition", "cosma"])
    assert rc == 0
    assert (tmp_path / "jobs.jsonl").exists()
    script = (tmp_path / "jena_eval.slurm").read_text()
    assert "#SBATCH --array=0-3" in script


def test_cli_aggregate_merges_to_csv(tmp_path):
    (tmp_path / "shard_0.jsonl").write_text('{"id": "a", "ok": true, "correct": true}\n')
    (tmp_path / "shard_1.jsonl").write_text('{"id": "b", "ok": true, "correct": false}\n')
    out = tmp_path / "results.csv"
    rc = bh.main(["aggregate", "--glob", str(tmp_path / "shard_*.jsonl"), "--out", str(out)])
    assert rc == 0 and out.exists()
    assert "a" in out.read_text() and "b" in out.read_text()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_batch_harness.py -k cli -v`
Expected: FAIL (`no attribute 'main'`).

- [ ] **Step 3: Implement the CLI**

```python
# add imports
import argparse
import sys


def _write_jobs(jobs: list[JobSpec], path: "str | Path") -> None:
    with open(path, "w") as fh:
        for j in jobs:
            fh.write(json.dumps({"id": j.id, "kind": j.kind, "payload": j.payload}) + "\n")


def _load_context(kinds: list[str]):
    """Load data/interfaces only when a kind needs them (keeps `plan`/`aggregate` dependency-free)."""
    from . import interface, orchestrator, preprocess
    raw = preprocess.clean(preprocess.load_data("data/jena_climate_2009_2016.csv"))
    df = preprocess.align(raw, freq="1h")
    iface = interface.Interface(df, raw=raw, use_llm=("understanding" in kinds))
    orch = orchestrator.Orchestrator(df, use_llm=True)
    return {"df": df, "raw": raw, "iface": iface, "orch": orch}


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(prog="batch_harness", description="COSMA eval batch harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="write jobs.jsonl + sbatch script")
    sp.add_argument("--kinds", required=True)
    sp.add_argument("--models", default="llama3.2:3b")
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--account", required=True)
    sp.add_argument("--partition", required=True)
    sp.add_argument("--time", default="01:00:00")

    sr = sub.add_parser("run-shard", help="run one shard (a SLURM array task)")
    sr.add_argument("--shard", type=int, required=True)
    sr.add_argument("--num-shards", type=int, required=True)
    sr.add_argument("--kinds", required=True)
    sr.add_argument("--models", default="llama3.2:3b")
    sr.add_argument("--out", required=True)

    sa = sub.add_parser("aggregate", help="merge shard JSONL into a results CSV")
    sa.add_argument("--glob", required=True)
    sa.add_argument("--out", required=True)

    sl = sub.add_parser("local", help="run all shards in-process (no SLURM) and aggregate")
    sl.add_argument("--kinds", required=True)
    sl.add_argument("--models", default="llama3.2:3b")
    sl.add_argument("--num-shards", type=int, default=4)
    sl.add_argument("--out", default="results.csv")

    args = p.parse_args(argv)
    kinds = getattr(args, "kinds", "").split(",") if getattr(args, "kinds", "") else []
    models = getattr(args, "models", "").split(",") if getattr(args, "models", "") else []

    if args.cmd == "plan":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = build_jobs(kinds, models)
        _write_jobs(jobs, out_dir / "jobs.jsonl")
        script = render_sbatch(args.num_shards, kinds, models, str(out_dir),
                               args.account, args.partition, args.time)
        (out_dir / "jena_eval.slurm").write_text(script)
        print(f"Wrote {len(jobs)} jobs and {out_dir/'jena_eval.slurm'}. Submit with:\n"
              f"  sbatch {out_dir/'jena_eval.slurm'}")
        return 0

    if args.cmd == "run-shard":
        ctx = _load_context(kinds)
        jobs = build_jobs(kinds, models)
        run_shard(jobs, args.shard, args.num_shards,
                  lambda j: default_runner(j, **ctx), out_path=args.out)
        return 0

    if args.cmd == "aggregate":
        rows = aggregate(collect_glob(args.glob))
        write_csv(rows, args.out)
        print(f"Aggregated {len(rows)} rows -> {args.out}: {summarise(rows)}")
        return 0

    if args.cmd == "local":
        ctx = _load_context(kinds)
        jobs = build_jobs(kinds, models)
        rows = run_local(jobs, args.num_shards, lambda j: default_runner(j, **ctx))
        write_csv(rows, args.out)
        print(f"Local run: {len(rows)} rows -> {args.out}: {summarise(rows)}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_batch_harness.py -v`
Expected: PASS (all harness tests).

- [ ] **Step 5: Commit** _(held)_

```bash
git add v1/src/batch_harness.py v1/tests/test_batch_harness.py
git commit -m "feat(harness): CLI (plan/run-shard/aggregate/local) for the SLURM fan-out"
```

---

### Task 7: Committed sbatch artifact + eval write-up + WORKLOG + gates

**Files:**
- Create: `v1/hpc/jena_eval.slurm` (generated, checked in)
- Create: `v1/docs/evaluation/11-cosma-batch-harness.md`
- Modify: `WORKLOG.md`

**Interfaces:** none (artifacts + docs).

- [ ] **Step 1: Generate the committed sbatch example + run the real local experiment**

```bash
cd v1
python3.12 -m src.batch_harness plan --kinds orchestrator,understanding --models llama3.2:3b \
  --num-shards 4 --out-dir hpc --account dpXXX --partition cosma --time 02:00:00
# move/keep hpc/jena_eval.slurm; hpc/jobs.jsonl is a byproduct (can be gitignored or kept as sample)
# result-preserving proof (offline-safe kinds only if Ollama down):
python3.12 - <<'PY'
from src import batch_harness as bh
jobs = bh.build_jobs(["orchestrator","understanding"], ["llama3.2:3b"])
print("jobs:", len(jobs))
# with a deterministic stub runner, prove shard-count invariance:
r1 = bh.run_local(jobs, 1, lambda j: {"id": j.id, "ok": True})
r4 = bh.run_local(jobs, 4, lambda j: {"id": j.id, "ok": True})
print("result-preserving across shards:", r1 == r4, "n=", len(r1))
PY
```

- [ ] **Step 2: Write the evaluation/design doc**

Write `v1/docs/evaluation/11-cosma-batch-harness.md` covering: purpose and why the eval workload is an ideal HPC fit (independent LLM calls, no cross-task comms, linear speed-up); the shard→run→aggregate flow with the committed `hpc/jena_eval.slurm`; the **result-preserving property** (aggregated output is independent of shard count — the key correctness claim, shown with real numbers from Step 1); the honest limits (LLM throughput / cluster queue time dominate; each array task still needs Ollama or an API key; cluster submission itself was not executed here — the Python core and generated script are verified, DiRAC submission is the reproducible next step); and the trust note (harness schedules/collects only; numbers come from the deterministic evaluators).

- [ ] **Step 3: Update WORKLOG.md**

Add a dated `2026-07-30` entry under a "sub-project 3 of 3" heading: the harness (JobSpec, round-robin sharding, run_shard, aggregate, render_sbatch, CLI), the result-preserving property, the committed sbatch artifact, and the honest "not run on the cluster yet" caveat.

- [ ] **Step 4: Run the full gate suite**

Run: `cd v1 && python3.12 -m pytest tests/ -q -p no:warnings && ruff check src tests apps && mypy src`
Expected: all tests pass; ruff + mypy clean.

- [ ] **Step 5: Commit** _(held; final commit batch)_

```bash
git add v1/hpc/jena_eval.slurm v1/docs/evaluation/11-cosma-batch-harness.md WORKLOG.md
git commit -m "docs(harness): committed sbatch artifact + COSMA eval write-up + worklog"
```

---

## Self-Review

**Spec coverage:** JobSpec + sharding (Task 1) ✓, run_shard + error capture + JSONL (Task 2) ✓, aggregate/summarise/write_csv (Task 3) ✓, render_sbatch (Task 4) ✓, build_jobs/default_runner/run_local result-preserving (Task 5) ✓, CLI plan/run-shard/aggregate/local (Task 6) ✓, committed sbatch + eval doc + worklog (Task 7) ✓. All spec sections covered.

**Placeholder scan:** No TBD/TODO. Task 7 Step 2 (doc body) is documentation written from real numbers at execution time — acceptable, not a placeholder.

**Type consistency:** `JobSpec(id, kind, payload)`, `assign(index, num_shards) -> int`, `shard(jobs, num_shards)`, `run_shard(jobs, shard_id, num_shards, runner, out_path=None)`, `aggregate(shard_paths) -> list[dict]`, `summarise(rows) -> dict`, `collect_glob(pattern)`, `write_csv(rows, path)`, `render_sbatch(n_shards, kinds, models, out_dir, account, partition, time, python)`, `build_jobs(kinds, models)`, `default_runner(job, df, raw, iface, orch)`, `run_local(jobs, num_shards, runner)`, `main(argv)` — names/signatures consistent across all tasks. `Runner = Callable[[JobSpec], dict]` used consistently.
