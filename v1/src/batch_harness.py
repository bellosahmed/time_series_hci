"""COSMA/DiRAC batch harness (sub-project 3): shard the project's embarrassingly-parallel eval
workloads across SLURM array tasks, run each shard, aggregate the per-shard outputs. The harness only
schedules and collects -- every number still comes from the existing deterministic evaluators. The
whole pipeline is offline-testable via ``local`` mode and an injected runner; no cluster required.

Flow:  build_jobs -> shard (round-robin) -> run_shard (per array task) -> aggregate -> CSV.
CLI:   python -m src.batch_harness {plan|run-shard|aggregate|local} ...
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import llm


@dataclass(frozen=True)
class JobSpec:
    """One unit of work: a stable unique id, a dispatch ``kind``, and kind-specific ``payload``."""

    id: str
    kind: str
    payload: dict = field(default_factory=dict)


Runner = Callable[["JobSpec"], dict]


# --- Sharding (deterministic, balanced) ---------------------------------------------------

def assign(index: int, num_shards: int) -> int:
    """Round-robin shard for the job at position ``index`` (single source of truth for sharding)."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    return index % num_shards


def shard(jobs: list[JobSpec], num_shards: int) -> list[list[JobSpec]]:
    """Partition ``jobs`` into ``num_shards`` balanced, disjoint, deterministic shards (round-robin).
    Every job appears exactly once; shard sizes differ by at most one."""
    buckets: list[list[JobSpec]] = [[] for _ in range(num_shards)]
    for i, job in enumerate(jobs):
        buckets[assign(i, num_shards)].append(job)
    return buckets


# --- Per-shard execution ------------------------------------------------------------------

def run_shard(jobs: list[JobSpec], shard_id: int, num_shards: int, runner: Runner,
              out_path: str | Path | None = None) -> list[dict]:
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


# --- Aggregation --------------------------------------------------------------------------

def aggregate(shard_paths: list[str | Path]) -> list[dict]:
    """Merge per-shard JSONL into one list, sorted by id and de-duplicated by id (last write wins, so
    a re-run shard is safe). Blank lines are ignored."""
    by_id: dict[str, dict] = {}
    for path in shard_paths:
        with open(path) as fh:
            for raw_line in fh:
                line = raw_line.strip()
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


def write_csv(rows: list[dict], path: str | Path) -> None:
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


# --- SLURM array script generation --------------------------------------------------------

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
#   {python} -m src.batch_harness aggregate --glob '{out_dir}/shard_*.jsonl' \\
#     --out {out_dir}/results.csv
"""


# --- Job enumeration + default dispatch ---------------------------------------------------

def build_jobs(kinds: list[str], models: list[str]) -> list[JobSpec]:
    """Enumerate eval jobs from the existing corpora, in a fixed order with stable unique ids. For
    ``benchmark`` each case is crossed with every model; the other corpora are model-agnostic."""
    from . import benchmark_external as be
    from . import eval_corpora as ec
    jobs: list[JobSpec] = []
    if "benchmark" in kinds:
        for model in models:
            for i in range(len(be.GOLD)):
                jobs.append(JobSpec(id=f"bench-{model}-{i:03d}", kind="benchmark",
                                    payload={"case_index": i, "model": model}))
    if "orchestrator" in kinds:
        for i, (q, expected) in enumerate(ec.MULTIPART_CASES):
            jobs.append(JobSpec(id=f"orch-{i:03d}", kind="orchestrator",
                                payload={"question": q, "expected": list(expected)}))
    if "understanding" in kinds:
        for i, case in enumerate(ec.MALFORMED_CASES):
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


# --- CLI ----------------------------------------------------------------------------------

def _write_jobs(jobs: list[JobSpec], path: str | Path) -> None:
    with open(path, "w") as fh:
        for j in jobs:
            fh.write(json.dumps({"id": j.id, "kind": j.kind, "payload": j.payload}) + "\n")


def _load_context(kinds: list[str]) -> dict:
    """Load data/interfaces only when a kind needs them (keeps `plan`/`aggregate` dependency-free)."""
    from . import interface, orchestrator, preprocess
    raw = preprocess.clean(preprocess.load_data("data/jena_climate_2009_2016.csv"))
    df = preprocess.align(raw, freq="1h")
    iface = interface.Interface(df, raw=raw, use_llm=("understanding" in kinds))
    orch = orchestrator.Orchestrator(df, use_llm=True)
    return {"df": df, "raw": raw, "iface": iface, "orch": orch}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="batch_harness", description="COSMA eval batch harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="write jobs.jsonl + sbatch script")
    sp.add_argument("--kinds", required=True)
    sp.add_argument("--models", default=llm.DEFAULT_MODEL)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--account", required=True)
    sp.add_argument("--partition", required=True)
    sp.add_argument("--time", default="01:00:00")

    sr = sub.add_parser("run-shard", help="run one shard (a SLURM array task)")
    sr.add_argument("--shard", type=int, required=True)
    sr.add_argument("--num-shards", type=int, required=True)
    sr.add_argument("--kinds", required=True)
    sr.add_argument("--models", default=llm.DEFAULT_MODEL)
    sr.add_argument("--out", required=True)

    sa = sub.add_parser("aggregate", help="merge shard JSONL into a results CSV")
    sa.add_argument("--glob", required=True)
    sa.add_argument("--out", required=True)

    sl = sub.add_parser("local", help="run all shards in-process (no SLURM) and aggregate")
    sl.add_argument("--kinds", required=True)
    sl.add_argument("--models", default=llm.DEFAULT_MODEL)
    sl.add_argument("--num-shards", type=int, default=4)
    sl.add_argument("--out", default="results.csv")

    args = p.parse_args(argv)
    kinds = args.kinds.split(",") if getattr(args, "kinds", "") else []
    models = args.models.split(",") if getattr(args, "models", "") else []

    if args.cmd == "plan":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = build_jobs(kinds, models)
        _write_jobs(jobs, out_dir / "jobs.jsonl")
        script = render_sbatch(args.num_shards, kinds, models, str(out_dir),
                               args.account, args.partition, args.time)
        (out_dir / "jena_eval.slurm").write_text(script)
        print(f"Wrote {len(jobs)} jobs and {out_dir / 'jena_eval.slurm'}. Submit with:\n"
              f"  sbatch {out_dir / 'jena_eval.slurm'}")
        return 0

    if args.cmd == "run-shard":
        ctx = _load_context(kinds)
        jobs = build_jobs(kinds, models)
        run_shard(jobs, args.shard, args.num_shards,
                  lambda j: default_runner(j, **ctx), out_path=args.out)
        return 0

    if args.cmd == "aggregate":
        rows = aggregate(list(collect_glob(args.glob)))
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
