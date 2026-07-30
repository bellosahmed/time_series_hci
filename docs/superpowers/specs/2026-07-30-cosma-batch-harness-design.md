# COSMA/DiRAC batch harness — design (sub-project 3)

_30 July 2026._

## Purpose

Fan the project's **embarrassingly-parallel evaluation workloads** (external-LLM benchmark, orchestrator
multi-part eval, understanding-agent eval) out across the COSMA/DiRAC cluster as a **SLURM array job**:
shard a list of jobs, run each shard on its own array task, then aggregate the per-shard outputs into
one results file. The same shape maps onto the parallel `dispatch` seam sub-project 2 exposed — the
cluster simply replaces the local `ThreadPoolExecutor` with many nodes.

Because a running evaluation over hundreds of (question × model) pairs is dominated by independent LLM
calls, it is the textbook HPC fit: no cross-task communication, linear speed-up with node count.

## Scope / non-goals

- **In:** a workload-agnostic Python core (job model, deterministic sharding, per-shard run,
  aggregation), a generated `sbatch` array script, a `--local` mode that runs every shard in-process
  (so the whole harness is testable offline without SLURM), and default runners that wrap the existing
  eval corpora.
- **Out:** no new evaluation *metrics* (reuse `benchmark_external`, `orchestrator_eval`,
  `understanding_eval`); no cluster-specific credentials/paths hard-coded (they are CLI/env inputs); no
  deep-model training sweep (a possible future job kind, explicitly deferred); no CI.
- **Trust invariant unchanged:** the harness only *schedules and collects*; every number still comes
  from the existing deterministic evaluators. The harness computes nothing itself.

## Architecture

New module `v1/src/batch_harness.py`. One committed example script `v1/hpc/jena_eval.slurm` (generated
by the harness, checked in as a worked artifact). Evaluation/design write-up
`v1/docs/evaluation/11-cosma-batch-harness.md`.

```
build_jobs()      -> list[JobSpec]        (from the eval corpora × options)
shard(jobs, N)    -> list[list[JobSpec]]  (deterministic round-robin; union == jobs, disjoint)
run_shard(...)    -> list[dict]           (run one shard's jobs via a runner; write JSONL)
aggregate(paths)  -> list[dict]           (merge per-shard JSONL, stable order, dedup by job id)
render_sbatch(...)-> str                  (SLURM --array=0-(N-1) script text)
```

### Data model

```python
@dataclass(frozen=True)
class JobSpec:
    id: str            # stable, unique (e.g. "orch-0002", "bench-T-2015-max")
    kind: str          # "benchmark" | "orchestrator" | "understanding" (dispatch key)
    payload: dict      # kind-specific inputs (question, model, expected, ...)

Runner = Callable[[JobSpec], dict]   # returns a result row dict (must include "id")
```

### Components

**1. `build_jobs(kinds, models) -> list[JobSpec]`**
- Enumerates jobs from the existing corpora: `benchmark_external.GOLD` (× each model),
  `orchestrator_eval.MULTIPART_CASES`, `understanding_eval.MALFORMED_CASES`. Each job gets a stable id.
- Pure and deterministic — the same inputs always give the same ordered job list, so shard assignment
  is reproducible across the submit node and every array task.

**2. `shard(jobs, num_shards) -> list[list[JobSpec]]`**
- Round-robin: job at index `i` goes to shard `i % num_shards`. Guarantees (a) every job appears
  exactly once, (b) shards differ in size by at most one (balanced), (c) determinism. A helper
  `assign(i, num_shards) -> int` is the single source of truth, reused by `run_shard`.

**3. `run_shard(jobs, shard_id, num_shards, runner, out_path=None) -> list[dict]`**
- Selects this shard's jobs (via `assign`), runs each through `runner`, collects result rows, and (if
  `out_path`) writes them as JSONL. A runner exception becomes a row with `"ok": False, "error": ...`
  so one bad job never sinks the shard (mirrors `agent.execute_step`).

**4. `default_runner(job, df=None, raw=None) -> dict`**
- Dispatches by `job.kind` to the real evaluators: `benchmark` → `ask_app`/`ground_truth`/`is_correct`;
  `orchestrator` → `Orchestrator(df).run` + expected-substring check; `understanding` →
  `Interface.ask` + intent/sensor check. Data frames are loaded once and passed in. Tests inject a stub
  runner instead, so the core needs no data/Ollama.

**5. `aggregate(shard_paths) -> list[dict]`**
- Reads every per-shard JSONL, concatenates, sorts by `id` for stable output, and de-duplicates by
  `id` (last write wins — tolerant of a re-run shard). Also emits a tiny summary
  (`{"n", "ok", "correct"}` where applicable). A thin `write_csv(rows, path)` renders the final table.

**6. `render_sbatch(n_shards, account, partition, time, out_dir, python="python3.12") -> str`**
- Produces a `#SBATCH --array=0-(n_shards-1)` script whose body runs
  `python -m src.batch_harness run-shard --shard $SLURM_ARRAY_TASK_ID --num-shards N --kinds ... --out
  $out_dir/shard_$SLURM_ARRAY_TASK_ID.jsonl`. Account/partition/time/out-dir are inputs (no Durham-
  specific values baked in). A final (optional) aggregation step is documented as a dependent job
  (`--dependency=afterok`), not auto-submitted.

### CLI (`python -m src.batch_harness ...`)

- `plan --kinds benchmark,orchestrator --models llama3.2:3b --num-shards 8 --out-dir runs/2026-07-30`
  → writes `jobs.jsonl` + `jena_eval.slurm`, prints the submit command.
- `run-shard --shard K --num-shards N --kinds ... --out shard_K.jsonl` → executes one shard (this is
  what each array task runs).
- `aggregate --glob 'runs/2026-07-30/shard_*.jsonl' --out results.csv` → merges + summarises.
- `local --kinds ... --num-shards 4` → runs every shard in-process (dev/offline mode; no SLURM), then
  aggregates — this is the path exercised by tests and by anyone without a cluster.

## Error handling / determinism

- A runner failure is captured per-job, never raised out of `run_shard`.
- Sharding is deterministic and complete (property-tested: union == all jobs, pairwise disjoint,
  balanced).
- Re-running a shard is safe: `aggregate` dedups by `id`.
- No cluster access is required to develop or test — `local` + a stub runner cover the whole pipeline.

## Testing (TDD, all offline)

1. `assign(i, N)` round-robins; `shard` produces a partition (union == jobs, disjoint, size-balanced).
2. `build_jobs` is deterministic and gives unique ids; job count = |corpus| (× models for benchmark).
3. `run_shard` with a stub runner returns only this shard's jobs, in order, and writes valid JSONL.
4. A runner that raises yields an `ok=False` row, not an exception.
5. `aggregate` merges multiple shard files, sorts by id, dedups a duplicated id, and summarises counts.
6. `render_sbatch` emits `--array=0-(N-1)`, the parametrised account/partition/time, and the
   `run-shard` command with `$SLURM_ARRAY_TASK_ID` — asserted by string checks (no cluster needed).
7. `local` mode runs N shards in-process and aggregates to the same rows as a single-shard run
   (sharding does not change results — the key correctness property).

## Evaluation / write-up deliverable

`v1/docs/evaluation/11-cosma-batch-harness.md`: what the harness is for and why the eval workload is an
ideal HPC fit; the shard→run→aggregate flow with the committed `jena_eval.slurm`; a **real local run**
(`local` mode over the actual corpora) showing identical results to the single-process eval (proving
sharding is result-preserving); the expected linear speed-up argument and its honest limits (LLM
throughput / queue time dominate; array tasks still each need Ollama or an API); and the explicit note
that cluster execution itself was not run here — the Python core and generated script are verified, the
DiRAC submission is the reproducible next step.

## Trust invariants (unchanged)

- The harness schedules and aggregates only; all numbers come from the existing deterministic
  evaluators. Fully offline-testable; no cluster credentials or paths are hard-coded.
