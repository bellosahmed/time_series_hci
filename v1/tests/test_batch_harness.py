"""The batch-harness core is workload-agnostic and offline: sharding is a deterministic, balanced
partition; run_shard/aggregate/render_sbatch need no SLURM or Ollama (stub runner)."""
import json

from src import batch_harness as bh


def _jobs(n):
    return [bh.JobSpec(id=f"j{i:03d}", kind="stub", payload={"i": i}) for i in range(n)]


# --- Task 1: sharding ---------------------------------------------------------------------

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


# --- Task 2: run_shard --------------------------------------------------------------------

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


# --- Task 3: aggregate / summarise / write_csv --------------------------------------------

def test_aggregate_merges_sorts_and_dedups(tmp_path):
    (tmp_path / "s0.jsonl").write_text(
        '{"id": "j001", "ok": true, "correct": true}\n{"id": "j000", "ok": true, "correct": false}\n')
    (tmp_path / "s1.jsonl").write_text(
        '{"id": "j000", "ok": true, "correct": true}\n')   # duplicate id -> last wins
    rows = bh.aggregate([tmp_path / "s0.jsonl", tmp_path / "s1.jsonl"])
    assert [r["id"] for r in rows] == ["j000", "j001"]      # sorted, deduped
    assert next(r for r in rows if r["id"] == "j000")["correct"] is True
    summ = bh.summarise(rows)
    assert summ == {"n": 2, "ok": 2, "correct": 2}  # last-wins made j000 correct, j001 already was


def test_write_csv_roundtrips(tmp_path):
    rows = [{"id": "a", "ok": True}, {"id": "b", "ok": False}]
    out = tmp_path / "r.csv"
    bh.write_csv(rows, out)
    text = out.read_text()
    assert "id" in text and "a" in text and "b" in text


# --- Task 4: render_sbatch ----------------------------------------------------------------

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


# --- Task 5: build_jobs / run_local -------------------------------------------------------

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


# --- Task 6: CLI --------------------------------------------------------------------------

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
