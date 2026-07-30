#!/bin/bash
#SBATCH --job-name=jena-eval
#SBATCH --account=dpXXX
#SBATCH --partition=cosma
#SBATCH --array=0-3
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=hpc/slurm_%A_%a.out

set -euo pipefail
mkdir -p hpc

python3.12 -m src.batch_harness run-shard \
  --shard $SLURM_ARRAY_TASK_ID --num-shards 4 \
  --kinds orchestrator,understanding,benchmark --models llama3.2:3b \
  --out hpc/shard_$SLURM_ARRAY_TASK_ID.jsonl

# After all array tasks finish, aggregate (submit as a dependent job):
#   sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID aggregate.slurm
# or run locally:
#   python3.12 -m src.batch_harness aggregate --glob 'hpc/shard_*.jsonl' \
#     --out hpc/results.csv
