#!/usr/bin/env bash
# One-command reproduction of the v3 quality gates.
set -euo pipefail
cd "$(dirname "$0")"

echo "=================================================================="
echo " v3 — Verifiable NL Analytics for a Data-Center IoT Network (DAD)"
echo " Python: $(python3.12 --version 2>&1)"
echo "=================================================================="

if [ ! -d ../data/dad ]; then
  echo "DAD dataset missing — fetching it..."
  bash tools/get_dad_dataset.sh
fi

echo; echo "1/3  pytest"
python3.12 -m pytest tests/ -q

echo; echo "2/3  verify_results.py (independent trust gate)"
python3.12 verify_results.py

echo; echo "3/3  ruff"
python3.12 -m ruff check src tests apps verify_results.py

echo; echo "All gates passed."
