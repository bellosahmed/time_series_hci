#!/usr/bin/env bash
# Reproduce every v1 result from a clean checkout, in one command.
#
#   bash reproduce.sh
#
# It runs the test suite (correctness) and the independent verification harness
# (numbers == pandas ground truth, real-event corroboration, statistical rigor).
# All randomness is seeded (see evaluation.inject_spikes / bootstrap_ci / benchmark),
# so the counted checks are deterministic across runs and machines.
#
# Requirements: Python 3.12 with requirements.txt installed, and the dataset at
#   data/jena_climate_2009_2016.csv  (gitignored; Jena Climate 2009-2016).
# The local LLM (Ollama) is optional — its faithfulness check is advisory only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$HERE"
PY="${PYTHON:-python3.12}"

echo "== environment =="
"$PY" --version
"$PY" -c "import numpy,pandas,sklearn,scipy,statsmodels; \
print('numpy',numpy.__version__,'| pandas',pandas.__version__,\
'| scikit-learn',sklearn.__version__,'| scipy',scipy.__version__,\
'| statsmodels',statsmodels.__version__)"
"$PY" -c "import torch; print('torch',torch.__version__,'| cuda',torch.cuda.is_available())" || \
  echo "torch not available (deep-model notebook only)"

echo
echo "== test suite =="
"$PY" -m pytest tests/ -q

echo
echo "== verification harness (independent ground-truth checks) =="
"$PY" verify_results.py

echo
echo "All done. If both sections passed, every counted result is reproduced."
