#!/usr/bin/env bash
# Fetch the REAL DAD dataset (Vigoya et al., 2020) into code/data/dad/.
#
#   Vigoya, L., Fernandez, D., Carneiro, V. and Cacheda, F. (2020)
#   "Annotated Dataset for Anomaly Detection in a Data Center with IoT Sensors",
#   Sensors 20(13), 3745.  doi:10.3390/s20133745
#   Source: https://github.com/dad-repository/dad
#
# 7 days of MQTT network capture (.pcap) + per-packet attack labels (.csv/.xml).
# ~18 MB total. The data dir is gitignored, so this reproduces it from scratch.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/../../data/dad"          # code/data/dad
mkdir -p "$DEST"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Cloning dad-repository/dad ..."
git clone --depth 1 https://github.com/dad-repository/dad.git "$TMP/dad"
cp "$TMP"/dad/*.pcap "$TMP"/dad/*.csv "$TMP"/dad/*.xml "$TMP"/dad/README.md "$DEST"/

echo "Done. Files in $DEST:"
ls -1 "$DEST" | sed 's/^/  /'
echo "Expected: 7 .pcap + 7 _labels.csv + 7 _labels.xml + README.md (22 files)."
