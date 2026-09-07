"""Shared configuration constants for v1.

A handful of values (chiefly the dataset path) were previously duplicated as identical
literals across several apps/scripts, which meant a path change had to be made in every
copy by hand. This module is the single place to change them.
"""

from __future__ import annotations

from pathlib import Path

# v1 project root (the directory containing this src/ package).
V1_ROOT = Path(__file__).resolve().parents[1]

# The Jena Climate CSV. Gitignored (too large) — see README.md "About the data" for where
# to download it; it must sit at exactly this path.
DATA_PATH = V1_ROOT / "data" / "jena_climate_2009_2016.csv"
