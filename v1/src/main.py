"""Pipeline orchestration / entry point (Phase 7).

Minimal CLI for now: load data -> preprocess -> analyse -> detect -> answer questions.
A Streamlit UI (app.py) can wrap this later.
"""

from __future__ import annotations


def run(data_path: str) -> None:
    """Run the end-to-end pipeline on a dataset. TODO: wire modules together."""
    raise NotImplementedError


if __name__ == "__main__":
    print("time_series_hci — pipeline entry point (not yet implemented).")
