"""Shared fixtures. The `week` MESSAGES frame is parsed once per test session (7 pcaps → ~101k
rows is a few seconds), then reused across test modules."""
import pytest

from src import preprocess


@pytest.fixture(scope="session")
def week():
    if not preprocess.DATA_DIR.exists():
        pytest.skip("DAD dataset not downloaded (run tools/get_dad_dataset.sh)")
    return preprocess.load_week()
