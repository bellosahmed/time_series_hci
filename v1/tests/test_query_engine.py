"""Unit tests for the query engine (the truth layer).

These confirm the exact computations match pandas ground truth. Once implemented,
remove the skip and assert real values.
"""

import pytest


@pytest.mark.skip(reason="query_engine not yet implemented")
def test_get_max_matches_pandas():
    """get_max should equal df[sensor].max() exactly."""
    raise NotImplementedError
