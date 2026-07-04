"""Run the client-side JS unit suite (apps/web/chart.test.js) as part of pytest.

A full browser is not available in this environment, so we cannot do true end-to-end browser
testing; this instead gives real *behavioural* coverage of the pure render logic (chart drawing +
adaptive legend) by executing it in Node. Skipped if Node is not installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "apps" / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_chart_js_unit_suite():
    r = subprocess.run(["node", "chart.test.js"], cwd=WEB, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all assertions passed" in r.stdout
