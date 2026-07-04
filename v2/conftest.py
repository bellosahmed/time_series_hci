"""Pytest bootstrap for v2.

Its mere presence makes this folder the test rootdir and puts it on ``sys.path``,
so ``from src import ...`` resolves whether you run ``pytest`` or ``python -m pytest``
from here or from the repository root. No fixtures needed.
"""
