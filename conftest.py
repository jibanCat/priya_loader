"""Root conftest.

Its presence lets pytest import the in-tree ``priya_loader`` package without an
install step: with the default ``prepend`` import mode and no ``tests/__init__.py``,
pytest inserts this rootdir onto ``sys.path``. The supported, robust path is
still ``pip install -e .`` (see environment.yml); this file is belt-and-suspenders.

It also wires the ``realdata`` marker: tests that need the real (multi-GB) PRIYA
tree are auto-skipped unless the ``PRIYA_DATA_ROOT`` environment variable points
at a data root. This keeps the default ``pytest`` run hermetic, fast, and
path-independent (no hard-coded machine paths).
"""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PRIYA_DATA_ROOT"):
        return
    skip_realdata = pytest.mark.skip(reason="set PRIYA_DATA_ROOT to run realdata tests")
    for item in items:
        if "realdata" in item.keywords:
            item.add_marker(skip_realdata)
