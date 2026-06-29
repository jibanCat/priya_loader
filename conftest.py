"""Root conftest.

Its presence lets pytest import the in-tree ``priya_loader`` package without an
install step: with the default ``prepend`` import mode and no ``tests/__init__.py``,
pytest inserts this rootdir onto ``sys.path``. The supported, robust path is
still ``pip install -e .`` (see environment.yml); this file is belt-and-suspenders.
"""
