"""Compatibility package for the bench harness.

The actual Python package lives in ``bench/bench``. This outer package keeps
old ``python -m bench.cli`` and ``import bench.*`` paths working from repo root.
"""
from pathlib import Path

_INNER = Path(__file__).resolve().parent / "bench"
if _INNER.is_dir():
    __path__.insert(0, str(_INNER))

__all__ = ["VARIANTS", "BenchResult"]

from .config import VARIANTS  # noqa: E402,F401
from .bench_core import BenchResult  # noqa: E402,F401
