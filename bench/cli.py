#!/usr/bin/env python3
"""Compatibility shim for ``python -m bench.cli`` from repo root."""
from .bench.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
