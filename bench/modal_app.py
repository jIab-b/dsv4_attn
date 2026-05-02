"""Re-export shim for the old `bench.modal_app` import path.

The actual app + functions now live in `bench/modal/`. This file exists
so existing code (and any external imports) continues to work.
"""
from bench.modal import app, gpu_bench, gpu_smoke, gpu_decode  # noqa: F401
