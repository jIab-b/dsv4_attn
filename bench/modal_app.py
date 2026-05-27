"""Compatibility shim for the old ``bench.modal_app`` import path."""
from .bench.modal_app import app, gpu_bench, gpu_compile, gpu_smoke, gpu_decode  # noqa: F401
