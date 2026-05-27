"""Modal entry points for the dsv4-attn bench.

Re-exports the app and the GPU-side bench functions:
    app          : modal.App (run via `with app.run(): ...`)
    gpu_bench    : prefill HCA/CSA bench against fp32 reference
    gpu_compile  : compile hca/build/libhca.so inside Modal
    gpu_smoke    : quick all-eager smoke test
    gpu_decode   : single-step HCA decode bench against the decode reference

Image baking (cached after first deploy):
    cu12.9 devel + Python 3.11 + torch>=2.7 (cu128 Blackwell sm_100).
    Local `bench/` and `hca/` (kernel source) are copied into /root.
    Kernel build is lazy: candidates that need a compiled extension
    should compile it on first import (so the image stays cheap when
    only the reference path is exercised).
"""
from .app import (
    app,           # noqa: F401
    gpu_bench,     # noqa: F401
    gpu_compile,   # noqa: F401
    gpu_smoke,     # noqa: F401
    gpu_decode,    # noqa: F401
)
