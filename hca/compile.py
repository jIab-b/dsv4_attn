#!/usr/bin/env python3
"""Build hca_decode_fwd as a standalone .so. No torch / pybind needed.

Default: emits hca/build/libhca.so containing the C ABI entry
`hca_decode_fwd` (see sm100/kernel.cu). Loadable via ctypes.

    python hca/compile.py                 # builds .so
    python hca/compile.py --object        # emits a .o instead (no link)
    python hca/compile.py --ptx           # emits .ptx
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCES = [
    HERE / "sm100" / "kernel.cu",
    HERE / "sm100" / "combine.cu",
]
DEFAULT_OUT_DIR = HERE / "build"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile hca to a .so / .o / .ptx")
    p.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    p.add_argument("--out", type=Path, help="output path (default: build/libhca.so)")
    p.add_argument("--arch", default="compute_100f")
    p.add_argument("--code", default="sm_100f")
    p.add_argument("--object", action="store_true", help="emit .o, no link")
    p.add_argument("--ptx", action="store_true", help="emit .ptx, no link")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra nvcc args after --")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    nvcc = shutil.which(args.nvcc)
    if nvcc is None:
        print(f"error: nvcc not found: {args.nvcc}", file=sys.stderr)
        return 127

    if args.ptx and args.object:
        print("--ptx and --object are mutually exclusive", file=sys.stderr)
        return 2

    if args.ptx:
        out = args.out or (DEFAULT_OUT_DIR / "kernel.ptx")
        mode = ["-ptx"]
    elif args.object:
        out = args.out or (DEFAULT_OUT_DIR / "kernel.o")
        mode = ["-c"]
    else:
        out = args.out or (DEFAULT_OUT_DIR / "libhca.so")
        mode = ["-shared", "-Xcompiler", "-fPIC"]

    out.parent.mkdir(parents=True, exist_ok=True)
    opt = ["-G", "-g"] if args.debug else ["-O3"]
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    cmd = [
        nvcc,
        "-std=c++17",
        *opt,
        "--use_fast_math",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-I", str(HERE),
        "-gencode", f"arch={args.arch},code={args.code}",
        *mode,
        *(str(s) for s in (SOURCES if not args.ptx else [SOURCES[0]])),
        "-o", str(out),
        "-lcuda",
        *extra,
    ]

    print(" ".join(cmd))
    if args.dry_run:
        return 0
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print(f"# wrote {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
