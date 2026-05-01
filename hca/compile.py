#!/usr/bin/env python3
"""Compile-only smoke test for the HCA SM100 CUDA source."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "sm100" / "instantiations" / "m128.cu"
DEFAULT_BUILD_DIR = HERE / "build" / "compile"


def local_path(path: Path) -> Path:
    return path if path.is_absolute() else HERE / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nvcc on the HCA SM100 instantiation without linking."
    )
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arch", default="compute_100f")
    parser.add_argument("--code", default="sm_100f")
    parser.add_argument("--ptx", action="store_true", help="emit PTX instead of an object")
    parser.add_argument("--debug", action="store_true", help="use -G -g instead of -O3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="extra nvcc args after --")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nvcc = shutil.which(args.nvcc)
    if nvcc is None:
        print(f"error: nvcc not found: {args.nvcc}", file=sys.stderr)
        return 127

    source = local_path(args.source)
    suffix = ".ptx" if args.ptx else ".o"
    out = local_path(args.out) if args.out else DEFAULT_BUILD_DIR / f"{source.stem}{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)

    opt_flags = ["-G", "-g"] if args.debug else ["-O3"]
    mode_flags = ["-ptx"] if args.ptx else ["-c"]
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    cmd = [
        nvcc,
        "-std=c++17",
        *opt_flags,
        "--use_fast_math",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-I",
        str(HERE),
        "-gencode",
        f"arch={args.arch},code={args.code}",
        *mode_flags,
        str(source),
        "-o",
        str(out),
        *extra,
    ]

    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
