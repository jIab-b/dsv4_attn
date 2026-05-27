#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import decode_argv, run_cli


def main() -> int:
    p = argparse.ArgumentParser(description="Run CUDA-graph HCA main kernel + combine on Modal.")
    p.add_argument("--M-cur", "--m-cur", dest="m_cur", type=int, default=9472)
    p.add_argument("--swa-len", type=int, default=0)
    p.add_argument("--B", type=int)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--dtype", default="fp8", choices=("fp8", "fp32", "fp8_dequant", "fp8_rope"))
    p.add_argument("--flush", action="store_true")
    args = p.parse_args()

    return run_cli(decode_argv(
        candidate="hca_custom_decode_graph",
        m_cur=args.m_cur,
        swa_len=args.swa_len,
        B=args.B,
        iters=args.iters,
        warmup=args.warmup,
        dtype=args.dtype,
        no_flush=not args.flush,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
