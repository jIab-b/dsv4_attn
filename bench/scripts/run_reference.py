#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import decode_argv, run_cli


def main() -> int:
    p = argparse.ArgumentParser(description="Run PyTorch decode reference against itself locally.")
    p.add_argument("--M-cur", "--m-cur", dest="m_cur", type=int, default=128)
    p.add_argument("--swa-len", type=int, default=0)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--n-h", type=int, default=4)
    p.add_argument("--c", type=int, default=64)
    p.add_argument("--n-rope", type=int, default=64)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--dtype", default="fp8", choices=("fp8", "fp32", "fp8_dequant", "fp8_rope"))
    p.add_argument("--flush", action="store_true")
    args = p.parse_args()

    return run_cli(decode_argv(
        candidate="hca_decode_oracle",
        m_cur=args.m_cur,
        swa_len=args.swa_len,
        B=args.B,
        n_h=args.n_h,
        c=args.c,
        n_rope=args.n_rope,
        iters=args.iters,
        warmup=args.warmup,
        dtype=args.dtype,
        no_flush=not args.flush,
        local=True,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
