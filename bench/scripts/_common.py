from __future__ import annotations

import sys
from pathlib import Path


BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent


def add_paths() -> None:
    for path in (REPO_ROOT, BENCH_ROOT):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def run_cli(argv: list[str]) -> int:
    add_paths()
    from bench.cli import main

    old_argv = sys.argv
    sys.argv = ["bench", *argv]
    try:
        return main()
    finally:
        sys.argv = old_argv


def decode_argv(
    *,
    candidate: str,
    m_cur: int,
    swa_len: int,
    iters: int,
    warmup: int,
    dtype: str,
    no_flush: bool,
    local: bool = False,
    n_h: int | None = None,
    c: int | None = None,
    n_rope: int | None = None,
    B: int | None = None,
) -> list[str]:
    argv = [
        "decode",
        "--candidate", candidate,
        "--dtype", dtype,
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--shape", f"M_cur={m_cur}",
        "--shape", f"swa_len={swa_len}",
    ]
    if no_flush:
        argv.append("--no-flush")
    if local:
        argv.append("--local")
    for key, value in (("B", B), ("n_h", n_h), ("c", c), ("n_rope", n_rope)):
        if value is not None:
            argv.extend(["--shape", f"{key}={value}"])
    return argv
