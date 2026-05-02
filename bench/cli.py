#!/usr/bin/env python3
"""bench CLI: list / selftest / bench / smoke commands.

  # Discover what's installed (no GPU needed)
  python -m bench.cli list

  # Local CPU correctness sanity (tiny shapes, no GPU)
  python -m bench.cli selftest

  # Run a candidate on a Modal B200 (compares vs reference)
  python -m bench.cli bench --variant hca --candidate hca_torch_eager
  python -m bench.cli bench --variant csa --candidate csa_custom \
      --iters 30 --warmup 5 --shape seq_kv=2048

  # Quick GPU smoke: run all eager candidates on Modal once
  python -m bench.cli smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_shape_overrides(items: list[str] | None) -> dict:
    if not items:
        return {}
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--shape entries must be key=value: {it!r}")
        k, v = it.split("=", 1)
        try:
            v_parsed = int(v)
        except ValueError:
            try:
                v_parsed = float(v)
            except ValueError:
                v_parsed = v
        out[k.strip()] = v_parsed
    return out


def cmd_list(_args) -> int:
    from bench.candidates import list_candidates
    from bench.config import VARIANTS

    print("Variants:")
    for name, spec in VARIANTS.items():
        print(f"  {name}  m={spec.m} n={spec.n} d={spec.d} "
              f"n_h={spec.n_h} c={spec.c} d_c={spec.d_c} n_win={spec.n_win}"
              + (f" top_k={spec.top_k}" if spec.name == "csa" else ""))
    print("\nCandidates:")
    for name, info in list_candidates().items():
        msg = info.get("error", info["variant"])
        print(f"  {name:24s}  variant={msg}")
    return 0


def cmd_selftest(_args) -> int:
    """Run a tiny CPU bench round-trip for all eager candidates.

    Doesn't need a GPU.  Verifies the math is internally consistent
    (candidate_eager == reference, since they are aliases).
    """
    from bench.bench_core import run_bench
    from bench.candidates import list_candidates, load_candidate
    from bench.config import VARIANTS

    table = list_candidates()
    rc = 0
    for variant in ("hca", "csa"):
        name = f"{variant}_torch_eager"
        if name not in table:
            print(f"[skip] {name} not found")
            continue
        mod = load_candidate(name)
        # Tiny shapes for CPU.
        if variant == "hca":
            overrides = dict(n=256, d=128, c=32, d_c=64, n_h=4, n_rope=8,
                             m=128, n_win=16)
        else:
            overrides = dict(n=64, d=128, c=32, d_c=64, n_h=4, n_rope=8,
                             m=4, n_win=16, n_h_I=2, c_I=16, top_k=4)
        res = run_bench(
            variant=variant,
            candidate=mod.candidate,
            candidate_name=name,
            spec_overrides=overrides,
            iters=2, warmup=1, dtype="fp32", flush_l2=False,
            seed=0, device="cpu",
        )
        ok = "OK" if res.correct else "FAIL"
        print(f"[{ok}] {name}  abs_err={res.max_abs_err:.2e}  "
              f"cand_ms={res.cand_ms_mean:.1f}  ref_ms={res.ref_ms_mean:.1f}")
        if not res.correct:
            print(f"       message: {res.message}")
            rc = 1
    return rc


def cmd_bench(args) -> int:
    overrides = parse_shape_overrides(args.shape)
    from bench.modal_app import app, gpu_bench
    print(f"# variant={args.variant} candidate={args.candidate} "
          f"overrides={overrides} iters={args.iters} dtype={args.dtype}")
    with app.run():
        result = gpu_bench.remote(
            variant=args.variant,
            candidate_name=args.candidate,
            spec_overrides=overrides or None,
            iters=args.iters,
            warmup=args.warmup,
            dtype=args.dtype,
            flush_l2=not args.no_flush,
            seed=args.seed,
        )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"# wrote {args.out}")
    if not result.get("correct"):
        return 3
    return 0


def cmd_decode(args) -> int:
    """Run a decode-step bench. Default: on Modal B200 (matches `bench`).
    Pass --local to run on the host CPU/GPU instead (needs torch).
    """
    overrides = parse_shape_overrides(args.shape)

    if args.local:
        from bench.candidates import list_candidates, load_candidate
        from bench.decode import run_decode_bench, HCADecodeSpec

        table = list_candidates()
        if args.candidate not in table:
            print(f"unknown candidate {args.candidate!r}; have: {list(table)}")
            return 2
        if table[args.candidate].get("variant") != "hca_decode":
            print(f"{args.candidate!r} is not VARIANT='hca_decode'")
            return 2
        mod = load_candidate(args.candidate)
        res = run_decode_bench(
            candidate=mod.candidate,
            candidate_name=args.candidate,
            spec=HCADecodeSpec(),
            spec_overrides=overrides or None,
            iters=args.iters, warmup=args.warmup,
            dtype=args.dtype,
            flush_l2=not args.no_flush,
            seed=args.seed,
        )
        result = {
            "candidate": res.candidate, "spec": res.spec, "dtype": res.dtype,
            "correct": res.correct, "max_abs_err": res.max_abs_err,
            "max_rel_err": res.max_rel_err, "cand_ms_mean": res.cand_ms_mean,
            "cand_ms_min": res.cand_ms_min, "ref_ms_mean": res.ref_ms_mean,
            "ratio": res.ratio, "message": res.message,
        }
    else:
        from bench.modal import app, gpu_decode
        print(f"# decode on Modal B200 candidate={args.candidate} "
              f"dtype={args.dtype} overrides={overrides}")
        with app.run():
            result = gpu_decode.remote(
                candidate_name=args.candidate,
                spec_overrides=overrides or None,
                iters=args.iters, warmup=args.warmup,
                dtype=args.dtype,
                flush_l2=not args.no_flush,
                seed=args.seed,
            )

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"# wrote {args.out}")
    return 0 if result.get("correct") else 3


def cmd_smoke(args) -> int:
    from bench.modal_app import app, gpu_smoke
    print("# running gpu_smoke on Modal (B200)…")
    with app.run():
        result = gpu_smoke.remote()
    print(json.dumps(result, indent=2))
    rc = 0
    for name, res in result.get("results", {}).items():
        if "error" in res or not res.get("correct"):
            rc = 1
    return rc


def main() -> int:
    p = argparse.ArgumentParser(prog="bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list variants and candidates")
    sub.add_parser("selftest", help="local CPU sanity check")
    sub.add_parser("smoke", help="run all eager candidates on Modal B200")

    pb = sub.add_parser("bench", help="run one candidate on Modal B200")
    pb.add_argument("--variant", required=True, choices=("hca", "csa"))
    pb.add_argument("--candidate", required=True,
                    help="candidate NAME (see `bench list`)")
    pb.add_argument("--iters", type=int, default=20)
    pb.add_argument("--warmup", type=int, default=5)
    pb.add_argument("--dtype", default="fp32", choices=("fp32",),
                    help="reference dtype (only fp32 supported for now)")
    pb.add_argument("--shape", action="append", metavar="K=V",
                    help="override a VariantSpec field (repeatable)")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--no-flush", action="store_true",
                    help="don't flush L2 between timed runs")
    pb.add_argument("--out", help="save bench result JSON")

    pd = sub.add_parser("decode",
                        help="single-step HCA decode bench (local, needs torch)")
    pd.add_argument("--candidate", required=True,
                    help="candidate NAME with VARIANT='hca_decode'")
    pd.add_argument("--iters", type=int, default=20)
    pd.add_argument("--warmup", type=int, default=5)
    pd.add_argument("--dtype", default="fp8", choices=("fp8", "fp32"),
                    help="fp8: faithful (K quantized); fp32: oracle")
    pd.add_argument("--shape", action="append", metavar="K=V",
                    help="override an HCADecodeSpec field (repeatable)")
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--no-flush", action="store_true")
    pd.add_argument("--local", action="store_true",
                    help="run on host (needs torch) instead of Modal B200")
    pd.add_argument("--out", help="save result JSON")

    args = p.parse_args()
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "selftest":
        return cmd_selftest(args)
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd == "bench":
        return cmd_bench(args)
    if args.cmd == "decode":
        return cmd_decode(args)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
