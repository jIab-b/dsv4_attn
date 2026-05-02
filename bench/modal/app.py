"""Modal app: B200 image + bench entry points.

The image mounts:
  /root/bench  ← bench/  (Python harness)
  /root/hca    ← hca/    (kernel source; compiled on demand by candidates)

Functions:
  gpu_bench(...)   prefill HCA/CSA → run_bench()
  gpu_smoke()      run all eager candidates with default shapes
  gpu_decode(...)  decode-step HCA → run_decode_bench()
"""
from __future__ import annotations

from pathlib import Path

import modal


APP_NAME       = "dsv4-attn-bench"
GPU_TYPE       = "B200"
LOCAL_BENCH    = Path(__file__).resolve().parent.parent          # bench/
LOCAL_KERNELS  = LOCAL_BENCH.parent / "hca"                       # hca/
REMOTE_ROOT    = Path("/root")
REMOTE_BENCH   = REMOTE_ROOT / "bench"
REMOTE_KERNELS = REMOTE_ROOT / "hca"


app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("git", "ca-certificates")
    # cu128 wheel: shipped with Blackwell (sm_100) since torch 2.7. Triton
    # bundled.
    .pip_install("numpy", "ninja", "torch>=2.7")
    .add_local_dir(LOCAL_BENCH,   remote_path=str(REMOTE_BENCH),   copy=True)
    .add_local_dir(LOCAL_KERNELS, remote_path=str(REMOTE_KERNELS), copy=True)
    .workdir(str(REMOTE_ROOT))
)


def _ensure_path():
    """Make `bench` and `hca` importable inside the Modal container."""
    import sys
    if str(REMOTE_ROOT) not in sys.path:
        sys.path.insert(0, str(REMOTE_ROOT))


# ── Prefill bench ───────────────────────────────────────────────────────


@app.function(image=image, gpu=GPU_TYPE, timeout=600)
def gpu_bench(
    variant: str,
    candidate_name: str,
    spec_overrides: dict | None,
    iters: int,
    warmup: int,
    dtype: str,
    flush_l2: bool,
    seed: int,
) -> dict:
    _ensure_path()
    from bench.bench_core import run_bench
    from bench.candidates import load_candidate

    mod = load_candidate(candidate_name)
    res = run_bench(
        variant=variant,
        candidate=mod.candidate,
        candidate_name=candidate_name,
        spec_overrides=spec_overrides,
        iters=iters,
        warmup=warmup,
        dtype=dtype,
        flush_l2=flush_l2,
        seed=seed,
    )
    return res.to_dict()


# ── Smoke test ──────────────────────────────────────────────────────────


@app.function(image=image, gpu=GPU_TYPE, timeout=600)
def gpu_smoke() -> dict:
    """Run all eager (non-compiled, non-stub) candidates with default shapes."""
    _ensure_path()
    from bench.bench_core import run_bench
    from bench.candidates import list_candidates, load_candidate

    table = list_candidates()
    out = {"candidates": table, "results": {}}
    for name, info in table.items():
        if info.get("variant") not in ("hca", "csa"):
            continue
        if "compiled" in name:
            continue
        try:
            mod = load_candidate(name)
            res = run_bench(
                variant=info["variant"],
                candidate=mod.candidate,
                candidate_name=name,
                iters=5, warmup=2,
                dtype="fp32", flush_l2=True, seed=0,
            )
            out["results"][name] = res.to_dict()
        except Exception as exc:  # pragma: no cover
            out["results"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# ── Decode bench (single-step HCA) ──────────────────────────────────────


@app.function(image=image, gpu=GPU_TYPE, timeout=600)
def gpu_decode(
    candidate_name: str,
    spec_overrides: dict | None,
    iters: int,
    warmup: int,
    dtype: str,
    flush_l2: bool,
    seed: int,
) -> dict:
    """Run a decode-step candidate against the decode reference.

    `candidate_name` must be a candidate with VARIANT='hca_decode'.
    `dtype` ∈ {'fp8', 'fp32'}: 'fp8' = faithful (K quantized), 'fp32' = oracle.
    """
    _ensure_path()
    from bench.candidates import list_candidates, load_candidate
    from bench.decode import run_decode_bench, HCADecodeSpec

    table = list_candidates()
    if candidate_name not in table:
        return {"error": f"unknown candidate {candidate_name!r}",
                "have": list(table)}
    if table[candidate_name].get("variant") != "hca_decode":
        return {"error": f"{candidate_name!r} is not VARIANT='hca_decode'"}

    mod = load_candidate(candidate_name)
    res = run_decode_bench(
        candidate=mod.candidate,
        candidate_name=candidate_name,
        spec=HCADecodeSpec(),
        spec_overrides=spec_overrides,
        iters=iters,
        warmup=warmup,
        dtype=dtype,
        flush_l2=flush_l2,
        seed=seed,
    )
    return {
        "candidate":    res.candidate,
        "spec":         res.spec,
        "dtype":        res.dtype,
        "correct":      res.correct,
        "max_abs_err":  res.max_abs_err,
        "max_rel_err":  res.max_rel_err,
        "cand_ms_mean": res.cand_ms_mean,
        "cand_ms_min":  res.cand_ms_min,
        "ref_ms_mean":  res.ref_ms_mean,
        "ratio":        res.ratio,
        "message":      res.message,
    }
