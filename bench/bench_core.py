"""Bench core: build inputs once, time candidate vs reference, report errors.

The bench framework knows nothing about specific kernels.  It accepts:
    * a `VariantSpec` (shape + variant config);
    * a candidate callable taking (weights, inputs, **kwargs) → output;
the reference is always taken from `bench.reference`.

Timing uses CUDA events on a non-default stream.  L2 is flushed between
timed runs by writing to a 64 MiB scratch buffer.

This file imports torch lazily so `python -m bench.cli list` works
without torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional, Any
import statistics

from .config import VariantSpec, VARIANTS


# ── Result type ─────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    variant: str
    candidate: str
    shape: dict
    dtype: str
    iters: int
    warmup: int
    correct: bool
    max_abs_err: float
    max_rel_err: float
    cand_ms_mean: float
    cand_ms_min: float
    cand_ms_max: float
    cand_ms_std: float
    ref_ms_mean: float
    ref_ms_min: float
    ratio: float    # cand / ref (lower = candidate faster)
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        ok = "OK" if self.correct else "FAIL"
        return (f"BenchResult({ok} variant={self.variant} cand={self.candidate} "
                f"cand_ms={self.cand_ms_mean:.3f} ref_ms={self.ref_ms_mean:.3f} "
                f"ratio={self.ratio:.2f} max_abs_err={self.max_abs_err:.2e})")


# ── Input/weight builders ───────────────────────────────────────────────


def _randn(shape, *, generator, device, scale=0.02):
    import torch
    return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32) * scale


def _build_compressor_overlap(spec: VariantSpec, out_dim: int, *, generator, device):
    from .reference import CompressorWeightsOverlap
    g = generator
    d, m = spec.d, spec.m
    return CompressorWeightsOverlap(
        W_a_KV=_randn((d, out_dim), generator=g, device=device),
        W_b_KV=_randn((d, out_dim), generator=g, device=device),
        W_a_Z=_randn((d, out_dim), generator=g, device=device),
        W_b_Z=_randn((d, out_dim), generator=g, device=device),
        B_a=_randn((m, out_dim), generator=g, device=device),
        B_b=_randn((m, out_dim), generator=g, device=device),
    )


def _build_compressor_simple(spec: VariantSpec, out_dim: int, *, generator, device):
    from .reference import CompressorWeightsSimple
    g = generator
    d, m = spec.d, spec.m
    return CompressorWeightsSimple(
        W_a_KV=_randn((d, out_dim), generator=g, device=device),
        W_a_Z=_randn((d, out_dim), generator=g, device=device),
        B_a=_randn((m, out_dim), generator=g, device=device),
    )


def _build_core_attn(spec: VariantSpec, *, generator, device):
    import torch
    from .reference import CoreAttnWeights
    g = generator
    return CoreAttnWeights(
        W_UQ=_randn((spec.d_c, spec.n_h * spec.c), generator=g, device=device),
        q_norm_w=torch.ones(spec.c, device=device, dtype=torch.float32),
        k_norm_w=torch.ones(spec.c, device=device, dtype=torch.float32),
        sink_logits=_randn((spec.n_h,), generator=g, device=device, scale=0.1),
        n_heads=spec.n_h,
        head_dim=spec.c,
        n_rope=spec.n_rope,
        eps=spec.rms_norm_eps,
    )


def build_weights(spec: VariantSpec, *, device, dtype, generator):
    """Allocate paper-faithful weights for HCA or CSA."""
    from .reference import HCAWeights, CSAWeights, IndexerWeights
    g = generator

    W_DQ = _randn((spec.d, spec.d_c), generator=g, device=device)
    W_kv_sw = _randn((spec.d, spec.c), generator=g, device=device)
    core = _build_core_attn(spec, generator=g, device=device)

    if spec.name == "hca":
        return HCAWeights(
            W_DQ=W_DQ,
            W_kv_sw=W_kv_sw,
            core_attn=core,
            kv_compressor=_build_compressor_simple(spec, spec.c, generator=g, device=device),
        )

    # CSA: needs overlapped compressors and indexer head.
    indexer = IndexerWeights(
        W_IUQ=_randn((spec.d_c, spec.n_h_I * spec.c_I), generator=g, device=device),
        W_w=_randn((spec.d, spec.n_h_I), generator=g, device=device),
        n_heads=spec.n_h_I,
        head_dim=spec.c_I,
    )
    return CSAWeights(
        W_DQ=W_DQ,
        W_kv_sw=W_kv_sw,
        core_attn=core,
        kv_compressor=_build_compressor_overlap(spec, spec.c, generator=g, device=device),
        indexer_compressor=_build_compressor_overlap(spec, spec.c_I, generator=g, device=device),
        indexer=indexer,
    )


def build_inputs(spec: VariantSpec, *, device, generator):
    """Build a fresh input batch (B=1, single sequence of length n).

    All inputs are fp32 on `device`; candidates that want bf16/fp8 should
    cast inside their own module — the bench compares output cast back to
    fp32 against the fp32 reference.
    """
    import torch
    from .reference import Inputs
    g = generator
    H = torch.randn(1, spec.n, spec.d, generator=g, device=device, dtype=torch.float32) * 0.5
    return Inputs(H=H)


# ── Reference dispatch ──────────────────────────────────────────────────


def reference_call(spec: VariantSpec, weights, inputs):
    from .reference import hca_forward, csa_forward
    if spec.name == "hca":
        return hca_forward(weights, inputs, m_prime=spec.m, n_win=spec.n_win)
    if spec.name == "csa":
        return csa_forward(weights, inputs, m=spec.m, top_k=spec.top_k, n_win=spec.n_win)
    raise ValueError(f"unknown variant: {spec.name}")


# ── Timing harness ──────────────────────────────────────────────────────


def _flush_l2(scratch):
    scratch.zero_()


def _time_one(fn, *, iters: int, warmup: int, device, flush_l2: bool):
    """Time `fn()` with CUDA events.  Returns list of ms."""
    import torch
    if device.type == "cuda":
        l2 = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int32, device=device) if flush_l2 else None
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            if l2 is not None:
                _flush_l2(l2)
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in zip(starts, ends)]
    # CPU fallback (selftest)
    import time
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _stats(times: list[float]) -> tuple[float, float, float, float]:
    n = len(times)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    if n == 1:
        t = times[0]
        return t, t, t, 0.0
    return (statistics.mean(times), min(times), max(times),
            statistics.stdev(times))


# ── Top-level bench ─────────────────────────────────────────────────────


def run_bench(
    variant: str,
    candidate: Callable,
    *,
    candidate_name: str = "<unnamed>",
    spec_overrides: Optional[dict] = None,
    iters: int = 20,
    warmup: int = 5,
    device: Any = None,
    dtype: str = "fp32",
    flush_l2: bool = True,
    seed: int = 0,
) -> BenchResult:
    """Run reference + candidate, return BenchResult."""
    import torch

    spec = VARIANTS[variant]
    if spec_overrides:
        spec = spec.with_overrides(**spec_overrides)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    g = torch.Generator(device=device).manual_seed(seed)
    weights = build_weights(spec, device=device, dtype=dtype, generator=g)
    inputs = build_inputs(spec, device=device, generator=g)

    # 1. Reference output (fp32, single call — also timed for ratio).
    ref_call = lambda: reference_call(spec, weights, inputs)
    ref_out = ref_call()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ref_times = _time_one(ref_call, iters=max(iters // 4, 1), warmup=max(warmup // 2, 1),
                          device=device, flush_l2=flush_l2)

    # 2. Candidate.
    cand_call = lambda: candidate(weights, inputs, spec=spec)
    try:
        cand_out = cand_call()
    except Exception as exc:
        return BenchResult(
            variant=variant, candidate=candidate_name, shape=spec.to_dict(),
            dtype=dtype, iters=iters, warmup=warmup, correct=False,
            max_abs_err=float("inf"), max_rel_err=float("inf"),
            cand_ms_mean=0.0, cand_ms_min=0.0, cand_ms_max=0.0, cand_ms_std=0.0,
            ref_ms_mean=_stats(ref_times)[0], ref_ms_min=_stats(ref_times)[1],
            ratio=float("inf"),
            message=f"candidate raised: {type(exc).__name__}: {exc}",
        )

    cand32 = cand_out.float()
    ref32 = ref_out.float()
    if cand32.shape != ref32.shape:
        return BenchResult(
            variant=variant, candidate=candidate_name, shape=spec.to_dict(),
            dtype=dtype, iters=iters, warmup=warmup, correct=False,
            max_abs_err=float("inf"), max_rel_err=float("inf"),
            cand_ms_mean=0.0, cand_ms_min=0.0, cand_ms_max=0.0, cand_ms_std=0.0,
            ref_ms_mean=_stats(ref_times)[0], ref_ms_min=_stats(ref_times)[1],
            ratio=float("inf"),
            message=f"shape mismatch: cand={tuple(cand32.shape)} ref={tuple(ref32.shape)}",
        )
    abs_err = (cand32 - ref32).abs()
    max_abs_err = float(abs_err.max().item())
    max_rel_err = float((abs_err / ref32.abs().clamp(min=1e-8)).max().item())

    if dtype == "fp32":
        atol, rtol = spec.atol_fp32, spec.rtol_fp32
    else:
        atol, rtol = spec.atol_bf16, spec.rtol_bf16
    correct = (max_abs_err <= atol) or (max_rel_err <= rtol)

    cand_times = _time_one(cand_call, iters=iters, warmup=warmup,
                           device=device, flush_l2=flush_l2)

    cm, ci, ca, cs = _stats(cand_times)
    rm, ri, ra, rs = _stats(ref_times)
    return BenchResult(
        variant=variant, candidate=candidate_name, shape=spec.to_dict(),
        dtype=dtype, iters=iters, warmup=warmup, correct=correct,
        max_abs_err=max_abs_err, max_rel_err=max_rel_err,
        cand_ms_mean=cm, cand_ms_min=ci, cand_ms_max=ca, cand_ms_std=cs,
        ref_ms_mean=rm, ref_ms_min=ri,
        ratio=cm / rm if rm > 0 else float("inf"),
        message="ok" if correct else
                f"correctness fail: max_abs={max_abs_err:.3e} max_rel={max_rel_err:.3e} (atol={atol}, rtol={rtol})",
    )
