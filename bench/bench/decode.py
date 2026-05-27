"""Single-step HCA decode bench: spec, input builder, runner.

Parallel to bench_core.py but specialized to the decode kernel boundary
(pre-projected Q + pre-built fp8 KV cache, output [B, n_h, c+n_rope]).

Candidates live in `bench/candidates/` and declare `VARIANT = "hca_decode"`.
The runner builds:
  - bf16 Q (per-row pre-RoPE'd, then RoPE applied to last n_rope dims)
  - bf16 K-nope which it quantizes to fp8 + e8m0 scales (matches the
    inference engine's per-64 absmax recipe)
  - bf16 K-rope tail
for both the compressed leg (M_cur tokens) and the SWA leg (swa_len tokens).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional
import math
import statistics


# ── Spec ────────────────────────────────────────────────────────────────


@dataclass
class HCADecodeSpec:
    """Decode-step shape.

    M_cur is the populated compressed-cache length (not max). swa_len is the
    populated SWA length (≤ W_SWA=128).

    The default n_h=128 matches the current hca/sm100 kernel, which hardcodes
    two 64-head CTAs. Override n_h=64 for V4-Flash reference-only checks until
    the CUDA kernel accepts the head count dynamically.
    """
    name: str = "hca_decode"

    B:        int = 1
    n_h:      int = 128               # current SM100 kernel head count
    c:        int = 448               # NoPE dim; full head_dim is c + n_rope = 512
    n_rope:   int = 64                # qk_rope_head_dim
    M_cur:    int = 256               # compressed cache fill
    swa_len:  int = 128               # SWA fill
    quant_tile: int = 64              # one e8m0 scale per 64 NoPE channels

    rms_norm_eps: float = 1e-6

    # Tolerance vs the fp8-faithful reference. The fp32 oracle requires a
    # much looser tol because the kernel quantizes K.
    atol_oracle: float = 1.5e-2
    rtol_oracle: float = 5e-2
    atol_faithful: float = 1e-3
    rtol_faithful: float = 5e-3

    @property
    def n_kv(self) -> int:
        return self.M_cur + self.swa_len

    def to_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, **kw) -> "HCADecodeSpec":
        d = self.to_dict()
        d.update(kw)
        return HCADecodeSpec(**d)


def validate_decode_spec(spec: HCADecodeSpec, *, dtype: str) -> None:
    if dtype not in ("fp32", "fp8", "fp8_dequant", "fp8_rope"):
        raise ValueError(f"dtype must be fp32 / fp8 / fp8_dequant / fp8_rope: {dtype!r}")
    if spec.B <= 0 or spec.n_h <= 0:
        raise ValueError(f"B and n_h must be positive: B={spec.B} n_h={spec.n_h}")
    if spec.c <= 0 or spec.n_rope < 0:
        raise ValueError(f"c must be positive and n_rope non-negative: c={spec.c} n_rope={spec.n_rope}")
    if spec.c % spec.quant_tile != 0:
        raise ValueError(f"c must be divisible by quant_tile: c={spec.c} quant_tile={spec.quant_tile}")
    if spec.quant_tile != 64:
        raise ValueError(f"bench/reference assumes e8m0 scales per 64 channels; got {spec.quant_tile}")
    if spec.n_rope % 2 != 0:
        raise ValueError(f"n_rope must be even for split-half RoPE: {spec.n_rope}")
    if dtype == "fp8_rope" and spec.n_rope == 0:
        raise ValueError("fp8_rope requires n_rope > 0")
    if spec.M_cur < 0 or spec.swa_len < 0:
        raise ValueError(f"M_cur and swa_len must be non-negative: M_cur={spec.M_cur} swa_len={spec.swa_len}")
    if spec.M_cur + spec.swa_len <= 0:
        raise ValueError("decode bench needs at least one KV entry")


# ── Quant util (matches kernel's helpers.h cvt.rp ue8m0 path) ───────────


def _quant_kv_to_fp8(K_bf16, *, tile: int):
    """[..., c] bf16 -> (fp8 e4m3, e8m0 byte scales [..., c/tile]).

    Per-tile absmax -> e8m0 = ceil_pow2(absmax / 448). Stored as torch.uint8.
    """
    import torch
    *lead, c = K_bf16.shape
    assert c % tile == 0
    nb = c // tile
    K = K_bf16.view(*lead, nb, tile)

    absmax = K.abs().amax(dim=-1)                     # [..., nb]
    ratio  = absmax.to(torch.float32) / 448.0
    # ceil_pow2 of a fp32 ratio = next power of 2 >= ratio.
    bits = ratio.contiguous().view(torch.int32)
    e    = ((bits >> 23) & 0xff)
    bump = ((bits & 0x7fffff) != 0).to(torch.int32)
    e_byte = (e + bump).clamp_(0, 254).to(torch.uint8)

    dequant = (e_byte.to(torch.int32) << 23).view(torch.float32)   # 2^(byte-127)
    scaled  = K.to(torch.float32) / dequant.unsqueeze(-1)
    fp8     = scaled.clamp_(-448.0, 448.0).to(torch.float8_e4m3fn).view(*lead, c)
    return fp8, e_byte                                # [..., c], [..., nb]


def _quant_rope_to_fp8(R_bf16, *, tile: int = 64):
    """Same recipe as `_quant_kv_to_fp8` but applied to the n_rope tail.

    Default tile = full rope width (one scale per row). Hypothetical: the
    real kernel keeps rope as bf16, this is for diagnostic A/B benches.
    """
    return _quant_kv_to_fp8(R_bf16, tile=tile)


# ── RoPE (matches reference.apply_rope, for decoding a single q_pos) ────


def _apply_rope(x, positions, n_rope):
    import torch
    if n_rope == 0:
        return x
    half = n_rope // 2
    x_pass = x[..., :-n_rope]
    x_rope = x[..., -n_rope:]
    freqs = 1.0 / (10000.0 ** (torch.arange(
        0, n_rope, 2, device=x.device, dtype=torch.float32) / n_rope))
    angles = positions.unsqueeze(-1).float() * freqs
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rot = torch.cat([x1*cos_a - x2*sin_a, x1*sin_a + x2*cos_a], dim=-1)
    return torch.cat([x_pass, x_rot], dim=-1)


# ── Builders ─────────────────────────────────────────────────────────────


def build_decode_inputs(spec: HCADecodeSpec, *, device, generator, dtype: str):
    """Build (HCADecodeInputs, oracle_inputs) where the oracle has bf16 K
    (no quant) and the faithful one has fp8 K + scales.

    `dtype` ∈ {"fp32", "fp8"}: which one to return. We always build the
    underlying bf16 K once so the two views stay numerically aligned.
    """
    import torch
    from .reference import HCADecodeInputs

    validate_decode_spec(spec, dtype=dtype)

    g = generator
    B, n_h, c, n_rope = spec.B, spec.n_h, spec.c, spec.n_rope
    M_cur, swa_len = spec.M_cur, spec.swa_len

    def randn(*shape, scale=0.5):
        return (torch.randn(*shape, generator=g, device=device, dtype=torch.float32)
                * scale).to(torch.bfloat16)

    # Q: random bf16 + RoPE on last n_rope at q_pos.
    Q_bf16 = randn(B, n_h, c + n_rope)
    q_pos  = torch.tensor([M_cur + swa_len - 1], device=device).expand(B)
    Q_bf16 = _apply_rope(Q_bf16.float(), q_pos.unsqueeze(-1).expand(B, n_h),
                         n_rope).to(torch.bfloat16)

    # K-nope (per-leg) + K-rope tail. K-rope already RoPE'd at its own pos.
    Kc_nope    = randn(B, M_cur,   c)
    Kswa_nope  = randn(B, swa_len, c)
    Kc_pos     = (torch.arange(M_cur,   device=device) * spec.quant_tile +
                  spec.quant_tile // 2).unsqueeze(0).expand(B, -1)
    Kswa_pos   = (torch.arange(swa_len, device=device) +
                  (M_cur * spec.quant_tile - swa_len + 1)).clamp(min=0)\
                                  .unsqueeze(0).expand(B, -1)
    Kc_rope    = _apply_rope(randn(B, M_cur,   n_rope).float(), Kc_pos, n_rope)\
                                  .to(torch.bfloat16)
    Kswa_rope  = _apply_rope(randn(B, swa_len, n_rope).float(), Kswa_pos, n_rope)\
                                  .to(torch.bfloat16)

    sink_logits = torch.randn(n_h, generator=g, device=device, dtype=torch.float32) * 0.1
    sm_scale    = 1.0 / math.sqrt(c + n_rope)

    if dtype == "fp32":
        Kc_q,   Kc_s   = Kc_nope,   None
        Kswa_q, Kswa_s = Kswa_nope, None
    elif dtype in ("fp8", "fp8_dequant", "fp8_rope"):
        Kc_q,   Kc_s   = _quant_kv_to_fp8(Kc_nope,   tile=spec.quant_tile)
        Kswa_q, Kswa_s = _quant_kv_to_fp8(Kswa_nope, tile=spec.quant_tile)

    # `fp8_rope` mode: also round-trip rope through fp8 to measure the cost.
    if dtype == "fp8_rope":
        Kc_rope_fp8, Kc_rope_s     = _quant_rope_to_fp8(Kc_rope,   tile=n_rope)
        Kswa_rope_fp8, Kswa_rope_s = _quant_rope_to_fp8(Kswa_rope, tile=n_rope)
        # Dequant in-place so the input still presents bf16 (the kernel reads
        # bf16 rope); only the *values* now reflect the round-trip loss.
        def _deq(fp8, s):
            import torch
            s_f32 = (s.to(torch.uint8).to(torch.int32) << 23).view(torch.float32)
            bf = fp8.to(torch.float32) * s_f32.unsqueeze(-1)
            return bf.to(torch.bfloat16)
        Kc_rope   = _deq(Kc_rope_fp8,   Kc_rope_s)
        Kswa_rope = _deq(Kswa_rope_fp8, Kswa_rope_s)

    return HCADecodeInputs(
        Q=Q_bf16,
        Kc=Kc_q, Kc_rope=Kc_rope, Kc_scales=Kc_s,
        Kswa=Kswa_q, Kswa_rope=Kswa_rope, Kswa_scales=Kswa_s,
        sink_logits=sink_logits, sm_scale=sm_scale,
    )


# ── Reference dispatch ──────────────────────────────────────────────────


def reference_decode(spec: HCADecodeSpec, inp, *, dtype: str):
    from .reference import hca_decode_forward
    return hca_decode_forward(inp, dtype=dtype)


# ── Runner ──────────────────────────────────────────────────────────────


@dataclass
class DecodeBenchResult:
    spec: dict
    candidate: str
    dtype: str
    correct: bool
    max_abs_err: float
    max_rel_err: float
    cand_ms_mean: float
    cand_ms_min: float
    ref_ms_mean: float
    ratio: float
    message: str = ""


def _time(fn, *, iters, warmup, device, flush_l2):
    import torch, time
    if device.type == "cuda":
        l2 = torch.empty(64*1024*1024//4, dtype=torch.int32, device=device) if flush_l2 else None
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            if l2 is not None: l2.zero_()
            starts[i].record(); fn(); ends[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in zip(starts, ends)]
    for _ in range(warmup): fn()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def run_decode_bench(
    candidate: Callable,
    *,
    candidate_name: str = "<unnamed>",
    spec: Optional[HCADecodeSpec] = None,
    spec_overrides: Optional[dict] = None,
    iters: int = 20,
    warmup: int = 5,
    device: Any = None,
    dtype: str = "fp8",                # "fp8" (faithful) or "fp32" (oracle)
    flush_l2: bool = True,
    seed: int = 0,
) -> DecodeBenchResult:
    import torch
    if spec is None: spec = HCADecodeSpec()
    if spec_overrides: spec = spec.with_overrides(**spec_overrides)
    validate_decode_spec(spec, dtype=dtype)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    g = torch.Generator(device=device).manual_seed(seed)
    inp = build_decode_inputs(spec, device=device, generator=g, dtype=dtype)

    # Reference: the same decode forward, run with matching dtype.
    ref_call  = lambda: reference_decode(spec, inp, dtype=dtype)
    ref_out   = ref_call()
    if device.type == "cuda": torch.cuda.synchronize()
    ref_times = _time(ref_call, iters=max(iters//4, 1),
                      warmup=max(warmup//2, 1), device=device, flush_l2=flush_l2)

    cand_call = lambda: candidate(inp, spec=spec)
    try:
        cand_out = cand_call()
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as exc:
        return DecodeBenchResult(
            spec=spec.to_dict(), candidate=candidate_name, dtype=dtype,
            correct=False, max_abs_err=float("inf"), max_rel_err=float("inf"),
            cand_ms_mean=0.0, cand_ms_min=0.0,
            ref_ms_mean=statistics.mean(ref_times) if ref_times else 0.0,
            ratio=float("inf"),
            message=f"candidate raised: {type(exc).__name__}: {exc}",
        )

    cf, rf = cand_out.float(), ref_out.float()
    if cf.shape != rf.shape:
        return DecodeBenchResult(
            spec=spec.to_dict(), candidate=candidate_name, dtype=dtype,
            correct=False, max_abs_err=float("inf"), max_rel_err=float("inf"),
            cand_ms_mean=0.0, cand_ms_min=0.0,
            ref_ms_mean=statistics.mean(ref_times),
            ratio=float("inf"),
            message=f"shape mismatch: cand={tuple(cf.shape)} ref={tuple(rf.shape)}",
        )
    abs_err = (cf - rf).abs()
    max_abs = float(abs_err.max().item())
    max_rel = float((abs_err / rf.abs().clamp(min=1e-8)).max().item())

    atol = spec.atol_faithful if dtype == "fp8" else spec.atol_oracle
    rtol = spec.rtol_faithful if dtype == "fp8" else spec.rtol_oracle
    correct = (max_abs <= atol) or (max_rel <= rtol)

    cand_times = _time(cand_call, iters=iters, warmup=warmup,
                       device=device, flush_l2=flush_l2)
    cand_mean = statistics.mean(cand_times) if cand_times else 0.0
    cand_min  = min(cand_times) if cand_times else 0.0
    ref_mean  = statistics.mean(ref_times) if ref_times else 0.0
    return DecodeBenchResult(
        spec=spec.to_dict(), candidate=candidate_name, dtype=dtype,
        correct=correct, max_abs_err=max_abs, max_rel_err=max_rel,
        cand_ms_mean=cand_mean, cand_ms_min=cand_min,
        ref_ms_mean=ref_mean, ratio=cand_mean / ref_mean if ref_mean else 0.0,
    )
