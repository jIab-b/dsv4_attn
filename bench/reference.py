"""Reference implementations of HCA and CSA — paper-faithful version.

Math follows DeepSeek-V4 paper §2.3.1 (CSA) and §2.3.3 (QK Norm, Partial
RoPE, Sliding Window, Attention Sink), as transcribed by Arjun Kocher's
faithful CSA reference at https://www.k-a.in/CSA.html.  Equation numbers
in comments refer to the paper.

Reductions vs. the article: we skip the SwiGLU FFN and the grouped output
projection (those aren't part of the attention kernel benchmark) and run
B=1 only.  Otherwise this is the article's CSA layer line-for-line, with
an HCA twin built on the same primitives but with the no-overlap
compressor and dense (no top-k) selection.

Top-level entry points used by the bench:
    hca_forward(weights, inputs, *, m_prime, n_win)
    csa_forward(weights, inputs, *, m, top_k, n_win)

Both return [B=1, n, n_h, c] attention output (post-merge with SWA + sink,
post-inverse-RoPE).  The grouped output projection is the candidate's
problem after this, so we stop here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F


# ── RoPE / RMSNorm primitives (article-style) ──────────────────────────


def apply_rope(x: torch.Tensor, positions: torch.Tensor, n_rope: int) -> torch.Tensor:
    """Partial RoPE on the last `n_rope` dims of `x` (split-halves form).

    Per the article and §2.3.3 of the paper:
      * `freqs = 1 / 10000^(2i/n_rope)` for i in [0, n_rope/2)
      * x split as (x1, x2) along the rope-slice (each n_rope/2 wide)
      * x_rot = [x1*cos - x2*sin, x1*sin + x2*cos]
    `positions` must broadcast to `x[..., 0]`.
    """
    assert n_rope % 2 == 0
    if n_rope == 0:
        return x
    x_pass = x[..., :-n_rope]
    x_rope = x[..., -n_rope:]

    half = n_rope // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(
        0, n_rope, 2, device=x.device, dtype=torch.float32) / n_rope))
    angles = positions.unsqueeze(-1).float() * freqs

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:]
    x_rot = torch.cat(
        [x1 * cos_a - x2 * sin_a,
         x1 * sin_a + x2 * cos_a],
        dim=-1,
    )
    return torch.cat([x_pass, x_rot], dim=-1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype)


# ── Token-Level Compressor (Eqs 9-12) ───────────────────────────────────


@dataclass
class CompressorWeightsSimple:
    """No-overlap (HCA-style) compressor weights — single-branch.

    Equivalent to article's `TokenLevelCompressor` with the C_b/Z_b/B_b
    branches removed.  Used for HCA where m'=128 and the paper drops the
    overlap (vLLM/SGLang `coff = 1` path).
    """
    W_a_KV: torch.Tensor   # [d, out_dim]
    W_a_Z: torch.Tensor    # [d, out_dim]
    B_a: torch.Tensor      # [m, out_dim]


@dataclass
class CompressorWeightsOverlap:
    """Overlapped (CSA-style) compressor weights — two-branch (Eqs 9-12).

    Mirrors article's `TokenLevelCompressor`:
        C_a, C_b = H · W^aKV,  H · W^bKV         (Eq 9)
        Z_a, Z_b = H · W^aZ,   H · W^bZ          (Eq 10)
        S = softmax_row([Z_a + B_a; Z_b_prev + B_b])  over 2m positions  (Eq 11)
        C^Comp = sum S_a ⊙ C_a + sum S_b ⊙ C_b_prev   (Eq 12)
    Block 0 pads C_b with zeros and Z_b with -inf (so only C_a contributes).
    """
    W_a_KV: torch.Tensor
    W_b_KV: torch.Tensor
    W_a_Z: torch.Tensor
    W_b_Z: torch.Tensor
    B_a: torch.Tensor      # [m, out_dim]
    B_b: torch.Tensor      # [m, out_dim]


def compressor_simple(
    H: torch.Tensor,
    w: CompressorWeightsSimple,
    m: int,
) -> torch.Tensor:
    """One-branch compressor for HCA.  H: [B, n, d] → [B, n//m, out_dim]."""
    B, n, _ = H.shape
    assert n % m == 0
    nb = n // m

    C_a = (H.float() @ w.W_a_KV.float()).reshape(B, nb, m, -1)   # [B, nb, m, od]
    Z_a = (H.float() @ w.W_a_Z.float()).reshape(B, nb, m, -1)
    Z = Z_a + w.B_a.float()                                       # broadcast [m, od]
    S = F.softmax(Z, dim=2)                                       # softmax over m
    return (S * C_a).sum(dim=2)                                   # [B, nb, od]


def compressor_overlap(
    H: torch.Tensor,
    w: CompressorWeightsOverlap,
    m: int,
) -> torch.Tensor:
    """Two-branch overlapped compressor for CSA (paper Eqs 9-12).

    H: [B, n, d] → [B, n//m, out_dim].
    """
    B, n, _ = H.shape
    assert n % m == 0
    nb = n // m

    Hf = H.float()
    C_a = (Hf @ w.W_a_KV.float()).reshape(B, nb, m, -1)           # [B, nb, m, od]
    C_b = (Hf @ w.W_b_KV.float()).reshape(B, nb, m, -1)
    Z_a = (Hf @ w.W_a_Z.float()).reshape(B, nb, m, -1)
    Z_b = (Hf @ w.W_b_Z.float()).reshape(B, nb, m, -1)

    # Block i takes C_b / Z_b from block i-1.  Block 0 pads C_b with 0,
    # Z_b with -inf so its softmax weight is zero.
    C_b_shift = F.pad(C_b[:, :-1], (0, 0, 0, 0, 1, 0))
    Z_b_shift = F.pad(Z_b[:, :-1], (0, 0, 0, 0, 1, 0), value=float("-inf"))

    Z_a_b = Z_a + w.B_a.float()
    Z_b_b = Z_b_shift + w.B_b.float()

    Z_cat = torch.cat([Z_a_b, Z_b_b], dim=2)                      # [B, nb, 2m, od]
    S = F.softmax(Z_cat, dim=2)
    S_a, S_b = S[:, :, :m], S[:, :, m:]
    return (S_a * C_a).sum(dim=2) + (S_b * C_b_shift).sum(dim=2)  # [B, nb, od]


# ── Lightning Indexer (Eqs 13-17) ──────────────────────────────────────


@dataclass
class IndexerWeights:
    W_IUQ: torch.Tensor    # [d_c, n_h_I * c_I]   indexer queries (Eq 14)
    W_w: torch.Tensor      # [d, n_h_I]           per-head weights (Eq 15)
    n_heads: int
    head_dim: int


def lightning_indexer_topk(
    H: torch.Tensor,
    c_Q: torch.Tensor,
    K_I_comp: torch.Tensor,
    w: IndexerWeights,
    m: int,
    top_k: int,
) -> torch.Tensor:
    """Per-query index scores → causal-masked top-k compressed-block ids.

    Mirrors article's `LightningIndexer.forward` (Eqs 13-17):
        q_I  = c_Q · W^IUQ           [B, n, n_h_I, c_I]                (Eq 14)
        w^I  = H   · W^w             [B, n, n_h_I]                     (Eq 15)
        I[t,s] = Σ_h w^I[t,h] · ReLU(q_I[t,h] · K^IComp[s])             (Eq 16)
        causal: token t may only see blocks s < ⌊t/m⌋                  (Eq 17)
    """
    B, n, _ = H.shape
    nb = K_I_comp.shape[1]
    n_h, c_I = w.n_heads, w.head_dim

    q_I = (c_Q.float() @ w.W_IUQ.float()).reshape(B, n, n_h, c_I)
    w_I = H.float() @ w.W_w.float()                               # [B, n, n_h]

    # Eq 16: per-head dot, ReLU, weighted sum
    scores = torch.einsum("bnhc,bsc->bnhs", q_I, K_I_comp.float())
    scores = F.relu(scores)
    index_scores = (w_I.unsqueeze(-1) * scores).sum(dim=2)        # [B, n, nb]

    # Eq 17: causal mask
    tok_pos = torch.arange(n, device=H.device)
    blk_idx = torch.arange(nb, device=H.device)
    causal = blk_idx.unsqueeze(0) < (tok_pos // m).unsqueeze(1)   # [n, nb]
    index_scores = index_scores.masked_fill(~causal.unsqueeze(0), float("-inf"))

    k = min(top_k, nb)
    _, sel = torch.topk(index_scores, k=k, dim=-1)                # [B, n, k]
    return sel


# ── Sliding-Window KV (paper §2.3.3) ───────────────────────────────────


def sliding_window_kv(
    H: torch.Tensor,
    n_win: int,
    W_kv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce uncompressed KV for the n_win most recent tokens per query.

    Returns:
      kv_sw:    [B, n, n_win, c]
      sw_pos:   [B, n, n_win]      absolute positions for RoPE
      sw_valid: [B, n, n_win]      True iff the entry is real (else padding)
    """
    B, n, _ = H.shape
    kv_all = H.float() @ W_kv.float()                              # [B, n, c]
    kv_pad = F.pad(kv_all, (0, 0, n_win - 1, 0))                  # [B, n+win-1, c]
    kv_sw = kv_pad.unfold(1, n_win, 1).permute(0, 1, 3, 2)        # [B, n, win, c]

    t = torch.arange(n, device=H.device).unsqueeze(1)
    off = torch.arange(n_win, device=H.device).unsqueeze(0)
    pos = t - (n_win - 1) + off                                   # [n, win]
    valid = pos >= 0
    pos = pos.clamp(min=0)

    kv_sw = kv_sw * valid.unsqueeze(0).unsqueeze(-1).float()
    return (kv_sw,
            pos.unsqueeze(0).expand(B, -1, -1).contiguous(),
            valid.unsqueeze(0).expand(B, -1, -1).contiguous())


# ── Core Attention with Sink (Eqs 18-19, 27) ───────────────────────────


@dataclass
class CoreAttnWeights:
    W_UQ: torch.Tensor          # [d_c, n_h * c]    Eq 18
    q_norm_w: torch.Tensor      # [c]               §2.3.3 QK norm
    k_norm_w: torch.Tensor      # [c]
    sink_logits: torch.Tensor   # [n_h]             Eq 27
    n_heads: int
    head_dim: int
    n_rope: int
    eps: float


def core_attn_with_sink(
    c_Q: torch.Tensor,
    KV: torch.Tensor,
    kv_pos: torch.Tensor,
    q_pos: torch.Tensor,
    w: CoreAttnWeights,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Shared-KV MQA with QK norm, partial RoPE, attention sink, inverse RoPE.

    Args:
        c_Q:    [B, n, d_c]        shared compressed latent queries
        KV:     [B, n, n_kv, c]    keys = values (MQA)
        kv_pos: [B, n, n_kv]       absolute positions for RoPE
        q_pos:  [B, n]             absolute query positions
        mask:   [B, n, n_kv] bool  True = valid
    Returns:
        o: [B, n, n_h, c]
    """
    B, n, _ = c_Q.shape
    n_kv = KV.shape[2]
    n_h, c, n_rope, eps = w.n_heads, w.head_dim, w.n_rope, w.eps

    # Eq 18: queries from shared latent
    q = (c_Q.float() @ w.W_UQ.float()).reshape(B, n, n_h, c)

    # Partial RoPE (last n_rope dims) on Q + KV
    q = apply_rope(q, q_pos.unsqueeze(2).expand(-1, -1, n_h), n_rope)
    KV_roped = apply_rope(KV.float(), kv_pos, n_rope)

    # QK norm (§2.3.3)
    q = rms_norm(q, w.q_norm_w, eps)
    KV_normed = rms_norm(KV_roped, w.k_norm_w, eps)

    # MQA logits, scaled
    logits = torch.einsum("bnhc,bnkc->bnhk", q, KV_normed) / math.sqrt(c)
    if mask is not None:
        logits = logits.masked_fill(~mask.unsqueeze(2), float("-inf"))

    # Eq 27: attention sink — exp(z'_h) added to softmax denominator per head.
    lmax = logits.amax(dim=-1, keepdim=True).detach()
    # If all keys masked → lmax = -inf; clamp to 0 to keep sink valid (sink alone wins).
    lmax = torch.where(torch.isfinite(lmax), lmax, torch.zeros_like(lmax))
    exp_l = torch.exp(logits - lmax)
    if mask is not None:
        exp_l = exp_l * mask.unsqueeze(2).float()
    sink_shift = w.sink_logits.float().reshape(1, 1, n_h) - lmax.squeeze(-1)
    exp_sink = torch.exp(sink_shift).unsqueeze(-1)
    attn = exp_l / (exp_l.sum(dim=-1, keepdim=True) + exp_sink)

    # Eq 19: weighted sum (value = RoPE'd KV; shared K/V)
    o = torch.einsum("bnhk,bnkc->bnhc", attn, KV_roped)

    # Inverse RoPE on output (§2.3.3 — produces relative position encoding)
    o = apply_rope(o, -q_pos.unsqueeze(2).expand(-1, -1, n_h), n_rope)
    return o


# ── Top-level layer assemblies (HCA / CSA) ─────────────────────────────


@dataclass
class _SharedHeads:
    """Pieces shared by both HCA and CSA layers."""
    W_DQ: torch.Tensor             # [d, d_c]   shared latent query (Eq 13)
    W_kv_sw: torch.Tensor          # [d, c]     SWA KV projection
    core_attn: CoreAttnWeights


@dataclass
class HCAWeights(_SharedHeads):
    kv_compressor: CompressorWeightsSimple


@dataclass
class CSAWeights(_SharedHeads):
    kv_compressor: CompressorWeightsOverlap
    indexer_compressor: CompressorWeightsOverlap
    indexer: IndexerWeights


@dataclass
class Inputs:
    H: torch.Tensor                # [B, n, d]


def hca_forward(
    weights: HCAWeights,
    inputs: Inputs,
    *,
    m_prime: int,
    n_win: int,
) -> torch.Tensor:
    """Heavily Compressed Attention: m'=128 simple compressor + dense MQA + SWA + sink.

    HCA isn't covered explicitly in the article, but per the paper §2.3
    and the production code it uses the *same* core-attn / SWA / sink
    machinery as CSA, only swapping the overlapped m=4 compressor for a
    simple m'=128 compressor and skipping the top-k selector — every
    query attends densely to *all* compressed blocks (whose count is
    small at m'=128).
    """
    H = inputs.H
    B, n, _ = H.shape

    # 1. Compress (simple, no overlap).
    C_comp = compressor_simple(H, weights.kv_compressor, m_prime)  # [B, nb, c]
    nb = C_comp.shape[1]

    # 2. Shared latent query.  Eq 13.
    c_Q = H.float() @ weights.W_DQ.float()                          # [B, n, d_c]

    # 3. Dense KV expansion: every query sees every compressed block.
    C_exp = C_comp.unsqueeze(1).expand(-1, n, -1, -1)               # [B, n, nb, c]

    # Block-center positions for RoPE.
    blk_pos = (torch.arange(nb, device=H.device).float() * m_prime
               + m_prime / 2).long()                                # [nb]
    blk_pos = blk_pos.unsqueeze(0).unsqueeze(0).expand(B, n, -1).contiguous()

    # Causal mask: query t sees compressed block s iff s < ⌊t/m'⌋.
    tok_pos = torch.arange(n, device=H.device)
    blk_idx = torch.arange(nb, device=H.device)
    blk_causal = (blk_idx.unsqueeze(0) < (tok_pos // m_prime).unsqueeze(1))  # [n, nb]
    blk_causal = blk_causal.unsqueeze(0).expand(B, -1, -1).contiguous()

    # 4. Sliding-window branch.
    kv_sw, sw_pos, sw_valid = sliding_window_kv(H, n_win, weights.W_kv_sw)

    # 5. Concatenate compressed + SWA into a single KV bank.
    KV_all = torch.cat([C_exp, kv_sw], dim=2)
    pos_all = torch.cat([blk_pos, sw_pos], dim=2)
    mask = torch.cat([blk_causal, sw_valid], dim=2)
    q_pos = tok_pos.unsqueeze(0).expand(B, -1).contiguous()

    # 6. Core attention (single softmax with sink in denominator).
    return core_attn_with_sink(c_Q, KV_all, pos_all, q_pos, weights.core_attn, mask)


# ── Decode-step reference (single query position, pre-built cache) ─────


@dataclass
class HCADecodeInputs:
    """Single decode-step inputs. Mirrors the SM100 kernel boundary.

    All tensors are torch tensors on the same device. dtype convention:
        Q             : bf16 — pre-projected, post q_norm, RoPE already applied
                                to the last `n_rope` dims.
        Kc / Kswa     : either bf16 (oracle) or torch.float8_e4m3fn (faithful).
                        Last dim is `c` (no rope; rope is the separate tail).
        Kc_scales /   : torch.uint8 with values interpreted as e8m0 bytes
        Kswa_scales     (= 2^(byte - 127)). One scale per 128-channel block.
                        None when Kc/Kswa are bf16.
        Kc_rope/swa   : bf16 [B, n_kv, n_rope], RoPE already applied.
    """
    Q:           torch.Tensor          # [B, n_h, c + n_rope]
    Kc:          torch.Tensor          # [B, M_cur, c]
    Kc_rope:     torch.Tensor          # [B, M_cur, n_rope]
    Kc_scales:   Optional[torch.Tensor]  # [B, M_cur, c // 128]  (e8m0 bytes)
    Kswa:        torch.Tensor          # [B, swa_len, c]
    Kswa_rope:   torch.Tensor          # [B, swa_len, n_rope]
    Kswa_scales: Optional[torch.Tensor]  # [B, swa_len, c // 128]
    sink_logits: torch.Tensor          # [n_h]
    sm_scale:    float                 # 1 / sqrt(c + n_rope), typically


def _dequant_block_fp8(latent_fp8: torch.Tensor,
                       scales_e8m0: torch.Tensor) -> torch.Tensor:
    """[..., c] fp8_e4m3 + [..., c/128] e8m0 bytes -> [..., c] bf16.

    Mirrors helpers.h `fp8x2_to_bf16x2_with_scale`: cast fp8 -> bf16, multiply
    by 2^(byte - 127). Done in bf16 to track kernel numerics.
    """
    *lead, c = latent_fp8.shape
    assert c % 128 == 0
    n_blocks = c // 128
    s_f32 = (scales_e8m0.to(torch.uint8).int() << 23).view(torch.float32)
    s_bf16 = s_f32.to(torch.bfloat16)                      # [..., n_blocks]
    bf = latent_fp8.to(torch.bfloat16)                     # [..., c]
    bf = bf.view(*lead, n_blocks, 128) * s_bf16.unsqueeze(-1)
    return bf.view(*lead, c)


def hca_decode_forward(
    inp: HCADecodeInputs,
    *,
    dtype: str = "fp8",
) -> torch.Tensor:
    """One decode step; returns O [B, n_h, c] in bf16.

    Modes:
      dtype="fp32" : everything in fp32; oracle.
      dtype="fp8"  : K dequanted via the same bf16 path the kernel uses; QK
                     and PV math then in fp32 for the accumulator (kernel
                     also accumulates fp32). Output narrowed to bf16.
    """
    assert dtype in ("fp32", "fp8")
    Q = inp.Q
    B, n_h, dq = Q.shape
    c = inp.Kc.shape[-1]
    n_rope = dq - c

    # Reconstruct full K (nope + rope) for both legs in bf16.
    if dtype == "fp32":
        Kc_nope   = inp.Kc.float()
        Kswa_nope = inp.Kswa.float()
    else:
        assert inp.Kc_scales is not None and inp.Kswa_scales is not None
        Kc_nope   = _dequant_block_fp8(inp.Kc,   inp.Kc_scales).float()
        Kswa_nope = _dequant_block_fp8(inp.Kswa, inp.Kswa_scales).float()

    Kc_full   = torch.cat([Kc_nope,   inp.Kc_rope.float()],   dim=-1)
    Kswa_full = torch.cat([Kswa_nope, inp.Kswa_rope.float()], dim=-1)
    KV = torch.cat([Kc_full, Kswa_full], dim=1)             # [B, M+W, c+n_rope]
    n_kv = KV.shape[1]

    # Logits in fp32 (matches kernel's fp32 P accumulator).
    logits = torch.einsum("bhc,bkc->bhk", Q.float(), KV) * float(inp.sm_scale)

    # Sink-augmented softmax (Eq 27).
    lmax = logits.amax(dim=-1, keepdim=True)
    lmax = torch.where(torch.isfinite(lmax), lmax, torch.zeros_like(lmax))
    exp_l = torch.exp(logits - lmax)
    sink_shift = inp.sink_logits.float().reshape(1, n_h) - lmax.squeeze(-1)
    denom = exp_l.sum(dim=-1, keepdim=True) + torch.exp(sink_shift).unsqueeze(-1)
    attn = exp_l / denom

    # PV: only the c-dim latent (matrix-absorbed V).
    o = torch.einsum("bhk,bkc->bhc", attn, KV[..., :c])
    return o.to(torch.bfloat16)


def csa_forward(
    weights: CSAWeights,
    inputs: Inputs,
    *,
    m: int,
    top_k: int,
    n_win: int,
) -> torch.Tensor:
    """Compressed Sparse Attention — paper §2.3.1 / Eqs 9-19, 27.

    Order matches the article's `CSALayer.forward` exactly.
    """
    H = inputs.H
    B, n, d = H.shape

    # 1. KV compressor (overlap, Eqs 9-12) → C^Comp.
    C_comp = compressor_overlap(H, weights.kv_compressor, m)        # [B, nb, c]
    nb = C_comp.shape[1]

    # 2. Indexer-key compressor (separate weights, same algorithm).
    K_I_comp = compressor_overlap(H, weights.indexer_compressor, m) # [B, nb, c_I]

    # 3. Shared latent query (Eq 13) + lightning indexer top-k (Eqs 14-17).
    c_Q = H.float() @ weights.W_DQ.float()                           # [B, n, d_c]
    sel_idx = lightning_indexer_topk(H, c_Q, K_I_comp, weights.indexer, m, top_k)
    k = sel_idx.shape[-1]

    # 4. Gather selected compressed KV.
    c = C_comp.shape[-1]
    idx_exp = sel_idx.unsqueeze(-1).expand(-1, -1, -1, c)
    C_exp_for_gather = C_comp.unsqueeze(1).expand(-1, n, -1, -1)
    C_sel = torch.gather(C_exp_for_gather, 2, idx_exp)              # [B, n, k, c]

    # Block-center positions for RoPE on selected blocks.
    blk_pos = (sel_idx.float() * m + m / 2).long()                  # [B, n, k]

    # 5. Sliding window (uncompressed, recent n_win tokens).
    kv_sw, sw_pos, sw_valid = sliding_window_kv(H, n_win, weights.W_kv_sw)

    # 6. Concat + mask (compressed entries valid only for tokens past block 0).
    KV_all = torch.cat([C_sel, kv_sw], dim=2)
    pos_all = torch.cat([blk_pos, sw_pos], dim=2)

    tok_idx = torch.arange(n, device=H.device)
    no_comp = (tok_idx // m == 0)                                   # [n]
    comp_ok = (~no_comp).unsqueeze(0).unsqueeze(-1).expand(B, -1, k).contiguous()
    mask = torch.cat([comp_ok, sw_valid], dim=2)
    q_pos = tok_idx.unsqueeze(0).expand(B, -1).contiguous()

    # 7. Core attention (Eqs 18-19, 27) with inverse RoPE on output.
    return core_attn_with_sink(c_Q, KV_all, pos_all, q_pos, weights.core_attn, mask)
