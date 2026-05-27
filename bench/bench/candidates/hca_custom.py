"""Stub for a fully custom HCA implementation. Edit me!

The bench expects a callable:
    candidate(weights, inputs, *, spec) -> torch.Tensor [B=1, n, n_h, c]

Inputs (paper-faithful, single-sequence B=1):
    inputs.H            [1, n, d]  fp32  hidden states for the full sequence

Weights (HCAWeights):
    weights.W_DQ           [d, d_c]      shared latent query (Eq 13)
    weights.W_kv_sw        [d, c]        SWA KV projection
    weights.kv_compressor  CompressorWeightsSimple (HCA, no overlap)
        .W_a_KV  [d, c]
        .W_a_Z   [d, c]
        .B_a     [m_prime, c]
    weights.core_attn  CoreAttnWeights
        .W_UQ        [d_c, n_h * c]
        .q_norm_w    [c]
        .k_norm_w    [c]
        .sink_logits [n_h]   (added inside softmax denominator, Eq 27)
        .n_heads / .head_dim / .n_rope / .eps

    spec : VariantSpec  (m=128 here, n_win=128, ...)

Output:
    [1, n, n_h, c] fp32 attention output (post inverse-RoPE, post sink merge).

Tolerance: max_abs_err < spec.atol_fp32 (1e-4) for fp32 implementations.
"""
import torch

NAME = "hca_custom"
VARIANT = "hca"


def candidate(weights, inputs, *, spec):
    # TODO: replace with a custom HCA forward (Triton, CUDA, fused PyTorch...).
    # As shipped, this is a copy of the reference so the smoke test passes.
    from bench.reference import hca_forward
    return hca_forward(weights, inputs, m_prime=spec.m, n_win=spec.n_win)
