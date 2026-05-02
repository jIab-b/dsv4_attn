"""Stub for a fully custom CSA implementation. Edit me!

The bench expects a callable:
    candidate(weights, inputs, *, spec) -> torch.Tensor [B=1, n, n_h, c]

Inputs (paper-faithful, single-sequence B=1):
    inputs.H            [1, n, d]  fp32  hidden states for the full sequence

Weights (CSAWeights):
    weights.W_DQ                [d, d_c]      shared latent query (Eq 13)
    weights.W_kv_sw             [d, c]        SWA KV projection
    weights.kv_compressor       CompressorWeightsOverlap (Eqs 9-12)
        .W_a_KV  [d, c]   .W_b_KV [d, c]
        .W_a_Z   [d, c]   .W_b_Z  [d, c]
        .B_a     [m, c]   .B_b    [m, c]
    weights.indexer_compressor  CompressorWeightsOverlap with c_I head
    weights.indexer  IndexerWeights
        .W_IUQ      [d_c, n_h_I * c_I]      indexer queries (Eq 14)
        .W_w        [d, n_h_I]              per-head weights (Eq 15)
        .n_heads / .head_dim
    weights.core_attn  CoreAttnWeights (same shape as HCA)

    spec : VariantSpec  (m=4, n_win=128, top_k=…, ...)

Output:
    [1, n, n_h, c] fp32 attention output (post inverse-RoPE, post sink merge).
"""
import torch

NAME = "csa_custom"
VARIANT = "csa"


def candidate(weights, inputs, *, spec):
    # TODO: replace with a custom CSA forward.  As shipped, this is a copy of
    # the reference so the smoke test passes.
    from bench.reference import csa_forward
    return csa_forward(weights, inputs, m=spec.m, top_k=spec.top_k, n_win=spec.n_win)
