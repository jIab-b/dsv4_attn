"""Hyperparameters for the HCA / CSA benchmark variants.

Names follow Arjun Kocher's CSA reference (https://www.k-a.in/CSA.html)
which mirrors the DeepSeek-V4 paper:

    d       hidden size
    c       KV / attention head dimension
    d_c     query compression dimension (q-LoRA rank)
    n_h     attention query heads
    n_h_I   lightning-indexer heads
    c_I     indexer head dimension
    m       compression rate (CSA: 4, HCA: 128)
    n_win   sliding-window size
    n_rope  partial-RoPE dimensions (last n_rope of each head)
    top_k   sparse selection top-k (CSA only)

Defaults are the article's scaled-down config (works on a single B200 in
sub-millisecond).  Paper values cited inline.  The bench accepts overrides
per call (`--shape K=V`).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class VariantSpec:
    name: str

    # Sequence (single batch only; B is implicitly 1).
    n: int = 256                 # full sequence length (paper: up to 1M)

    # Attention dimensions
    d: int = 256                 # hidden size                       (paper: 4096)
    c: int = 64                  # KV / attention head dim           (paper: 192)
    d_c: int = 128               # query compression dim             (paper: ~512)
    n_h: int = 8                 # attention query heads             (paper: 128)
    n_rope: int = 16             # partial-RoPE dim                  (paper: 64)

    # Compressor
    m: int = 4                   # compression rate                  (CSA: 4, HCA: 128)
    n_win: int = 32              # sliding-window size               (paper: 128)

    # Lightning indexer (CSA only)
    n_h_I: int = 4               # indexer heads                     (paper: ~16)
    c_I: int = 32                # indexer head dim                  (paper: ~64)
    top_k: int = 8               # sparse top-k                      (paper: 512)

    # Numerics
    rms_norm_eps: float = 1e-6

    # Tolerances for correctness vs fp32 reference.
    atol_fp32: float = 1e-4
    rtol_fp32: float = 1e-4
    atol_bf16: float = 5e-2
    rtol_bf16: float = 5e-2

    @property
    def nb(self) -> int:
        return self.n // self.m

    def to_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, **kw) -> "VariantSpec":
        d = self.to_dict()
        d.update(kw)
        return VariantSpec(**d)


VARIANTS: dict[str, VariantSpec] = {
    # CSA: ratio-4 overlapped compressor + lightning indexer + top-k sparse MQA.
    "csa": VariantSpec(name="csa", m=4, n=512, n_win=64, top_k=16),
    # HCA: ratio-128 simple compressor + dense MQA over compressed entries.
    # Need n divisible by 128 → keep modest at 1024.
    "hca": VariantSpec(name="hca", m=128, n=1024, n_win=64),
}
