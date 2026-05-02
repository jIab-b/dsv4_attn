"""bench: benchmark harness for DeepSeek V4 attention variants (HCA, CSA).

Reference implementations follow SGLang's `Compressor` / `C4Indexer` /
`MQALayer` (sgl-project/sglang@deepseek_v4) and vLLM's
`DeepseekCompressor` / `DeepseekV4MLAAttention` (vllm-project/vllm
PR#40760), reduced to a clean fp32 PyTorch eager form so candidate
kernels can be compared for both correctness and speed.

Variants:
    hca  - Heavily Compressed Attention (m'=128, dense MQA over
           compressed entries + sliding-window + attn_sink).
    csa  - Compressed Sparse Attention (m=4 + lightning-indexer top-k
           sparse MQA + sliding-window + attn_sink).

Shape preset matches the public DeepSeek-V4-Pro config (qk_rope=64,
head_dim=512, MQA single KV head). All sizes are reduced for a fast
benchmark; override via `--shape`.
"""
__all__ = ["VARIANTS", "BenchResult"]

from .config import VARIANTS  # noqa: F401
from .bench_core import BenchResult  # noqa: F401
