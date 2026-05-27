"""bench: benchmark harness for DeepSeek V4 attention variants (HCA, CSA).

Reference implementations follow SGLang's `Compressor` / `C4Indexer` /
`MQALayer` and vLLM's DeepSeek V4 MLA attention path, reduced to clean
PyTorch references so candidate kernels can be compared for correctness
and speed.
"""
__all__ = ["VARIANTS", "BenchResult"]

from .config import VARIANTS  # noqa: F401
from .bench_core import BenchResult  # noqa: F401
