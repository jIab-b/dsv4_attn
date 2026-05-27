"""CUDA-graph HCA decode candidate.

Captures the same host path as hca_custom_decode: main HCA kernel followed by
combine, both on the current PyTorch stream. Repeated calls replay the graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import hca_custom_decode as base

NAME = "hca_custom_decode_graph"
VARIANT = "hca_decode"


@dataclass
class _GraphEntry:
    key: tuple
    graph: torch.cuda.CUDAGraph
    O: torch.Tensor
    partial_O: torch.Tensor
    partial_lse: torch.Tensor


_ENTRY: _GraphEntry | None = None


def _ptr(t):
    return 0 if t is None else int(t.data_ptr())


def _key(inputs, spec):
    B, n_h, c, n_rope, head_dim, M_cur, swa_len, num_splits = base._decode_shape(inputs, spec)
    return (
        int(inputs.Q.device.index or 0),
        B, n_h, c, n_rope, head_dim, M_cur, swa_len, num_splits,
        _ptr(inputs.Q), _ptr(inputs.Kc), _ptr(inputs.Kc_scales), _ptr(inputs.Kc_rope),
        _ptr(inputs.Kswa), _ptr(inputs.Kswa_scales), _ptr(inputs.Kswa_rope),
        _ptr(inputs.sink_logits),
        float(inputs.sm_scale),
    )


def _capture(inputs, *, spec, key):
    base._load()
    O, partial_O, partial_lse = base.make_workspace(inputs, spec=spec)
    graph = torch.cuda.CUDAGraph()

    torch.cuda.synchronize(inputs.Q.device)
    with torch.cuda.graph(graph):
        base.launch(inputs, spec=spec, O=O, partial_O=partial_O, partial_lse=partial_lse)

    return _GraphEntry(
        key=key,
        graph=graph,
        O=O,
        partial_O=partial_O,
        partial_lse=partial_lse,
    )


def candidate(inputs, *, spec):
    global _ENTRY

    if not inputs.Q.is_cuda:
        raise ValueError("hca_custom_decode_graph requires CUDA tensors")

    key = _key(inputs, spec)
    if _ENTRY is None or _ENTRY.key != key:
        _ENTRY = _capture(inputs, spec=spec, key=key)

    _ENTRY.graph.replay()
    return _ENTRY.O
