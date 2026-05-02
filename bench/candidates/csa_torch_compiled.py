"""CSA candidate via torch.compile (with sticky eager fallback)."""
import torch

from bench.reference import csa_forward

NAME = "csa_torch_compiled"
VARIANT = "csa"

_runner = None


def _build_runner(weights, inputs, *, m, top_k, n_win):
    for kwargs in (
        dict(mode="reduce-overhead", dynamic=False, fullgraph=False),
        dict(mode="default",         dynamic=False, fullgraph=False),
        dict(backend="aot_eager",    dynamic=False, fullgraph=False),
    ):
        try:
            fn = torch.compile(csa_forward, **kwargs)
            _ = fn(weights, inputs, m=m, top_k=top_k, n_win=n_win)
            return fn
        except Exception:
            continue
    return csa_forward


def candidate(weights, inputs, *, spec):
    global _runner
    if _runner is None:
        _runner = _build_runner(weights, inputs,
                                m=spec.m, top_k=spec.top_k, n_win=spec.n_win)
    return _runner(weights, inputs, m=spec.m, top_k=spec.top_k, n_win=spec.n_win)
