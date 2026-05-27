"""HCA candidate via torch.compile.

The paper-faithful reference uses unfold + F.pad + gather + RoPE that
some torch versions of Inductor can't trace (`'-X + (-Y)' is not
comparable`).  We try several modes; the *first call* may fall back to
eager and the candidate sticks with eager for the rest of the run so
the bench reports an honest eager timing instead of paying the
exception cost on every iteration.
"""
import torch

from bench.reference import hca_forward

NAME = "hca_torch_compiled"
VARIANT = "hca"

_runner = None  # will be either a torch.compile'd fn or the eager fn


def _build_runner(weights, inputs, *, m_prime, n_win):
    """Try compile modes top-to-bottom; first one that *runs* once wins."""
    for kwargs in (
        dict(mode="reduce-overhead", dynamic=False, fullgraph=False),
        dict(mode="default",         dynamic=False, fullgraph=False),
        dict(backend="aot_eager",    dynamic=False, fullgraph=False),
    ):
        try:
            fn = torch.compile(hca_forward, **kwargs)
            _ = fn(weights, inputs, m_prime=m_prime, n_win=n_win)
            return fn
        except Exception:
            continue
    return hca_forward  # eager fallback


def candidate(weights, inputs, *, spec):
    global _runner
    if _runner is None:
        _runner = _build_runner(weights, inputs,
                                m_prime=spec.m, n_win=spec.n_win)
    return _runner(weights, inputs, m_prime=spec.m, n_win=spec.n_win)
