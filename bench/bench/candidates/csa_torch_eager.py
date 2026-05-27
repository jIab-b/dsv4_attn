"""CSA candidate that delegates to the fp32 reference (sanity test)."""
from bench.reference import csa_forward

NAME = "csa_torch_eager"
VARIANT = "csa"


def candidate(weights, inputs, *, spec):
    return csa_forward(weights, inputs, m=spec.m, top_k=spec.top_k, n_win=spec.n_win)
