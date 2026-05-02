"""HCA candidate that delegates to the fp32 reference (sanity test).

Bench-against-itself: this should produce zero error vs the reference
and timing within noise.  Useful as the "golden" candidate when wiring
up new infrastructure.
"""
from bench.reference import hca_forward

NAME = "hca_torch_eager"
VARIANT = "hca"


def candidate(weights, inputs, *, spec):
    return hca_forward(weights, inputs, m_prime=spec.m, n_win=spec.n_win)
