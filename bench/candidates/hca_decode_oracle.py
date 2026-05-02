"""HCA decode oracle: reference-as-candidate (zero numerical error).

Useful smoke test for the bench harness itself. Use with `dtype="fp32"`
or `dtype="fp8"`; either way it returns the reference's own output.
"""
from bench.reference import hca_decode_forward

NAME = "hca_decode_oracle"
VARIANT = "hca_decode"


def candidate(inputs, *, spec):
    # The runner picks dtype; we just dispatch to whichever the input was
    # built for. Detect via presence of scale tensors.
    dtype = "fp8" if inputs.Kc_scales is not None else "fp32"
    return hca_decode_forward(inputs, dtype=dtype)
